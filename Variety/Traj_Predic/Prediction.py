from typing import Dict,Callable

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import optim,Tensor,unsqueeze
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.autograd
import torch.nn as nn
from torchvision.models.resnet import resnet50,resnet18
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
import torch.nn.functional as F

from newdataset import MyTrainDataset, my_dataset_worker_init_func

from tqdm import tqdm

from Traj_Predic.l5kit.configs import load_config_data
from Traj_Predic.l5kit.data import LocalDataManager, ChunkedDataset
from Traj_Predic.l5kit.dataset import AgentDataset, EgoDataset
from Traj_Predic.l5kit.rasterization import build_rasterizer
from Traj_Predic.l5kit.evaluation import write_pred_csv, compute_metrics_csv, write_gt_csv, read_gt_csv
from Traj_Predic.l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from metrics import neg_multi_log_likelihood, time_displace, average_displacement_error_oracle, average_displacement_error_mean, final_displacement_error_oracle, final_displacement_error_mean
from Traj_Predic.l5kit.geometry import transform_points
from Traj_Predic.l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "E:/Downloads/lyft-motion-prediction-autonomous-vehicles"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("./agent_motion_config.yaml")
print(cfg)

if not cfg['mode']['load_mode']:    
    # ===== INIT DATASET
    train_cfg = cfg["train_data_loader"]
    rasterizer = build_rasterizer(cfg, dm)
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    print(len(train_dataset))
    print(train_dataset)
    print(train_dataset[0].keys())

    train_dataset = MyTrainDataset(cfg, dm, len(train_dataset),raster_mode = cfg["raster_params"]["raster_mode"], num_classes=cfg["model_params"]["num_classes"])
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=train_cfg["shuffle"], 
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        prefetch_factor = 2,
        pin_memory = True,
        persistent_workers=True,
        worker_init_fn=my_dataset_worker_init_func
    )

    # 基本参数
if cfg["train_params"]["device"] == 1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
torch.backends.cudnn.benchmark = True

epochs = cfg["train_params"]["epochs"]
latent_dim = cfg["model_params"]["latent_dim"]  # LSTM 的单元个数
encoder_fc = 64
num_layers = cfg["model_params"]["num_layers"]
bidirectional = cfg["model_params"]["bidirectional"]

encoder_length = cfg["model_params"]["history_num_frames"]
decoder_length = cfg["model_params"]["future_num_frames"]
num_encoder_tokens = 2
num_decoder_tokens = 2
z_dimension = 32
accumulation_steps = 5 # 梯度累积步数

num_classes = cfg["model_params"]["num_classes"] # 类数
modal_fc = latent_dim*(1+bidirectional) 

def neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"
    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    del gt, avails, max_value
    return torch.mean(error)

class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        # 定义序列编码器
        self.encoder = nn.LSTM(
            num_encoder_tokens, latent_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.encoder2 = nn.Linear(latent_dim*(1+bidirectional), encoder_fc)
#         self.encoder2 = nn.Linear(latent_dim*(1+bidirectional)+modal_fc, encoder_fc)
#         self.encoder_mean1 = nn.Linear(latent_dim*(1+bidirectional), 64)
        self.encoder_mean2 = nn.Linear(encoder_fc, z_dimension)
#         self.encoder_std1 = nn.Linear(latent_dim*(1+bidirectional), 32)
        self.encoder_std2 = nn.Linear(encoder_fc, z_dimension)

        # 定义序列解码器
        self.decoder = nn.LSTM(z_dimension*2, latent_dim, num_layers=num_layers,
                               bidirectional=bidirectional, batch_first=True)
        self.decoder_fc = nn.Linear(latent_dim*(1+bidirectional), 64)
        self.decoder_fc1 = nn.Linear(64, num_decoder_tokens*num_classes)
#         self.decoder_fc2 = nn.Linear(64, num_decoder_tokens)
#         self.decoder_fc3 = nn.Linear(64, num_decoder_tokens)
        self.decoder_confi = nn.Linear(num_decoder_tokens*num_classes, num_classes)

        # 定义图像编码器
        # load pre-trained Conv2D model
        self.resnet = resnet50(pretrained=True)
        # change input channels number to match the rasterizer's output
        num_history_channels = (
            cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        self.resnet.conv1 = nn.Conv2d(
            num_in_channels,
            self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size,
            stride=self.resnet.conv1.stride,
            padding=self.resnet.conv1.padding,
            bias=False,
        )
        # change output size to (X, Y) * number of future states
        num_targets = z_dimension * cfg["model_params"]["future_num_frames"]
        self.resnet.fc = nn.Linear(in_features=2048, out_features=512)
        self.encoder_mean3 = nn.Linear(512, num_targets)
        self.encoder_std3 = nn.Linear(512, num_targets)
        
        #定义采样器
#         self.sampler_fc1 = nn.Linear(3,1024)
#         self.sampler_fc2 = nn.Linear(1024,512)
#         self.sampler_fc3 = nn.Linear(512,z_dimension * cfg["model_params"]["future_num_frames"])
        
        #定义行为预测
        self.modal1 = nn.LSTM(
            num_encoder_tokens, latent_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.modal2 = nn.Linear(latent_dim*(1+bidirectional), num_classes)
        
    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        del eps
        return z

    def forward(self, data):
        inputs1 = torch.FloatTensor(data["history_positions"]).to(device)
#         yaw = torch.FloatTensor(data["history_yaws"]).to(device)
        if inputs1.dim() == 2:
            inputs1 = torch.unsqueeze(inputs1, 0)

        h0 = torch.autograd.Variable(torch.randn(
            num_layers*(1+bidirectional), inputs1.size()[0], latent_dim)).to(device)
        c0 = torch.autograd.Variable(torch.randn(
            num_layers*(1+bidirectional), inputs1.size()[0], latent_dim)).to(device)

        inputs2 = torch.FloatTensor(data["image"]).to(device)
        if inputs2.dim() == 3:
            inputs2 = torch.unsqueeze(inputs2, 0)

        out1, _ = self.encoder(inputs1, (h0, c0))
#         out1 = out1[:,-1,:]
#         out1 = torch.unsqueeze(out1, 1)
#         out1 = out1.expand(out1.size()[0],decoder_length,out1.size()[-1])
        out1 = F.relu(self.encoder2(out1), inplace=True)
        
        out_modal, _ = self.modal1(inputs1, (h0, c0))
        out_modal = F.softmax(self.modal2(out_modal[:, -1, :]), dim = -1)

#         mean1 = F.relu(self.encoder_mean1(out1), inplace=True)
        mean2 = F.relu(self.encoder_mean2(out1), inplace=True)
#         logstd1 = F.relu(self.encoder_std1(out1), inplace=True)
        logstd2 = F.relu(self.encoder_std2(out1), inplace=True)
        # prevent from poster vanish
#         logstd2 = torch.abs(logstd2) + 0.6

        z1 = self.noise_reparameterize(mean2, logstd2)
        z1 = z1[:, -1, :]
        z1 = torch.unsqueeze(z1, 1)
        z1 = z1.expand(z1.size()[0], decoder_length, z1.size()[-1])

        out12 = self.resnet(inputs2)
        mean3 = F.relu(self.encoder_mean3(out12), inplace=True)
        logstd3 = F.relu(self.encoder_std3(out12), inplace=True)
        z2 = self.noise_reparameterize(mean3, logstd3)
        z2 = z2.reshape(z1.size())
        z = torch.cat([z1, z2], -1)
        out2, _ = self.decoder(z)
        out2 = F.relu(self.decoder_fc(out2), inplace=True)

        out21 = F.relu(self.decoder_fc1(out2), inplace=True)
#         out22 = F.relu(self.decoder_fc2(out2), inplace=True)
#         out23 = F.relu(self.decoder_fc3(out2), inplace=True)
        confidences = F.softmax(self.decoder_confi(out21)[:, -1, :], dim=-1)
        confidences = F.softmax(confidences * out_modal, dim=-1)
        
        out3 = torch.split(out21,2,dim=-1)
        y_hat = torch.Tensor([]).to(device)
        for i in out3:
            i = torch.unsqueeze(i, 1) 
            y_hat=torch.cat([y_hat,i],dim=1)

        return y_hat, confidences, mean2, logstd2, mean3, logstd3


# def label_maker(yaw,target_yaw,num_classes):
# #     target_yaw = target_yaw.squeeze()
# #     print(yaw.size())
#     target_yaw = torch.max(target_yaw, dim=1)
#     target_yaw = target_yaw.squeeze(dim=1)
#     label = torch.zeros_like(yaw)
#     phi = 2*np.pi / num_classes
#     diff = target_yaw - yaw
#     for k in range(num_classes):
#         if np.pi-(k+1)*phi<diff<np.pi-k*phi:
#             label[k]=1
#     del phi, diff
#     del yaw, ind, one, label1, label2, label3
#     return w


def loss_function(y_hat, confidences, data, label, mean1, std1, mean2, std2):
    y_availabilities = data["target_availabilities"].to(device)
#     yaw = data["target_yaws"].to(device)
#     label = label_maker(yaw)
    y_true = data["target_positions"].to(device)
#     MSE = F.mse_loss(y_hat, y_true, reduction='none')
#     MSE = MSE * y_availabilities
#     MSE = MSE.mean()
    label = label.to(device)
    Cross = F.binary_cross_entropy_with_logits(confidences, label)
    NLL = neg_multi_log_likelihood_batch(
        y_true, y_hat, confidences, y_availabilities)
    # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
    var1 = torch.pow(torch.exp(std1), 2)
    var2 = torch.pow(torch.exp(std2), 2)
    KLD1 = -0.5 * torch.mean(1+torch.log(var1)-torch.pow(mean1, 2)-var1)
    KLD1 = torch.max(KLD1,torch.ones_like(KLD1))
    KLD2 = -0.5 * torch.mean(1+torch.log(var2)-torch.pow(mean2, 2)-var2)
    KLD2 = torch.max(KLD2,torch.ones_like(KLD2))
    KLD = KLD1 + KLD2
#     print('KLD: ',KLD,' NLL: ',NLL,' Cross: ', Cross)
    del KLD1, KLD2, var1, var2, y_availabilities, y_true
    return NLL, KLD, Cross


# 创建对象
cvae = CVAE().to(device)
# vae.load_state_dict(torch.load('./VAE_z2.pth'))
cvae_optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-4)

if not cfg['mode']['load_mode']:    
    # ==== TRAIN LOOP
    losses_avg = []
    for epoch in range(epochs):  # 进行多个epoch的训练
        tr_it = iter(train_dataloader)
        progress_bar = tqdm(range(len(train_dataloader)//cfg['scale']),position=0)
        losses_train = []
        cvae_optimizer.zero_grad(set_to_none = True)
        for i in progress_bar:
            try:
                data,label = next(tr_it)
            except StopIteration:
                tr_it = iter(train_dataloader)
                data,label = next(tr_it)
            cvae.train() # 设置为训练模式
            torch.set_grad_enabled(True)
            y_hat, confidences, mean1, std1, mean2, std2 = cvae(data)  # 输入
            if cfg["train_params"]["device"] == 1:
                with torch.cuda.amp.autocast():
                    NLL,KLD,Cross = loss_function(y_hat, confidences, data, label, mean1, std1, mean2, std2)
                    loss = NLL + (25)*KLD + 20*Cross
#                     if i + 1>= len(train_dataloader)//1:
#                         print(NLL,KLD,Cross)
            else:
                NLL,KLD,Cross = loss_function(y_hat, confidences, data, label, mean1, std1, mean2, std2)
                loss = NLL + (25)*KLD + 20*Cross
#                 if i + 1>= len(train_dataloader)//1:
#                     print(NLL,KLD,Cross)

            # Backward pass
            # 梯度累积模式
#             loss = loss / accumulation_steps
#             loss.backward() 
#             if (i+1) % accumulation_steps == 0:
#                 cvae_optimizer.step()
#                 cvae_optimizer.zero_grad(set_to_none = True)

            # 无梯度累积模式
            cvae_optimizer.zero_grad(set_to_none = True)
            loss.backward()
            cvae_optimizer.step()
            losses_train.append(loss.item())
            progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
            del data, y_hat, confidences, mean1, std1, mean2, std2, NLL, KLD, Cross, loss
        losses_avg.append(np.mean(losses_train))

eval_cfg = cfg["val_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer)
print(len(eval_dataset))
print(eval_dataset[0].keys())
# print(len(eval_dataset))

eval_dataset = MyTrainDataset(cfg, dm, len(eval_dataset),raster_mode = cfg["raster_params"]["raster_mode"])
eval_dataloader = DataLoader(
    eval_dataset,
    shuffle=eval_cfg["shuffle"], 
    batch_size=eval_cfg["batch_size"],
    num_workers=eval_cfg["num_workers"],
    prefetch_factor = 2,
    persistent_workers=True,
    pin_memory = True,
    worker_init_fn=my_dataset_worker_init_func
)
pred_path = "E:/Downloads/lyft-motion-prediction-autonomous-vehicles/pred.csv"
eval_gt_path = "E:/Downloads/lyft-motion-prediction-autonomous-vehicles/gt.csv"
cvae.load_state_dict(torch.load('E:/Downloads/lyft-motion-prediction-autonomous-vehicles/cvae.pth'))
print(len(eval_dataloader))

# ==== EVAL LOOP
cvae.eval()
torch.set_grad_enabled(False)
losses_test = []

# store information for evaluation
future_coords_offsets_pd = []
gt_coords_offsets_pd = []
timestamps = []
agent_ids = []
availability = []
confs = []
tr_it = iter(eval_dataloader)
progress_bar = tqdm(range(len(eval_dataloader)//cfg['scale']),position=0)

for i in progress_bar:
    try:
        data,_ = next(tr_it)
    except StopIteration:
        tr_it = iter(eval_dataloader)
        data,_ = next(tr_it)
    y_hat, confidences,mean1,std1,mean2,std2 = cvae(data)
#     if cfg["train_params"]["device"] == 1:
#         with torch.cuda.amp.autocast():
#             NLL,KLD,Cross = loss_function(y_hat, confidences, data, mean1, std1, mean2, std2)
#             loss = NLL + (25)*KLD + 20*Cross
#     losses_test.append(loss.item())
#     progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_test)}")
#     print(data)
    # convert agent coordinates into world offsets
    agents_coords = y_hat.detach().cpu().numpy()
    gt_coords = data['target_positions'].numpy()
    world_from_agents = data['world_from_agent'].numpy()
    centroids = data["centroid"].numpy()
    coords_off = []
    for i in range(num_classes):
        coords_off.append(transform_points(agents_coords[:,i,:,:], world_from_agents) - centroids[:, None, :2])
#         coords_offset2 = transform_points(agents_coords[:,1,:,:], world_from_agents) - centroids[:, None, :2]
#         coords_offset3 = transform_points(agents_coords[:,2,:,:], world_from_agents) - centroids[:, None, :2]
    coords_offset = np.stack([coords_offseti for coords_offseti in coords_off],1)
    gt_offset = transform_points(gt_coords, world_from_agents) - centroids[:, None, :2]
    
    future_coords_offsets_pd.append(np.stack(coords_offset))
    gt_coords_offsets_pd.append(np.stack(gt_offset))
    timestamps.append(data["timestamp"].numpy().copy())
    agent_ids.append(data["track_id"].numpy().copy())
    availability.append(data["target_availabilities"].numpy().copy())
    confs.append(confidences.detach().cpu().numpy().copy())

write_pred_csv(pred_path,
               timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(future_coords_offsets_pd),
               confs=np.concatenate(confs)
              )

write_gt_csv(eval_gt_path,timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(gt_coords_offsets_pd),avails=np.concatenate(availability))

metrics = compute_metrics_csv(eval_gt_path, pred_path, [
                              final_displacement_error_oracle, final_displacement_error_mean, average_displacement_error_oracle, average_displacement_error_mean, time_displace])
for metric_name, metric_mean in metrics.items():
    print(metric_name, metric_mean)
    if metric_name=="time_displace":
        FDE = metric_mean
print('FDE1s: {}, FDE3s: {}, FDE5s: {}, ADE1s: {}, ADE3s: {}, ADE5s: {} '.format(
    FDE[10//cfg["model_params"]["future_step_size"]-1], FDE[30//cfg["model_params"]["future_step_size"]-1], FDE[50//cfg["model_params"]["future_step_size"]-1], np.mean(FDE[:10//cfg["model_params"]["future_step_size"]]), np.mean(FDE[:30//cfg["model_params"]["future_step_size"]]), np.mean(FDE[:50//cfg["model_params"]["future_step_size"]])))

multi_vis = False
cvae.eval()
torch.set_grad_enabled(False)

# build a dict to retrieve future trajectories from GT
gt_rows = {}
for row in read_gt_csv(eval_gt_path):
    gt_rows[row["track_id"] + row["timestamp"]] = row["coord"]

eval_ego_dataset = EgoDataset(cfg, eval_zarr, rasterizer)
eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer)

for frame_number in range(99, len(eval_zarr.frames), 100):  # start from last frame of scene_0 and increase by 100
    agent_indices = eval_dataset.get_frame_indices(frame_number) 
    if not len(agent_indices):
        continue

    # get AV point-of-view frame
    data_ego = eval_ego_dataset[frame_number]
    im_ego = rasterizer.to_rgb(data_ego["image"].transpose(1, 2, 0))
    center = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
    
    predicted_positions = []
    predicted_positions1 = []
    predicted_positions2 = []
    predicted_positions3 = []
    target_positions = []

    if multi_vis == True:
        for v_index in agent_indices:
            data_agent = eval_dataset[v_index]
            out_net,confs,_,_,_,_ = cvae(data_agent)
            confs = confs.detach().cpu().numpy()
            print(confs)
            out_net1 = out_net[0][0]
            out_net2 = out_net[0][1]
            out_net3 = out_net[0][2]
            out_pos1 = out_net.reshape(-1, 2).detach().cpu().numpy()
            out_pos2 = out_net.reshape(-1, 2).detach().cpu().numpy()
            out_pos3 = out_net.reshape(-1, 2).detach().cpu().numpy()
            # store absolute world coordinates
            predicted_positions1.append(transform_points(out_pos1, data_agent["world_from_agent"]))
            predicted_positions2.append(transform_points(out_pos2, data_agent["world_from_agent"]))
            predicted_positions3.append(transform_points(out_pos3, data_agent["world_from_agent"]))
            # retrieve target positions from the GT and store as absolute coordinates
            track_id, timestamp = data_agent["track_id"], data_agent["timestamp"]
            target_positions.append(gt_rows[str(track_id) + str(timestamp)] + data_agent["centroid"][:2])


        # convert coordinates to AV point-of-view so we can draw them
        predicted_positions1 = transform_points(np.concatenate(predicted_positions1), data_ego["raster_from_world"])
        predicted_positions2 = transform_points(np.concatenate(predicted_positions2), data_ego["raster_from_world"])
        predicted_positions3 = transform_points(np.concatenate(predicted_positions3), data_ego["raster_from_world"])
        target_positions = transform_points(np.concatenate(target_positions), data_ego["raster_from_world"])

        draw_trajectory(im_ego, predicted_positions1, (34,222,79))
        draw_trajectory(im_ego, predicted_positions2, (220,235,21))
        draw_trajectory(im_ego, predicted_positions3, PREDICTED_POINTS_COLOR)
        draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)

        plt.imshow(im_ego)
        plt.show()

    else:
        for v_index in agent_indices:
            data_agent = eval_dataset[v_index]
            out_net,confs,_,_,_,_ = cvae(data_agent)
            confs = confs.detach().cpu().numpy()
    #         print(confs)
            out_net = out_net[0][np.argmax(confs)]
            out_pos = out_net.reshape(-1, 2).detach().cpu().numpy()
            # store absolute world coordinates
            predicted_positions.append(transform_points(out_pos, data_agent["world_from_agent"]))
            # retrieve target positions from the GT and store as absolute coordinates
            track_id, timestamp = data_agent["track_id"], data_agent["timestamp"]
            target_positions.append(gt_rows[str(track_id) + str(timestamp)] + data_agent["centroid"][:2])


        # convert coordinates to AV point-of-view so we can draw them
        predicted_positions = transform_points(np.concatenate(predicted_positions), data_ego["raster_from_world"])
        target_positions = transform_points(np.concatenate(target_positions), data_ego["raster_from_world"])

        draw_trajectory(im_ego, target_positions, TARGET_POINTS_COLOR)
        draw_trajectory(im_ego, predicted_positions, PREDICTED_POINTS_COLOR)


        plt.imshow(im_ego)
        plt.show()
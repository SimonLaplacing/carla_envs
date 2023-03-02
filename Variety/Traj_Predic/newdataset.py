from l5kit import dataset
from l5kit.data import ChunkedDataset
from l5kit.dataset import AgentDataset #, EgoDataset
from l5kit.rasterization import build_rasterizer
from torch.utils.data import get_worker_info
from torch.utils.data.dataset import Dataset
from typing import Optional
import numpy as np

class MyTrainDataset(Dataset):
    def __init__(self, mode, cfg, dm, length, agents_mask: Optional[np.ndarray] = None, raster_mode: Optional[np.int] = 0, num_classes = 3):
        self.mode = mode
        self.cfg = cfg
        self.dm = dm
        self.length = length
        self.has_init = False
        self.agents_mask = agents_mask
        self.raster_mode = raster_mode
        self.num_classes = num_classes
    def initialize(self, worker_id):
        print('initialize called with worker_id', worker_id)
        if self.raster_mode:
            rasterizer = build_rasterizer(self.cfg, self.dm)
        else:
            rasterizer = None
        train_cfg = self.cfg["train_data_loader"]
        val_cfg = self.cfg["val_data_loader"]
        train_zarr = ChunkedDataset(self.dm.require(train_cfg["key"])).open(cached=False)  # try to turn off cache
        val_zarr = ChunkedDataset(self.dm.require(val_cfg["key"])).open(cached=False)  # try to turn off cache
        if self.mode == 'train':
            self.dataset = AgentDataset(train_zarr, rasterizer=rasterizer, agents_mask=self.agents_mask)
        else:
            self.dataset = AgentDataset(val_zarr, rasterizer=rasterizer, agents_mask=self.agents_mask)
        self.has_init = True
    def reset(self):
        self.dataset = None
        self.has_init = False
    def __len__(self):
        # note you have to figure out the actual length beforehand since once the rasterizer and/or AgentDataset been constructed, you cannot pickle it anymore! So we can't compute the size from the real dataset. However, DataLoader require the len to determine the sampling.
        return self.length

    def get_label(self, cur_yaw, future_yaw):
        phi = 2*np.pi / self.num_classes
        label = np.zeros(self.num_classes)
        diff = future_yaw[-1] - cur_yaw[-1]
        for k in range(self.num_classes):
            if np.pi-(k+1)*phi<diff<np.pi-k*phi:
                label[k]=1
        del phi, diff
        return label

    def __getitem__(self, index):
        label = self.get_label(self.dataset[index]['history_yaws'],self.dataset[index]['target_yaws'])
        return self.dataset[index], label   

def my_dataset_worker_init_func(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.initialize(worker_id)
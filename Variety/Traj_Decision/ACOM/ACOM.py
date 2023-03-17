from ast import Delete
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from torch.distributions import Normal, MultivariateNormal
import copy
from memory_profiler import profile



################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

    return layer

class Actor_Critic_RNN(nn.Module):
    def __init__(self, args):
        super(Actor_Critic_RNN, self).__init__()
        self.args = args
        self.activate_func1 = [nn.CELU(), nn.Softsign()][args.use_tanh]  # Trick10: use tanh
        self.activate_func2 = nn.Softsign()
        # BEV
        # self.BEV_layer = resnet18(pretrained=True)
        # self.BEV_layer.fc = torch.nn.Linear(512,args.hidden_dim1)
        # self.Norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # self.BEV_layer = efficientnet_b0(pretrained=True)
        self.BEV_fc = nn.Linear(1000, args.hidden_dim1)
        # self.qq = 0

        # actor_init
        self.share_rnn_hidden = None
        self.share_fc11 = nn.Linear(args.state_dim+(args.max_agent_num-1)*args.action_dim, args.hidden_dim1)
        self.share_fc12 = nn.Linear(args.hidden_dim1, args.hidden_dim1)
        if args.use_gru:
            self.share_rnn = nn.GRU(2*args.hidden_dim1, 2*args.hidden_dim1, batch_first=True)
        elif args.use_lstm:
            self.share_rnn = nn.LSTM(2*args.hidden_dim1, 2*args.hidden_dim1, batch_first=True)
        self.share_fc2 = nn.Linear(2*args.hidden_dim1, args.hidden_dim2)
        self.mean_layer = nn.Linear(args.hidden_dim2, args.action_dim)
        # self.std_layer = nn.Linear(args.hidden_dim2, args.action_dim*args.action_dim)
        self.std_layer = nn.Linear(args.hidden_dim2, args.action_dim)

        # critic_init
        # self.critic_rnn_hidden = None
        # self.critic_fc11 = nn.Linear(args.state_dim+(args.max_agent_num-1)*args.action_dim, args.hidden_dim1)
        # self.critic_fc12 = nn.Linear(args.hidden_dim1, args.hidden_dim1)
        # if args.use_gru:
        #     self.critic_rnn = nn.GRU(2*args.hidden_dim1, 2*args.hidden_dim1, batch_first=True)
        # elif args.use_lstm:
        #     self.critic_rnn = nn.LSTM(2*args.hidden_dim1, 2*args.hidden_dim1, batch_first=True)
        # self.critic_fc2 = nn.Linear(2*args.hidden_dim1, args.hidden_dim2)
        self.critic_fc3 = nn.Linear(args.hidden_dim2, 1)

        # OM_init
        # self.OM_rnn_hidden = None
        self.OM_fc11 = nn.Linear(args.state_dim, args.hidden_dim1)
        # self.OM_fc12 = nn.Linear(args.hidden_dim1, args.hidden_dim1)
        # if args.use_gru:
        #     self.OM_rnn = nn.GRU(2*args.hidden_dim1, 2*args.hidden_dim1, batch_first=True)
        # elif args.use_lstm:
        #     self.OM_rnn = nn.LSTM(2*args.hidden_dim1, 2*args.hidden_dim1, batch_first=True)
        # self.OM_fc2 = nn.Linear(2*args.hidden_dim1, args.hidden_dim2)
        self.OMmean_layer = nn.Linear(args.hidden_dim2, (args.max_agent_num-1)*args.action_dim)
        # self.tanh = torch.nn.Hardtanh()
        # self.OMstd_layer = nn.Linear(args.hidden_dim2, args.action_dim)

        if args.use_orthogonal_init:
            # print("------use orthogonal init------")
            orthogonal_init(self.BEV_fc)
            orthogonal_init(self.share_fc11)
            orthogonal_init(self.share_fc12)
            orthogonal_init(self.share_fc2)
            orthogonal_init(self.mean_layer,gain=0.1)
            orthogonal_init(self.std_layer)
            # orthogonal_init(self.critic_fc11)
            # orthogonal_init(self.critic_fc12)
            # orthogonal_init(self.critic_fc2)
            orthogonal_init(self.critic_fc3)
            orthogonal_init(self.OM_fc11)
            # orthogonal_init(self.OM_fc12)
            # orthogonal_init(self.OM_fc2)
            orthogonal_init(self.OMmean_layer,gain=0.1)

            if self.args.use_gru or self.args.use_lstm:
                orthogonal_init(self.share_rnn)
                # orthogonal_init(self.critic_rnn)
                # orthogonal_init(self.OM_rnn)

    def actor(self, s, p, om_a):
        # p = p.numpy()
        # p = np.ascontiguousarray(p)
        p = torch.as_tensor(p, dtype=torch.float, device=device)
        p = self.BEV(p)
        p = p.view(s.shape[0],s.shape[1],self.args.hidden_dim1)
        s_ = torch.cat([s,om_a],-1)
        s_ = self.activate_func1(self.share_fc11(s_))
        p_ = self.activate_func1(self.share_fc12(p))
        t_ = torch.cat([s_,p_],-1)

        if self.args.use_gru or self.args.use_lstm:
            t_, self.share_rnn_hidden = self.share_rnn(t_, self.share_rnn_hidden)
        # output = self.activate_func(t_)
        output = self.activate_func1(self.share_fc2(t_))
        mean = self.activate_func2(self.mean_layer(output))
        std = torch.exp(self.activate_func2(self.std_layer(output)))  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        std = std.view(std.size()[0],std.size()[1],self.args.action_dim)
        std = self.args.init_std * torch.diag_embed(std)
        print('std: ', std)
        # std = std.view(std.size()[0],std.size()[1],self.args.action_dim,self.args.action_dim)
        # std = self.args.init_std * torch.tril(std)
        if torch.isnan(mean).any():             
            print('error1:',mean)
        # print(std,std.shape)
        # dist = MultivariateNormal(mean, scale_tril=std)  # Get the Gaussian distribution, Using scale_tril will be more efficient
        dist = MultivariateNormal(mean, covariance_matrix=std)
        return mean, std, dist

    def critic(self, s, p, om_a):
        # p = p.numpy()
        # p = np.ascontiguousarray(p)
        p = torch.as_tensor(p, dtype=torch.float, device=device)
        # p = p.view(-1,p.shape[-3],p.shape[-2],p.shape[-1])
        p = self.BEV(p)
        p = p.view(s.shape[0],s.shape[1],self.args.hidden_dim1)
        s_ = torch.cat([s,om_a],-1)
        s_ = self.activate_func1(self.share_fc11(s_))
        p_ = self.activate_func1(self.share_fc12(p))
        t_ = torch.cat([s_,p_],-1)
        
        if self.args.use_gru or self.args.use_lstm:
            t_, self.share_rnn_hidden = self.share_rnn(t_, self.share_rnn_hidden)
        # output = self.activate_func(t_)
        output = self.activate_func1(self.share_fc2(t_))
        value = self.critic_fc3(output)
        if torch.isnan(value).any():
            print('error2:',value)
        return value

    def OM(self, s, p):
        # p = p.numpy()
        # p = np.ascontiguousarray(p)
        p = torch.as_tensor(p, dtype=torch.float, device=device)
        # p = p.view(-1,p.shape[-3],p.shape[-2],p.shape[-1])
        p = self.BEV(p)
        p = p.view(s.shape[0],s.shape[1],self.args.hidden_dim1)
        s1_ = self.OM_fc11(s)
        s_ = self.activate_func1(s1_)
        p1_ = self.share_fc12(p)
        p_ = self.activate_func1(p1_)
        t_ = torch.cat([s_,p_],-1)

        if self.args.use_gru or self.args.use_lstm:
            t_, self.OM_rnn_hidden = self.share_rnn(t_, self.share_rnn_hidden)
        # output = self.activate_func(t_)
        output = self.activate_func1(self.share_fc2(t_))
        output = self.OMmean_layer(output)
        mean = self.activate_func2(output)
        if torch.isnan(mean).any():
            print('error3:',s1_,'\n',p1_,'\n',s_,'\n',p_)


        # scale = torch.tensor([0.03,0.0075,0.005]).to(device)
        # mean = scale*mean
        # std = torch.exp(torch.tanh(self.OMstd_layer(output)))  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        # std = self.args.init_std * torch.diag_embed(std)
        # dist = MultivariateNormal(mean, std)  # Get the Gaussian distribution
        return mean, None, None

    def BEV(self, p):
        # p = self.Norm(p/255)
        # p = self.BEV_layer(p)
        # # self.qq = p
        # p.detach_()
        p = self.activate_func1(self.BEV_fc(p))
        return p

class PPO_RNN:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_length_of_trajectory * args.max_episode/2
        self.lr = args.lr  # Learning rate of actor
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.mse_loss = nn.MSELoss()
        self.mask = torch.zeros((self.args.max_agent_num-1)*self.args.action_dim).to(device)
        self.mask[0:(self.args.agent_num-1)*self.args.action_dim] = 1
        self.ac = Actor_Critic_RNN(args).to(device)
        self.mse = nn.MSELoss()
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer = torch.optim.AdamW(self.ac.parameters(), lr=self.lr, eps=1e-5)
        else:
            self.optimizer = torch.optim.AdamW(self.ac.parameters(), lr=self.lr)

    def reset_rnn_hidden(self):
        if self.args.use_gru or self.args.use_lstm: 
            self.ac.share_rnn_hidden = None

    # @profile(stream=open('memory_profile.log','w+'))
    def choose_action(self, s, p, evaluate=False):
        # self.ac.eval()
        # mask = torch.zeros([self.args.max_agent_num, self.args.action_dim]).to(device)
        # mask[0:self.args.agent_num] = 1
        with torch.no_grad():
            # p = np.ascontiguousarray(p)
            # p = torch.as_tensor(p, dtype=torch.float).unsqueeze(0)
            # p = p.to(device)
            # p = self.ac.BEV(p)
            s = torch.as_tensor(s, dtype=torch.float).unsqueeze(0)
            s = torch.as_tensor(s, dtype=torch.float).unsqueeze(0)
            s = s.to(device)
            om_mean, _, _ = self.ac.OM(s,p)
            om_mean2 = om_mean*self.mask
            mean, _, dist = self.ac.actor(s, p, om_mean2)
            # om_a = om_dist.sample()  # Sample the action according to the probability distribution
            # om_a = torch.clamp(om_a, -1, 1)  # [-max,max]
            # om_mean_logprob = om_dist.log_prob(om_mean)  # The log probability density of the action
            if evaluate:
                a = torch.clamp(mean, -1, 1)  # [-max,max]
                a_logprob = dist.log_prob(a)
                return a.cpu().numpy().flatten(), None
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -1, 1)  # [-max,max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def get_value(self, s, p, _):
        # self.ac.eval()
        with torch.no_grad():
            # p = np.ascontiguousarray(p)
            # p = torch.as_tensor(p, dtype=torch.float).unsqueeze(0)
            # p = p.to(device)
            # p = self.ac.BEV(p)
            s = torch.as_tensor(s, dtype=torch.float).unsqueeze(0)
            s = torch.as_tensor(s, dtype=torch.float).unsqueeze(0)
            s = s.to(device)
            om_mean1, _, _ = self.ac.OM(s,p)
            om_mean = om_mean1*self.mask
            value = self.ac.critic(s, p, om_mean)
            # print('value:  ', value)
            # value = torch.squeeze(s,-2)
            return value.cpu().numpy().flatten()

    def train(self, total_steps, replay_buffer, ind):
        self.ac.train()
        if not self.args.shared_policy:
            OM_buffer = replay_buffer.copy()
            OM_buffer.pop(ind)
            batch = replay_buffer[ind].get_training_data(replay_buffer[ind].max_episode_len,OM_buffer)  # Get training data
            batch_size = self.batch_size
            mini_batch_size = self.mini_batch_size
        else:
            max_episode_len = max([x.max_episode_len for x in replay_buffer])
            # batch = replay_buffer.get_training_data(max_episode_len,OM_buffer)  # Get training data
            for j in range(self.args.agent_num):
                OM_buffer = replay_buffer.copy()
                OM_buffer.pop(j)
                print('1111:  ', len(replay_buffer),len(OM_buffer))
                if j == 0:
                    batch = replay_buffer[j].get_training_data(max_episode_len,OM_buffer)
                else:
                    batch2 = replay_buffer[j].get_training_data(max_episode_len,OM_buffer)  # Get training data
                    for key in batch:
                        batch[key] = torch.cat([batch[key],batch2[key]],0)
                batch_size = self.args.agent_num * self.batch_size
                mini_batch_size = self.args.agent_num * self.mini_batch_size
        # Optimize policy for K epochs:
        try:
            for _ in range(self.K_epochs):
                for index in BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, False):
                    # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                    if self.args.use_gru or self.args.use_lstm:
                        self.reset_rnn_hidden()
                    om_a_now1,_,_ = self.ac.OM(batch['s'][index].to(device),batch['p'][index].to(device))
                    om_a_now = om_a_now1*self.mask
                    _,_,dist_now = self.ac.actor(batch['s'][index].to(device),batch['p'][index].to(device),om_a_now)  # logits_now.shape=(mini_batch_size, max_episode_len, action_dim)
                    values_now = self.ac.critic(batch['s'][index].to(device),batch['p'][index].to(device),om_a_now).squeeze(-1)  # values_now.shape=(mini_batch_size, max_episode_len)

                    dist_entropy = dist_now.entropy()  # shape(mini_batch_size, max_episode_len)
                    a_logprob_now = dist_now.log_prob(batch['a'][index].to(device))  # shape(mini_batch_size, max_episode_len)
                    # a/b=exp(log(a)-log(b))
                    ratios = torch.exp(a_logprob_now - batch['a_logprob'][index].to(device))  # shape(mini_batch_size, max_episode_len)
                    # print('ratio: ',ratios.size())
                    # actor loss
                    surr1 = ratios * batch['adv'][index].to(device)
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch['adv'][index].to(device)
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size, max_episode_len)
                    actor_loss = (actor_loss * batch['active'][index].to(device)).sum() / batch['active'][index].to(device).sum()
                    # print('actor_loss: ',surr1,surr2,ratios,dist_entropy)

                    # critic_loss
                    critic_loss = self.mse(values_now,batch['v_target'][index].to(device))
                    critic_loss = (critic_loss * batch['active'][index].to(device)).sum() / batch['active'][index].to(device).sum()

                    # om_loss
                    om_loss = self.mse(om_a_now,batch['om_real_a'][index].to(device))
                    om_loss = (om_loss.mean(axis=-1) * batch['active'][index].to(device)).sum() / batch['active'][index].to(device).sum()

                    # Update
                    self.optimizer.zero_grad()
                    loss = actor_loss + critic_loss + 1.1*om_loss
                    print('all kinds of loss:            ', actor_loss,critic_loss,om_loss)
                    loss.backward()
                    if self.use_grad_clip:  # Trick 7: Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.8)
                    self.optimizer.step()
        except ValueError:
            s = batch['s'].detach().cpu().numpy()
            p = batch['p'].detach().cpu().numpy()
            om = om_a_now.detach().cpu().numpy().reshape(1,-1)
            # print(s,'\n',p,'\n',om)

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def save(self, checkpoint_path):
        torch.save(self.ac.state_dict(), checkpoint_path + 'ac.pkl')
   
    def load(self, checkpoint_path):
        self.ac.load_state_dict(torch.load(checkpoint_path + 'ac.pkl', map_location=lambda storage, loc: storage))


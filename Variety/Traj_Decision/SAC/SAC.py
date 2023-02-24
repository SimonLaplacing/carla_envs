# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
# from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
# from torchvision.models import resnet18, efficientnet_b0
# import torchvision.transforms as transforms
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

# class BEV(nn.Module):
#     def __init__(self, args):
#         self.args = args
#         super(BEV, self).__init__()
#         self.BEV_fc = nn.Linear(1000, self.args.hidden_dim1)
    
class Actor(nn.Module):
    def __init__(self, args, max_action=1):
        super(Actor, self).__init__()
        self.args = args
        self.max_action = max_action
        self.BEV_fc = nn.Linear(1000, self.args.hidden_dim1)
        self.l1 = nn.Linear(self.args.state_dim, self.args.hidden_dim1)
        self.l2 = nn.Linear(2*self.args.hidden_dim1, self.args.hidden_dim1)
        self.mean_layer = nn.Linear(self.args.hidden_dim1, self.args.action_dim)
        self.log_std_layer = nn.Linear(self.args.hidden_dim1, self.args.action_dim)

    def forward(self, x, p, evaluate=False, with_logprob=True):
        
        p = torch.as_tensor(p, dtype=torch.float, device=device)
        p = self.BEV_fc(p)

        x = F.relu(self.l1(x))
        x = torch.cat([x,p],-1)
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if evaluate:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = self.max_action * torch.tanh(a)  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.BEV_fc = nn.Linear(1000, self.args.hidden_dim1)
        # Q1
        self.l1 = nn.Linear(self.args.state_dim + self.args.action_dim, self.args.hidden_dim1)
        self.l2 = nn.Linear(2*self.args.hidden_dim1, self.args.hidden_dim1)
        self.l3 = nn.Linear(self.args.hidden_dim1, 1)
        # Q2
        self.l4 = nn.Linear(self.args.state_dim + self.args.action_dim, self.args.hidden_dim1)
        self.l5 = nn.Linear(2*self.args.hidden_dim1, self.args.hidden_dim1)
        self.l6 = nn.Linear(self.args.hidden_dim1, 1)

    def forward(self, s, p, a):
        s_a = torch.cat([s, a], -1)
        p = torch.as_tensor(p, dtype=torch.float, device=device)
        p = self.BEV_fc(p)

        q1 = F.relu(self.l1(s_a))
        q1 = torch.cat([q1,p],-1)
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = torch.cat([q2,p],-1)
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2
    


class ReplayBuffer(object):
    def __init__(self, args):
        self.args = args
        self.max_size = int(1e5)
        self.episode_num = 0
        self.size = 0
        self.s = np.zeros((self.max_size, self.args.state_dim))
        self.p = np.zeros((self.max_size, 1000))
        self.a = np.zeros((self.max_size, self.args.action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, self.args.state_dim))
        self.p_ = np.zeros((self.max_size, 1000))
        self.dw = np.zeros((self.max_size, 1))

    def store_transition(self, step, s, p, v, a, a_pro, r, s_, p_, dw):
        self.s[self.episode_num] = s
        self.p[self.episode_num] = p
        self.a[self.episode_num] = a
        self.r[self.episode_num] = r
        self.s_[self.episode_num] = s_
        self.p_[self.episode_num] = p_
        self.dw[self.episode_num] = dw
        self.episode_num = (self.episode_num + 1) % self.max_size  # When the 'episode_num' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_p = torch.tensor(self.p[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_p_ = torch.tensor(self.p_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_p, batch_a, batch_r, batch_s_, batch_p_, batch_dw


class SAC(object):
    def __init__(self, args, max_action=1):
        self.args = args
        self.max_action = max_action
        self.hidden_width = args.hidden_dim1  # The number of neurons in hidden layers of the neural network
        self.batch_size = args.mini_batch_size  # batch size
        self.GAMMA = args.gamma  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = args.lr  # learning rate
        self.adaptive_alpha = True  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -args.action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2
        self.actor = Actor(args).to(device)
        self.critic = Critic(args).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1.1*self.lr)

    def choose_action(self, s, p, evaluate=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        a, _ = self.actor(s, p, evaluate, False)  # When choosing actions, we do not need to compute log_pi
        return a.data.cpu().numpy().flatten(),_

    def train(self, void1, replay_buffer, ind):
        for _ in range(self.args.K_epochs):
            if self.args.shared_policy:
                for i in range(self.args.agent_num):
                    if i ==0:
                        batch_s, batch_p, batch_a, batch_r, batch_s_, batch_p_, batch_dw = replay_buffer[i].sample(self.batch_size)  # Sample a batch
                    else:
                        batch_s2, batch_p2, batch_a2, batch_r2, batch_s_2, batch_p_2, batch_dw2 = replay_buffer[i].sample(self.batch_size)  # Sample a batch
                        batch_s = torch.cat([batch_s,batch_s2],0)
                        batch_p = torch.cat([batch_p,batch_p2],0)
                        batch_a = torch.cat([batch_a,batch_a2],0)
                        batch_r = torch.cat([batch_r,batch_r2],0)
                        batch_s_ = torch.cat([batch_s_,batch_s_2],0)
                        batch_p_ = torch.cat([batch_p_,batch_p_2],0)
                        batch_dw = torch.cat([batch_dw,batch_dw2],0)
            else:
                batch_s, batch_p, batch_a, batch_r, batch_s_, batch_p_, batch_dw = replay_buffer[ind].sample(self.batch_size)  # Sample a batch

            with torch.no_grad():
                batch_a_, log_pi_ = self.actor(batch_s_.to(device),batch_p_.to(device))  # a' from the current policy
                # Compute target Q
                target_Q1, target_Q2 = self.critic_target(batch_s_.to(device), batch_p_.to(device), batch_a_.to(device))
                target_Q = batch_r.to(device) + self.GAMMA * (1 - batch_dw.to(device)) * (torch.min(target_Q1, target_Q2).to(device) - self.alpha.to(device) * log_pi_)

            # Compute current Q
            current_Q1, current_Q2 = self.critic(batch_s.to(device), batch_p.to(device), batch_a.to(device))
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Freeze critic networks so you don't waste computational effort
            for params in self.critic.parameters():
                params.requires_grad = False

            # Compute actor loss
            a, log_pi = self.actor(batch_s.to(device),batch_p.to(device))
            Q1, Q2 = self.critic(batch_s.to(device), batch_p.to(device), a)
            Q = torch.min(Q1, Q2)
            actor_loss = (self.alpha.to(device) * log_pi - Q).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze critic networks
            for params in self.critic.parameters():
                params.requires_grad = True

            # Update alpha
            if self.adaptive_alpha:
                # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
                alpha_loss = -(self.log_alpha.exp().to(device) * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()

            # Softly update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def save(self, checkpoint_path):
        torch.save(self.actor.state_dict(), checkpoint_path + 'actor.pkl')
        torch.save(self.critic.state_dict(), checkpoint_path + 'critic.pkl')
   
    def load(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(checkpoint_path + 'actor.pkl', map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load(checkpoint_path + 'critic.pkl', map_location=lambda storage, loc: storage))
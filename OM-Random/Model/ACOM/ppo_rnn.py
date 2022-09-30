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
        self.use_gru = args.use_gru
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        self.actor_rnn_hidden = None
        self.actor_fc1 = nn.Linear(args.state_dim, args.hidden_dim1)
        if args.use_gru:
            # print("------use GRU------")
            self.actor_rnn = nn.GRU(args.hidden_dim1, args.hidden_dim1, batch_first=True)
        elif args.use_lstm:
            # print("------use LSTM------")
            self.actor_rnn = nn.LSTM(args.hidden_dim1, args.hidden_dim1, batch_first=True)
        self.actor_fc2 = nn.Linear(args.hidden_dim1, args.hidden_dim2)
        self.mean_layer = nn.Linear(args.hidden_dim2, args.action_dim)
        self.log_std = nn.Parameter(self.args.init_logstd * torch.eye(args.action_dim, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        # self.log_std *= self.args.init_logstd
        self.critic_rnn_hidden = None
        self.critic_fc1 = nn.Linear(args.state_dim, args.hidden_dim1)
        if args.use_gru:
            self.critic_rnn = nn.GRU(args.hidden_dim1, args.hidden_dim1, batch_first=True)
        else:
            self.critic_rnn = nn.LSTM(args.hidden_dim1, args.hidden_dim1, batch_first=True)
        self.critic_fc2 = nn.Linear(args.hidden_dim1, args.hidden_dim2)
        self.critic_fc3 = nn.Linear(args.hidden_dim2, 1)

        if args.use_orthogonal_init:
            # print("------use orthogonal init------")
            orthogonal_init(self.actor_fc1)
            orthogonal_init(self.actor_fc2)
            orthogonal_init(self.mean_layer, gain=0.01)
            orthogonal_init(self.critic_fc1)
            orthogonal_init(self.critic_fc2)
            orthogonal_init(self.critic_fc3)
            if self.args.use_gru or self.args.use_lstm:
                orthogonal_init(self.actor_rnn)
                orthogonal_init(self.critic_rnn)

    def actor(self, s):
        s = self.activate_func(self.actor_fc1(s))
        if self.args.use_gru or self.args.use_lstm:
            s, self.actor_rnn_hidden = self.actor_rnn(s, self.actor_rnn_hidden)
        output = self.activate_func(s)
        output = self.activate_func(self.actor_fc2(output))
        mean = torch.tanh(self.mean_layer(output))
        # log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(self.log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = MultivariateNormal(mean, std)  # Get the Gaussian distribution
        return mean, std, dist

    def critic(self, s):
        s = self.activate_func(self.critic_fc1(s))
        if self.args.use_gru or self.args.use_lstm:
            s, self.critic_rnn_hidden = self.critic_rnn(s, self.critic_rnn_hidden)
        output = self.activate_func(s)
        output = self.activate_func(self.critic_fc2(output))
        value = self.critic_fc3(output)
        return value


class PPO_RNN:
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
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

        self.ac = Actor_Critic_RNN(args).to(device)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.lr)

    def reset_rnn_hidden(self):
        if self.args.use_gru or self.args.use_lstm: 
            self.ac.actor_rnn_hidden = None
            self.ac.critic_rnn_hidden = None
    # @profile(stream=open('memory_profile.log','w+'))
    def choose_action(self, s, evaluate=False):
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float).unsqueeze(0)
            s = torch.as_tensor(s, dtype=torch.float).unsqueeze(0)
            s = s.to(device)
            mean, _, dist = self.ac.actor(s)
            if evaluate:
                a = torch.clamp(mean, -1, 1)  # [-max,max]
                a_logprob = dist.log_prob(a)
                return a.cpu().numpy().flatten(), None
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -1, 1)  # [-max,max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def get_value(self, s):
        with torch.no_grad():
            s = torch.as_tensor(s, dtype=torch.float).unsqueeze(0)
            s = torch.as_tensor(s, dtype=torch.float).unsqueeze(0)
            s = s.to(device)
            value = self.ac.critic(s)
            # print('value:  ', value)
            # value = torch.squeeze(s,-2)
            return value.cpu().numpy().flatten()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # Get training data
        # batch = batch.to(device)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                # If use RNN, we need to reset the rnn_hidden of the actor and critic.
                if self.args.use_gru or self.args.use_lstm:
                    self.reset_rnn_hidden()
                _,_,dist_now = self.ac.actor(batch['s'][index].to(device))  # logits_now.shape=(mini_batch_size, max_episode_len, action_dim)
                values_now = self.ac.critic(batch['s'][index].to(device)).squeeze(-1)  # values_now.shape=(mini_batch_size, max_episode_len)

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

                # critic_loss
                critic_loss = (values_now - batch['v_target'][index].to(device)) ** 2
                critic_loss = (critic_loss * batch['active'][index].to(device)).sum() / batch['active'][index].to(device).sum()

                # Update
                self.optimizer.zero_grad()
                loss = actor_loss + critic_loss * 0.5
                loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()
        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def save(self, checkpoint_path):
        torch.save(self.ac.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.ac.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


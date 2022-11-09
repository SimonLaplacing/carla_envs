import torch
import numpy as np
import copy


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.use_adv_norm = args.use_adv_norm
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {'s': np.zeros([self.batch_size, self.episode_limit + 1, self.state_dim]),
                       'p': np.zeros([self.batch_size, self.episode_limit + 1, self.args.hidden_dim1]),
                       'v': np.zeros([self.batch_size, self.episode_limit + 1]),
                       'a': np.zeros([self.batch_size, self.episode_limit,self.action_dim]),
                       'a_logprob': np.zeros([self.batch_size, self.episode_limit]),
                       'r': np.zeros([self.batch_size, self.episode_limit]),
                       'dw': np.ones([self.batch_size, self.episode_limit]),  # Note: We use 'np.ones' to initialize 'dw'
                       'active': np.zeros([self.batch_size, self.episode_limit])
                       }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, s, p, v, a, a_logprob, r, dw):
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['p'][self.episode_num][episode_step] = p
        self.buffer['v'][self.episode_num][episode_step] = v
        self.buffer['a'][self.episode_num][episode_step] = a
        self.buffer['a_logprob'][self.episode_num][episode_step] = a_logprob
        self.buffer['r'][self.episode_num][episode_step] = r
        self.buffer['dw'][self.episode_num][episode_step] = dw

        self.buffer['active'][self.episode_num][episode_step] = 1.0

    def store_last_sv(self, episode_step, v, s, p):
        self.buffer['v'][self.episode_num][episode_step] = v
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['p'][self.episode_num][episode_step] = p
        self.episode_num += 1
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def get_adv(self,max_episode_len):
        # Calculate the advantage using GAE
        v = self.buffer['v'][:, :max_episode_len]
        v_next = self.buffer['v'][:, 1:max_episode_len + 1]
        r = self.buffer['r'][:, :max_episode_len]
        dw = self.buffer['dw'][:, :max_episode_len]
        active = self.buffer['active'][:, :max_episode_len]
        adv = np.zeros_like(r)  # adv.shape=(batch_size,max_episode_len)
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len)
            deltas = r + self.gamma * v_next * (1 - dw) - v
            for t in reversed(range(max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae  # gae.shape=(batch_size)
                adv[:, t] = gae
            v_target = adv + v  # v_target.shape(batch_size,max_episode_len)
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv_copy = copy.deepcopy(adv)
                adv_copy[active == 0] = np.nan  # 忽略掉active=0的那些adv
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
        return adv, v_target
    
    # def get_om_real_a(self,max_episode_len):
    #     s = self.buffer['s'][:,:max_episode_len]
    #     s_next = self.buffer['s'][:,1:max_episode_len + 1]
    #     delta = s_next - s
    #     om_real_a = delta[:,:,10:13]
    #     return om_real_a

    def get_training_data(self,max_episode_len, OM_buffer):
        adv, v_target = self.get_adv(max_episode_len)
        batch = {'s': torch.tensor(self.buffer['s'][:, :max_episode_len], dtype=torch.float32),
                 'p': torch.tensor(self.buffer['p'][:, :max_episode_len], dtype=torch.float32),
                 'a': torch.tensor(self.buffer['a'][:, :max_episode_len], dtype=torch.float32),  
                 'a_logprob': torch.tensor(self.buffer['a_logprob'][:, :max_episode_len], dtype=torch.float32),
                 'om_real_a': torch.tensor(OM_buffer.buffer['a'][:, :max_episode_len], dtype=torch.float32),
                 'active': torch.tensor(self.buffer['active'][:, :max_episode_len], dtype=torch.float32),
                 'adv': torch.tensor(adv, dtype=torch.float32),
                 'v_target': torch.tensor(v_target, dtype=torch.float32)}

        return batch

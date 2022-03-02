import glob
import os
import sys

import argparse
from itertools import count
from collections import namedtuple

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import IPPO2_ENVS
import time

try:
    sys.path.append(glob.glob('D:/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla



parser = argparse.ArgumentParser()

parser.add_argument('--c_tau',  default=1, type=float) # action软更新系数
parser.add_argument('--Alearning_rate', default=1e-4, type=float) # Actor学习率
parser.add_argument('--Clearning_rate', default=5e-4, type=float) # Critic学习率
parser.add_argument('--gamma', default=0.95, type=int) # discounted factor
parser.add_argument('--capacity', default=500, type=int) # replay buffer size
parser.add_argument('--batch_size', default=32, type=int) # mini batch size

parser.add_argument('--envs_create', default=False, type=bool) # 建立环境开关
parser.add_argument('--synchronous_mode', default=True, type=bool) # 同步模式开关
parser.add_argument('--no_rendering_mode', default=False, type=bool) # 无渲染模式开关
parser.add_argument('--fixed_delta_seconds', default=0.05, type=float) # 步长,步长建议不大于0.1，为0时代表可变步长

# parser.add_argument('--log_interval', default=50, type=int) # 网络保存间隔
# parser.add_argument('--load', default=False, type=bool) # 训练模式下是否load model
 
parser.add_argument('--max_episode', default=50000, type=int) # 仿真次数
parser.add_argument('--update_iteration', default = 10, type=int) # 网络迭代次数
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)

Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob'])

# 环境建立
create_envs = IPPO2_ENVS.Create_Envs(args.synchronous_mode,args.no_rendering_mode,args.fixed_delta_seconds) # 设置仿真模式以及步长

# 状态、动作空间定义
action_space = create_envs.get_action_space()
state_space = create_envs.get_state_space()
state_dim = len(state_space)
action_dim = len(action_space)
actor_num = 2

directory = './carla-IPPO2./'

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.action_head = nn.Linear(32, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.state_value = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPO(nn.Module):
    clip_param = 0.2
    max_grad_norm = 0.5

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('../exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.Alearning_rate)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), args.Clearning_rate)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % args.capacity == 0

    def update(self):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(device)

        R = 0
        Gt = []
        for r in reward[::-1]: # 返回包含原列表中所有元素的逆序列表
            R = r + args.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(device)
        #print("The agent is updating....")
        for i in range(args.update_iteration):
            for index in BatchSampler(SubsetRandomSampler(range(args.capacity)), args.batch_size, True):
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience

def main():
    ego_PPO = PPO()
    npc_PPO = PPO()
    ego_PPO.to(device)
    npc_PPO.to(device)
    if args.envs_create:
        client, world, blueprint_library = create_envs.connection()
    main_writer = SummaryWriter(directory)
    reward_list1 = []
    reward_list2 = []
    action_list1 = []
    action_list2 = []

    try:
        for i in range(args.max_episode):
            print('%dth time learning begins'%i)
            if args.envs_create:
                ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)
            # sim_time = 0  # 仿真时间
            # start_time = time.time()  # 初始时间
            state = state_space
            # egocol = sensor_list[0].get_collision_history()
            # npccol = sensor_list[1].get_collision_history()

            ego_action,ego_action_prob = ego_PPO.select_action(state)
            npc_action,npc_action_prob = npc_PPO.select_action(state)

            action = [int(ego_action),int(npc_action)]
            print('action is',action)
            reward1,reward2 = create_envs.get_reward(action)

            ego_trans = Transition(state, action, ego_action_prob, reward1)
            npc_trans = Transition(state, action, npc_action_prob, reward2)

            print('reward of %dth episode is %d,%d'%(i, reward1,reward2))
            reward_list1.append(reward1)
            reward_list2.append(reward2)
            action_list1.append(ego_action)
            action_list2.append(npc_action)
            main_writer.add_scalar('Reward/reward1', reward1, global_step=i)
            main_writer.add_scalar('Reward/reward2', reward2, global_step=i)
            if ego_PPO.store_transition(ego_trans):
                ego_PPO.update()
            if npc_PPO.store_transition(npc_trans):
                npc_PPO.update()
                print('parameters updated')            

            # time.sleep(1)
            
            # action
            # flag = 1
            # while state:  
            #     sim_time = time.time() - start_time
                
            #     create_envs.get_ego_step(ego_list[0],ego_action,sim_time,flag)       
            #     create_envs.get_npc_step(npc_list[0],npc_action,sim_time,flag)
            #     flag = 0
            #     if egocol_list[0] and npccol_list[0] or sim_time > 8: # 发生碰撞，重置场景
            #         state = state_space[0]

            # print(reward)
            # time.sleep(1)
            if args.envs_create:
                for x in sensor_list:
                    if x.sensor.is_alive:
                        x.sensor.destroy()            
                for x in ego_list:
                    if x.is_alive:
                        client.apply_batch([carla.command.DestroyActor(x)])
                for x in npc_list:
                    if x.is_alive:
                        client.apply_batch([carla.command.DestroyActor(x)])
                for x in obstacle_list:
                    if x.is_alive:
                        client.apply_batch([carla.command.DestroyActor(x)])

            print('Reset')
    finally:
        # rew = open(directory + 'reward.txt','w+')
        # rew.write(str(reward_list1))
        # rew.close()
        plt.subplot(2,  1,  1)  
        x = np.linspace(0,len(reward_list1),len(reward_list1))
        a=[]
        b=[]
        for i in range(len(reward_list1)):
            if i == 0:
                a.append(reward_list1[i])
                b.append(reward_list2[i])
            else:
                a.append((reward_list1[i]+a[-1]*i)/(i+1))
                b.append((reward_list2[i]+b[-1]*i)/(i+1))
        plt.plot(x,a,color='blue')
        plt.plot(x,b,color='red')
        plt.title('reward')
        # plt.subplot(3,  1,  2)
        # x2 = np.linspace(0,len(action_list1),len(action_list1))
        # plt.scatter(x2,action_list1)
        # plt.title('ego_action')
        # plt.subplot(3,  1,  3)
        # plt.scatter(x2,action_list2)
        # plt.title('npc_action')
        
        a11, a12, a21, a22 = [], [], [], []
        for i in range(len(action_list1)//args.capacity):
            a1 = [int(x) for x in action_list1[args.capacity*i:args.capacity*(i+1)-1]]
            a2 = [int(x) for x in action_list2[args.capacity*i:args.capacity*(i+1)-1]]
            a11.append((a1.count(0))/args.capacity)
            a12.append((a1.count(1))/args.capacity)
            a21.append((a2.count(0))/args.capacity)
            a22.append((a2.count(1))/args.capacity)
        plt.subplot(2,  1,  2)
        x2 = np.linspace(0,len(a11),len(a11))
        plt.plot(x2,a11)
        plt.plot(x2,a12)
        plt.plot(x2,a21)
        plt.plot(x2,a22)
        plt.legend(('ego_0','ego_1','npc_0','npc_1'))
        plt.title('action')
        plt.show()
        # 清洗环境
        if args.envs_create:
            print('Start Cleaning Envs')
            for x in sensor_list:
                if x.sensor.is_alive:
                    x.sensor.destroy()
            for x in ego_list:
                if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])
            for x in npc_list:
                if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])
            for x in obstacle_list:
                if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])
            print('all clean, simulation done!')


if __name__ == '__main__':
    main()

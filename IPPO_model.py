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
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import copy
import PPO_ENVS
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
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'

parser.add_argument('--c_tau',  default=1, type=float) # action软更新系数
parser.add_argument('--test_iteration', default=3, type=int) # 测试次数
parser.add_argument('--max_length_of_trajectory', default=200, type=int) # 最大仿真步数
parser.add_argument('--Alearning_rate', default=1e-4, type=float) # Actor学习率
parser.add_argument('--Clearning_rate', default=1e-3, type=float) # Critic学习率
parser.add_argument('--gamma', default=0.98, type=int) # discounted factor
parser.add_argument('--capacity', default=50, type=int) # replay buffer size
parser.add_argument('--batch_size', default=16, type=int) # mini batch size

parser.add_argument('--seed', default=False, type=bool) # 随机种子模式
parser.add_argument('--random_seed', default=1227, type=int) # 种子值

parser.add_argument('--synchronous_mode', default=True, type=bool) # 同步模式开关
parser.add_argument('--no_rendering_mode', default=False, type=bool) # 无渲染模式开关
parser.add_argument('--fixed_delta_seconds', default=0.05, type=float) # 步长,步长建议不大于0.1，为0时代表可变步长

parser.add_argument('--log_interval', default=50, type=int) # 网络保存间隔
parser.add_argument('--load', default=False, type=bool) # 训练模式下是否load model
 
parser.add_argument('--max_episode', default=2000, type=int) # 仿真次数
parser.add_argument('--update_iteration', default = 8, type=int) # 网络迭代次数
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)

Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state'])
TrainRecord = namedtuple('TrainRecord',['episode', 'reward'])

# 随机值
if args.seed:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

# 环境建立
if args.mode == 'train':
    create_envs = PPO_ENVS.Create_Envs(args.synchronous_mode,args.no_rendering_mode,args.fixed_delta_seconds) # 设置仿真模式以及步长
    print('==========training mode is activated==========')
elif args.mode == 'test':
    create_envs = PPO_ENVS.Create_Envs(args.synchronous_mode,False,args.fixed_delta_seconds)
    print('===========testing mode is activated===========')
else:
    raise NameError("wrong mode!!!")

# 状态、动作空间定义
action_space = create_envs.get_action_space()
state_space = create_envs.get_state_space()
state_dim = len(state_space)
action_dim = len(action_space)
actor_num = 2
max_action = torch.tensor(action_space[...,1]).float()
min_action = torch.tensor(action_space[...,0]).float()

directory = './carla-PPO./'

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128,16)
        self.mu_head = nn.Linear(16, action_dim)
        self.sigma_head = nn.Linear(16, action_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        mu = F.tanh(self.mu_head(x))
        sigma = F.relu(self.sigma_head(x)) + 1e-8

        return mu, sigma

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 16)
        self.state_value= nn.Linear(16, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.state_value(x)
        return value

class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().float()
        self.critic_net = Critic().float()
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.Alearning_rate)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), args.Clearning_rate)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            mu, sigma = self.actor_net(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-1, 1)
        return action, action_log_prob


    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net'+str(time.time())[:10],+'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net'+str(time.time())[:10],+'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % args.capacity == 0

    def update(self):
        self.training_step +=1

        state = torch.tensor([t.state for t in self.buffer ], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob.cpu().detach().numpy() for t in self.buffer], dtype=torch.float)

        reward = (reward - reward.mean())/(reward.std() + 1e-10)
        with torch.no_grad():
            target_v = reward + args.gamma * self.critic_net(next_state)

        advantage = (target_v - self.critic_net(state)).detach()
        for _ in range(args.update_iteration): # iteration ppo_epoch 
            for index in BatchSampler(SubsetRandomSampler(range(args.capacity)), args.batch_size, True):
                # epoch iteration, PPO core!!!
                mu, sigma = self.actor_net(state[index])
                n = Normal(mu, sigma)
                action_log_prob = n.log_prob(action[index])
                # print(action_log_prob,'-----',old_action_log_prob)
                ratio = torch.exp(action_log_prob - old_action_log_prob[index])
                
                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage[index]
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]

    # def save(self, name):
    #     # torch.save(self.actor.state_dict(), directory + name + '_actor.pth')
    #     # torch.save(self.critic.state_dict(), directory + name + '_critic.pth')
    #     torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net'+str(time.time())[:10],+'.pkl')
    #     torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net'+str(time.time())[:10],+'.pkl')
    #     print("====================================")
    #     print("Model has been saved...")
    #     print("====================================")

    # def load(self, name):
    #     self.actor.load_state_dict(torch.load(directory + name + '_actor.pth'))
    #     self.critic.load_state_dict(torch.load(directory + name + '_critic.pth'))
    #     print("====================================")
    #     print("model has been loaded...")
    #     print("====================================")

def main():
    ego_PPO = PPO()
    npc_PPO = PPO()
    # sim_time = args.fixed_delta_seconds  # 每步仿真时间
    client, world, blueprint_library = create_envs.connection()
    # main_writer = SummaryWriter(directory)
    ego_reward_list = []
    npc_reward_list = []

    try:
        if args.mode == 'test':
            ego_PPO.load('ego')
            npc_PPO.load('npc')
            for i in range(args.test_iteration):
                #---------动作决策----------
                ego_total_reward = 0
                npc_total_reward = 0
                print('---------%dth time learning begins---------'%i)
                ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)
                # start_time = time.time()  # 初始时间
                
                ego_transform = ego_list[0].get_transform()
                npc_transform = npc_list[0].get_transform()
                # ego_state = np.array([(ego_transform.location.x-120)/125,(ego_transform.location.y+375)/4,ego_transform.rotation.yaw/90,
                # (npc_transform.location.x-120)/125,(npc_transform.location.y+375)/4,npc_transform.rotation.yaw/90])
                # npc_state = np.array([(npc_transform.location.x-120)/125,(npc_transform.location.y+375)/4,npc_transform.rotation.yaw/90,
                # (ego_transform.location.x-120)/125,(ego_transform.location.y+375)/4,ego_transform.rotation.yaw/90])
                ego_state = np.array([(ego_transform.location.x-120)/125,(ego_transform.location.y+375)/4,ego_transform.rotation.yaw/90])
                npc_state = np.array([(npc_transform.location.x-120)/125,(npc_transform.location.y+375)/4,npc_transform.rotation.yaw/90])

                egosen_list = sensor_list[0]
                npcsen_list = sensor_list[1]

                for t in count():
                    ego_action = ego_PPO.select_action(ego_state)
                    npc_action = npc_PPO.select_action(npc_state)

                    # ego_action = np.array(ego_action).clip(min_action.cpu().numpy(), max_action.cpu().numpy())
                    # npc_action = np.array(npc_action).clip(min_action.cpu().numpy(), max_action.cpu().numpy())
                    if i<=args.max_episode/3:
                        c_tau = args.c_tau
                    elif i<=args.max_episode/2:
                        c_tau = args.c_tau/2
                    else:
                        c_tau = args.c_tau/3
                    create_envs.set_vehicle_control(ego_list[0], npc_list[0], ego_action, npc_action, args.c_tau, args.fixed_delta_seconds, t)
                    #---------和环境交互动作反馈---------
                    if args.synchronous_mode:
                        world.tick() # 客户端主导，tick
                        # print(world.tick())
                    else:
                        world.wait_for_tick() # 服务器主导，tick
                        # world_snapshot = world.wait_for_tick()
                        # print(world_snapshot.frame)
                        # world.on_tick(lambda world_snapshot: func(world_snapshot))
                    ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done = create_envs.get_vehicle_step(ego_list[0], npc_list[0], egosen_list, npcsen_list)
                    
                    ego_total_reward += ego_reward
                    npc_total_reward += npc_reward

                    if t >= args.max_length_of_trajectory: # 总结束条件
                        break
                    if ego_done or npc_done: # 结束条件
                        break
                    # period = time.time() - start_time                    
                    ego_state = ego_next_state
                    npc_state = npc_next_state

                ego_total_reward /= t
                npc_total_reward /= t
                print("Episode: {} step: {} ego Total Reward: {:0.3f} npc Total Reward: {:0.3f}".format(i+1, t, ego_total_reward, npc_total_reward))
                ego_reward_list.append(ego_total_reward)
                npc_reward_list.append(npc_total_reward)
                
                for x in sensor_list[0]:
                    if x.sensor.is_alive:
                        x.sensor.destroy()
                for x in sensor_list[1]:
                    if x.sensor.is_alive:
                        x.sensor.destroy()            
                for x in ego_list:
                    # if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])
                for x in npc_list:
                    # if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])
                for x in obstacle_list:
                    # if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])
                print('Reset')

        elif args.mode == 'train':
            # if args.load: ego_PPO.load('ego')
            # if args.load: npc_PPO.load('npc')
            for i in range(args.max_episode):
                ego_total_reward = 0
                npc_total_reward = 0
                print('------------%dth time learning begins-----------'%i)
                ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)

                # noise1 = OrnsteinUhlenbeckActionNoise(sigma=args.sigma,theta=0.5)
                # noise2 = OrnsteinUhlenbeckActionNoise(sigma=args.sigma,theta=0.5)

                ego_transform = ego_list[0].get_transform()
                npc_transform = npc_list[0].get_transform()
                # ego_state = np.array([(ego_transform.location.x-120)/125,(ego_transform.location.y+375)/4,ego_transform.rotation.yaw/90,
                # (npc_transform.location.x-120)/125,(npc_transform.location.y+375)/4,npc_transform.rotation.yaw/90])
                # npc_state = np.array([(npc_transform.location.x-120)/125,(npc_transform.location.y+375)/4,npc_transform.rotation.yaw/90,
                # (ego_transform.location.x-120)/125,(ego_transform.location.y+375)/4,ego_transform.rotation.yaw/90])
                ego_state = np.array([(ego_transform.location.x-120)/125,(ego_transform.location.y+375)/4,ego_transform.rotation.yaw/90])
                npc_state = np.array([(npc_transform.location.x-120)/125,(npc_transform.location.y+375)/4,npc_transform.rotation.yaw/90])

                egosen_list = sensor_list[0]
                npcsen_list = sensor_list[1]
                # start_time = time.time()

                for t in count():
                    #---------动作决策----------
                    ego_action,ego_action_log_prob = ego_PPO.select_action(ego_state)
                    npc_action,npc_action_log_prob = npc_PPO.select_action(npc_state)
                    
                    # ego_action = np.array(ego_action + np.random.normal(0, args.exploration_noise, size=(action_dim,))).clip(
                    #     min_action.cpu().numpy(), max_action.cpu().numpy()) #将输出tensor格式的action，因此转换为numpy格式
                    # npc_action = np.array(npc_action + np.random.normal(0, args.exploration_noise, size=(action_dim,))).clip(
                    #     min_action.cpu().numpy(), max_action.cpu().numpy()) #将输出tensor格式的action，因此转换为numpy格式
                    ego_action = np.array(ego_action) #将输出tensor格式的action，因此转换为numpy格式
                    npc_action = np.array(npc_action) #将输出tensor格式的action，因此转换为numpy格式
                    # period = time.time() - start_time
                    create_envs.set_vehicle_control(ego_list[0], npc_list[0], ego_action, npc_action, args.c_tau, args.fixed_delta_seconds, t)
                    #---------和环境交互动作反馈---------
                    if args.synchronous_mode:
                        world.tick() # 客户端主导，tick
                    #     # print(world.tick())
                    else:
                        world.wait_for_tick() # 服务器主导，tick
                        # world_snapshot = world.wait_for_tick()
                        # print(world_snapshot.frame)
                        # world.on_tick(lambda world_snapshot: func(world_snapshot))
                    ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done = create_envs.get_vehicle_step(ego_list[0], npc_list[0], egosen_list, npcsen_list)
                    ego_trans = Transition(ego_state, ego_action, ego_reward, ego_action_log_prob, ego_next_state)
                    npc_trans = Transition(npc_state, npc_action, npc_reward, npc_action_log_prob, npc_next_state)
                    # start_time = time.time()  # 开始时间
                    # print('period:',period)
                    # ego_next_action = ego_PPO.select_next_action(np.concatenate((ego_next_state, npc_next_state)))
                    # npc_next_action = npc_PPO.select_next_action(np.concatenate((npc_next_state, ego_next_state)))
                    # 数据储存
                    # ego_PPO.replay_buffer.push((np.concatenate((ego_state, npc_state)), np.concatenate((ego_next_state, npc_next_state)), 
                    #     np.concatenate((ego_action, npc_action)), np.concatenate((ego_next_action, npc_next_action)), ego_reward, ego_done))
                    # npc_PPO.replay_buffer.push((np.concatenate((npc_state, ego_state)), np.concatenate((npc_next_state, ego_next_state)), 
                    #     np.concatenate((npc_action, ego_action)), np.concatenate((npc_next_action, ego_next_action)), npc_reward, npc_done))

                    ego_state = ego_next_state
                    npc_state = npc_next_state
                    
                    ego_total_reward += ego_reward
                    npc_total_reward += npc_reward

                    if t >= args.max_length_of_trajectory: # 总结束条件
                        break
                    if ego_done or npc_done: # 结束条件
                        break

                ego_total_reward /= t
                npc_total_reward /= t
                # main_writer.add_scalar('reward/ego_reward', ego_total_reward, global_step=i)
                # main_writer.add_scalar('reward/npc_reward', npc_total_reward, global_step=i)
                # ego_reward_list.append(ego_total_reward)
                # npc_reward_list.append(npc_total_reward)
                print("Episode: {} step: {} ego_Total_Reward: {:0.3f} npc_Total_Reward: {:0.3f}".format(i+1, t, ego_total_reward, npc_total_reward))
                # if i % args.update_interval == 0:
                if ego_PPO.store_transition(ego_trans):
                    ego_PPO.update()
                if npc_PPO.store_transition(npc_trans):
                    npc_PPO.update()
                # if i % args.log_interval == 0:
                #     ego_PPO.save('ego')
                #     npc_PPO.save('npc')

                for x in sensor_list[0]:
                    if x.sensor.is_alive:
                        x.sensor.destroy()
                for x in sensor_list[1]:
                    if x.sensor.is_alive:
                        x.sensor.destroy()            
                for x in ego_list:
                    # if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])
                for x in npc_list:
                    # if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])
                for x in obstacle_list:
                    # if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])
                print('Reset')

    finally:
        # 清洗环境
        print('Start Cleaning Envs')
        for x in sensor_list[0]:
            if x.sensor.is_alive:
                x.sensor.destroy()
        for x in sensor_list[1]:
            if x.sensor.is_alive:
                x.sensor.destroy()
        for x in ego_list:
            # if x.is_alive:
            client.apply_batch([carla.command.DestroyActor(x)])
        for x in npc_list:
            # if x.is_alive:
            client.apply_batch([carla.command.DestroyActor(x)])
        for x in obstacle_list:
            # if x.is_alive:
            client.apply_batch([carla.command.DestroyActor(x)])
        print('all clean, simulation done!')


if __name__ == '__main__':
    main()

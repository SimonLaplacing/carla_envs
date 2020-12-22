import glob
import os
import sys

import argparse
from itertools import count

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import copy
import DDPG_ENVS
import time

try:
    sys.path.append(glob.glob('D:/CARLA_0.9.10-Pre_Win/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla



parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument('--tau',  default=0.005, type=float) # 目标网络软更新系数
parser.add_argument('--target_update_interval', default=4, type=int) # 目标网络更新间隔
parser.add_argument('--warmup_step', default=10, type=int) # 网络参数训练更新预备回合数
parser.add_argument('--test_iteration', default=10, type=int) # 测试次数
parser.add_argument('--max_length_of_trajectory', default=300, type=int) # 最大仿真步数
parser.add_argument('--Alearning_rate', default=5e-5, type=float) # Actor学习率
parser.add_argument('--Clearning_rate', default=5e-4, type=float) # Critic学习率
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=100000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=50, type=int) # mini batch size

parser.add_argument('--seed', default=False, type=bool) # 随机种子模式
parser.add_argument('--random_seed', default=1227, type=int) # 种子值

parser.add_argument('--no_rendering_mode', default=True, type=bool) # 无渲染模式开关
parser.add_argument('--fixed_delta_seconds', default=0.05, type=float) # 无渲染模式下步长,步长建议不大于0.1

parser.add_argument('--log_interval', default=50, type=int) # 目标网络保存间隔
parser.add_argument('--load', default=False, type=bool) # 训练模式下是否load model
parser.add_argument('--exploration_noise', default=0.3, type=float) # 探索偏移分布 
parser.add_argument('--max_episode', default=500, type=int) # num of games
parser.add_argument('--update_iteration', default = 5, type=int) # 网络迭代次数
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)

# 随机值
if args.seed:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

# 环境建立
if args.mode == 'train':
    create_envs = DDPG_ENVS.Create_Envs(args.no_rendering_mode,args.fixed_delta_seconds) # 设置仿真模式以及步长
    print('==========training mode is activated==========')
elif args.mode == 'test':
    create_envs = DDPG_ENVS.Create_Envs()
    print('===========testing mode is activated===========')
else:
    raise NameError("mode wrong!!!")

# 状态、动作空间定义
action_space = create_envs.get_action_space()
state_space = create_envs.get_state_space()
state_dim = len(state_space)
action_dim = len(action_space)
max_action = torch.tensor(action_space[...,1]).float().to(device)
min_action = torch.tensor(action_space[...,0]).float().to(device)

directory = './carla-' + 'DDPG' +'./'

class Replay_buffer():

    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 300)
        # self.l1.weight.data.normal_(1,1)
        self.l2 = nn.Linear(300, 100)
        # self.l2.weight.data.normal_(1,1)
        self.l3 = nn.Linear(100, action_dim)
        # self.l3.weight.data.normal_(1,1)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 300)
        # self.l1.weight.data.normal_(1,1)
        self.l2 = nn.Linear(300 , 100)
        # self.l2.weight.data.normal_(1,1)
        self.l3 = nn.Linear(100, 1)
        # self.l3.weight.data.normal_(1,1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.Alearning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.Clearning_rate)
        
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self,curr_epi):

        if curr_epi > args.warmup_step:
            for it in range(args.update_iteration):
                # Sample replay buffer
                x, y, u, r, d = self.replay_buffer.sample(args.batch_size)

                state = torch.FloatTensor(x).to(device)
                action = torch.FloatTensor(u).to(device)
                next_state = torch.FloatTensor(y).to(device)
                done = torch.FloatTensor(1-d).to(device)
                reward = torch.FloatTensor(r).to(device)

                # Compute the target Q value
                target_Q = self.critic_target(next_state, self.actor_target(next_state))
                target_Q = reward + (done * args.gamma * target_Q).detach()

                # Get current Q estimate
                current_Q = self.critic(state, action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q, target_Q)
                self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Compute actor loss
                actor_loss = -self.critic(state, self.actor(state)).mean()
                self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                if it % args.target_update_interval == 0:
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                self.num_actor_update_iteration += 1
                self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def main():
    ego_DDPG = DDPG(state_dim, action_dim, max_action)
    npc_DDPG = DDPG(state_dim, action_dim, max_action)
    sim_time = args.fixed_delta_seconds  # 每步仿真时间
    client, world, blueprint_library = create_envs.connection()
    reward_list = []

    try:
        if args.mode == 'test':
            ego_DDPG.load()
            npc_DDPG.load()
            for i in range(args.test_iteration):
                ego_total_reward = 0
                npc_total_reward = 0
                print('---------%dth time learning begins---------'%i)
                ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)
                start_time = time.time()  # 初始时间
                
                ego_state = ego_list[0].get_transform()
                ego_state = np.array([ego_state.location.x,ego_state.location.y,ego_state.rotation.yaw])
                npc_state = npc_list[0].get_transform()
                npc_state = np.array([npc_state.location.x,npc_state.location.y,npc_state.rotation.yaw])

                egocol_list = sensor_list[0].get_collision_history()
                npccol_list = sensor_list[1].get_collision_history()

                for t in count():
                    ego_action = ego_DDPG.select_action(ego_state)
                    npc_action = npc_DDPG.select_action(npc_state)

                    ego_action = np.array(ego_action + np.random.normal(0, 0, size=(action_dim,))).clip(
                        min_action.cpu().numpy(), max_action.cpu().numpy())
                    npc_action = np.array(npc_action + np.random.normal(0, 0, size=(action_dim,))).clip(
                        min_action.cpu().numpy(), max_action.cpu().numpy())
                    ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done = create_envs.get_vehicle_step(ego_list[0], npc_list[0], egocol_list, npccol_list,ego_action, npc_action, sim_time)
                    
                    ego_total_reward += ego_reward
                    npc_total_reward += npc_reward

                    if t >= args.max_length_of_trajectory: # 总结束条件
                        print("Episode: {} step: {} ego Total Reward: {:0.2f} npc Total Reward: {:0.2f}".format(i, t, ego_total_reward, npc_total_reward))
                        break
                    if ego_done: # ego结束条件ego_done
                        print("Episode: {} step: {} ego Total Reward: {:0.2f} npc Total Reward: {:0.2f}".format(i, t, ego_total_reward, npc_total_reward))
                        break
                    if npc_done: # npc结束条件npc_done
                        print("Episode: {} step: {} ego Total Reward: {:0.2f} npc Total Reward: {:0.2f}".format(i, t, ego_total_reward, npc_total_reward))
                        break
                    period = time.time() - start_time                    
                    ego_state = ego_next_state
                    npc_state = npc_next_state
                    reward_list.append(ego_total_reward)
                
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

        elif args.mode == 'train':
            if args.load: ego_DDPG.load()
            if args.load: npc_DDPG.load()
            for i in range(args.max_episode):
                ego_total_reward = 0
                npc_total_reward = 0
                step =0
                print('------------%dth time learning begins-----------'%i)
                ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)
                start_time = time.time()  # 开始时间

                ego_state = ego_list[0].get_transform()
                ego_state = np.array([ego_state.location.x,ego_state.location.y,ego_state.rotation.yaw])
                npc_state = npc_list[0].get_transform()
                npc_state = np.array([npc_state.location.x,npc_state.location.y,npc_state.rotation.yaw])
                egocol_list = sensor_list[0].get_collision_history()
                npccol_list = sensor_list[1].get_collision_history()
                for t in count():
                    ego_action = ego_DDPG.select_action(ego_state)
                    npc_action = npc_DDPG.select_action(npc_state)
                    if i<=args.max_episode/3:
                        noise = args.exploration_noise
                    elif i<=args.max_episode/2:
                        noise = args.exploration_noise/2
                    else:
                        noise = args.exploration_noise/3
                    ego_action = np.array(ego_action + np.random.normal(0, args.exploration_noise, size=(action_dim,))).clip(
                        min_action.cpu().numpy(), max_action.cpu().numpy()) #将输出tensor格式的action，因此转换为numpy格式
                    npc_action = np.array(npc_action + np.random.normal(0, args.exploration_noise, size=(action_dim,))).clip(
                        min_action.cpu().numpy(), max_action.cpu().numpy()) #将输出tensor格式的action，因此转换为numpy格式

                    ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done = create_envs.get_vehicle_step(ego_list[0], npc_list[0], egocol_list, npccol_list,ego_action, npc_action, sim_time)

                    # 数据储存
                    ego_DDPG.replay_buffer.push((ego_state, ego_next_state, ego_action, ego_reward, ego_done))
                    npc_DDPG.replay_buffer.push((npc_state, npc_next_state, npc_action, npc_reward, npc_done))
                    ego_state = ego_next_state
                    npc_state = npc_next_state
                    
                    ego_total_reward += ego_reward
                    npc_total_reward += npc_reward

                    if t >= args.max_length_of_trajectory: # 总结束条件
                        break
                    if ego_done: # ego结束条件ego_done
                        break
                    if npc_done: # npc结束条件npc_done
                        break
                    # period = time.time() - start_time
                    # print('period:',period)
                ego_total_reward /= t
                npc_total_reward /= t
                reward_list.append(ego_total_reward)
                print("Episode: {} step: {} ego_Total_Reward: {:0.2f} npc_Total_Reward: {:0.2f}".format(i, t, ego_total_reward, npc_total_reward))
                ego_DDPG.update(curr_epi=i)
                npc_DDPG.update(curr_epi=i)
                # "Total T: %d Episode Num: %d Episode T: %d Reward: %f
                if i % args.log_interval == 0:
                    ego_DDPG.save()
                    npc_DDPG.save()

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
        # reward图
        x=np.arange(len(reward_list))
        y=reward_list
        plt.plot(x,y)
        plt.show()
        # 清洗环境
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

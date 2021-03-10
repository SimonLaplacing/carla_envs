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
    sys.path.append(glob.glob('D:/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla



parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument('--tau',  default=0.01, type=float) # 目标网络软更新系数
parser.add_argument('--c_tau',  default=1, type=float) # action软更新系数
parser.add_argument('--update_interval', default=4, type=int) # 目标网络更新间隔
parser.add_argument('--target_update_interval', default=8, type=int) # 目标网络更新间隔
parser.add_argument('--warmup_step', default=8, type=int) # 网络参数训练更新预备回合数
parser.add_argument('--test_iteration', default=10, type=int) # 测试次数
parser.add_argument('--max_length_of_trajectory', default=150, type=int) # 最大仿真步数
parser.add_argument('--Alearning_rate', default=1e-3, type=float) # Actor学习率
parser.add_argument('--Clearning_rate', default=1e-3, type=float) # Critic学习率
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=100000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=128, type=int) # mini batch size

parser.add_argument('--seed', default=False, type=bool) # 随机种子模式
parser.add_argument('--random_seed', default=1227, type=int) # 种子值

parser.add_argument('--synchronous_mode', default=True, type=bool) # 同步模式开关
parser.add_argument('--no_rendering_mode', default=False, type=bool) # 无渲染模式开关
parser.add_argument('--fixed_delta_seconds', default=0.05, type=float) # 步长,步长建议不大于0.1，为0时代表可变步长

parser.add_argument('--log_interval', default=50, type=int) # 目标网络保存间隔
parser.add_argument('--load', default=True, type=bool) # 训练模式下是否load model
parser.add_argument('--sigma', default=0.4, type=float) # 探索偏移分布 
parser.add_argument('--max_episode', default=500, type=int) # 仿真次数
parser.add_argument('--update_iteration', default = 8, type=int) # 网络迭代次数
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)

# 随机值
if args.seed:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

# 环境建立
if args.mode == 'train':
    create_envs = DDPG_ENVS.Create_Envs(args.synchronous_mode,args.no_rendering_mode,args.fixed_delta_seconds) # 设置仿真模式以及步长
    print('==========training mode is activated==========')
elif args.mode == 'test':
    create_envs = DDPG_ENVS.Create_Envs(args.synchronous_mode,False,args.fixed_delta_seconds)
    print('===========testing mode is activated===========')
else:
    raise NameError("wrong mode!!!")

# 状态、动作空间定义
action_space = create_envs.get_action_space()
state_space = create_envs.get_state_space()
state_dim = len(state_space)
action_dim = len(action_space)
actor_num = 2
max_action = torch.tensor(action_space[...,1]).float().to(device)
min_action = torch.tensor(action_space[...,0]).float().to(device)

directory = './carla-DDPG./'

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
        x, y, u, uy, r, d = [], [], [], [], [], []

        for i in ind:
            X, Y, U, UY, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            uy.append(np.array(UY, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(uy), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu=0, sigma=2, theta=0.05, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim*actor_num, 128)
        # self.l1.weight.data.normal_(1e-1,1e-1)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 16)
        self.l4 = nn.Linear(16, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.max_action * torch.tanh(self.l4(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear((state_dim + action_dim)*actor_num, 128)
        self.l2 = nn.Linear(128 , 64)
        self.l3 = nn.Linear(64 , 16)
        self.l4 = nn.Linear(16, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
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
    
    def select_next_action(self, next_state):
        next_state = torch.FloatTensor(next_state.reshape(1, -1)).to(device)
        return self.actor_target(next_state).cpu().data.numpy().flatten()

    def update(self,curr_epi,vehicle):

        if curr_epi > args.warmup_step:
            for it in range(args.update_iteration):
                # Sample replay buffer
                x, y, u, uy, r, d = self.replay_buffer.sample(args.batch_size) # 状态、下个状态、动作、下个动作、奖励、是否结束标志
                state = torch.FloatTensor(x).to(device)
                action = torch.FloatTensor(u).to(device)
                next_state = torch.FloatTensor(y).to(device)
                next_action = torch.FloatTensor(uy).to(device)
                done = torch.FloatTensor(1-d).to(device)
                reward = torch.FloatTensor(r).to(device)

                # Compute the target Q value
                target_Q = self.critic_target(next_state, next_action.detach())
                target_Q = reward + (done * args.gamma * target_Q).detach()

                # Get current Q estimate
                current_Q = self.critic(state, action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q, target_Q)
                self.writer.add_scalar('Loss/%s_critic_loss'%vehicle, critic_loss, global_step=self.num_critic_update_iteration)
                
                # Optimize the critic
                self.critic_optimizer.zero_grad() # 梯度初始化，使得batch梯度不积累
                critic_loss.backward() # 损失反向传播
                self.critic_optimizer.step() # 更新

                # Compute actor loss
                actor_loss = -1*self.critic(state, action).mean()
                self.writer.add_scalar('Loss/%s_actor_loss'%vehicle, actor_loss, global_step=self.num_actor_update_iteration)

                # Optimize the actor
                self.actor_optimizer.zero_grad() # # 梯度初始化，使得batch梯度不积累
                actor_loss.backward() # 损失反向传播
                self.actor_optimizer.step() # 更新

                # Update the frozen target models
                if it % args.target_update_interval == 0:
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                self.num_actor_update_iteration += 1
                self.num_critic_update_iteration += 1

    def save(self, name):
        torch.save(self.actor.state_dict(), directory + name + '_actor.pth')
        torch.save(self.critic.state_dict(), directory + name + '_critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, name):
        self.actor.load_state_dict(torch.load(directory + name + '_actor.pth'))
        self.critic.load_state_dict(torch.load(directory + name + '_critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def main():
    ego_DDPG = DDPG(state_dim, action_dim, max_action)
    npc_DDPG = DDPG(state_dim, action_dim, max_action)
    sim_time = args.fixed_delta_seconds  # 每步仿真时间
    client, world, blueprint_library = create_envs.connection()
    ego_reward_list = []
    npc_reward_list = []

    try:
        if args.mode == 'test':
            ego_DDPG.load('ego')
            npc_DDPG.load('npc')
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
                    ego_action = ego_DDPG.select_action(np.concatenate((ego_state, npc_state)))
                    npc_action = npc_DDPG.select_action(np.concatenate((npc_state, ego_state)))

                    ego_action = np.array(ego_action).clip(min_action.cpu().numpy(), max_action.cpu().numpy())
                    npc_action = np.array(npc_action).clip(min_action.cpu().numpy(), max_action.cpu().numpy())
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
            if args.load: ego_DDPG.load('ego')
            if args.load: npc_DDPG.load('npc')
            for i in range(args.max_episode):
                ego_total_reward = 0
                npc_total_reward = 0
                print('------------%dth time learning begins-----------'%i)
                ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)

                noise1 = OrnsteinUhlenbeckActionNoise(mu=np.array([0,0]),sigma=args.sigma,theta=0.15)
                noise2 = OrnsteinUhlenbeckActionNoise(mu=np.array([0,0]),sigma=args.sigma,theta=0.15)

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
                    ego_action = ego_DDPG.select_action(np.concatenate((ego_state, npc_state)))
                    npc_action = npc_DDPG.select_action(np.concatenate((npc_state, ego_state)))
                    # 探索偏差调节（先高后低）：
                    if i<=args.max_episode/4:
                        b = 1
                        c_tau = args.c_tau
                    elif i<=args.max_episode/2:
                        b = 0.75
                        c_tau = args.c_tau
                    elif i<=args.max_episode*0.75:
                        b = 0.5
                        c_tau = args.c_tau
                    else:
                        b = 0.25
                        c_tau = args.c_tau
                    
                    # ego_action = np.array(ego_action + np.random.normal(0, args.exploration_noise, size=(action_dim,))).clip(
                    #     min_action.cpu().numpy(), max_action.cpu().numpy()) #将输出tensor格式的action，因此转换为numpy格式
                    # npc_action = np.array(npc_action + np.random.normal(0, args.exploration_noise, size=(action_dim,))).clip(
                    #     min_action.cpu().numpy(), max_action.cpu().numpy()) #将输出tensor格式的action，因此转换为numpy格式
                    ego_action = np.array(ego_action + noise1()*b).clip(min_action.cpu().numpy(), max_action.cpu().numpy()) #将输出tensor格式的action，因此转换为numpy格式
                    npc_action = np.array(npc_action + noise2()*b).clip(min_action.cpu().numpy(), max_action.cpu().numpy()) #将输出tensor格式的action，因此转换为numpy格式
                    # period = time.time() - start_time
                    create_envs.set_vehicle_control(ego_list[0], npc_list[0], ego_action, npc_action, c_tau, args.fixed_delta_seconds, t)
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
                    # start_time = time.time()  # 开始时间
                    # print('period:',period)
                    ego_next_action = ego_DDPG.select_next_action(np.concatenate((ego_next_state, npc_next_state)))
                    npc_next_action = npc_DDPG.select_next_action(np.concatenate((npc_next_state, ego_next_state)))
                    # 数据储存
                    ego_DDPG.replay_buffer.push((np.concatenate((ego_state, npc_state)), np.concatenate((ego_next_state, npc_next_state)), 
                        np.concatenate((ego_action, npc_action)), np.concatenate((ego_next_action, npc_next_action)), ego_reward, ego_done))
                    npc_DDPG.replay_buffer.push((np.concatenate((npc_state, ego_state)), np.concatenate((npc_next_state, ego_next_state)), 
                        np.concatenate((npc_action, ego_action)), np.concatenate((npc_next_action, ego_next_action)), npc_reward, npc_done))

                    ego_state = ego_next_state
                    npc_state = npc_next_state
                    
                    ego_total_reward += ego_reward
                    npc_total_reward += npc_reward

                    if t >= args.max_length_of_trajectory: # 总结束条件
                        break
                    if ego_done or npc_done: # 结束条件
                        break

                # ego_total_reward /= t
                # npc_total_reward /= t
                ego_reward_list.append(ego_total_reward)
                npc_reward_list.append(npc_total_reward)
                print("Episode: {} step: {} ego_Total_Reward: {:0.3f} npc_Total_Reward: {:0.3f}".format(i+1, t, ego_total_reward, npc_total_reward))
                if i % args.update_interval == 0:
                    ego_DDPG.update(curr_epi=i,vehicle='ego')
                    npc_DDPG.update(curr_epi=i,vehicle='npc')
                if i % args.log_interval == 0:
                    ego_DDPG.save('ego')
                    npc_DDPG.save('npc')

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

        # reward图
        x1=np.arange(len(ego_reward_list))
        y1=ego_reward_list
        x2=np.arange(len(npc_reward_list))
        y2=npc_reward_list
        plt.figure(figsize=(8,8), dpi=80)
        plt.figure(1)
        ax1 = plt.subplot(211)
        ax1.plot(x1,y1)
        ax2 = plt.subplot(212)
        ax2.plot(x2,y2)
        plt.show()


if __name__ == '__main__':
    main()

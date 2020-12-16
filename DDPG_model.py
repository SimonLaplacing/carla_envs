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
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)
parser.add_argument('--max_length_of_trajectory', default=60, type=int) # 仿真步长
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=10000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=80, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=1227, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=True, type=bool) # load model
parser.add_argument('--exploration_noise', default=0.5, type=float)
parser.add_argument('--max_episode', default=2000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=8, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)


if args.seed:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

create_envs = DDPG_ENVS.Create_Envs()
action_space = create_envs.get_action_space()
state_space = create_envs.get_state_space()
state_dim = len(state_space)
action_dim = len(action_space)
max_action = torch.tensor(action_space[...,1]).float().to(device)
min_action = torch.tensor(action_space[...,0]).float().to(device) # min value

directory = './carla-' + 'DDPG' +'./'

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
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
        x, y, u, r = [], [], [], []

        for i in ind:
            X, Y, U, R = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l1.weight.data.normal_(0,0.1)
        self.l2 = nn.Linear(400, 200)
        self.l2.weight.data.normal_(0,0.1)
        self.l3 = nn.Linear(200, action_dim)
        self.l3.weight.data.normal_(0,0.1)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 500)
        self.l1.weight.data.normal_(0,0.1)
        self.l2 = nn.Linear(500 , 300)
        self.l2.weight.data.normal_(0,0.1)
        self.l3 = nn.Linear(300, 1)
        self.l3.weight.data.normal_(0,0.1)

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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (args.gamma * target_Q).detach()

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
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def main():
    ego_DDPG = DDPG(state_dim, action_dim, max_action)
    npc_DDPG = DDPG(state_dim, action_dim, max_action)
    ep_r = 0
    client, world, blueprint_library = create_envs.connection()
    print("Collecting Experience....")
    reward_list = []

    try:
        if args.mode == 'test':
            ego_DDPG.load()
            ego_DDPG.load()
            for i in range(args.test_iteration):
                print('%dth time learning begins'%i)
                ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)
                sim_time = 0.5  # 每步仿真时间
                start_time = time.time()  # 初始时间
                
                ego_state = ego_list[0].get_transform()
                ego_state = np.array([ego_state.location.x,ego_state.location.y,ego_state.location.z,ego_state.rotation.pitch,ego_state.rotation.yaw,ego_state.rotation.roll])
                npc_state = npc_list[0].get_transform()
                npc_state = np.array([npc_state.location.x,npc_state.location.y,npc_state.location.z,npc_state.rotation.pitch,npc_state.rotation.yaw,npc_state.rotation.roll])

                egocol_list = sensor_list[0].get_collision_history()
                npccol_list = sensor_list[1].get_collision_history()

                for t in count():
                    ego_action = ego_DDPG.select_action(ego_state)
                    npc_action = npc_DDPG.select_action(npc_state)
                    ego_next_state,ego_reward,npc_next_state,npc_reward = create_envs.get_vehicle_step(ego_list[0], npc_list[0], egocol_list, npccol_list,ego_action, npc_action, sim_time)
                    ep_r += ego_reward
                    if t >= args.max_length_of_trajectory: # 总结束条件
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                        ep_r = 0
                        break
                    if egocol_list[0]==1 or ego_next_state[0] > 245 or ego_next_state[1] > -367: # ego结束条件
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                        ep_r = 0
                        break
                    if npccol_list[0]==1 or npc_next_state[0] > 245 or npc_next_state[1] > -370: # npc结束条件
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                        ep_r = 0
                        break
                    period = time.time() - start_time                    
                    ego_state = ego_next_state

        elif args.mode == 'train':
            if args.load: ego_DDPG.load()
            if args.load: npc_DDPG.load()
            total_step = 0
            for i in range(args.max_episode):
                ego_total_reward = 0
                npc_total_reward = 0
                step =0
                print('%dth time learning begins'%i)
                ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)
                sim_time = 0.5  # 仿真时间
                start_time = time.time()  # 开始时间

                ego_state = ego_list[0].get_transform()
                ego_state = np.array([ego_state.location.x,ego_state.location.y,ego_state.location.z,ego_state.rotation.pitch,ego_state.rotation.yaw,ego_state.rotation.roll])
                npc_state = npc_list[0].get_transform()
                npc_state = np.array([npc_state.location.x,npc_state.location.y,npc_state.location.z,npc_state.rotation.pitch,npc_state.rotation.yaw,npc_state.rotation.roll])

                egocol_list = sensor_list[0].get_collision_history()
                npccol_list = sensor_list[1].get_collision_history()
                for t in count():
                    ego_action = ego_DDPG.select_action(ego_state)
                    npc_action = npc_DDPG.select_action(npc_state)
                    ego_action = np.array(ego_action + np.random.normal(0, args.exploration_noise, size=(action_dim,))).clip(
                        min_action.cpu().numpy(), max_action.cpu().numpy()) #将输出tensor格式的action，因此转换为numpy格式
                    npc_action = np.array(npc_action + np.random.normal(0, args.exploration_noise, size=(action_dim,))).clip(
                        min_action.cpu().numpy(), max_action.cpu().numpy()) #将输出tensor格式的action，因此转换为numpy格式
                    ego_next_state,ego_reward,npc_next_state,npc_reward = create_envs.get_vehicle_step(ego_list[0], npc_list[0], egocol_list, npccol_list,ego_action, npc_action, sim_time)
                    
                    ego_DDPG.replay_buffer.push((ego_state, ego_next_state, ego_action, ego_reward))
                    npc_DDPG.replay_buffer.push((npc_state, npc_next_state, npc_action, npc_reward))
                    ego_state = ego_next_state
                    npc_state = npc_next_state
                    
                    step += 1
                    ego_total_reward += ego_reward
                    npc_total_reward += npc_reward

                    if t >= args.max_length_of_trajectory: # 总结束条件
                        break
                    if egocol_list[0]==1 or ego_next_state[0] > 245 or ego_next_state[1] > -367: # ego结束条件
                        break
                    if npccol_list[0]==1 or npc_next_state[0] > 245 or npc_next_state[1] > -370: # npc结束条件
                        break
                    period = time.time() - start_time
                    # print('period:',period)
                ego_total_reward /= step
                npc_total_reward /= step
                total_step += step+1
                print("T: {} Episode: {} ego Total Reward: {:0.2f} npc Total Reward: {:0.2f}".format(step, i, ego_total_reward, npc_total_reward))
                ego_DDPG.update()
                npc_DDPG.update()
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
        else:
            raise NameError("mode wrong!!!")

    finally:
        # rew = open('reward.txt','w+')
        # rew.write(str(reward_list))
        # rew.close()
        # x = np.linspace(0,len(reward_list),len(reward_list))
        # plt.plot(x,reward_list)
        # plt.show()
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

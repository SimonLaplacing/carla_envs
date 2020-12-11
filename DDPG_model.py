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
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
# parser.add_argument("--env_name", default="Pendulum-v0")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
# env = gym.make(args.env_name)

if args.seed:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

create_envs = DDPG_ENVS.Create_Envs()
action_space = create_envs.get_action_space()
state_space = create_envs.get_state_space()
state_dim = len(state_space)
action_dim = len(action_space)
max_action = float(action_space[...,1])
min_Val = float(action_space[...,0])
# min_Val = torch.tensor(1e-7).float().to(device) # min value

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

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

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
                sim_time = 0  # 仿真时间
                start_time = time.time()  # 初始时间

                egocol_list = sensor_list[0].get_collision_history()
                npccol_list = sensor_list[1].get_collision_history()

                for t in count():
                    action = ego_DDPG.select_action(state)
                    next_state, reward, done, info = create_envs.get_vehicle_step(np.float32(action))
                    ep_r += reward
                    if done or t >= args.max_length_of_trajectory:
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                        ep_r = 0
                        break
                    state = next_state

        elif args.mode == 'train':
            if args.load: ego_DDPG.load()
            total_step = 0
            for i in range(args.max_episode):
                total_reward = 0
                step =0
                print('%dth time learning begins'%i)
                ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)
                sim_time = 0  # 仿真时间
                start_time = time.time()  # 初始时间

                egocol_state = sensor_list[0].get_collision_history()
                npccol_state = sensor_list[1].get_collision_history()
                for t in count():
                    action = ego_DDPG.select_action(state)
                    action = (action + np.random.normal(0, args.exploration_noise, size=action_space)).clip(
                        max_action, min_Val)

                    next_state = create_envs.get_vehicle_step(ego_list[0],action,t)
                    
                    # if args.render and i >= args.render_interval : env.render()
                    ego_DDPG.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                    state = next_state
                    if done:
                        break
                    step += 1
                    total_reward += reward
                total_step += step+1
                print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
                ego_DDPG.update()
            # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

                if i % args.log_interval == 0:
                    ego_DDPG.save()

        else:
            raise NameError("mode wrong!!!")

        time.sleep(1)
            
        # action
        while state:  
            sim_time = time.time() - start_time
            if egocol_list[0] or npccol_list[0] or sim_time > 12: # 发生碰撞，重置场景
                state = state_space[0]

        # print(reward)
        time.sleep(1)

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
        rew = open('reward.txt','w+')
        rew.write(str(reward_list))
        rew.close()
        x = np.linspace(0,len(reward_list),len(reward_list))
        plt.plot(x,reward_list)
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

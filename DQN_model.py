import glob
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import DQN_ENVS
import time
from tensorboardX import SummaryWriter
import argparse

try:
    sys.path.append(glob.glob('D:/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', default = 3000, type=int)
parser.add_argument('--MEMORY_CAPACITY', default = 20000, type=int)
parser.add_argument('--BATCH_SIZE', default = 32, type=int)
parser.add_argument('--LR', default = 1e-4, type=float)
parser.add_argument('--GAMMA', default = 0.9, type=float) 
parser.add_argument('--EPISILO', default = 0.9, type=float) # greedy

parser.add_argument('--synchronous_mode', default=False, type=bool) # 同步模式开关
parser.add_argument('--no_rendering_mode', default=True, type=bool) # 无渲染模式开关
parser.add_argument('--fixed_delta_seconds', default=0.05, type=float) # 步长,步长建议不大于0.1，为0时代表可变步长

parser.add_argument('--log_interval', default=50, type=int) # 网络保存间隔
parser.add_argument('--update_interval', default=1, type=int) # 网络更新间隔
parser.add_argument('--warmup_step', default=0, type=int) # 网络参数训练更新预备回合数
parser.add_argument('--load', default=False, type=bool) # 是否load model
parser.add_argument('--Q_NETWORK_ITERATION', default = 3, type=int)
args = parser.parse_args()

create_envs = DQN_ENVS.Create_Envs(args.synchronous_mode,args.no_rendering_mode,args.fixed_delta_seconds)
action_space = create_envs.get_action_space()
state_space = create_envs.get_state_space()
NUM_ACTIONS = len(action_space)
NUM_STATES = len(state_space) - 1

directory = './carla-DQN./'

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 64) # linear层1
        # self.fc1.weight.data.normal_(0,0.1)  # 初始化参数
        self.fc2 = nn.Linear(64,16)           # linear层2
        # self.fc2.weight.data.normal_(0,0.1)  # 初始化参数
        self.out = nn.Linear(16,NUM_ACTIONS)  # 输出层
        # self.out.weight.data.normal_(0,0.1)  # 初始化参数

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((args.MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 + 2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.LR)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter(directory)
        self.num_update_iteration = 0

    def choose_action(self, state, EPISILO):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
        else: # random policy
            action = np.random.choice(action_space)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % args.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self,vehicle):

        #update the parameters
        if self.learn_step_counter % args.Q_NETWORK_ITERATION == 0:
            print('update network parameters')
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(args.MEMORY_CAPACITY, args.BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        # q_next = self.target_net(batch_next_state).detach()
        # q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = batch_reward
        loss = self.loss_func(q_eval, q_target)
        # self.writer.add_scalar('Loss/%s_loss'%vehicle, loss, global_step=self.num_update_iteration)
        self.num_update_iteration += 1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, name):
        torch.save(self.eval_net.state_dict(), directory + name + '.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, name):
        self.eval_net.load_state_dict(torch.load(directory + name + '.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    ego_dqn = DQN()
    npc_dqn = DQN()
    if args.load: ego_dqn.load('ego')
    if args.load: npc_dqn.load('npc')
    main_writer = SummaryWriter(directory)
    client, world, blueprint_library = create_envs.connection()
    print("Collecting Experience....")
    reward_list1 = []
    reward_list2 = []

    try:
        for i in range(args.episodes):
            print('%dth time learning begins'%i)
            ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)
            # sim_time = 0  # 仿真时间
            # start_time = time.time()  # 初始时间
            state = state_space[1]

            # egocol = sensor_list[0].get_collision_history()
            # npccol = sensor_list[1].get_collision_history()

            # adaptive EPISILO
            if i <= 0.5 * args.episodes:
                EPISILO = args.EPISILO
            elif 0.5 * args.episodes < i < 0.9 * args.episodes:
                EPISILO = 0.5 + 0.5*args.EPISILO
            else:
                EPISILO = 0.9 + 0.1*args.EPISILO
            ego_action = ego_dqn.choose_action(state,EPISILO)
            npc_action = npc_dqn.choose_action(state,EPISILO)

            action = [int(ego_action),int(npc_action)]
            print('action is',action)
            reward1,reward2 = create_envs.get_reward(action)

            ego_dqn.store_transition(state_space[1], ego_action, reward1, state_space[0])
            npc_dqn.store_transition(state_space[1], npc_action, reward2, state_space[0])

            print('reward of %dth episode is %d,%d'%(i, reward1,reward2))
            reward_list1.append(reward1)
            reward_list2.append(reward2)
            main_writer.add_scalar('Reward/reward1', reward1, global_step=i)
            main_writer.add_scalar('Reward/reward2', reward2, global_step=i)
            if i > args.warmup_step: 
                if i % args.update_interval == 0:
                    ego_dqn.learn('ego')
                    npc_dqn.learn('npc')
                    print('update parameters')            

            if i % args.log_interval == 0:
                    ego_dqn.save('ego')
                    npc_dqn.save('npc')
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
        plt.plot(x,a)
        plt.plot(x,b)
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
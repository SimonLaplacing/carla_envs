import glob
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import ENVS
import time

try:
    sys.path.append(glob.glob('D:/CARLA_0.9.10-Pre_Win/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# hyper-parameters
create_envs = ENVS.Create_Envs()
BATCH_SIZE = 10
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 20
Q_NETWORK_ITERATION = 80

# env = gym.make("CartPole-v0")
# env = env.unwrapped
action_space = create_envs.get_action_space()
state_space = create_envs.get_state_space()
NUM_ACTIONS = len(action_space)
NUM_STATES = len(state_space)-1

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 10)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(10,5)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(10,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

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
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
        else: # random policy
            action = np.random.choice(action_space)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        # q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = batch_reward
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    ego_dqn = DQN()
    npc_dqn = DQN()
    episodes = 50
    print("Collecting Experience....")
    # reward_list = []

    try:
        for i in range(episodes):
            ep_reward = 0
            client, world, blueprint_library = create_envs.connection()
            ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)
            sim_time = 0  # 仿真时间
            start_time = time.time()  # 初始时间
            state = state_space[1]
            # state = env.reset()
            egocol_list = sensor_list[0].get_collision_history()
            npccol_list = sensor_list[1].get_collision_history()
            ego_action = ego_dqn.choose_action(state)
            npc_action = npc_dqn.choose_action(state)
            reward = create_envs.get_reward([ego_action,npc_action])

            dqn.store_transition(1, action, reward, 0)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
            r = copy.copy(reward)
            # reward_list.append(r)
            
            # action
            while state:  
                sim_time = time.time() - start_time
                
                create_envs.get_ego_step(ego_list[0],action,sim_time)       
                create_envs.get_npc_step(npc_list[0],action,sim_time)
                # npc.apply_control(set_control)
                # for vehicle in actor_list:
                #     vehicle.set_autopilot(True)
                if egocol_list[0] or npccol_list[0] or sim_time > 12: # 发生碰撞，重置场景
                    state = state_space[0]

            # print(reward)
            time.sleep(1)

            for x in sensor_list:
                if x is not None:
                    x.sensor.destroy()            
            for x in ego_list:
                if x is not None:
                    client.apply_batch([carla.command.DestroyActor(x)])
            for x in npc_list:
                if x is not None:
                    client.apply_batch([carla.command.DestroyActor(x)])
            for x in obstacle_list:
                if x is not None:
                    client.apply_batch([carla.command.DestroyActor(x)])

            print('Reset')
    except:
    # 清洗环境
        print('Start Cleaning Envs')
        for x in sensor_list:
            if x is not None:
                x.sensor.destroy()
        for x in ego_list:
            if x is not None:
                client.apply_batch([carla.command.DestroyActor(x)])
        for x in npc_list:
            if x is not None:
                client.apply_batch([carla.command.DestroyActor(x)])
        for x in obstacle_list:
            if x is not None:
                client.apply_batch([carla.command.DestroyActor(x)])
        print('all clean, simulation done!')

if __name__ == '__main__':

    main()
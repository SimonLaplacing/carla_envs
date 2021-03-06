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
parser.add_argument('--tau',  default=0.005, type=float) # 目标网络软更新系数
parser.add_argument('--c_tau',  default=1, type=float) # action软更新系数
parser.add_argument('--update_interval', default=4, type=int) # 网络更新间隔
parser.add_argument('--target_update_interval', default=8, type=int) # 目标网络更新间隔
parser.add_argument('--warmup_step', default=6, type=int) # 网络参数训练更新预备回合数
parser.add_argument('--test_iteration', default=3, type=int) # 测试次数
parser.add_argument('--max_length_of_trajectory', default=300, type=int) # 最大仿真步数
parser.add_argument('--Alearning_rate', default=1e-4, type=float) # Actor学习率
parser.add_argument('--Clearning_rate', default=1e-3, type=float) # Critic学习率
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=30000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=32, type=int) # mini batch size

parser.add_argument('--seed', default=False, type=bool) # 随机种子模式
parser.add_argument('--random_seed', default=1227, type=int) # 种子值

parser.add_argument('--synchronous_mode', default=True, type=bool) # 同步模式开关
parser.add_argument('--no_rendering_mode', default=True, type=bool) # 无渲染模式开关
parser.add_argument('--fixed_delta_seconds', default=0.05, type=float) # 步长,步长建议不大于0.1，为0时代表可变步长

parser.add_argument('--log_interval', default=50, type=int) # 网络保存间隔
parser.add_argument('--load', default=False, type=bool) # 训练模式下是否load model
parser.add_argument('--sigma', default=0.8, type=float) # 探索偏移分布 
parser.add_argument('--max_episode', default=1000, type=int) # 仿真次数
parser.add_argument('--update_iteration', default = 6, type=int) # 网络迭代次数
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
max_action = torch.tensor(action_space[...,1]).float()
min_action = torch.tensor(action_space[...,0]).float()

directory = './carla-DDPG./'


def main():
    sim_time = args.fixed_delta_seconds  # 每步仿真时间
    client, world, blueprint_library = create_envs.connection()
    main_writer = SummaryWriter(directory)
    ego_reward_list = []
    npc_reward_list = []

    try:
        if args.mode == 'test':
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
                    ego_action = 2*np.random.random_sample(2,) - 1 
                    npc_action = 2*np.random.random_sample(2,) - 1 

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
            for i in range(args.max_episode):
                ego_total_reward = 0
                npc_total_reward = 0
                print('------------%dth time learning begins-----------'%i)
                ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)

                ego_transform = ego_list[0].get_transform()
                npc_transform = npc_list[0].get_transform()

                ego_state = np.array([(ego_transform.location.x-120)/125,(ego_transform.location.y+375)/4,ego_transform.rotation.yaw/90])
                npc_state = np.array([(npc_transform.location.x-120)/125,(npc_transform.location.y+375)/4,npc_transform.rotation.yaw/90])

                egosen_list = sensor_list[0]
                npcsen_list = sensor_list[1]
                # start_time = time.time()

                for t in count():
                    #---------动作决策----------
                    ego_action = 2*np.random.random_sample(2,) - 1 
                    npc_action = 2*np.random.random_sample(2,) - 1 
                    # period = time.time() - start_time
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
                    # start_time = time.time()  # 开始时间
                    # print('period:',period)

                    ego_state = ego_next_state
                    npc_state = npc_next_state
                    
                    ego_total_reward += ego_reward
                    npc_total_reward += npc_reward

                    if t >= args.max_length_of_trajectory: # 总结束条件
                        break
                    if ego_done and npc_done: # 结束条件
                        break

                ego_total_reward /= t
                npc_total_reward /= t
                main_writer.add_scalar('reward/ego_reward', ego_total_reward, global_step=i)
                main_writer.add_scalar('reward/npc_reward', npc_total_reward, global_step=i)
                # ego_reward_list.append(ego_total_reward)
                # npc_reward_list.append(npc_total_reward)
                print("Episode: {} step: {} ego_Total_Reward: {:0.3f} npc_Total_Reward: {:0.3f}".format(i+1, t, ego_total_reward, npc_total_reward))

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

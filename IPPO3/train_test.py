import glob
import os
import sys

import argparse

import numpy as np

import torch

from tensorboardX import SummaryWriter

import IPPO_ENVS
from PPO_model import PPO

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
parser.add_argument('--max_length_of_trajectory', default=300, type=int) # 最大仿真步数
parser.add_argument('--Alearning_rate', default=1e-5, type=float) # Actor学习率
parser.add_argument('--Clearning_rate', default=5e-5, type=float) # Critic学习率
parser.add_argument('--gamma', default=0.95, type=int) # discounted factor

parser.add_argument('--fig_size', default=[20,20], type=list) # BEV尺寸

parser.add_argument('--synchronous_mode', default=True, type=bool) # 同步模式开关
parser.add_argument('--no_rendering_mode', default=True, type=bool) # 无渲染模式开关
parser.add_argument('--fixed_delta_seconds', default=0.1, type=float) # 步长,步长建议不大于0.1，为0时代表可变步长

parser.add_argument('--log_interval', default=50, type=int) # 网络保存间隔
parser.add_argument('--update_interval', default=15, type=int) # 网络更新间隔
parser.add_argument('--load', default=False, type=bool) # 训练模式下是否load model
 
parser.add_argument('--max_episode', default=2000, type=int) # 仿真次数
parser.add_argument('--update_iteration', default = 5, type=int) # 网络迭代次数
args = parser.parse_args()

script_name = os.path.basename(__file__)

# 环境建立
if args.mode == 'train':
    create_envs = IPPO_ENVS.Create_Envs(args.synchronous_mode,args.no_rendering_mode,args.fixed_delta_seconds,args.fig_size)
    print('==========training mode is activated==========')
elif args.mode == 'test':
    create_envs = IPPO_ENVS.Create_Envs(args.synchronous_mode,False,args.fixed_delta_seconds,args.fig_size)
    print('===========testing mode is activated===========')
else:
    raise NameError("wrong mode!!!")

# 状态、动作空间定义
action_space = create_envs.get_action_space()
state_dim = create_envs.get_state_space()

action_dim = len(action_space)
actor_num = 2
max_action = torch.tensor(action_space[...,1]).float()
min_action = torch.tensor(action_space[...,0]).float()

directory = './carla-IPPO3./'

def main():
    ego_PPO = PPO(state_dim, action_dim, args.Alearning_rate, args.Clearning_rate, args.gamma, args.update_iteration, 0.2, True, action_std_init=0.6)
    npc_PPO = PPO(state_dim, action_dim, args.Alearning_rate, args.Clearning_rate, args.gamma, args.update_iteration, 0.2, True, action_std_init=0.6)
    client, world, blueprint_library = create_envs.connection()
    main_writer = SummaryWriter(directory)

    try:
        if args.load or args.mode == 'test': 
            ego_PPO.load(directory + 'ego.pkl')
            npc_PPO.load(directory + 'npc.pkl')
        for i in range(args.max_episode):
            ego_total_reward = 0
            npc_total_reward = 0
            print('------------%dth time learning begins-----------'%i)
            ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)

            egosen_list = sensor_list[0]
            npcsen_list = sensor_list[1]

            # ego_transform = ego_list[0].get_transform()
            # npc_transform = npc_list[0].get_transform()
            ego_velocity = ego_list[0].get_velocity().x
            npc_velocity = npc_list[0].get_velocity().x

            ego_angular = ego_list[0].get_angular_velocity().z
            npc_angular = npc_list[0].get_angular_velocity().z
            ego_camera = egosen_list[2].get_BEV()
            npc_camera = npcsen_list[2].get_BEV()
            # print('1111111111:  ',ego_camera.shape)
            ego_state = [[ego_velocity/25,ego_angular/2],[ego_camera]]
            npc_state = [[npc_velocity/25,npc_angular/2],[npc_camera]]
            # ego_state2 = np.array([ego_camera])
            # npc_state2 = np.array([npc_camera])

            

            # if (i+1) % int(200) == 0:
            #         ego_PPO.decay_action_std(0.05, 0.1)
            #         npc_PPO.decay_action_std(0.05, 0.1)

            for t in range(args.max_length_of_trajectory):
                #---------动作决策----------
                if args.mode == 'test':
                    ego_action = ego_PPO.select_best_action(ego_state)
                    npc_action = npc_PPO.select_best_action(npc_state)
                else:
                    ego_action = ego_PPO.select_action(ego_state)
                    npc_action = npc_PPO.select_action(npc_state)
                
                create_envs.set_vehicle_control(ego_list[0], npc_list[0], ego_action, npc_action, args.c_tau, t)
                #---------和环境交互动作反馈---------
                frames = 1 # 步长 
                if args.synchronous_mode:
                    for _ in range(frames):
                        world.tick() # 客户端主导，tick
                else:
                    world.wait_for_tick() # 服务器主导，tick

                ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done = create_envs.get_vehicle_step(ego_list[0], npc_list[0], egosen_list, npcsen_list, t)

                # 数据储存
                ego_PPO.buffer.rewards.append(ego_reward)
                ego_PPO.buffer.is_terminals.append(ego_done)
                npc_PPO.buffer.rewards.append(npc_reward)
                npc_PPO.buffer.is_terminals.append(npc_done)

                ego_state = ego_next_state
                # ego_state2 = ego_next_state2
                npc_state = npc_next_state
                # ego_state2 = npc_next_state2
                
                ego_total_reward += ego_reward
                npc_total_reward += npc_reward

                if ego_done or npc_done: # 结束条件
                    break

            ego_total_reward /= t+1
            npc_total_reward /= t+1
            main_writer.add_scalar('reward/ego_reward', ego_total_reward, global_step=i)
            main_writer.add_scalar('reward/npc_reward', npc_total_reward, global_step=i)

            print("Episode: {} step: {} ego_Total_Reward: {:0.3f} npc_Total_Reward: {:0.3f}".format(i+1, t, ego_total_reward, npc_total_reward))
            
            if args.mode == 'train':    
                if i > 0 and (i+1) % args.update_interval == 0:
                    ego_PPO.update()
                    print('ego_updated')
                    npc_PPO.update()
                    print('npc_updated')
                if i > 0 and (i+1) % args.log_interval == 0:
                    ego_PPO.save(directory + 'ego.pkl')
                    npc_PPO.save(directory + 'npc.pkl')
                    print('Network Saved')

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

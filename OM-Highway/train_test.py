import argparse

import numpy as np

import torch

from tensorboardX import SummaryWriter

import Highway_ENVS as IPPO_ENVS
from PPO_model import PPO

import carla
from carla import Transform, Location, Rotation
import misc
import random
import os
import time


parser = argparse.ArgumentParser()
# parser.add_argument('--env', type=str, default='highway')
parser.add_argument('--mode', default='test', type=str) # mode = 'train' or 'test'
parser.add_argument('--load_seed', default=379, type=str) # seed

parser.add_argument('--c_tau',  default=0.8, type=float) # action软更新系数,1代表完全更新，0代表不更新
parser.add_argument('--max_length_of_trajectory', default=300, type=int) # 最大仿真步数
parser.add_argument('--Alearning_rate', default=5e-5, type=float) # Actor学习率
parser.add_argument('--Clearning_rate', default=8e-5, type=float) # Critic学习率
parser.add_argument('--gamma', default=0.98, type=int) # discounted factor

parser.add_argument('--synchronous_mode', default=True, type=bool) # 同步模式开关
parser.add_argument('--no_rendering_mode', default=True, type=bool) # 无渲染模式开关
parser.add_argument('--fixed_delta_seconds', default=0.03, type=float) # 步长,步长建议不大于0.1，为0时代表可变步长

parser.add_argument('--log', default=True, type=bool) # 日志开关
parser.add_argument('--log_interval', default=50, type=int) # 网络保存间隔
parser.add_argument('--update_interval', default=800, type=int) # 网络更新step间隔
parser.add_argument('--load', default=True, type=bool) # 训练模式下是否load model
 
parser.add_argument('--max_episode', default=200, type=int) # 仿真次数
parser.add_argument('--update_iteration', default = 10, type=int) # 网络迭代次数
args = parser.parse_args()

# 环境建立

create_envs = IPPO_ENVS.Create_Envs(args.synchronous_mode,args.no_rendering_mode,args.fixed_delta_seconds) # 设置仿真模式以及步长

# 状态、动作空间定义
action_space = create_envs.get_action_space()
state_space = create_envs.get_state_space()
state_dim = len(state_space)
action_dim = len(action_space)
actor_num = 2
# max_action = torch.tensor(action_space[...,1]).float()
# min_action = torch.tensor(action_space[...,0]).float()

directory = './carla-Highway/'

def main():
    ego_PPO = PPO(state_dim, action_dim, args.Alearning_rate, args.Clearning_rate, args.gamma, args.update_iteration, 0.2, True, action_std_init=1)
    npc_PPO = PPO(state_dim, action_dim, args.Alearning_rate, args.Clearning_rate, args.gamma, args.update_iteration, 0.2, True, action_std_init=1)
    if not misc.judgeprocess('CarlaUE4.exe'):
        os.startfile('D:\CARLA_0.9.11\WindowsNoEditor\CarlaUE4.exe')
        time.sleep(10)
    client, world, blueprint_library = create_envs.connection()
    seed = random.randint(0,10000)
    if args.log == True:
        main_writer = SummaryWriter(directory + '/' + args.mode + str(seed))
    count = 0
    pre_t = 0
    try:
        if args.load or args.mode == 'test': 
            ego_PPO.load(directory + '/' + 'train' + str(args.load_seed) + '/' + 'ego.pkl')
            npc_PPO.load(directory + '/' + 'train' + str(args.load_seed) + '/' + 'npc.pkl')

        for i in range(args.max_episode):
            ego_total_reward = 0
            npc_total_reward = 0
            ego_offsety = 0
            npc_offsety = 0
            
            print('------------%dth time learning begins-----------'%i)
            ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)
            egosen_list = sensor_list[0]
            npcsen_list = sensor_list[1]
            if args.synchronous_mode:
                world.tick() # 客户端主导，tick
            else:
                world.wait_for_tick() # 服务器主导，tick
            ego_transform = ego_list[0].get_transform()
            npc_transform = npc_list[0].get_transform()

            if i == 0:
                # 全局路径
                ego_start_location = ego_transform.location
                ego_end_location = ego_transform.location + carla.Location(x=135)
                ego_route = create_envs.route_positions_generate(ego_start_location,ego_end_location)
                misc.draw_waypoints(world,ego_route)

                npc_start_location = npc_transform.location
                npc_end_location = npc_transform.location + carla.Location(x=135)
                npc_route = create_envs.route_positions_generate(npc_start_location,npc_end_location)
                misc.draw_waypoints(world,npc_route)

            ego_step, npc_step = 1, 1
            ego_state,_,_,npc_state,_,_ = create_envs.get_vehicle_step(ego_list[0], npc_list[0], egosen_list, npcsen_list, ego_route[ego_step], npc_route[npc_step], obstacle_list[0], 0)
            
            for t in range(99999):
                #---------动作决策----------
                # if args.mode == 'test':
                #     ego_action = ego_PPO.select_best_action(ego_state,npc_state)
                #     npc_action = npc_PPO.select_best_action(npc_state,ego_state)
                # else:
                ego_action = ego_PPO.select_action(ego_state,npc_state)
                npc_action = npc_PPO.select_action(npc_state,ego_state)
                
                create_envs.set_vehicle_control(ego_list[0], npc_list[0], ego_action, npc_action, args.c_tau, t)
                #---------和环境交互动作反馈---------
                frames = 1 # 步长 
                if args.synchronous_mode:
                    for _ in range(frames):
                        world.tick() # 客户端主导，tick
                else:
                    world.wait_for_tick() # 服务器主导，tick
                # print('step:',ego_step)

                     
                ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done = create_envs.get_vehicle_step(ego_list[0], npc_list[0], egosen_list, npcsen_list, ego_route[ego_step], npc_route[npc_step], obstacle_list[0], t)
                
                if ego_next_state[0] < 0.2:
                    ego_step += 1                    
                if npc_next_state[0] < 0.2:
                    npc_step += 1
                # print('state: ', ego_next_state)
                # 数据储存
                
                count += 1
                ego_state = ego_next_state
                npc_state = npc_next_state
                # print('ego_state: ', ego_state, 'npc_state: ', npc_state)
                ego_total_reward += ego_reward
                npc_total_reward += npc_reward
                ego_offsety += np.abs(ego_state[1])
                npc_offsety += np.abs(npc_state[1])
                ego_PPO.buffer.rewards.append(ego_reward)
                npc_PPO.buffer.rewards.append(npc_reward)
                # main_writer.add_scalar('offset/ego_step_offsety', npc_offsety/ego_state[1], global_step=pre_t+t)
                # main_writer.add_scalar('offset/npc_step_offsety', npc_offsety/npc_state[1], global_step=pre_t+t)

                if args.mode == 'test':
                    if ego_done or npc_done or t==args.max_length_of_trajectory-1:
                        break

                if args.mode == 'train':
                    if ego_done or npc_done or t==args.max_length_of_trajectory-1: # 结束条件
                        ego_PPO.buffer.is_terminals.append(True)
                        npc_PPO.buffer.is_terminals.append(True)
                        break
                    else:
                        ego_PPO.buffer.is_terminals.append(False)
                        npc_PPO.buffer.is_terminals.append(False)

            pre_t += t
            ego_total_reward /= t+1
            npc_total_reward /= t+1
                

            print("Episode: {} step: {} ego_Total_Reward: {:0.3f} npc_Total_Reward: {:0.3f}".format(i+1, t+1, ego_total_reward, npc_total_reward))
            
            if args.log == True:  
                main_writer.add_scalar('reward/ego_reward', ego_total_reward, global_step=i)
                main_writer.add_scalar('reward/npc_reward', npc_total_reward, global_step=i)
                main_writer.add_scalar('step/step', t+1, global_step=i)
                main_writer.add_scalar('step/ego_step', ego_step, global_step=i)
                main_writer.add_scalar('step/npc_step', npc_step, global_step=i)
                main_writer.add_scalar('offset/ego_offsety', ego_offsety/(t+1), global_step=i)
                main_writer.add_scalar('offset/npc_offsety', npc_offsety/(t+1), global_step=i)  
                if args.mode == 'train':
                    if i > 0 and (count+1) >= args.update_interval:
                        ego_loss = ego_PPO.update(npc_PPO)
                        print('ego_updated')
                        npc_loss = npc_PPO.update(ego_PPO)
                        print('npc_updated')
                        main_writer.add_scalar('loss/ego_loss', npc_loss, global_step=i)
                        main_writer.add_scalar('loss/npc_loss', npc_loss, global_step=i)
                        ego_PPO.clear()
                        npc_PPO.clear()
                        count = 0
                    
                    if i > 0 and (i+1) % args.log_interval == 0:
                        ego_PPO.save(directory + '/' + args.mode + str(seed) + '/' + 'ego.pkl')
                        npc_PPO.save(directory + '/' + args.mode + str(seed) + '/' + 'npc.pkl')
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
            # del ego_list, npc_list, obstacle_list, sensor_list, egosen_list, npcsen_list
            print('Reset')

    finally:
        # 清洗环境
        print('Start Cleaning Envs')
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
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

# import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from Model.ACOM.normalization import Normalization, RewardScaling
from Model.ACOM.replaybuffer import ReplayBuffer
from Model.ACOM.ppo_discrete_rnn import PPO_discrete_RNN as PPO

import Envs.Crossroad_ENVS as IPPO_ENVS
# import carla
import utils.misc as misc
# import random
import os
import time


parser = argparse.ArgumentParser()
# parser.add_argument('--env', type=str, default='highway')
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument('--seed', default=7304, type=str) # seed
parser.add_argument('--load_seed', default=7304, type=str) # seed

parser.add_argument('--c_tau',  default=1, type=float) # action软更新系数,1代表完全更新，0代表不更新
parser.add_argument('--max_length_of_trajectory', default=400, type=int) # 最大仿真步数
parser.add_argument('--Alearning_rate', default=1e-5, type=float) # Actor学习率
parser.add_argument('--Clearning_rate', default=4e-5, type=float) # Critic学习率
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor

parser.add_argument('--synchronous_mode', default=True, type=bool) # 同步模式开关
parser.add_argument('--no_rendering_mode', default=True, type=bool) # 无渲染模式开关
parser.add_argument('--fixed_delta_seconds', default=0.03, type=float) # 步长,步长建议不大于0.1，为0时代表可变步长

parser.add_argument('--log', default=False, type=bool) # 日志开关
parser.add_argument('--log_interval', default=50, type=int) # 网络保存间隔
parser.add_argument('--update_interval', default=500, type=int) # 网络更新step间隔
parser.add_argument('--load', default=False, type=bool) # 训练模式下是否load model
 
parser.add_argument('--max_episode', default=1800, type=int) # 仿真次数
parser.add_argument('--update_iteration', default = 10, type=int) # 网络迭代次数
args = parser.parse_args()



# 环境建立
if args.mode == 'train':
    print('==========training mode is activated==========')
elif args.mode == 'test':
    print('===========testing mode is activated===========')
else:
    raise NameError("wrong mode!!!")

create_envs = IPPO_ENVS.Create_Envs(args) # 设置仿真模式以及步长
# 状态、动作空间定义
action_space = create_envs.get_action_space()
state_space = create_envs.get_state_space()
state_dim = len(state_space)
action_dim = len(action_space)
actor_num = 2

directory = './carla-Random/'

def main():
    Model = PPO(state_dim, action_dim, args.Alearning_rate, args.Clearning_rate, args.gamma, args.update_iteration, 0.1, True, action_std_init=0.2)
    # npc_PPO = PPO(state_dim, action_dim, args.Alearning_rate, args.Clearning_rate, args.gamma, args.update_iteration, 0.1, True, action_std_init=0.2)
    if not misc.judgeprocess('CarlaUE4.exe'):
        os.startfile('D:\CARLA_0.9.11\WindowsNoEditor\CarlaUE4.exe')
        time.sleep(8)
    world = create_envs.connection()
    # seed = random.randint(0,10000)
    if args.log == True:
        main_writer = SummaryWriter(directory + '/' + args.mode + str(args.seed))
    count = 0
    try:
        if args.load or args.mode == 'test': 
            Model.load(directory + '/' + 'train' + str(args.load_seed) + '/Model.pkl')

        for i in range(args.max_episode):
            ego_total_reward = 0
            npc_total_reward = 0
            ego_offsetx = 0
            npc_offsetx = 0
            ego_offsety = 0
            npc_offsety = 0

            print('------------%dth time learning begins-----------'%i)
            create_envs.Create_actors()

            if args.synchronous_mode:
                world.tick() # 客户端主导，tick
            else:
                world.wait_for_tick() # 服务器主导，tick

            if i == 0:
                # 全局路径
                ego_route, npc_route = create_envs.ego_route()
                misc.draw_waypoints(world,ego_route)
                misc.draw_waypoints(world,npc_route)

            ego_step, npc_step = 1, 1

            ego_state,_,_,npc_state,_,_,_,_,_,_ = create_envs.get_vehicle_step(ego_step, npc_step, 0)
            
            for t in range(99999):
                #---------动作决策----------
                ego_action = Model.select_action(ego_state,npc_state)
                npc_action = Model.select_action(npc_state,ego_state)
                create_envs.set_vehicle_control(ego_action, npc_action)
                #---------和环境交互动作反馈---------
                frames = 1 # 步长 
                if args.synchronous_mode:
                    for _ in range(frames):
                        world.tick() # 客户端主导，tick
                else:
                    world.wait_for_tick() # 服务器主导，tick
                     
                ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done,egocol,ego_fin,npccol,npc_fin = create_envs.get_vehicle_step(ego_step, npc_step, t)

                if ego_next_state[0] > -0.1:
                    ego_step += 1                    
                if npc_next_state[0] > -0.1:
                    npc_step += 1
                # 数据储存
                
                count += 1
                ego_state = ego_next_state
                npc_state = npc_next_state

                ego_total_reward += ego_reward
                npc_total_reward += npc_reward
                ego_offsetx += np.abs(ego_state[0])
                npc_offsetx += np.abs(npc_state[0])
                ego_offsety += np.abs(ego_state[1])
                npc_offsety += np.abs(npc_state[1])
                Model.buffer.rewards.append(ego_reward)
                Model.buffer.rewards.append(npc_reward)

                if args.mode == 'test':
                    if ego_done or npc_done or t==args.max_length_of_trajectory-1: # 结束条件
                        break

                if args.mode == 'train':
                    if ego_done or npc_done or t==args.max_length_of_trajectory-1: # 结束条件
                        Model.buffer.is_terminals.append(True)
                        Model.buffer.is_terminals.append(True)
                        break
                    else:
                        Model.buffer.is_terminals.append(False)
                        Model.buffer.is_terminals.append(False)


            ego_total_reward /= t+1
            npc_total_reward /= t+1
                

            print("Episode: {} step: {} ego_Total_Reward: {:0.3f} npc_Total_Reward: {:0.3f}".format(i+1, t+1, ego_total_reward, npc_total_reward))
            
            if args.log == True:  
                main_writer.add_scalar('reward/ego_reward', ego_total_reward, global_step=i)
                main_writer.add_scalar('reward/npc_reward', npc_total_reward, global_step=i)
                main_writer.add_scalar('step/step', t+1, global_step=i)
                main_writer.add_scalar('step/ego_step', ego_step, global_step=i)
                main_writer.add_scalar('step/npc_step', npc_step, global_step=i)
                main_writer.add_scalar('offset/ego_offsetx', ego_offsetx/(t+1), global_step=i)
                main_writer.add_scalar('offset/ego_offsety', ego_offsety/(t+1), global_step=i)
                main_writer.add_scalar('offset/npc_offsetx', npc_offsetx/(t+1), global_step=i)
                main_writer.add_scalar('offset/npc_offsety', npc_offsety/(t+1), global_step=i)
                main_writer.add_scalar('rate/ego_col', egocol, global_step=i)
                main_writer.add_scalar('rate/npc_col', npccol, global_step=i)
                main_writer.add_scalar('rate/ego_finish', ego_fin, global_step=i)
                main_writer.add_scalar('rate/npc_finsh', npc_fin, global_step=i)    
                
                if args.mode == 'train':
                    if i > 0 and (count+1) >= args.update_interval:
                        ego_loss = Model.update(Model)
                        print('ego_updated')
                        npc_loss = Model.update(Model)
                        print('npc_updated')
                        main_writer.add_scalar('loss/ego_loss', ego_loss, global_step=i)
                        main_writer.add_scalar('loss/npc_loss', npc_loss, global_step=i)
                        Model.clear()
                        Model.clear()
                        count = 0
                    
                    if i > 0 and (i+1) % args.log_interval == 0:
                        Model.save(directory + '/' + args.mode + str(args.seed) + '/' + 'ego.pkl')
                        Model.save(directory + '/' + args.mode + str(args.seed) + '/' + 'npc.pkl')
                        print('Network Saved')

            create_envs.reset()

    finally:
        create_envs.reset()
        print('all clean, simulation done!')


if __name__ == '__main__':
    main()

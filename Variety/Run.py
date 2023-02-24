import numpy as np
from torch.utils.tensorboard import SummaryWriter

import argparse
import random
import utils.misc as misc
import os
import time
from memory_profiler import profile
from Path_Decision.frenet_optimal_trajectory import FrenetPlanner as PathPlanner
from Traj_Decision.BEV_handle.BEV_handle import BEV_handle

class Runner:
    def __init__(self, args):
        # 环境建立
        self.args = args

        if self.args.envs == 'crossroad':
            import Envs.Crossroad_MENVS as ENVS
        elif self.args.envs == 'highway':
            import Envs.Highway_MENVS as ENVS
        elif self.args.envs == 'straight':
            import Envs.Straight_MENVS as ENVS
        elif self.args.envs == 'onramp':
            import Envs.OnRamp_MENVS as ENVS
        elif self.args.envs == 'roundabout':
            import Envs.Roundabout_MENVS as ENVS
        elif self.args.envs == 'tjunction':
            import Envs.Tjunction_MENVS as ENVS
        elif self.args.envs == 'circle':
            import Envs.Circle_MENVS as ENVS
        
        if self.args.model == 'OMAC':
            from Traj_Decision.ACOM.normalization import Normalization, RewardScaling
            from Traj_Decision.ACOM.replaybuffer import ReplayBuffer
            from Traj_Decision.ACOM.ACOM import PPO_RNN as Model
        elif self.args.model == 'IPPO':
            from Traj_Decision.IPPO.normalization import Normalization, RewardScaling
            from Traj_Decision.IPPO.replaybuffer import ReplayBuffer
            from Traj_Decision.IPPO.ppo_rnn import PPO_RNN as Model
        elif self.args.model == 'PR2AC':
            from Traj_Decision.PR2AC.normalization import Normalization, RewardScaling
            from Traj_Decision.PR2AC.replaybuffer import ReplayBuffer
            from Traj_Decision.PR2AC.PR2AC import PPO_RNN as Model
        elif self.args.model == 'SAC':
            from Traj_Decision.SAC.normalization import Normalization, RewardScaling
            from Traj_Decision.SAC.SAC import SAC as Model
            from Traj_Decision.SAC.SAC import ReplayBuffer
        elif self.args.model == 'MADDPG':
            from Traj_Decision.MADDPG.maddpg import MADDPG as Model
            from Traj_Decision.MADDPG.buffer import ReplayBuffer

        self.directory = './carla-Variety/'
        self.save_directory = self.directory + '/' + self.args.mode + '_' + self.args.envs + '_' + self.args.model + '_' + 'gru'*self.args.use_gru + 'lstm'*self.args.use_lstm + 'nornn'*(1-self.args.use_gru-self.args.use_lstm) + '_' + str(self.args.save_seed)
        self.args.save_directory = self.save_directory

        # Create env
        if not misc.judgeprocess('CarlaUE4.exe'):
            os.startfile('D:\CARLA_0.9.11\WindowsNoEditor\CarlaUE4.exe')
            time.sleep(15)
        self.create_envs = ENVS.Create_Envs(self.args,self.save_directory) # 设置仿真模式以及步长
        self.world = self.create_envs.connection()
        self.agent_num = self.args.agent_num # 根据场景创建agent_num个agent

        # Create a tensorboard
        self.writer = SummaryWriter(self.save_directory)

        # 状态、动作空间定义
        action_space = self.create_envs.get_action_space()
        state_space = self.create_envs.get_state_space()
        self.args.state_dim = len(state_space)
        self.args.action_dim = len(action_space)
        self.args.episode_limit = self.args.max_length_of_trajectory # Maximum number of steps per episode
        # self.ego_evaluate_rewards = []  # Record the rewards during the evaluating
        # self.npc_evaluate_rewards = []
        self.total_steps = 0
        self.total_episode = 0
        self.buffer = list(np.zeros(self.args.max_agent_num,dtype=int))
        self.policy = list(np.zeros(self.args.max_agent_num,dtype=int))
        self.pp = list(np.zeros(self.args.max_agent_num,dtype=int))
        self.num = list(999*np.ones(self.args.max_agent_num,dtype=int))
        for i in range(self.args.max_agent_num):
            self.buffer[i] = ReplayBuffer(self.args)
            if self.args.shared_policy:
                # self.ego = Model(self.args)
                self.policy[i] = Model(self.args) if i == 0 else self.policy[i - 1]
            else:
                self.policy[i] = Model(self.args)
            self.pp[i] = PathPlanner(self.args)
        self.BEV_handle = BEV_handle()

        if self.args.use_state_norm:
            print("------use state normalization------")
            self.state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)                
        
    # @profile(stream=open('memory_profile.log','w+'))
    def run(self, ):
        try:
            if self.args.mode == 'test':
                for i in range(self.agent_num): 
                    self.policy[i].load(self.directory + '/' + 'train' + '_' + self.args.envs + '_' + self.args.model + '_' + 'gru'*self.args.use_gru + 'lstm'*self.args.use_lstm + 'nornn'*(1-self.args.use_gru-self.args.use_lstm) + '_' + str(self.args.load_seed) + '/' + str(i))
                # if self.args.npc_exist:
                #     self.npc.load(self.directory + '/' + 'train' + '_' + self.args.envs + '_' + self.args.model + '_' + 'gru'*self.args.use_gru + 'lstm'*self.args.use_lstm + 'nornn'*(1-self.args.use_gru-self.args.use_lstm) + '_' + str(self.args.load_seed) + '/npc.pkl')
            
            evaluate_num = -1  # Record the number of evaluations
            save_num = 1
            flag = 0
            
            for i in range(self.args.max_episode):
                if self.args.random:
                    if self.args.envs == 'highway':
                        import Envs.Highway_MENVS as ENVS
                    rnd = random.randint(2,3)
                    self.args.agent_num = rnd
                    if not rnd==self.agent_num:
                        flag = 1
                        self.create_envs = ENVS.Create_Envs(self.args,self.save_directory) # 设置仿真模式以及步长
                        self.world = self.create_envs.connection()
                        self.agent_num = self.args.agent_num
                self.create_envs.Create_actors()
                frames = 1 # 步长 
                if self.args.synchronous_mode:
                    for _ in range(frames):
                        self.world.tick() # 客户端主导，tick
                        # print('frame:', ff)
                else:
                    self.world.wait_for_tick() # 服务器主导，tick
                # print('settings:',self.world.get_settings())
                self.total_episode += 1
                if i==0 or flag==1:
                    # 全局路径
                    route, num = self.create_envs.get_route()
                    self.num = num
                    # print('route_num:',ego_num,npc_num)
                    for j in range(self.agent_num):
                        if self.args.Start_Path:
                            self.pp[j].start(route[j])
                        # self.npc_pp.start(npc_route)                    
                        misc.draw_waypoints(self.world,route[j])

                if self.total_episode // self.args.evaluate_freq > evaluate_num:
                    self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                    evaluate_num += 1
                
                episode_steps = self.run_episode()  # Run an episode
                self.total_steps += episode_steps

                if self.args.mode == 'train':
                    if self.buffer[0].episode_num >= self.args.batch_size:
                        print('------start_training-------')
                        if self.args.shared_policy:
                            self.policy[0].train(self.total_steps, self.buffer, 0)  # Training
                        else:
                            for j in range(self.agent_num):
                                self.policy[j].train(self.total_steps, self.buffer, j)  # Training                                                                
                    
                        if not self.args.model=='SAC':
                            for j in range(self.args.max_agent_num):
                                self.buffer[j].reset_buffer()

                    # Save the models
                    if self.total_episode // self.args.save_freq > save_num:
                        for j in range(self.agent_num):
                            self.policy[j].save(self.save_directory + '/' + str(j))
                        save_num += 1
                        print('Network Saved')
            
            self.create_envs.Create_actors()
            self.evaluate_policy()
        finally:    
            self.create_envs.reset()
    # @profile(stream=open('memory_profile.log','w+'))
    def run_episode(self, ):
        print('------start_simulating-------')
        total_reward = list(np.zeros(self.agent_num,dtype=int))
        # npc_total_reward = 0
        offsetx = list(np.zeros(self.agent_num,dtype=int))
        # npc_offsetx = 0
        offsety = list(np.zeros(self.agent_num,dtype=int))
        # npc_offsety = 0
        dw = list(np.zeros(self.agent_num,dtype=int))
        # ego_wps,npc_wps = 0, 0
        step_list = list(np.ones(self.agent_num,dtype=int))
        if self.args.synchronous_mode:
            self.world.tick() # 客户端主导，tick
        else:
            self.world.wait_for_tick() # 服务器主导，tick
        data = self.create_envs.get_vehicle_step(step_list, 0)
        last_state = list(np.ones(self.agent_num,dtype=int))
        last_BEV = list(np.ones(self.agent_num,dtype=int))
        a_list = list(np.zeros([self.args.max_agent_num,self.args.action_dim],dtype=int))
        v_list = list(np.zeros(self.agent_num,dtype=int))
        a_logprob = list(np.ones(self.agent_num,dtype=int))
        p = list(np.ones(self.agent_num,dtype=int))
        last_step = list(np.ones(self.agent_num,dtype=int))
        episode_reward = list(np.ones(self.agent_num,dtype=int))
        final_step = list(np.ones(self.agent_num,dtype=int))
        for j in range(self.agent_num):
            data[j][-1] = self.BEV_handle.act(data[j][-1])
            last_state[j] = data[j][0]
            last_BEV[j] = data[j][-1]
            if self.args.use_gru or self.args.use_lstm:
                self.policy[j].reset_rnn_hidden()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        
        # self.policy.reset_rnn_hidden()
        for episode_step in range(999999):
                # npc_state = self.state_norm(npc_state)
            for j in range(self.agent_num):
                if self.args.use_state_norm:
                    data[j][0] = self.state_norm(data[j][0])
                a_list[j], a_logprob[j] = self.policy[j].choose_action(data[j][0], data[j][-1], evaluate=False)
                # npc_a, npc_a_logprob, p = self.npc.choose_action(npc_state, npc_BEV, evaluate=False)
                
                # npc_v = self.npc.get_value(npc_state, npc_BEV, ego_a)
                # a_list[j] = ego_a
            if not self.args.model=='SAC': 
                for j in range(self.agent_num):
                    om_list = a_list.copy()
                    om_list.pop(j)
                    v_list[j] = self.policy[j].get_value(data[j][0], data[j][-1], np.array(om_list).ravel())
            self.create_envs.set_vehicle_control(a_list, step_list)

            #---------和环境交互动作反馈---------
            frames = 1 # 步长 
            if self.args.synchronous_mode:
                for _ in range(frames):
                    self.world.tick() # 客户端主导，tick
                    # print('frame:', ff)
            else:
                self.world.wait_for_tick() # 服务器主导，tick
            # self.data_queue.put(self.create_envs.get_vehicle_step(ego_step,npc_step,episode_step))
            data = self.create_envs.get_vehicle_step(step_list,episode_step)

            for j in range(self.agent_num):
                data[j][-1] = self.BEV_handle.act(data[j][-1])
                if data[j][0][0] > -0.1:
                    step_list[j] += 1                    
            # if npc_next_state[0] > -0.1:
            #     npc_step += 1
            
                total_reward[j] += data[j][1]
            # npc_total_reward += npc_reward

                if data[j][3]:
                    dw[j] += True
            # if npc_fin:
            #     npc_dw += True        
                if data[j][2] or episode_step==self.args.max_length_of_trajectory-1:
                    if not dw[j]:
                        dw[j] += True
                # if not npc_dw:
                #     npc_dw += True

                if self.args.use_reward_scaling:
                    data[j][1] = self.reward_scaling(data[j][1])
                # npc_reward = self.reward_scaling(npc_reward)

            # Store the transition, only store once after finish
                if dw[j] <= 1:
                    # print([episode_step, ego_state, p, ego_v, ego_a, ego_a_logprob, ego_reward, ego_dw])
                    # print('aaaaaaaa: ',len(data),len(p),len(v_list),len(a_list),len(a_logprob),len(dw))
                    self.buffer[j].store_transition(episode_step, last_state[j], last_BEV[j], v_list[j], a_list[j], a_logprob[j], data[j][1], data[j][0], data[j][-1], dw[j])
                    last_state[j] = data[j][0]
                    last_BEV[j] = data[j][-1]
                    last_step[j] = episode_step
                    episode_reward[j] = total_reward[j]
                    final_step[j] = step_list[j]
                    offsetx[j] += np.abs(data[j][0][0])
                    offsety[j] += np.abs(data[j][0][1])

            # if npc_dw <= 1:
            #     self.npc_buffer.store_transition(episode_step, npc_state, p, npc_v, npc_a, npc_a_logprob, npc_reward, npc_dw)
            #     npc_last_state = npc_next_state
            #     npc_last_BEV = npc_next_BEV
            #     npc_last_step = episode_step
            #     npc_episode_reward = npc_total_reward
            #     npc_final_step = npc_step
            #     npc_offsetx += np.abs(npc_next_state[0])
            #     npc_offsety += np.abs(npc_next_state[1])

            # ego_state = ego_next_state
            # npc_state = npc_next_state
            # ego_BEV = ego_next_BEV
            # npc_BEV = npc_next_BEV

            if any(dw): # 结束条件
                break

        # An episode is over, store v in the last step
        for j in range(self.agent_num):
            if self.args.use_state_norm:
                last_state[j] = self.state_norm(last_state[j])
                # npc_last_state = self.state_norm(npc_last_state)
            if not self.args.model=='SAC':
                v = self.policy[j].get_value(last_state[j],last_BEV[j],None)
                # npc_v = self.npc.get_value(npc_last_state,npc_last_BEV, None)
                self.buffer[j].store_last_sv(last_step[j] + 1, v, last_state[j], last_BEV[j])
                # self.npc_buffer.store_last_sv(npc_last_step + 1, npc_v, npc_last_state, p)

            self.writer.add_scalar('reward/'+str(j)+'_train_rewards', episode_reward[j], global_step=self.total_episode)
            # self.writer.add_scalar('reward/npc_train_rewards', npc_episode_reward/(npc_last_step + 1), global_step=self.total_episode)
            self.writer.add_scalar('step/'+str(j)+'_train_step', final_step[j]/(self.num[j]-3), global_step=self.total_episode)
            # self.writer.add_scalar('step/npc_train_step', npc_final_step/(self.npc_num-3), global_step=self.total_episode)
            self.writer.add_scalar('offset/'+str(j)+'_train_offsetx', offsetx[j]/(last_step[j] + 1), global_step=self.total_episode)
            self.writer.add_scalar('offset/'+str(j)+'_train_offsety', offsety[j]/(last_step[j] + 1), global_step=self.total_episode)
            # self.writer.add_scalar('offset/npc_train_offsetx', npc_offsetx/(ego_last_step + 1), global_step=self.total_episode)
            # self.writer.add_scalar('offset/npc_train_offsety', npc_offsety/(ego_last_step + 1), global_step=self.total_episode)
            self.writer.add_scalar('rate/'+str(j)+'_train_col', data[j][2], global_step=self.total_episode)
            # self.writer.add_scalar('rate/npc_train_col', npccol, global_step=self.total_episode)
            self.writer.add_scalar('rate/'+str(j)+'_train_finish', data[j][3], global_step=self.total_episode)
            # self.writer.add_scalar('rate/npc_train_finsh', npc_fin, global_step=self.total_episode)
        self.writer.add_scalar('step/train_step', episode_step, global_step=self.total_episode)
        print("Episode: {} step: {} ".format(self.total_episode, episode_step+1))
        self.create_envs.clean()
        return episode_step + 1
    
    def evaluate_policy(self, ):
        print('------start_evaluating-------')
        evaluate_reward = list(np.zeros(self.agent_num,dtype=int))
        # npc_evaluate_reward = 0
        
        total_offsetx = list(np.zeros(self.agent_num,dtype=int))
        # npc_total_offsetx = 0
        total_offsety = list(np.zeros(self.agent_num,dtype=int))
        # npc_total_offsety = 0
        evaluate_step = 0
        total_fin = list(np.zeros(self.agent_num,dtype=int))
        total_col = list(np.zeros(self.agent_num,dtype=int))
        # npc_total_fin = 0
        # npc_total_col = 0
        dw = list(np.zeros(self.agent_num,dtype=int))
        a_list = list(np.ones([self.agent_num,self.args.action_dim],dtype=int))
        last_step = list(np.ones(self.agent_num,dtype=int))
        episode_reward = list(np.ones(self.agent_num,dtype=int))
        final_step = list(np.ones(self.agent_num,dtype=int))

        for _ in range(self.args.evaluate_times):
            episode_reward = list(np.zeros(self.agent_num,dtype=int))
            # npc_episode_reward = 0
            offsetx = list(np.zeros(self.agent_num,dtype=int))
            # npc_offsetx = 0
            offsety = list(np.zeros(self.agent_num,dtype=int))
            # npc_offsety = 0
            step_list = list(np.ones(self.agent_num,dtype=int))
            frames = 1 # 步长 
            if self.args.synchronous_mode:
                for _ in range(frames):
                    self.world.tick() # 客户端主导，tick
                    # print('frame:', ff)
            else:
                self.world.wait_for_tick() # 服务器主导，tick
            data = self.create_envs.get_vehicle_step(step_list, 0)
            
            for j in range(self.agent_num):
                if self.args.use_gru or self.args.use_lstm:
                    self.policy[j].reset_rnn_hidden()
            # self.npc.reset_rnn_hidden()

            for episode_step in range(99999):
                for j in range(self.agent_num):
                    if self.args.use_state_norm:
                        data[j][0] = self.state_norm(data[j][0], update=False)
                        # npc_state = self.state_norm(npc_state, update=False)
                    # t1 = time.time()
                    data[j][-1] = self.BEV_handle.act(data[j][-1])
                    a_list[j], _ = self.policy[j].choose_action(data[j][0],data[j][-1], evaluate=True)
                    # print('time:    ',time.time()-t1)
                    # npc_a, _, _ = self.npc.choose_action(npc_state,npc_BEV, evaluate=True)
                self.create_envs.set_vehicle_control(a_list, step_list)
                frames = 1 # 步长 
                if self.args.synchronous_mode:
                    for _ in range(frames):
                        self.world.tick() # 客户端主导，tick
                else:
                    self.world.wait_for_tick() # 服务器主导，tick
                    # self.data_queue.put(self.create_envs.get_vehicle_step(ego_step,npc_step,episode_step))
                data = self.create_envs.get_vehicle_step(step_list,episode_step)

                for j in range(self.agent_num):
                    if data[j][0][0] > -0.1:
                        step_list[j] += 1
                    # if npc_next_state[0] > -0.1:
                    #     npc_step += 1
                    if data[j][3]:
                        dw[j] += True
                    # if npc_fin:
                    #     npc_dw += True
                    if data[j][2] or episode_step==self.args.max_length_of_trajectory-1:
                        if not dw[j]:
                            dw[j] += True
                        # if not npc_dw:
                        #     npc_dw += True

                    if dw[j] <= 1:
                        last_step[j] = episode_step
                        episode_reward[j] += data[j][1]
                        final_step[j] = step_list[j]
                        offsetx[j] += np.abs(data[j][0][0])
                        offsety[j] += np.abs(data[j][0][1])
                    # if npc_dw < 2:
                    #     npc_last_step = episode_step
                    #     npc_offsetx += np.abs(npc_next_state[0])
                    #     npc_offsety += np.abs(npc_next_state[1])
                    #     npc_episode_reward += npc_reward
                    #     npc_final_step = npc_step

                    # ego_state = data[j][0]
                    # npc_state = npc_next_state
                    # ego_BEV = data[j][-1]
                    # npc_BEV = npc_next_BEV

                if any(dw): # 结束条件
                    break
            print("Episode: {} step: {}".format(self.total_episode, episode_step+1))

            for j in range(self.agent_num):
                evaluate_reward[j] += (episode_reward[j])
                # npc_evaluate_reward += (npc_episode_reward/(npc_last_step + 1))
                total_offsetx[j] += (offsetx[j]/(last_step[j] + 1))
                # npc_total_offsetx += (npc_offsetx/(npc_last_step + 1))
                total_offsety[j] += (offsety[j]/(last_step[j] + 1))
                # npc_total_offsety += (npc_offsety/(npc_last_step + 1))
                total_fin[j] += data[j][3]
                total_col[j] += data[j][2]
                # npc_total_fin += npc_fin
                # npc_total_col += npccol
            evaluate_step += (episode_step+1)

        evaluate_reward = [a/self.args.evaluate_times for a in evaluate_reward]
        # npc_evaluate_reward /= self.args.evaluate_times
        total_offsetx = [a/self.args.evaluate_times for a in total_offsetx]
        # npc_total_offsetx /= self.args.evaluate_times
        total_offsety = [a/self.args.evaluate_times for a in total_offsety]
        # npc_total_offsety /= self.args.evaluate_times
        evaluate_step /= self.args.evaluate_times
        total_fin = [a/self.args.evaluate_times for a in total_fin]
        total_col = [a/self.args.evaluate_times for a in total_col]
        # npc_total_fin /= self.args.evaluate_times
        # npc_total_col /= self.args.evaluate_times

        for j in range(self.agent_num):
            self.writer.add_scalar('reward/'+str(j)+'_evaluate_rewards', evaluate_reward[j], global_step=self.total_episode)
            # self.writer.add_scalar('reward/npc_evaluate_rewards', npc_evaluate_reward, global_step=self.total_episode)
            self.writer.add_scalar('step/'+str(j)+'_step', final_step[j]/(self.num[j]-3), global_step=self.total_episode)
            # self.writer.add_scalar('step/npc_step', npc_final_step/(self.npc_num-3), global_step=self.total_episode)
            self.writer.add_scalar('offset/'+str(j)+'_offsetx', total_offsetx[j], global_step=self.total_episode)
            self.writer.add_scalar('offset/'+str(j)+'_offsety', total_offsety[j], global_step=self.total_episode)
            # self.writer.add_scalar('offset/npc_offsetx', npc_total_offsetx, global_step=self.total_episode)
            # self.writer.add_scalar('offset/npc_offsety', npc_total_offsety, global_step=self.total_episode)
            self.writer.add_scalar('rate/'+str(j)+'_col', total_col[j], global_step=self.total_episode)
            # self.writer.add_scalar('rate/npc_col', npc_total_col, global_step=self.total_episode)
            self.writer.add_scalar('rate/'+str(j)+'_finish', total_fin[j], global_step=self.total_episode)
            # self.writer.add_scalar('rate/npc_finsh', npc_total_fin, global_step=self.total_episode)
        self.writer.add_scalar('step/step', evaluate_step, global_step=self.total_episode)

        self.create_envs.clean()
        self.create_envs.Create_actors()
        frames = 1 # 步长 
        if self.args.synchronous_mode:
            for _ in range(frames):
                self.world.tick() # 客户端主导，tick
                # print('frame:', ff)
        else:
            self.world.wait_for_tick() # 服务器主导，tick


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting")
    parser.add_argument("--evaluate_freq", type=float, default=16, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=200, help="Save frequency")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")

    parser.add_argument("--Start_Path", type=bool, default=False, help="Start_Path")
    # parser.add_argument("--carla_dt", type=float, default=0.05, help="dt")
    parser.add_argument("--carla_lane_width", type=float, default=3.5, help="lane_width")
    parser.add_argument("--carla_max_s", type=int, default=8, help="max_s")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--hidden_dim1", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--hidden_dim2", type=int, default=32, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--init_std", type=float, default=0.15, help="std_initialization")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.97, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=1, help="PPO parameter")
    parser.add_argument("--M", type=int, default=10, help="sample_times")
    parser.add_argument("--N", type=int, default=20, help="sample_times")
    parser.add_argument("--use_adv_norm", type=bool, default=False, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")
    parser.add_argument("--use_gru", type=bool, default=False, help="Whether to use GRU") # Priority for GRU
    parser.add_argument("--use_lstm", type=bool, default=False, help="Whether to use LSTM")
    parser.add_argument("--shared_policy", type=bool, default=True, help="Whether to share policy")
    parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
    parser.add_argument('--save_seed', default=1, type=str) # seed
    parser.add_argument('--load_seed', default=1, type=str) # seed
    parser.add_argument('--c_tau',  default=1, type=float) # action软更新系数,1代表完全更新，0代表不更新
    parser.add_argument('--max_length_of_trajectory', default=300, type=int) # 最大仿真步数
    parser.add_argument('--res', default=5, type=int) # pixel per meter
    parser.add_argument('--H', default=224, type=int) # BEV_Height
    parser.add_argument('--W', default=56, type=int) # BEV_Width

    parser.add_argument('--envs', default='highway', type=str) # 环境选择crossroad,highway,straight,onramp,roundabout,tjunction,circle
    parser.add_argument('--random', default=False, type=bool) # random-training
    parser.add_argument('--model', default='SAC', type=str) # 模型选择OMAC、IPPO、MAPPO、MADDPG、PR2AC、Rules
    parser.add_argument('--agent_num', default=2, type=int) # 当前智能体个数
    parser.add_argument('--max_agent_num', default=2, type=int) # 最大智能体个数
    parser.add_argument('--controller', default=2, type=int) # /单点跟踪控制：1/双点跟踪控制：2
    parser.add_argument('--pure_track', default=False, type=bool) # 纯跟踪/
    parser.add_argument('--control_mode', default=1, type=int) # /PID控制（优化点）：0/直接控制：1/混合控制（优化控制量）：2
    parser.add_argument('--synchronous_mode', default=True, type=bool) # 同步模式开关
    parser.add_argument('--no_rendering_mode', default=True, type=bool) # 无渲染模式开关
    parser.add_argument('--fixed_delta_seconds', default=0.03, type=float) # 步长,步长建议不大于0.1，为0时代表可变步长
    parser.add_argument('--max_episode', default=5000, type=int) # 仿真次数
    args = parser.parse_args()

    runner = Runner(args)
    runner.run()

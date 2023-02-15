import numpy as np
from torch.utils.tensorboard import SummaryWriter

import argparse

import utils.misc as misc
import os
import time
from memory_profiler import profile
from Path_Decision.frenet_optimal_trajectory import FrenetPlanner as PathPlanner

class Runner:
    def __init__(self, args):
        # 环境建立
        self.args = args

        if self.args.envs == 'crossroad':
            import Envs.Crossroad_ENVS as ENVS
        elif self.args.envs == 'highway':
            import Envs.Highway_ENVS as ENVS
        elif self.args.envs == 'straight':
            import Envs.Straight_ENVS as ENVS
        
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

        self.directory = './carla-Random/'
        self.save_directory = self.directory + '/' + self.args.mode + '_' + self.args.envs + '_' + self.args.model + '_' + 'gru'*self.args.use_gru + 'lstm'*self.args.use_lstm + 'nornn'*(1-self.args.use_gru-self.args.use_lstm) + '_' + str(self.args.save_seed)
        # Create env
        if not misc.judgeprocess('CarlaUE4.exe'):
            os.startfile('D:\CARLA_0.9.11\WindowsNoEditor\CarlaUE4.exe')
            time.sleep(15)
        self.create_envs = ENVS.Create_Envs(self.args,self.save_directory) # 设置仿真模式以及步长
        self.world = self.create_envs.connection()

        # Create a tensorboard
        self.writer = SummaryWriter(self.save_directory)

        # 状态、动作空间定义
        action_space = self.create_envs.get_action_space()
        state_space = self.create_envs.get_state_space()
        self.args.state_dim = len(state_space)
        self.args.action_dim = len(action_space)
        self.args.episode_limit = self.args.max_length_of_trajectory # Maximum number of steps per episode
        self.ego_evaluate_rewards = []  # Record the rewards during the evaluating
        self.npc_evaluate_rewards = []
        self.total_steps = 0
        self.total_episode = 0
        self.ego_buffer = ReplayBuffer(self.args)
        self.npc_buffer = ReplayBuffer(self.args)
        
        if self.args.shared_policy:
            self.ego = Model(self.args)
            self.npc = self.ego
        else:
            self.ego = Model(self.args)
            self.npc = Model(self.args)

        if self.args.use_state_norm:
            print("------use state normalization------")
            self.state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)
        
        self.ego_num,self.npc_num = 999, 999

        self.ego_pp = PathPlanner(self.args)
        self.npc_pp = PathPlanner(self.args)
    # @profile(stream=open('memory_profile.log','w+'))
    def run(self, ):
        try:
            if self.args.mode == 'test': 
                self.ego.load(self.directory + '/' + 'train' + '_' + self.args.envs + '_' + self.args.model + '_' + 'gru'*self.args.use_gru + 'lstm'*self.args.use_lstm + 'nornn'*(1-self.args.use_gru-self.args.use_lstm) + '_' + str(self.args.load_seed) + '/ego.pkl')
                if self.args.npc_exist:
                    self.npc.load(self.directory + '/' + 'train' + '_' + self.args.envs + '_' + self.args.model + '_' + 'gru'*self.args.use_gru + 'lstm'*self.args.use_lstm + 'nornn'*(1-self.args.use_gru-self.args.use_lstm) + '_' + str(self.args.load_seed) + '/npc.pkl')
            
            evaluate_num = -1  # Record the number of evaluations
            save_num = 1

            
            for i in range(self.args.max_episode):
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
                if i == 0:
                    # 全局路径
                    ego_route, npc_route, ego_num, npc_num = self.create_envs.get_route()
                    self.ego_num,self.npc_num = ego_num, npc_num
                    # print('route_num:',ego_num,npc_num)
                    if self.args.Start_Path:
                        self.ego_pp.start(ego_route)
                        self.npc_pp.start(npc_route)
                    
                    misc.draw_waypoints(self.world,ego_route)
                    misc.draw_waypoints(self.world,npc_route)

                if self.total_episode // self.args.evaluate_freq > evaluate_num:
                    self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                    evaluate_num += 1
                
                _, _, episode_steps = self.run_episode()  # Run an episode
                self.total_steps += episode_steps

                if self.args.mode == 'train':
                    if self.ego_buffer.episode_num >= self.args.batch_size:
                        print('------start_training-------')
                        if self.args.shared_policy:
                            self.ego.train(self.total_steps, self.ego_buffer, self.npc_buffer)  # Training
                        else:
                            self.ego.train(self.total_steps, self.ego_buffer, self.npc_buffer)  # Training
                            if self.args.npc_exist:
                                self.npc.train(self.total_steps, self.npc_buffer, self.ego_buffer)  # Training
                    
                        self.ego_buffer.reset_buffer()
                        self.npc_buffer.reset_buffer()

                    # Save the models
                    if self.total_episode // self.args.save_freq > save_num:
                        self.ego.save(self.save_directory + '/ego.pkl')
                        if self.args.npc_exist:
                            self.npc.save(self.save_directory + '/npc.pkl')
                        save_num += 1
                        print('Network Saved')
            
            self.create_envs.Create_actors()
            self.evaluate_policy()
        finally:    
            self.create_envs.reset()
    # @profile(stream=open('memory_profile.log','w+'))
    def run_episode(self, ):
        print('------start_simulating-------')
        ego_total_reward = 0
        npc_total_reward = 0
        ego_offsetx = 0
        npc_offsetx = 0
        ego_offsety = 0
        npc_offsety = 0
        ego_dw, npc_dw = False, False
        ego_wps,npc_wps = 0, 0

        ego_step, npc_step = 1, 1
        ego_state,_,npc_state,_,_,_,_,_,ego_BEV,npc_BEV = self.create_envs.get_vehicle_step(ego_step, npc_step, 0)
        ego_last_state, npc_last_state = ego_state, npc_state
        ego_last_BEV, npc_last_BEV = ego_BEV, npc_BEV

        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        self.ego.reset_rnn_hidden()
        self.npc.reset_rnn_hidden()
        for episode_step in range(999999):
            if self.args.use_state_norm:
                ego_state = self.state_norm(ego_state)
                npc_state = self.state_norm(npc_state)

            ego_a, ego_a_logprob, p = self.ego.choose_action(ego_state, ego_BEV, evaluate=False)
            npc_a, npc_a_logprob, p = self.npc.choose_action(npc_state, npc_BEV, evaluate=False)
            ego_v = self.ego.get_value(ego_state, ego_BEV, npc_a)
            npc_v = self.npc.get_value(npc_state, npc_BEV, ego_a)
            self.create_envs.set_vehicle_control(ego_a, npc_a, ego_step, npc_step)

            #---------和环境交互动作反馈---------
            frames = 1 # 步长 
            if self.args.synchronous_mode:
                for _ in range(frames):
                    self.world.tick() # 客户端主导，tick
                    # print('frame:', ff)
            else:
                self.world.wait_for_tick() # 服务器主导，tick
            # self.data_queue.put(self.create_envs.get_vehicle_step(ego_step,npc_step,episode_step))
            ego_next_state,ego_reward,npc_next_state,npc_reward,egocol,ego_fin,npccol,npc_fin,ego_next_BEV,npc_next_BEV = self.create_envs.get_vehicle_step(ego_step,npc_step,episode_step)

            if ego_next_state[0] > -0.1:
                ego_step += 1                    
            if npc_next_state[0] > -0.1:
                npc_step += 1
            
            ego_total_reward += ego_reward
            npc_total_reward += npc_reward         

            if ego_fin:
                ego_dw += True
            if npc_fin:
                npc_dw += True
            if egocol or npccol or episode_step==self.args.max_length_of_trajectory-1 or ego_step >= self.ego_num-1 or npc_step >= self.npc_num-1:
                if not ego_dw:
                    ego_dw += True
                if not npc_dw:
                    npc_dw += True

            if self.args.use_reward_scaling:
                ego_reward = self.reward_scaling(ego_reward)
                npc_reward = self.reward_scaling(npc_reward)

            # Store the transition, only store once after finish
            if ego_dw <= 1:
                # print([episode_step, ego_state, p, ego_v, ego_a, ego_a_logprob, ego_reward, ego_dw])
                self.ego_buffer.store_transition(episode_step, ego_state, p, ego_v, ego_a, ego_a_logprob, ego_reward, ego_dw)
                ego_last_state = ego_next_state
                ego_last_BEV = ego_next_BEV
                ego_last_step = episode_step
                ego_episode_reward = ego_total_reward
                ego_final_step = ego_step
                ego_offsetx += np.abs(ego_next_state[0])
                ego_offsety += np.abs(ego_next_state[1])

            if npc_dw <= 1:
                self.npc_buffer.store_transition(episode_step, npc_state, p, npc_v, npc_a, npc_a_logprob, npc_reward, npc_dw)
                npc_last_state = npc_next_state
                npc_last_BEV = npc_next_BEV
                npc_last_step = episode_step
                npc_episode_reward = npc_total_reward
                npc_final_step = npc_step
                npc_offsetx += np.abs(npc_next_state[0])
                npc_offsety += np.abs(npc_next_state[1])

            ego_state = ego_next_state
            npc_state = npc_next_state
            ego_BEV = ego_next_BEV
            npc_BEV = npc_next_BEV

            if ego_dw and npc_dw: # 结束条件
                break

        # An episode is over, store v in the last step
        if self.args.use_state_norm:
            ego_last_state = self.state_norm(ego_last_state)
            npc_last_state = self.state_norm(npc_last_state)
        ego_v = self.ego.get_value(ego_last_state,ego_last_BEV, None)
        npc_v = self.npc.get_value(npc_last_state,npc_last_BEV, None)
        self.ego_buffer.store_last_sv(ego_last_step + 1, ego_v, ego_last_state, p)
        self.npc_buffer.store_last_sv(npc_last_step + 1, npc_v, npc_last_state, p)

        self.writer.add_scalar('reward/ego_train_rewards', ego_episode_reward/(ego_last_step + 1), global_step=self.total_episode)
        self.writer.add_scalar('reward/npc_train_rewards', npc_episode_reward/(npc_last_step + 1), global_step=self.total_episode)
        self.writer.add_scalar('step/train_step', episode_step, global_step=self.total_episode)
        self.writer.add_scalar('step/ego_train_step', ego_final_step/(self.ego_num-3), global_step=self.total_episode)
        self.writer.add_scalar('step/npc_train_step', npc_final_step/(self.npc_num-3), global_step=self.total_episode)
        self.writer.add_scalar('offset/ego_train_offsetx', ego_offsetx/(ego_last_step + 1), global_step=self.total_episode)
        self.writer.add_scalar('offset/ego_train_offsety', ego_offsety/(ego_last_step + 1), global_step=self.total_episode)
        self.writer.add_scalar('offset/npc_train_offsetx', npc_offsetx/(ego_last_step + 1), global_step=self.total_episode)
        self.writer.add_scalar('offset/npc_train_offsety', npc_offsety/(ego_last_step + 1), global_step=self.total_episode)
        self.writer.add_scalar('rate/ego_train_col', egocol, global_step=self.total_episode)
        self.writer.add_scalar('rate/npc_train_col', npccol, global_step=self.total_episode)
        self.writer.add_scalar('rate/ego_train_finish', ego_fin, global_step=self.total_episode)
        self.writer.add_scalar('rate/npc_train_finsh', npc_fin, global_step=self.total_episode)
        print("Episode: {} step: {} ego_Total_Reward: {:0.3f} npc_Total_Reward: {:0.3f}".format(self.total_episode, episode_step+1, ego_episode_reward/(ego_last_step + 1), npc_episode_reward/(npc_last_step + 1)))

        self.create_envs.clean()
        return ego_total_reward/(episode_step + 1), npc_total_reward/(episode_step + 1), episode_step + 1
    
    def evaluate_policy(self, ):
        print('------start_evaluating-------')
        ego_evaluate_reward = 0
        npc_evaluate_reward = 0
        
        ego_total_offsetx = 0
        npc_total_offsetx = 0
        ego_total_offsety = 0
        npc_total_offsety = 0
        evaluate_step = 0
        ego_total_fin = 0
        ego_total_col = 0
        npc_total_fin = 0
        npc_total_col = 0
        ego_dw, npc_dw = False, False

        for _ in range(self.args.evaluate_times):
            ego_episode_reward = 0
            npc_episode_reward = 0
            ego_offsetx = 0
            npc_offsetx = 0
            ego_offsety = 0
            npc_offsety = 0
            ego_step, npc_step = 1, 1
            frames = 1 # 步长 
            if self.args.synchronous_mode:
                for _ in range(frames):
                    self.world.tick() # 客户端主导，tick
                    # print('frame:', ff)
            else:
                self.world.wait_for_tick() # 服务器主导，tick
            ego_state,_,npc_state,_,_,_,_,_,ego_BEV,npc_BEV = self.create_envs.get_vehicle_step(ego_step, npc_step, 0)
            
            self.ego.reset_rnn_hidden()
            self.npc.reset_rnn_hidden()

            for episode_step in range(99999):
                if self.args.use_state_norm:
                    ego_state = self.state_norm(ego_state, update=False)
                    npc_state = self.state_norm(npc_state, update=False)
                t1 = time.time()
                ego_a, _, _ = self.ego.choose_action(ego_state,ego_BEV, evaluate=True)
                # print('time:    ',time.time()-t1)
                npc_a, _, _ = self.npc.choose_action(npc_state,npc_BEV, evaluate=True)
                self.create_envs.set_vehicle_control(ego_a, npc_a, ego_step, npc_step)
                frames = 1 # 步长 
                if self.args.synchronous_mode:
                    for _ in range(frames):
                        self.world.tick() # 客户端主导，tick
                else:
                    self.world.wait_for_tick() # 服务器主导，tick
                # self.data_queue.put(self.create_envs.get_vehicle_step(ego_step,npc_step,episode_step))
                ego_next_state,ego_reward,npc_next_state,npc_reward,egocol,ego_fin,npccol,npc_fin,ego_next_BEV,npc_next_BEV = self.create_envs.get_vehicle_step(ego_step,npc_step,episode_step)

                if ego_next_state[0] > -0.1:
                    ego_step += 1
                if npc_next_state[0] > -0.1:
                    npc_step += 1
                if ego_fin:
                    ego_dw += True
                if npc_fin:
                    npc_dw += True
                if egocol or npccol or episode_step==self.args.max_length_of_trajectory-1 or ego_step >= self.ego_num-1 or npc_step >= self.npc_num-1:
                    if not ego_dw:
                        ego_dw += True
                    if not npc_dw:
                        npc_dw += True

                if ego_dw < 2:
                    ego_last_step = episode_step
                    ego_offsetx += np.abs(ego_next_state[0])
                    ego_offsety += np.abs(ego_next_state[1])
                    ego_episode_reward += ego_reward
                    ego_final_step = ego_step
                    
                if npc_dw < 2:
                    npc_last_step = episode_step
                    npc_offsetx += np.abs(npc_next_state[0])
                    npc_offsety += np.abs(npc_next_state[1])
                    npc_episode_reward += npc_reward
                    npc_final_step = npc_step

                ego_state = ego_next_state
                npc_state = npc_next_state
                ego_BEV = ego_next_BEV
                npc_BEV = npc_next_BEV

                if ego_dw and npc_dw: # 结束条件
                    break
            print("Episode: {} step: {} ego_Total_Reward: {:0.3f} npc_Total_Reward: {:0.3f}".format(self.total_episode, episode_step+1, ego_episode_reward/(ego_last_step + 1), npc_episode_reward/(npc_last_step + 1)))

            ego_evaluate_reward += (ego_episode_reward/(ego_last_step + 1))
            npc_evaluate_reward += (npc_episode_reward/(npc_last_step + 1))
            ego_total_offsetx += (ego_offsetx/(ego_last_step + 1))
            npc_total_offsetx += (npc_offsetx/(npc_last_step + 1))
            ego_total_offsety += (ego_offsety/(ego_last_step + 1))
            npc_total_offsety += (npc_offsety/(npc_last_step + 1))
            evaluate_step += (episode_step+1)
            ego_total_fin += ego_fin
            ego_total_col += egocol
            npc_total_fin += npc_fin
            npc_total_col += npccol

        ego_evaluate_reward /= self.args.evaluate_times
        npc_evaluate_reward /= self.args.evaluate_times

        ego_total_offsetx /= self.args.evaluate_times
        npc_total_offsetx /= self.args.evaluate_times
        ego_total_offsety /= self.args.evaluate_times
        npc_total_offsety /= self.args.evaluate_times
        evaluate_step /= self.args.evaluate_times
        ego_total_fin /= self.args.evaluate_times
        ego_total_col /= self.args.evaluate_times
        npc_total_fin /= self.args.evaluate_times
        npc_total_col /= self.args.evaluate_times

        self.writer.add_scalar('reward/ego_evaluate_rewards', ego_evaluate_reward, global_step=self.total_episode)
        self.writer.add_scalar('reward/npc_evaluate_rewards', npc_evaluate_reward, global_step=self.total_episode)
        self.writer.add_scalar('step/step', evaluate_step, global_step=self.total_episode)
        self.writer.add_scalar('step/ego_step', ego_final_step/(self.ego_num-3), global_step=self.total_episode)
        self.writer.add_scalar('step/npc_step', npc_final_step/(self.npc_num-3), global_step=self.total_episode)
        self.writer.add_scalar('offset/ego_offsetx', ego_total_offsetx, global_step=self.total_episode)
        self.writer.add_scalar('offset/ego_offsety', ego_total_offsety, global_step=self.total_episode)
        self.writer.add_scalar('offset/npc_offsetx', npc_total_offsetx, global_step=self.total_episode)
        self.writer.add_scalar('offset/npc_offsety', npc_total_offsety, global_step=self.total_episode)
        self.writer.add_scalar('rate/ego_col', ego_total_col, global_step=self.total_episode)
        self.writer.add_scalar('rate/npc_col', npc_total_col, global_step=self.total_episode)
        self.writer.add_scalar('rate/ego_finish', ego_total_fin, global_step=self.total_episode)
        self.writer.add_scalar('rate/npc_finsh', npc_total_fin, global_step=self.total_episode)
        
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
    parser.add_argument("--evaluate_freq", type=float, default=64, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=200, help="Save frequency")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")

    parser.add_argument("--Start_Path", type=bool, default=False, help="Start_Path")
    # parser.add_argument("--carla_dt", type=float, default=0.05, help="dt")
    parser.add_argument("--carla_lane_width", type=float, default=3.5, help="lane_width")
    parser.add_argument("--carla_max_s", type=int, default=300, help="max_s")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--hidden_dim1", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--hidden_dim2", type=int, default=32, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--init_std", type=float, default=0.3, help="std_initialization")
    parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.97, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.02, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=12, help="PPO parameter")
    parser.add_argument("--M", type=int, default=10, help="sample_times")
    parser.add_argument("--N", type=int, default=20, help="sample_times")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")
    parser.add_argument("--use_gru", type=bool, default=True, help="Whether to use GRU") # Priority for GRU
    parser.add_argument("--use_lstm", type=bool, default=False, help="Whether to use LSTM")
    parser.add_argument("--shared_policy", type=bool, default=False, help="Whether to share policy")
    parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
    parser.add_argument('--save_seed', default=5, type=str) # seed
    parser.add_argument('--load_seed', default=1, type=str) # seed
    parser.add_argument('--c_tau',  default=0.8, type=float) # action软更新系数,1代表完全更新，0代表不更新
    parser.add_argument('--max_length_of_trajectory', default=180, type=int) # 最大仿真步数
    parser.add_argument('--res', default=5, type=int) # pixel per meter
    parser.add_argument('--H', default=448, type=int) # BEV_Height
    parser.add_argument('--W', default=112, type=int) # BEV_Width

    parser.add_argument('--envs', default='straight', type=str) # 环境选择crossroad,highway,straight
    parser.add_argument('--npc_exist', default=False, type=bool) # multi-agent bool
    parser.add_argument('--model', default='OMAC', type=str) # 模型选择OMAC、IPPO、MAPPO、MADDPG、PR2AC、Rules

    parser.add_argument('--direct_control', default=True, type=bool) # 直接控制/PID跟踪
    parser.add_argument('--synchronous_mode', default=True, type=bool) # 同步模式开关
    parser.add_argument('--no_rendering_mode', default=True, type=bool) # 无渲染模式开关
    parser.add_argument('--fixed_delta_seconds', default=0.03, type=float) # 步长,步长建议不大于0.1，为0时代表可变步长
    parser.add_argument('--max_episode', default=10000, type=int) # 仿真次数
    args = parser.parse_args()

    runner = Runner(args)
    runner.run()

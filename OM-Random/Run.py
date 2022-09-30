import numpy as np
from torch.utils.tensorboard import SummaryWriter

import argparse

import utils.misc as misc
import os
import time
from memory_profiler import profile

class Runner:
    def __init__(self, args):
        # 环境建立
        self.args = args

        if self.args.envs == 'crossroad':
            import Envs.Crossroad_ENVS as ENVS
        elif self.args.envs == 'highway':
            import Envs.Highway_ENVS as ENVS
        
        if self.args.model == 'OMAC':
            from Model.ACOM.normalization import Normalization, RewardScaling
            from Model.ACOM.replaybuffer import ReplayBuffer
            from Model.ACOM.ppo_rnn import PPO_RNN as Model

        self.create_envs = ENVS.Create_Envs(self.args) # 设置仿真模式以及步长
        # 状态、动作空间定义
        action_space = self.create_envs.get_action_space()
        state_space = self.create_envs.get_state_space()
        self.args.state_dim = len(state_space)
        self.args.action_dim = len(action_space)
        self.args.episode_limit = self.args.max_length_of_trajectory # Maximum number of steps per episode
        self.directory = './carla-Random/'

        # Create env
        if not misc.judgeprocess('CarlaUE4.exe'):
            os.startfile('D:\CARLA_0.9.11\WindowsNoEditor\CarlaUE4.exe')
            time.sleep(15)
        self.world = self.create_envs.connection()

        self.ego_buffer = ReplayBuffer(self.args)
        self.npc_buffer = ReplayBuffer(self.args)
        self.ego = Model(self.args)
        self.npc = Model(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(self.directory + '/' + self.args.mode + '_' + self.args.envs + '_' + self.args.model + '_' + str(self.args.save_seed))

        self.ego_evaluate_rewards = []  # Record the rewards during the evaluating
        self.npc_evaluate_rewards = []
        self.total_steps = 0
        self.total_episode = 0

        if self.args.use_state_norm:
            print("------use state normalization------")
            self.state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)
    # @profile(stream=open('memory_profile.log','w+'))
    def run(self, ):
        
        if self.args.load or self.args.mode == 'test': 
            self.ego.load(self.directory + '/' + 'train' + '_' + self.args.envs + '_' + self.args.model + '_' + str(self.args.load_seed) + '/ego.pkl')
            self.npc.load(self.directory + '/' + 'train' + '_' + self.args.envs + '_' + self.args.model + '_' + str(self.args.load_seed) + '/npc.pkl')
        
        evaluate_num = -1  # Record the number of evaluations
        save_num = 1
    
        for i in range(self.args.max_episode):
            self.create_envs.Create_actors()
            # print('settings:',self.world.get_settings())
            self.total_episode += 1
            if i == 0:
                # 全局路径
                ego_route, npc_route, _, _ = self.create_envs.get_route()
                misc.draw_waypoints(self.world,ego_route)
                misc.draw_waypoints(self.world,npc_route)

            if self.total_episode // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1
            
            _, _, episode_steps = self.run_episode()  # Run an episode
            self.total_steps += episode_steps

            if self.ego_buffer.episode_num >= self.args.batch_size:
                print('------start_training-------')
                self.ego.train(self.ego_buffer, self.total_steps)  # Training
                self.npc.train(self.npc_buffer, self.total_steps)  # Training
            
                self.ego_buffer.reset_buffer()
                self.npc_buffer.reset_buffer()

            # Save the models
            if self.total_episode // self.args.save_freq > save_num:
                self.ego.save(self.directory + '/' + self.args.mode + '_' + self.args.envs + '_' + self.args.model + '_' + str(self.args.save_seed) + '/' + 'ego.pkl')
                self.npc.save(self.directory + '/' + self.args.mode + '_' + self.args.envs + '_' + self.args.model + '_' + str(self.args.save_seed) + '/' + 'npc.pkl')
                save_num += 1
                print('Network Saved')
            

        self.evaluate_policy()
        
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

        ego_step, npc_step = 1, 1
        ego_state,_,_,npc_state,_,_,_,_,_,_ = self.create_envs.get_vehicle_step(ego_step, npc_step, 0)

        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        self.ego.reset_rnn_hidden()
        self.npc.reset_rnn_hidden()
        for episode_step in range(self.args.episode_limit):
            if self.args.use_state_norm:
                ego_state = self.state_norm(ego_state)
                npc_state = self.state_norm(npc_state)
            ego_a, ego_a_logprob = self.ego.choose_action(ego_state, evaluate=False)
            npc_a, npc_a_logprob = self.npc.choose_action(npc_state, evaluate=False)
            ego_v = self.ego.get_value(ego_state)
            npc_v = self.npc.get_value(npc_state)
            self.create_envs.set_vehicle_control(ego_a, npc_a)

            #---------和环境交互动作反馈---------
            frames = 1 # 步长 
            if self.args.synchronous_mode:
                for _ in range(frames):
                    self.world.tick() # 客户端主导，tick
                    # print('frame:', ff)
            else:
                self.world.wait_for_tick() # 服务器主导，tick
            # self.data_queue.put(self.create_envs.get_vehicle_step(ego_step,npc_step,episode_step))
            ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done,egocol,ego_fin,npccol,npc_fin = self.create_envs.get_vehicle_step(ego_step,npc_step,episode_step)

            if ego_next_state[0] > -0.1:
                    ego_step += 1                    
            if npc_next_state[0] > -0.1:
                    npc_step += 1
            
            ego_state = ego_next_state.copy()
            npc_state = npc_next_state.copy()

            ego_total_reward += ego_reward
            npc_total_reward += npc_reward
            ego_offsetx += np.abs(ego_state[0])
            npc_offsetx += np.abs(npc_state[0])
            ego_offsety += np.abs(ego_state[1])
            npc_offsety += np.abs(npc_state[1])
            
            if self.args.mode == 'test':
                if ego_done or npc_done or episode_step==args.max_length_of_trajectory-1: # 结束条件
                    break

            if self.args.mode == 'train':
                if ego_done or npc_done or episode_step==args.max_length_of_trajectory-1: # 结束条件
                    ego_dw = True
                    npc_dw = True
                    break
                else:
                    ego_dw = False
                    npc_dw = False

        if self.args.use_reward_scaling:
            ego_reward = self.reward_scaling(ego_reward)
            npc_reward = self.reward_scaling(npc_reward)
        # Store the transition
        self.ego_buffer.store_transition(episode_step, ego_state, ego_v, ego_a, ego_a_logprob, ego_reward, ego_dw)
        self.npc_buffer.store_transition(episode_step, npc_state, npc_v, npc_a, npc_a_logprob, npc_reward, npc_dw)

        # An episode is over, store v in the last step
        if self.args.use_state_norm:
            ego_state = self.state_norm(ego_state)
            npc_state = self.state_norm(npc_state)
        ego_v = self.ego.get_value(ego_state)
        npc_v = self.npc.get_value(npc_state)
        self.ego_buffer.store_last_value(episode_step + 1, ego_v)
        self.npc_buffer.store_last_value(episode_step + 1, npc_v)
        print("Episode: {} step: {} ego_Total_Reward: {:0.3f} npc_Total_Reward: {:0.3f}".format(self.total_episode, episode_step+1, ego_total_reward/(episode_step + 1), npc_total_reward/(episode_step + 1)))

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

        for _ in range(self.args.evaluate_times):
            ego_done, npc_done = False, False
            ego_episode_reward = 0
            npc_episode_reward = 0
            ego_offsetx = 0
            npc_offsetx = 0
            ego_offsety = 0
            npc_offsety = 0
            ego_step, npc_step = 1, 1
            ego_state,_,_,npc_state,_,_,_,_,_,_ = self.create_envs.get_vehicle_step(ego_step, npc_step, 0)
            
            self.ego.reset_rnn_hidden()
            self.npc.reset_rnn_hidden()

            for episode_step in range(99999):
                if self.args.use_state_norm:
                    ego_state = self.state_norm(ego_state, update=False)
                    npc_state = self.state_norm(npc_state, update=False)
                ego_a, _ = self.ego.choose_action(ego_state, evaluate=True)
                npc_a, _ = self.npc.choose_action(npc_state, evaluate=True)
                self.create_envs.set_vehicle_control(ego_a, npc_a)
                frames = 1 # 步长 
                if self.args.synchronous_mode:
                    for _ in range(frames):
                        self.world.tick(seconds=20) # 客户端主导，tick
                else:
                    self.world.wait_for_tick() # 服务器主导，tick
                # self.data_queue.put(self.create_envs.get_vehicle_step(ego_step,npc_step,episode_step))
                ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done,egocol,ego_fin,npccol,npc_fin = self.create_envs.get_vehicle_step(ego_step,npc_step,episode_step)

                if ego_next_state[0] > -0.1:
                    ego_step += 1                    
                if npc_next_state[0] > -0.1:
                    npc_step += 1

                ego_state = ego_next_state.copy()
                npc_state = npc_next_state.copy()
                ego_episode_reward += ego_reward
                npc_episode_reward += npc_reward
                ego_offsetx += np.abs(ego_state[0])
                npc_offsetx += np.abs(npc_state[0])
                ego_offsety += np.abs(ego_state[1])
                npc_offsety += np.abs(npc_state[1])
                
                if ego_done or npc_done or episode_step==args.max_length_of_trajectory-1: # 结束条件
                    break
            
            ego_evaluate_reward += ego_episode_reward/(episode_step + 1)
            npc_evaluate_reward += npc_episode_reward/(episode_step + 1)
            ego_total_offsetx += ego_offsetx/(episode_step + 1)
            npc_total_offsetx += npc_offsetx/(episode_step + 1)
            ego_total_offsety += ego_offsety/(episode_step + 1)
            npc_total_offsety += npc_offsety/(episode_step + 1)
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

        # print("total_episodes:{} \t evaluate_reward:{}".format(self.total_episode, ego_evaluate_reward))
        self.writer.add_scalar('reward/ego_evaluate_rewards', ego_evaluate_reward, global_step=self.total_episode)
        self.writer.add_scalar('reward/npc_evaluate_rewards', npc_evaluate_reward, global_step=self.total_episode)
        self.writer.add_scalar('step/step', evaluate_step, global_step=self.total_episode)
        self.writer.add_scalar('step/ego_step', ego_step, global_step=self.total_episode)
        self.writer.add_scalar('step/npc_step', npc_step, global_step=self.total_episode)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting")
    parser.add_argument("--max_train_steps", type=int, default=int(5e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_dim1", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--hidden_dim2", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--init_logstd", type=float, default=0.4, help="logstd_initialization")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
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
    parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
    parser.add_argument('--save_seed', default=8, type=str) # seed
    parser.add_argument('--load_seed', default=7, type=str) # seed
    parser.add_argument('--c_tau',  default=1, type=float) # action软更新系数,1代表完全更新，0代表不更新
    parser.add_argument('--max_length_of_trajectory', default=400, type=int) # 最大仿真步数

    parser.add_argument('--envs', default='crossroad', type=str) # 环境选择crossroad,highway
    parser.add_argument('--model', default='OMAC', type=str) # 模型选择OMAC、DRON、IPPO、MAPPO、MADDPG、PR2AC、Rules

    parser.add_argument('--synchronous_mode', default=True, type=bool) # 同步模式开关
    parser.add_argument('--no_rendering_mode', default=True, type=bool) # 无渲染模式开关
    parser.add_argument('--fixed_delta_seconds', default=0.05, type=float) # 步长,步长建议不大于0.1，为0时代表可变步长
    parser.add_argument('--load', default=True, type=bool) # 训练模式下是否load model  
    parser.add_argument('--max_episode', default=3000, type=int) # 仿真次数
    args = parser.parse_args()
    for _ in range(1):
        try:
            runner = Runner(args)
            runner.run()
        except RuntimeError:
            args.save_seed += 1
            args.load_seed += 1

import glob
import os
import sys
import numpy as np
try:
    sys.path.append(glob.glob('D:/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import Transform, Location, Rotation

import time

import Simple_Sensors as SS

directory = './carla-IPPO3/'

class Create_Envs(object):
    def __init__(self,synchronous_mode = False,no_rendering_mode=False,fixed_delta_seconds = 0.05,size=[300,200]):
        self.synchronous_mode = synchronous_mode
        self.no_rendering_mode = no_rendering_mode
        self.fixed_delta_seconds = fixed_delta_seconds
        self.H = size[0]
        self.W = size[1]

    def connection(self):
        # 连接客户端:localhost:2000\192.168.199.238:2000
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # 连接世界
        world = client.load_world('Town04')
        settings = world.get_settings()
        settings.synchronous_mode = self.synchronous_mode
        settings.no_rendering_mode = self.no_rendering_mode
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        world.apply_settings(settings)

        # 蓝图
        blueprint_library = world.get_blueprint_library()
        return client, world, blueprint_library

    def Create_actors(self,world, blueprint_library): 
        ego_list = []
        npc_list = []
        obstacle_list = []
        sensor_list = []
        # ego车辆设置---------------------------------------------------------------
        ego_bp = blueprint_library.find(id='vehicle.lincoln.mkz2017')
        # 坐标建立
        ego_transform = Transform(Location(x=160.341522, y=-371.640472, z=0.281942), 
                    Rotation(pitch=0.000000, yaw=0.500910, roll=0.000000))
        # 车辆从蓝图定义以及坐标生成
        ego = world.spawn_actor(ego_bp, ego_transform)
        ego_list.append(ego)
        print('created %s' % ego.type_id)

        # 视角设置------------------------------------------------------------------
        spectator = world.get_spectator()
        # spec_transform = ego.get_transform()
        spec_transform = Transform(Location(x=140.341522, y=-375.140472, z=15.281942), 
                    Rotation(pitch=0.000000, yaw=0.500910, roll=0.000000))
        spec_transform.location += carla.Location(x=60,z=45)
        spec_transform.rotation = carla.Rotation(pitch=-90, yaw=90)
        spectator.set_transform(spec_transform)

        # npc设置--------------------------------------------------------------------
        npc_transform = ego_transform
        for i in range(1):
            npc_transform.location += carla.Location(x=-15,y=-3.5)
            npc_bp = blueprint_library.find(id='vehicle.lincoln.mkz2017')
            # print(npc_bp.get_attribute('color').recommended_values)
            npc_bp.set_attribute('color', '229,28,0')
            npc = world.try_spawn_actor(npc_bp, npc_transform)
            if npc is None:
                print('%s npc created failed' % i)
            else:
                npc_list.append(npc)
                print('created %s' % npc.type_id)

        # 障碍物设置------------------------------------------------------------------
        obstacle_transform = ego_transform
        for i in range(28):
            if i == 0:
                obsta_bp = blueprint_library.find(id='vehicle.mercedes-benz.coupe')
                obstacle_transform.location += carla.Location(x=95,y=3.8)
                obstacle = world.try_spawn_actor(obsta_bp, obstacle_transform)
                obstacle_transform.location += carla.Location(x=0,y=-5.3)
                if obstacle is None:
                    print('%s obstacle created failed' % i)
                else:
                    obstacle_list.append(obstacle)
                    print('created %s' % obstacle.type_id)
            else:
                obsta_bp = blueprint_library.find(id='static.prop.streetbarrier')
                obstacle_transform.location += carla.Location(x=-3.5,y=7.4)
                obstacle1 = world.try_spawn_actor(obsta_bp, obstacle_transform)
                obstacle_list.append(obstacle1)
                obstacle_transform.location += carla.Location(y=-7.4)
                obstacle2 = world.try_spawn_actor(obsta_bp, obstacle_transform)
                obstacle_list.append(obstacle2)

        # 传感器设置-------------------------------------------------------------------
        ego_collision = SS.CollisionSensor(ego)
        npc_collision = SS.CollisionSensor(npc)
        ego_invasion = SS.LaneInvasionSensor(ego)
        npc_invasion = SS.LaneInvasionSensor(npc)
        ego_camera = SS.Camera(ego, 'ego', directory, self.H, self.W)
        npc_camera = SS.Camera(npc, 'npc', directory, self.H, self.W)
        sensor_list.extend([[ego_collision,ego_invasion,ego_camera],[npc_collision,npc_invasion,npc_camera]])
        return ego_list,npc_list,obstacle_list,sensor_list

    # 车辆控制
    def set_vehicle_control(self,ego,npc,ego_action,npc_action,c_tau,step):
        if step == 0:
            # 初始速度设定
            ego_target_speed = carla.Vector3D(16.5,0,0)
            npc_target_speed = carla.Vector3D(20,0,0)
            ego.set_target_velocity(ego_target_speed)
            npc.set_target_velocity(npc_target_speed)
            print('target velocity is set!')

        else: 
            ego_move,ego_steer = ego_action
            npc_move,npc_steer = npc_action
            ego_steer = c_tau*ego_steer + (1-c_tau)*ego.get_control().steer
            npc_steer = c_tau*npc_steer + (1-c_tau)*npc.get_control().steer
            if ego_move >= 0:
                ego_throttle = c_tau*ego_move + (1-c_tau)*ego.get_control().throttle
                ego_control = carla.VehicleControl(throttle = ego_throttle, steer = ego_steer, brake = 0)
            elif ego_move < 0:
                ego_brake = -c_tau*ego_move + (1-c_tau)*ego.get_control().brake
                ego_control = carla.VehicleControl(throttle = 0, steer = ego_steer, brake = ego_brake)
            if npc_move >= 0:
                npc_throttle = c_tau*npc_move + (1-c_tau)*npc.get_control().throttle
                npc_control = carla.VehicleControl(throttle = npc_throttle, steer = 0, brake = 0)
            elif npc_move < 0:
                npc_brake = -c_tau*npc_move + (1-c_tau)*npc.get_control().brake
                npc_control = carla.VehicleControl(throttle = 0, steer = 0, brake = npc_brake)
            ego.apply_control(ego_control)
            npc.apply_control(npc_control)

            print('ego:%f,%f,%f,npc:%f,%f,%f'%(ego.get_control().throttle,ego_steer,ego.get_control().brake,
                                            npc.get_control().throttle,npc_steer,npc.get_control().brake))
    
    # 车辆信息反馈
    def get_vehicle_step(self,ego,npc,ego_sensor,npc_sensor, step):
        ego_next_transform = ego.get_transform()
        npc_next_transform = npc.get_transform()

        # 碰撞、变道检测
        ego_col = ego_sensor[0].get_collision_history()
        npc_col = npc_sensor[0].get_collision_history()
        # ego_inv = ego_sensor[1].get_invasion_history()
        # npc_inv = npc_sensor[1].get_invasion_history()
        ego_camera = ego_sensor[2].get_BEV()
        npc_camera = npc_sensor[2].get_BEV()

        # 速度、加速度
        ego_velocity = ego.get_velocity().x
        npc_velocity = npc.get_velocity().x
        ego_angular = ego.get_angular_velocity().z
        npc_angular = npc.get_angular_velocity().z
        ego_next_state1 = np.array([ego_velocity, ego_angular])
        npc_next_state1 = np.array([npc_velocity, npc_angular])
        ego_next_state2 = [ego_camera]
        npc_next_state2 = [npc_camera]
        
        ego_acceleration = abs(ego.get_acceleration().y)
        npc_acceleration = abs(npc.get_acceleration().y)
        # 回报设置:碰撞惩罚、纵向奖励、最低速度惩罚、存活奖励 
        eb=-0.5 if ego_velocity <= 2 else 0

        ego_reward = (-10)*ego_col[0] + (0)*ego_acceleration + (5)*(ego_next_transform.location.x-200)/125 + eb + step/100
        npc_reward = (-10)*npc_col[0] + (0)*npc_acceleration + (5)*(npc_next_transform.location.x-200)/125 + eb + step/100
        # ego_reward = (-20)*ego_col[0] + eb
        # npc_reward = (-20)*npc_col[0] + nb
        # ego_sensor[1].reset()
        # npc_sensor[1].reset()

        # done结束状态判断
        if ego_col[0]==1 or (ego_next_transform.location.x-200)/125 > 1: # ego结束条件ego_done
            ego_done = True
        else:
            ego_done = False
        if npc_col[0]==1 or (npc_next_transform.location.x-200)/125 > 1: # npc结束条件npc_done
            npc_done = True
        else:
            npc_done = False  
        return [ego_next_state1,ego_next_state2,ego_reward,ego_done,npc_next_state1,npc_next_state2,npc_reward,npc_done]

    # 车辆动作空间
    def get_action_space(self):
        action_space = np.array([[-1,1],[-1,1]],dtype=np.float16) # 油门、方向盘、刹车,油门刹车合并
        return action_space
    
    # 车辆状态空间
    def get_state_space(self):
        state_dim = [2,[self.H,self.W]] # ego_velocity,angular_velocity;
        return state_dim

import glob
import os
import sys
import numpy as np
try:
    sys.path.append(glob.glob('D:/CARLA_0.9.10-Pre_Win/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import Transform, Location, Rotation

import random
import time
import copy

import Simple_Sensors as SS

class Create_Envs(object):
    def __init__(self,synchronous_mode = False,no_rendering_mode=False,fixed_delta_seconds = 0.05):
        self.no_rendering_mode = no_rendering_mode
        self.fixed_delta_seconds = fixed_delta_seconds

    def connection(self):
        # 连接客户端:localhost:2000\192.168.199.238:2000
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # 连接世界
        world = client.load_world('Town04')
        settings = world.get_settings()
        if self.no_rendering_mode:
            settings.synchronous_mode = False
            settings.no_rendering_mode = True
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
        for i in range(1):
            obstacle_transform.location += carla.Location(x=95,y=3.7)
            obsta_bp = blueprint_library.find(id='vehicle.mercedes-benz.coupe')
            # bp = random.choice(blueprint_library.filter('vehicle'))
            obstacle = world.try_spawn_actor(obsta_bp, obstacle_transform)
            if obstacle is None:
                print('%s obstacle created failed' % i)
            else:
                obstacle_list.append(obstacle)
                print('created %s' % obstacle.type_id)

        # 传感器设置-------------------------------------------------------------------
        ego_collision = SS.CollisionSensor(ego)
        npc_collision = SS.CollisionSensor(npc)
        ego_invasion = SS.LaneInvasionSensor(ego)
        npc_invasion = SS.LaneInvasionSensor(npc)
        sensor_list.extend([ego_collision,npc_collision,ego_invasion,npc_invasion])
        return ego_list,npc_list,obstacle_list,sensor_list

    # 车辆控制
    def set_vehicle_control(self,ego,npc,ego_action,npc_action):  
        ego_move,ego_steer = ego_action
        npc_move,npc_steer = npc_action
        print('ego:%f,%f,npc:%f,%f'%(ego_move,ego_steer,npc_move,npc_steer))
        if ego_move >= 0:
            ego_control = carla.VehicleControl(throttle = ego_move, steer = ego_steer, brake = 0)
        elif ego_move < 0:
            ego_control = carla.VehicleControl(throttle = 0, steer = ego_steer, brake = -ego_move)
        if npc_move >= 0:
            npc_control = carla.VehicleControl(throttle = npc_move, steer = 0, brake = 0)
        elif npc_move < 0:
            npc_control = carla.VehicleControl(throttle = 0, steer = 0, brake = -npc_move)
        ego.apply_control(ego_control)
        npc.apply_control(npc_control)
    
    # 车辆信息反馈
    def get_vehicle_step(self,ego,npc,ego_sensor,npc_sensor):
        ego_next_state = ego.get_transform()
        npc_next_state = npc.get_transform()
        ego_next_state = np.array([ego_next_state.location.x/250,ego_next_state.location.y/400,ego_next_state.rotation.yaw])
        npc_next_state = np.array([npc_next_state.location.x/250,npc_next_state.location.y/400,npc_next_state.rotation.yaw])
         # 回报设置:碰撞惩罚、横向惩罚、纵向奖励
        ego_reward = ego_sensor[0]*(-100) - 2*(ego_next_state[1] - (-370.640472/400)) + (ego_next_state[0] - 245/250) + 1
        npc_reward = npc_sensor[0]*(-100) - 2*abs(npc_next_state[1] - (-375.140472/400)) + (npc_next_state[0] - 245/250) + 1
        # done结束状态判断
        if ego_sensor[0]==1 or ego_next_state[0] > 245/250 or ego_next_state[1] > -367/400: # ego结束条件ego_done
            ego_done = True
        else:
            ego_done = False
        if npc_sensor[0]==1 or npc_next_state[0] > 245/250 or npc_next_state[1] > -370/400: # npc结束条件npc_done
            npc_done = True
        else:
            npc_done = False  
        return [ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done]

    # 车辆动作空间
    def get_action_space(self):
        action_space = np.array([[-1,1],[-1,1]],dtype=np.float16) # 油门、方向盘、刹车,油门刹车合并
        return action_space
    
    # 车辆状态空间
    def get_state_space(self):
        state_space = [0,0,0] # x,y,yaw
        return state_space

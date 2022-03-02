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

import random
import time
import copy

import Simple_Sensors as SS

class Create_Envs(object):
    def __init__(self,synchronous_mode = False,no_rendering_mode = True,fixed_delta_seconds = 0.05):
        self.synchronous_mode = synchronous_mode
        self.no_rendering_mode = no_rendering_mode
        self.fixed_delta_seconds = fixed_delta_seconds

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
        # lane_invasion = SS.LaneInvasionSensor(ego)
        sensor_list.append(ego_collision)
        sensor_list.append(npc_collision)
        return ego_list,npc_list,obstacle_list,sensor_list

    # 车辆控制
    def get_ego_step(self,ego,action,sim_time,flag): # 0加速变道 1刹车变道
        if flag == 1:
            ego_target_speed = carla.Vector3D(18,0,0)
            ego.set_target_velocity(ego_target_speed)
            print('ego velocity is set!')
        if action == 0: 
            if 1 < sim_time <= 1.55:
                ego_control = carla.VehicleControl(throttle = 1, steer = -0.1)
                ego.apply_control(ego_control)                
            elif 1.55 < sim_time <= 2.1:
                ego_control = carla.VehicleControl(throttle = 1, steer = 0.1)
                ego.apply_control(ego_control)
            else:
                ego_control = carla.VehicleControl(throttle = 1, brake = 0)
                ego.apply_control(ego_control)
        
        if action == 1:
            if 1.25 <= sim_time <= 2:
                ego_control = carla.VehicleControl(throttle = 0, brake = 1)
                ego.apply_control(ego_control)
            elif 2 < sim_time <= 2.7:
                ego_control = carla.VehicleControl(throttle = 1, steer = -0.1)
                ego.apply_control(ego_control)                
            elif 2.7 < sim_time <= 3.4:
                ego_control = carla.VehicleControl(throttle = 1, steer = 0.1)
                ego.apply_control(ego_control)
            elif sim_time > 3.4:
                ego_control = carla.VehicleControl(throttle = 1, steer = 0)
                ego.apply_control(ego_control)
            else:
                ego_control = carla.VehicleControl(throttle = 0, brake = 0)
                ego.apply_control(ego_control)
            

    def get_npc_step(self,npc,action,sim_time,flag): # 0刹车 1加速
        if flag == 1:
            npc_target_speed = carla.Vector3D(28,0,0)
            npc.set_target_velocity(npc_target_speed)
            print('npc velocity is set!')
        if action == 0:
            if 0.5 < sim_time <= 2:
                npc_control = carla.VehicleControl(throttle = 0, brake = 0.4)
                npc.apply_control(npc_control)                   
            else:
                npc_control = carla.VehicleControl(throttle = 1, brake = 0)
                npc.apply_control(npc_control)

        if action == 1:
            npc_control = carla.VehicleControl(throttle = 1)
            npc.apply_control(npc_control)

    def get_action_space(self):
        action_space = np.array([0,1])
        return action_space
    
    def get_state_space(self):
        state_space = np.array([0,1])
        return state_space

    def get_reward(self,action):  
        if action == [0,1]:
            reward1,reward2 = 1,4
        elif action == [0,0]:
            reward1,reward2 = 1,1
        elif action == [1,1]:
            reward1,reward2 = -20,-20
        else:
            reward1,reward2 = 4,1
        return reward1,reward2

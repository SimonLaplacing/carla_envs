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

    def connection(self):
        # 连接客户端:localhost:2000\192.168.199.238:2000
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)

        # 连接世界
        world = client.load_world('Town04')

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
        spec_transform.location += carla.Location(x=60,z=35)
        spec_transform.rotation = carla.Rotation(pitch=-90, yaw=90)
        spectator.set_transform(spec_transform)

        # npc设置--------------------------------------------------------------------
        npc_transform = ego_transform
        for i in range(1):
            npc_transform.location += carla.Location(x=-15,y=-3.5)
            npc_bp = blueprint_library.find(id='vehicle.lincoln.mkz2017')
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
        # lane_invasion = SS.LaneInvasionSensor(ego)
        sensor_list.append(ego_collision)
        sensor_list.append(npc_collision)
        return ego_list,npc_list,obstacle_list,sensor_list

    # 车辆控制
    def get_ego_step(self,ego,sim_time):      
        if 0 < sim_time <= 6:
            ego_control = carla.VehicleControl(throttle = 1, steer = 0, brake = 0)
            ego.apply_control(ego_control) 
        else:
            ego_control = carla.VehicleControl(throttle = 0, steer = -0.2, brake = 0.1)
            ego.apply_control(ego_control)

def main():
    create_envs = Create_Envs()
    episodes = 5
    client, world, blueprint_library = create_envs.connection()

    try:
        for i in range(episodes):
            print('%dth Test begins'%i)
            client, world, blueprint_library = create_envs.connection()
            ego_list,npc_list,obstacle_list,sensor_list = create_envs.Create_actors(world,blueprint_library)
            sim_time = 0  # 仿真时间
            start_time = time.time()  # 初始时间

            egocol_list = sensor_list[0].get_collision_history()
            npccol_list = sensor_list[1].get_collision_history()
            
            # action
            while 1:
                sim_time = time.time() - start_time
                create_envs.get_ego_step(ego_list[0],sim_time)
                if sim_time > 12:
                    break

            for x in sensor_list:
                if x.sensor.is_alive:
                    x.sensor.destroy()            
            for x in ego_list:
                if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])
            for x in npc_list:
                if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])
            for x in obstacle_list:
                if x.is_alive:
                    client.apply_batch([carla.command.DestroyActor(x)])

            print('Reset')
    finally:
        # 清洗环境
        print('Start Cleaning Envs')
        for x in sensor_list:
            if x.sensor.is_alive:
                x.sensor.destroy()
        for x in ego_list:
            if x.is_alive:
                client.apply_batch([carla.command.DestroyActor(x)])
        for x in npc_list:
            if x.is_alive:
                client.apply_batch([carla.command.DestroyActor(x)])
        for x in obstacle_list:
            if x.is_alive:
                client.apply_batch([carla.command.DestroyActor(x)])
        print('all clean, simulation done!')

if __name__ == '__main__':

    main()
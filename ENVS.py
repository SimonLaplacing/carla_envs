import glob
import os
import sys
import Simple_Sensors as SS

try:
    sys.path.append(glob.glob('E:/CARLA_0.9.10-Pre_Win/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
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



def main():
    actor_list = []
    obstacle_list = []
    sensor_list = []

    try:
        # 连接客户端:localhost:2000\192.168.199.238:2000
        client = carla.Client('localhost', 2000)
        # client = carla.Client('192.168.199.238', 2000) 
        client.set_timeout(30.0)

        # 连接世界
        world = client.load_world('Town04')

        # 蓝图
        blueprint_library = world.get_blueprint_library()

        # 车辆定义
        bp = blueprint_library.find(id='vehicle.lincoln.mkz2017')

        # 坐标建立
        ego_transform = Transform(Location(x=160.341522, y=-371.640472, z=0.281942), 
                    Rotation(pitch=0.000000, yaw=0.500910, roll=0.000000))

        # 车辆从蓝图定义以及坐标生成
        ego = world.spawn_actor(bp, ego_transform)
        actor_list.append(ego)
        print('created %s' % ego.type_id)

        # 视角设置
        spectator = world.get_spectator()
        # spec_transform = ego.get_transform()
        spec_transform = Transform(Location(x=140.341522, y=-375.140472, z=15.281942), 
                    Rotation(pitch=0.000000, yaw=0.500910, roll=0.000000))
        spec_transform.location += carla.Location(x=60,z=35)
        spec_transform.rotation = carla.Rotation(pitch=-90, yaw=90)
        spectator.set_transform(spec_transform)

        # npc设置
        npc_transform = ego_transform
        for i in range(1):
            npc_transform.location += carla.Location(x=-20,y=-3.5)
            bp = blueprint_library.find(id='vehicle.lincoln.mkz2017')
            npc = world.try_spawn_actor(bp, npc_transform)
            if npc is None:
                print('%s npc created failed' % i)
            else:
                actor_list.append(npc)
                print('created %s' % npc.type_id)

        # 障碍物设置
        obstacle_transform = ego_transform
        for i in range(1):
            obstacle_transform.location += carla.Location(x=100,y=3.7)
            bp = blueprint_library.find(id='vehicle.mercedes-benz.coupe')
            # bp = random.choice(blueprint_library.filter('vehicle'))
            obstacle = world.try_spawn_actor(bp, obstacle_transform)
            if obstacle is None:
                print('%s obstacle created failed' % i)
            else:
                obstacle_list.append(obstacle)
                print('created %s' % obstacle.type_id)

        # 传感器设置
        collision = SS.CollisionSensor(ego)
        lane_invasion = SS.LaneInvasionSensor(ego)
        sensor_list.append(collision.sensor)
        sensor_list.append(lane_invasion.sensor)

        # 车辆控制
        def ego_step(mode,simtime): # 1变道 2刹车
            if mode == 1:
                if 0 < sim_time <= 1.2:
                    ego_control = carla.VehicleControl(throttle = 0, steer = 0)
                    ego.apply_control(ego_control) 
                elif 7 < sim_time <= 7.7:
                    ego_control = carla.VehicleControl(throttle = 0.5, steer = -0.1)
                    ego.apply_control(ego_control)                
                elif 7.7 < sim_time <= 8.4:
                    ego_control = carla.VehicleControl(throttle = 0.5, steer = 0.1)
                    ego.apply_control(ego_control)
                elif sim_time > 8.4:
                    ego_control = carla.VehicleControl(throttle = 0.5, steer = 0)
                    ego.apply_control(ego_control)
                else:
                    ego_control = carla.VehicleControl(throttle = 1, steer = 0)
                    ego.apply_control(ego_control)
            
            if mode == 2:
                if 0 < sim_time <= 6:
                    ego_control = carla.VehicleControl(throttle = 0.9, steer = 0)
                    ego.apply_control(ego_control) 
                else:
                    ego_control = carla.VehicleControl(throttle = 0, steer = 0, brake = 0.3)
                    ego.apply_control(ego_control)

        def npc_step(mode,simtime): # 1刹车 2加速
            if mode == 1:
                if 0 < sim_time <= 6:
                    npc_control = carla.VehicleControl(throttle = 1)
                    npc.apply_control(npc_control)
                elif 6 < sim_time <= 7:
                    npc_control = carla.VehicleControl(throttle = 0, brake = 0.3)
                    npc.apply_control(npc_control)                     
                else:
                    npc_control = carla.VehicleControl(throttle = 0.5, brake = 0)
                    npc.apply_control(npc_control)

            if mode == 2:
                if 0 < sim_time <= 2:
                    npc_control = carla.VehicleControl(throttle = 1)
                    npc.apply_control(npc_control) 
                else:
                    npc_control = carla.VehicleControl(throttle = 1)
                    npc.apply_control(npc_control)

        sim_time = 0  # 仿真时间
        start_time = time.time()  # 初始时间
        while sim_time < 20:  # 仿真时间限制
            sim_time = time.time() - start_time
            ego_step(2,sim_time)
            npc_step(2,sim_time)             
                        
            # npc.apply_control(set_control)
            # for vehicle in actor_list:
            #     vehicle.set_autopilot(True)  

        # 仿真时间设置
        time.sleep(5)

    finally:
        # 删除车辆
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in obstacle_list])
        for sensor in sensor_list:
            if sensor is not None:
                sensor.destroy()
        print('done!')


if __name__ == '__main__':

    main()

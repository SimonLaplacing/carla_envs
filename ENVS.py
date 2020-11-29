import glob
import os
import sys
import Simple_Sensors as SS

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



def main():
    actor_list = []
    obstacle_list = []

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
        transform = Transform(Location(x=160.341522, y=-375.140472, z=0.281942), 
                    Rotation(pitch=0.000000, yaw=0.500910, roll=0.000000))

        # 车辆从蓝图定义以及坐标生成
        ego = world.spawn_actor(bp, transform)
        actor_list.append(ego)
        print('created %s' % ego.type_id)

        # 视角设置
        spectator = world.get_spectator()
        # spec_transform = ego.get_transform()
        spec_transform = Transform(Location(x=140.341522, y=-375.140472, z=15.281942), 
                    Rotation(pitch=0.000000, yaw=0.500910, roll=0.000000))
        spec_transform.location += carla.Location(x=-10,z=10)
        spec_transform.rotation = carla.Rotation(pitch=-45)
        spectator.set_transform(spec_transform)

        # npc设置
        for i in range(1):
            transform.location += carla.Location(x=12)
            bp = blueprint_library.find(id='vehicle.lincoln.mkz2017')
            npc = world.try_spawn_actor(bp, transform)
            if npc is None:
                print('%s npc created failed' % i)
            else:
                actor_list.append(npc)
                print('created %s' % npc.type_id)

        # 障碍物设置
        for i in range(1):
            transform.location += carla.Location(x=50)
            bp = blueprint_library.find(id='vehicle.mercedes-benz.coupe')
            # bp = random.choice(blueprint_library.filter('vehicle'))
            obstacle = world.try_spawn_actor(bp, transform)
            if obstacle is None:
                print('%s obstacle created failed' % i)
            else:
                obstacle_list.append(obstacle)
                print('created %s' % obstacle.type_id)

        # 车辆控制
        set_control = carla.VehicleControl(throttle = 0.5, steer = 0.5)
        ego.apply_control(set_control)
        # npc.apply_control(set_control)
        # for vehicle in actor_list:
        #     vehicle.set_autopilot(True)

        # 传感器设置
        collision = SS.CollisionSensor(ego)
        lane_invasion = SS.LaneInvasionSensor(ego)
        sensor_list = [collision.sensor, lane_invasion.sensor]  

        # 仿真时间设置
        time.sleep(50)

    finally:
        # 删除车辆
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in obstacle_list])
        for sensor in sensor_list:
            if sensor is not None:
                sensor.destroy()
        print('done.')


if __name__ == '__main__':

    main()

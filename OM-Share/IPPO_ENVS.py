import numpy as np

import carla
from carla import Transform, Location, Rotation

import time

import Simple_Sensors as SS
from global_route_planner_dao import GlobalRoutePlannerDAO
from global_route_planner import GlobalRoutePlanner

import misc

class Create_Envs(object):
    def __init__(self,synchronous_mode = False,no_rendering_mode=False,fixed_delta_seconds = 0.05):
        self.synchronous_mode = synchronous_mode
        self.no_rendering_mode = no_rendering_mode
        self.fixed_delta_seconds = fixed_delta_seconds

    def connection(self):
        # 连接客户端:localhost:2000\192.168.199.238:2000
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # 连接世界
        self.world = client.load_world('Town04')
        settings = self.world.get_settings()
        settings.synchronous_mode = self.synchronous_mode
        settings.no_rendering_mode = self.no_rendering_mode
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

        # 蓝图
        blueprint_library = self.world.get_blueprint_library()
        return client, self.world, blueprint_library

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
            # else:
            #     obsta_bp = blueprint_library.find(id='static.prop.streetbarrier')
            #     obstacle_transform.location += carla.Location(x=-3.5,y=7.4)
            #     obstacle1 = world.try_spawn_actor(obsta_bp, obstacle_transform)
            #     obstacle_list.append(obstacle1)
            #     obstacle_transform.location += carla.Location(y=-7.4)
            #     obstacle2 = world.try_spawn_actor(obsta_bp, obstacle_transform)
            #     obstacle_list.append(obstacle2)

        # 传感器设置-------------------------------------------------------------------
        ego_collision = SS.CollisionSensor(ego)
        npc_collision = SS.CollisionSensor(npc)
        ego_invasion = SS.LaneInvasionSensor(ego)
        npc_invasion = SS.LaneInvasionSensor(npc)
        sensor_list.extend([[ego_collision,ego_invasion],[npc_collision,npc_invasion]])
        return ego_list,npc_list,obstacle_list,sensor_list

    # 全局规划
    def route_positions_generate(self,start_pos,end_pos):
        dao = GlobalRoutePlannerDAO(self.world.get_map(), sampling_resolution=3)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        self._grp = grp
        route = self._grp.trace_route(start_pos, end_pos)

        positions = []
        for i in range((len(route))):
            # xi = route[i][0].transform.location.x
            # yi = route[i][0].transform.location.y
            position = route[i][0].transform
            positions.append(position)

        return positions

    # 车辆控制
    def set_vehicle_control(self,ego,npc,ego_action,npc_action,c_tau,step):
        if step == 0:
            # 初始速度设定
            ego_target_speed = carla.Vector3D(16.5,0,0)
            npc_target_speed = carla.Vector3D(20,0,0)
            ego.set_target_velocity(ego_target_speed)
            npc.set_target_velocity(npc_target_speed)
            # print('target velocity is set!')

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
                npc_control = carla.VehicleControl(throttle = npc_throttle, steer = npc_steer, brake = 0)
            elif npc_move < 0:
                npc_brake = -c_tau*npc_move + (1-c_tau)*npc.get_control().brake
                npc_control = carla.VehicleControl(throttle = 0, steer = npc_steer, brake = npc_brake)
            ego.apply_control(ego_control)
            npc.apply_control(npc_control)

            print('ego:%f,%f,%f,npc:%f,%f,%f'%(ego.get_control().throttle,ego_steer,ego.get_control().brake,
                                            npc.get_control().throttle,npc_steer,npc.get_control().brake))
    
    # 车辆信息反馈
    def get_vehicle_step(self,ego,npc,ego_sensor,npc_sensor,ego_route,npc_route,extra_reward):

        ego_extra, npc_extra = extra_reward
        ego_next_transform = ego.get_transform()
        npc_next_transform = npc.get_transform()
        # 速度、加速度
        ego_velocity = ego.get_velocity()
        npc_velocity = npc.get_velocity()
        ego_yaw = ego_next_transform.rotation.yaw
        npc_yaw = npc_next_transform.rotation.yaw

        ego_target_disX = (ego_route.location.x - ego_next_transform.location.x)
        ego_target_disY = (ego_route.location.y - ego_next_transform.location.y)
        ego_dis_x = (npc_next_transform.location.x-ego_next_transform.location.x)
        ego_dis_y = (npc_next_transform.location.y-ego_next_transform.location.y)
        ego_dis = np.sqrt(ego_dis_x**2+ego_dis_y**2)
        ego_next_state = np.array([ego_target_disX/5,ego_target_disY/5,ego_dis_x/20,ego_dis_y/5,
            ego_velocity.x/30,ego_velocity.y/20,np.sin(ego_yaw/2),npc_velocity.x/30,npc_velocity.y/20,np.sin(npc_yaw/2)])

        npc_target_disX = (npc_route.location.x - npc_next_transform.location.x)
        npc_target_disY = (npc_route.location.y - npc_next_transform.location.y)
        npc_dis_x = (ego_next_transform.location.x-npc_next_transform.location.x)
        npc_dis_y = (ego_next_transform.location.y-npc_next_transform.location.y)
        npc_dis = np.sqrt(npc_dis_x**2+npc_dis_y**2)
        npc_next_state = np.array([npc_target_disX/5,npc_target_disY/5,npc_dis_x/20,npc_dis_y/5,
            npc_velocity.x/30,npc_velocity.y/20,np.sin(npc_yaw/2),ego_velocity.x/30,ego_velocity.y/20,np.sin(ego_yaw/2)])
        
        # ego_acceleration = abs(ego.get_acceleration().y)
        # npc_acceleration = abs(npc.get_acceleration().y)
        # 碰撞、变道检测
        ego_col = ego_sensor[0].get_collision_history()
        npc_col = npc_sensor[0].get_collision_history()
        # ego_inv = ego_sensor[1].get_invasion_history()
        # npc_inv = npc_sensor[1].get_invasion_history()

        # 回报设置:碰撞惩罚、纵向奖励、最低速度惩罚、存活奖励 
        # ev=-1 if ego_velocity <= 2 else 0
        # nv=-1 if npc_velocity <= 2 else 0

        ego_reward = (-5)*ego_col[0] + (-2)*np.abs(ego_target_disX) + (-10)*np.abs(ego_target_disY) + (0.001)*ego_dis + 5*ego_extra
        npc_reward = (-5)*npc_col[0] + (-2)*np.abs(npc_target_disX) + (-15)*np.abs(npc_target_disY) + (0.001)*npc_dis + 5*npc_extra
        # ego_reward = (-20)*ego_col[0] + eb
        # npc_reward = (-20)*npc_col[0] + nb
        ego_sensor[1].reset()
        npc_sensor[1].reset()

        # done结束状态判断
        if ego_col[0]==1 or (ego_next_transform.location.x-120)/125 > 1: # ego结束条件ego_done
            ego_done = True
        else:
            ego_done = False
        if npc_col[0]==1 or (npc_next_transform.location.x-120)/125 > 1: # npc结束条件npc_done
            npc_done = True
        else:
            npc_done = False  
        return [ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done]

    # 车辆动作空间
    def get_action_space(self):
        action_space = [0,0] # 油门、方向盘、刹车,油门刹车合并
        return action_space
    
    # 车辆状态空间
    def get_state_space(self):
        state_space = [0,0,0,0,0,0,0,0,0,0]
        return state_space

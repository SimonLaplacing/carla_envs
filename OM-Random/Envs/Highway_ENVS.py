import numpy as np

import carla
from carla import Transform, Location, Rotation

import utils.Simple_Sensors as SS
from utils.global_route_planner_dao import GlobalRoutePlannerDAO
from utils.global_route_planner import GlobalRoutePlanner

import utils.misc as misc

class Create_Envs(object):
    def __init__(self,args):
        self.synchronous_mode = args.synchronous_mode
        self.no_rendering_mode = args.no_rendering_mode
        self.fixed_delta_seconds = args.fixed_delta_seconds
        self.sensor_list = []
        self.ego_list = []
        self.npc_list = []
        self.obstacle_list = []

        self.world = None
        self.client = None
        self.blueprint_library = None

        self.ego_route = None
        self.npc_route = None
        self.ego_num = 999
        self.npc_num = 999
        self.c_tau = args.c_tau

    def connection(self):
        # 连接客户端:localhost:2000\192.168.199.238:2000
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        # 连接世界
        self.world = self.client.load_world('Town03')
        settings = self.world.get_settings()
        settings.synchronous_mode = self.synchronous_mode
        settings.no_rendering_mode = self.no_rendering_mode
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

        # 蓝图
        self.blueprint_library = self.world.get_blueprint_library()
        return self.world

    def Create_actors(self,world, blueprint_library): 
        self.ego_list = []
        self.npc_list = []
        self.obstacle_list = []
        self.sensor_list = []
        # ego车辆设置---------------------------------------------------------------
        ego_bp = blueprint_library.find(id='vehicle.lincoln.mkz2017')
        # 坐标建立
        ego_transform = Transform(Location(x=160.341522, y=-371.640472, z=0.281942), 
                    Rotation(pitch=0.000000, yaw=0.500910, roll=0.000000))
        # 车辆从蓝图定义以及坐标生成
        ego = world.spawn_actor(ego_bp, ego_transform)
        self.ego_list.append(ego)
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
                self.npc_list.append(npc)
                print('created %s' % npc.type_id)

        # 障碍物设置------------------------------------------------------------------
        obstacle_transform = ego_transform
        for i in range(28):
            if i == 0:
                obsta_bp = blueprint_library.find(id='vehicle.mercedes-benz.coupe')
                obstacle_transform.location += carla.Location(x=55,y=3.8)
                obstacle = world.try_spawn_actor(obsta_bp, obstacle_transform)
                obstacle_transform.location += carla.Location(x=50,y=-5.3)
                if obstacle is None:
                    print('%s obstacle created failed' % i)
                else:
                    self.obstacle_list.append(obstacle)
                    # print('created %s' % obstacle.type_id)
            else:
                obsta_bp = blueprint_library.find(id='static.prop.streetbarrier')
                obstacle_transform.location += carla.Location(x=-3.5,y=7.4)
                obstacle1 = world.try_spawn_actor(obsta_bp, obstacle_transform)
                self.obstacle_list.append(obstacle1)
                obstacle_transform.location += carla.Location(y=-7.4)
                obstacle2 = world.try_spawn_actor(obsta_bp, obstacle_transform)
                self.obstacle_list.append(obstacle2)

        # 传感器设置-------------------------------------------------------------------
        ego_collision = SS.CollisionSensor(ego)
        npc_collision = SS.CollisionSensor(npc)
        ego_invasion = SS.LaneInvasionSensor(ego)
        npc_invasion = SS.LaneInvasionSensor(npc)
        self.sensor_list.extend([[ego_collision,ego_invasion],[npc_collision,npc_invasion]])

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

    def get_route(self):
        ego_transform = self.ego_list[0].get_transform()
        npc_transform = self.npc_list[0].get_transform()

        # 全局路径
        ego_start_location = ego_transform.location
        ego_end_location = ego_transform.location + carla.Location(x=150)
        self.ego_route = self.route_positions_generate(ego_start_location,ego_end_location)
        self.ego_num = len(self.ego_route)

        npc_start_location = npc_transform.location
        npc_end_location = npc_transform.location + carla.Location(x=150)
        self.npc_route = self.route_positions_generate(npc_start_location,npc_end_location)
        self.npc_num = len(self.npc_route)
        return self.ego_route, self.npc_route, self.ego_num, self.npc_num

    # 车辆控制
    def set_vehicle_control(self,ego_action,npc_action):
        ego_move,ego_steer = ego_action
        npc_move,npc_steer = npc_action
        ego_steer = self.c_tau*ego_steer + (1-self.c_tau)*self.ego_list[0].get_control().steer
        npc_steer = self.c_tau*npc_steer + (1-self.c_tau)*self.npc_list[0].get_control().steer
        if ego_move >= 0:
            ego_throttle = self.c_tau*ego_move + (1-self.c_tau)*self.ego_list[0].get_control().throttle
            ego_control = carla.VehicleControl(throttle = ego_throttle, steer = ego_steer, brake = 0)
        elif ego_move < 0:
            ego_brake = -self.c_tau*ego_move + (1-self.c_tau)*self.ego_list[0].get_control().brake
            ego_control = carla.VehicleControl(throttle = 0, steer = ego_steer, brake = ego_brake)
        if npc_move >= 0:
            npc_throttle = self.c_tau*npc_move + (1-self.c_tau)*self.npc_list[0].get_control().throttle
            npc_control = carla.VehicleControl(throttle = npc_throttle, steer = npc_steer, brake = 0)
        elif npc_move < 0:
            npc_brake = -self.c_tau*npc_move + (1-self.c_tau)*self.npc_list[0].get_control().brake
            npc_control = carla.VehicleControl(throttle = 0, steer = npc_steer, brake = npc_brake)
        self.ego_list[0].apply_control(ego_control)
        self.npc_list[0].apply_control(npc_control)

        print('ego:%f,%f,%f,npc:%f,%f,%f'%(self.ego_list[0].get_control().throttle,ego_steer,self.ego_list[0].get_control().brake,
                                        self.npc_list[0].get_control().throttle,npc_steer,self.npc_list[0].get_control().brake))
    

        # 车辆信息反馈
    def get_vehicle_step(self, ego_step, npc_step, step):

        ego_route = self.ego_route[ego_step]
        npc_route = self.npc_route[npc_step]
        ego_next_route = self.ego_route[ego_step + 1]
        npc_next_route = self.npc_route[npc_step + 1]

        ego_next_transform = self.ego_list[0].get_transform()
        npc_next_transform = self.npc_list[0].get_transform()
        obstacle_next_transform = self.obstacle_list[0].get_transform()
        # 速度、加速度
        ego_velocity = self.ego_list[0].get_velocity()
        npc_velocity = self.npc_list[0].get_velocity()
        ego_yaw = ego_next_transform.rotation.yaw * np.pi/180
        npc_yaw = npc_next_transform.rotation.yaw * np.pi/180
        

        ego_f_loc,ego_vec,ego_yaw = misc.inertial_to_frenet(ego_route,ego_next_transform.location.x,ego_next_transform.location.y,ego_velocity.x,ego_velocity.y,ego_yaw)
        npc_f_loc,npc_vec,npc_yaw = misc.inertial_to_frenet(npc_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw)

        ego_next_loc,ego_next_vec,ego_next_yaw = misc.inertial_to_frenet(ego_next_route,ego_next_transform.location.x,ego_next_transform.location.y,ego_velocity.x,ego_velocity.y,ego_yaw)
        npc_next_loc,npc_next_vec,npc_next_yaw = misc.inertial_to_frenet(npc_next_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw)

        ego_target_disX = ego_f_loc[0]
        ego_target_disY = ego_f_loc[1]
        ego_next_disX = ego_next_loc[0]
        ego_next_disY = ego_next_loc[1]

        ego_dis_x = (npc_next_transform.location.x-ego_next_transform.location.x)
        ego_dis_y = (npc_next_transform.location.y-ego_next_transform.location.y)
        ego_dis = np.sqrt(ego_dis_x**2+ego_dis_y**2)
        ego_ob_x = (obstacle_next_transform.location.x-ego_next_transform.location.x)
        ego_ob_y = (obstacle_next_transform.location.y-ego_next_transform.location.y)
        ego_ob = np.sqrt(ego_ob_x**2+ego_ob_y**2)

        ego_next_state = np.array([ego_target_disX/5,ego_target_disY/10,ego_next_disX/10,ego_next_disY/20,ego_dis_x/20,ego_dis_y/20,ego_ob_y/25,
            ego_vec[0]/30,ego_vec[1]/30,np.sin(ego_yaw/2),np.sin(ego_next_yaw/2),npc_vec[0]/30,npc_vec[1]/30,np.sin(npc_yaw/2)])

        npc_target_disX = npc_f_loc[0]
        npc_target_disY = npc_f_loc[1]
        npc_next_disX = npc_next_loc[0]
        npc_next_disY = npc_next_loc[1]

        npc_dis_x = (ego_next_transform.location.x-npc_next_transform.location.x)
        npc_dis_y = (ego_next_transform.location.y-npc_next_transform.location.y)
        npc_dis = np.sqrt(npc_dis_x**2+npc_dis_y**2)
        npc_ob_x = (obstacle_next_transform.location.x-npc_next_transform.location.x)
        npc_ob_y = (obstacle_next_transform.location.y-npc_next_transform.location.y)
        npc_ob = np.sqrt(npc_ob_x**2+npc_ob_y**2)

        npc_next_state = np.array([npc_target_disX/5,npc_target_disY/10,npc_next_disX/10,npc_next_disY/20,npc_dis_x/20,npc_dis_y/20,npc_ob_y/25,
            npc_vec[0]/30,npc_vec[1]/30,np.sin(npc_yaw/2),np.sin(npc_next_yaw/2),ego_vec[0]/30,ego_vec[1]/30,np.sin(ego_yaw/2)])
        
        # ego_acceleration = abs(ego.get_acceleration().y)
        # npc_acceleration = abs(npc.get_acceleration().y)
        # 碰撞、变道检测
        ego_col = self.sensor_list[0][0].get_collision_history()
        npc_col = self.sensor_list[1][0].get_collision_history()
        # ego_inv = ego_sensor[1].get_invasion_history()
        # npc_inv = npc_sensor[1].get_invasion_history()

        # 回报设置:碰撞惩罚、纵向奖励、最低速度惩罚、存活奖励 
        # ev=-1 if ego_velocity <= 2 else 0
        # nv=-1 if npc_velocity <= 2 else 0
        ego_bonus,npc_bonus = 0, 0
        
        if ego_target_disX > -0.5:
            ego_bonus += 1                    
        if npc_target_disX > -0.5:
            npc_bonus += 1

        ego_reward = (-50)*ego_col[0] + (-1)*(ego_target_disX/5)**2 + (-10)*(ego_target_disY/10)**2 + (0.0001)*(ego_dis + ego_ob) + 30*ego_bonus - 0.002*step
        npc_reward = (-50)*npc_col[0] + (-1)*(npc_target_disX/5)**2 + (-10)*(npc_target_disY/10)**2 + (0.0001)*(npc_dis + npc_ob) + 30*npc_bonus - 0.002*step
        # ego_reward = (-80)*ego_col[0] + (-5)*(ego_target_disX/5)**2 + (-10)*(ego_target_disY/10)**2 + (-30)*np.abs(np.sin(ego_yaw/2)) + (-2.5)*(ego_next_disX/10)**2 + (-5)*(ego_next_disY/20)**2 + (-15)*np.abs(np.sin(ego_next_yaw/2)) + (0.002)*(ego_dis) + 50*ego_bonus - 0.0005*step
        # npc_reward = (-80)*npc_col[0] + (-5)*(npc_target_disX/5)**2 + (-10)*(npc_target_disY/10)**2 + (-30)*np.abs(np.sin(npc_yaw/2)) + (-2.5)*(npc_next_disX/10)**2 + (-5)*(npc_next_disY/20)**2 + (-15)*np.abs(np.sin(npc_next_yaw/2)) + (0.002)*(npc_dis) + 50*npc_bonus - 0.0005*step
        # ego_reward = (-20)*ego_col[0] + eb
        # npc_reward = (-20)*npc_col[0] + nb
        # self.sensor_list[0][1].reset()
        # self.sensor_list[1][1].reset()

        # done结束状态判断
        if ego_col[0]==1: # ego结束条件ego_done
            ego_done = True
            egocol_num = 1
            ego_finish = 0
        elif ego_step == self.ego_num - 2:
            ego_done = True
            egocol_num = 0
            ego_finish = 1
        else:
            ego_done = False
            egocol_num = 0
            ego_finish = 0

        if npc_col[0]==1: # npc结束条件npc_done
            npc_done = True
            npccol_num = 1
            npc_finish = 0
        elif npc_step == self.npc_num - 2:
            npc_done = True
            npccol_num = 0
            npc_finish = 1
        else:
            npc_done = False
            npccol_num = 0
            npc_finish = 0  
        return [ego_next_state,ego_reward,ego_done,npc_next_state,npc_reward,npc_done,egocol_num,ego_finish,npccol_num,npc_finish]

    # 车辆动作空间
    def get_action_space(self):
        action_space = [0,0] # 油门、方向盘、刹车,油门刹车合并
        return action_space
    
    # 车辆状态空间
    def get_state_space(self):
        state_space = [0,0,0,0,0,0,0,0,0,0,0,0]
        return state_space

    def clean(self):
        # 清洗环境
        print('Start Cleaning Envs')
        for x in self.sensor_list[0]:
            if x.sensor.is_alive:
                x.sensor.destroy()
        for x in self.sensor_list[1]:
            if x.sensor.is_alive:
                x.sensor.destroy()
        for x in self.ego_list:
            # if x.is_alive:
            self.client.apply_batch([carla.command.DestroyActor(x)])
        for x in self.npc_list:
            # if x.is_alive:
            self.client.apply_batch([carla.command.DestroyActor(x)])
        for x in self.obstacle_list:
            # if x.is_alive:
            self.client.apply_batch([carla.command.DestroyActor(x)])
        print('all clean!')

    def reset(self):
        # 清洗环境
        print('Start Cleaning Envs')
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        for x in self.sensor_list[0]:
            if x.sensor.is_alive:
                x.sensor.destroy()
        for x in self.sensor_list[1]:
            if x.sensor.is_alive:
                x.sensor.destroy()
        for x in self.ego_list:
            # if x.is_alive:
            self.client.apply_batch([carla.command.DestroyActor(x)])
        for x in self.npc_list:
            # if x.is_alive:
            self.client.apply_batch([carla.command.DestroyActor(x)])
        for x in self.obstacle_list:
            # if x.is_alive:
            self.client.apply_batch([carla.command.DestroyActor(x)])
        print('all clean!')
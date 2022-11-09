import numpy as np

import carla
from carla import Transform, Location, Rotation

import Envs.Simple_Sensors as SS
from Lane_Decision.global_route_planner_dao import GlobalRoutePlannerDAO
from Lane_Decision.global_route_planner import GlobalRoutePlanner
from Track_Controller.controller import VehiclePIDController
import utils.misc as misc
from memory_profiler import profile

class Create_Envs(object):
    def __init__(self,args,directory):
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
        self.ego_transform = 0
        self.npc_transform = 0
        self.H = 640
        self.W = 480
        self.directory = directory
        self.ego_controller = None
        self.npc_controller = None

        # PID
        self._dt = 1.0 / 20.0
        self._max_brake = 1
        self._max_throt = 1
        self._max_steer = 1
        self.lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.2,
            'K_I': 0.07,
            'dt': self._dt}
        self.longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0.05,
            'dt': self._dt}
        self._offset = 0

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
    # @profile(stream=open('memory_profile.log','w+'))
    def Create_actors(self): 
        self.sensor_list = []
        self.ego_list = []
        self.npc_list = []
        self.obstacle_list = []
        # ego车辆设置---------------------------------------------------------------
        ego_bp = self.blueprint_library.find(id='vehicle.lincoln.mkz2017')
        # 坐标建立
        self.ego_transform = Transform(Location(x=9, y=-110.350967, z=0.2), 
                    Rotation(pitch=0, yaw=-90, roll=-0.000000))
        # 车辆从蓝图定义以及坐标生成
        ego = self.world.spawn_actor(ego_bp, self.ego_transform)
        self.ego_list.append(ego)
        self.ego_controller = VehiclePIDController(self.ego_list[0],
                                                args_lateral=self.lateral_dict,
                                                args_longitudinal=self.longitudinal_dict,
                                                offset=self._offset,
                                                max_throttle=self._max_throt,
                                                max_brake=self._max_brake,
                                                max_steering=self._max_steer)
        print('created %s' % ego.type_id)

        # 视角设置------------------------------------------------------------------
        spectator = self.world.get_spectator()
        # spec_transform = ego.get_transform()
        spec_transform = Transform(Location(x=9, y=-115.350967, z=0), 
                    Rotation(pitch=0, yaw=180, roll=-0.000000))
        spec_transform.location += carla.Location(x=-5,z=60)
        spec_transform.rotation = carla.Rotation(pitch=-90,yaw=1.9,roll=-0.000000)
        spectator.set_transform(spec_transform)
        
        # npc设置--------------------------------------------------------------------
        self.npc_transform = Transform(Location(x=9, y=-110.350967, z=0.2), 
                    Rotation(pitch=0, yaw=-90, roll=-0.000000))
        for i in range(1):
            self.npc_transform.location += carla.Location(x=-18,y=-24)
            self.npc_transform.rotation = carla.Rotation(pitch=0,yaw=0, roll=-0.000000)
            npc_bp = self.blueprint_library.find(id='vehicle.lincoln.mkz2017')
            # print(npc_bp.get_attribute('color').recommended_values)
            npc_bp.set_attribute('color', '229,28,0')
            npc = self.world.try_spawn_actor(npc_bp, self.npc_transform)
            if npc is None:
                print('%s npc created failed' % i)
            else:
                self.npc_list.append(npc)
                self.npc_controller = VehiclePIDController(self.npc_list[i],
                                                args_lateral=self.lateral_dict,
                                                args_longitudinal=self.longitudinal_dict,
                                                offset=self._offset,
                                                max_throttle=self._max_throt,
                                                max_brake=self._max_brake,
                                                max_steering=self._max_steer)
                print('created %s' % npc.type_id)

        # 障碍物设置------------------------------------------------------------------
        obsta_bp = self.blueprint_library.find(id='static.prop.streetbarrier')
        # 障碍物1
        obstacle_transform1 = Transform(Location(x=9, y=-110.350967,z=0), 
                    Rotation(pitch=0, yaw=-90, roll=-0.000000))
        obstacle_transform1.location += carla.Location(x=50,y=-27,z=3)
        obstacle_transform1.rotation = carla.Rotation(pitch=0, yaw=0, roll=0.000000)
        for i in range(30):
            obstacle1 = self.world.try_spawn_actor(obsta_bp, obstacle_transform1)
            obstacle_transform1.location += carla.Location(x=-2.5,y=-0.05,z=-0.12)
            self.obstacle_list.append(obstacle1)

        # 传感器设置-------------------------------------------------------------------
        ego_collision = SS.CollisionSensor(ego)
        npc_collision = SS.CollisionSensor(npc)
        # ego_invasion = SS.LaneInvasionSensor(ego)
        # npc_invasion = SS.LaneInvasionSensor(npc)
        # ego_camera = SS.Camera(ego, 'ego', self.directory, self.H, self.W)
        # npc_camera = SS.Camera(npc, 'npc', self.directory, self.H, self.W)
        
        self.sensor_list.extend([[ego_collision,ego_collision],[npc_collision,npc_collision]])

        ego_target_speed = carla.Vector3D(0,-10,0)
        npc_target_speed = carla.Vector3D(12,0,0) 
        self.ego_list[0].set_target_velocity(ego_target_speed)
        self.npc_list[0].set_target_velocity(npc_target_speed)


    # 全局规划
    def route_positions_generate(self,start_pos,end_pos):
        dao = GlobalRoutePlannerDAO(self.world.get_map(), sampling_resolution=3.5)
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

        return np.array(positions)

    def get_route(self):
        # 全局路径
        ego_start_location = self.ego_transform.location
        ego_end_location = self.ego_transform.location + carla.Location(x=65,y=-20.5)
        self.ego_route = self.route_positions_generate(ego_start_location,ego_end_location)
        self.ego_num = len(self.ego_route)

        npc_start_location = self.npc_transform.location
        npc_end_location = self.npc_transform.location + carla.Location(x=83,y=-2)
        self.npc_route = self.route_positions_generate(npc_start_location,npc_end_location)
        self.npc_num = len(self.npc_route)
        # print(npc_route[3])
        return self.ego_route, self.npc_route, self.ego_num, self.npc_num

    # 车辆控制
    def set_vehicle_control(self,ego_action,npc_action,ego_step,npc_step):
        a,b,c,d = 0.5,0.15,0.15,0
        ego_x,ego_y,ego_speed = ego_action
        npc_x,npc_y,npc_speed = npc_action
        if ego_speed >= 0:
            ego_color = carla.Color(0,154+100*ego_speed,0)
        else:
            ego_color = carla.Color(154-100*ego_speed,0,0)
        if npc_speed >= 0:
            npc_color = carla.Color(0,154+100*ego_speed,0)
        else:
            npc_color = carla.Color(154-100*ego_speed,0,0)

        ego_speed = c * ego_speed + d + misc.get_speed(self.ego_list[0]) # more acceleration
        npc_speed = c * npc_speed + d + misc.get_speed(self.npc_list[0]) # more acceleration
        ego_waypoint = carla.Transform()
        npc_waypoint = carla.Transform()

        ego_route = self.ego_route[ego_step]
        npc_route = self.npc_route[npc_step]

        ego_next_transform = self.ego_list[0].get_transform()
        npc_next_transform = self.npc_list[0].get_transform()
        # 速度、加速度
        ego_velocity = self.ego_list[0].get_velocity()
        npc_velocity = self.npc_list[0].get_velocity()
        ego_yaw = ego_next_transform.rotation.yaw * np.pi/180
        npc_yaw = npc_next_transform.rotation.yaw * np.pi/180
        
        ego_f_loc,ego_vec,ego_yaw = misc.inertial_to_frenet(ego_route,ego_next_transform.location.x,ego_next_transform.location.y,ego_velocity.x,ego_velocity.y,ego_yaw)
        npc_f_loc,npc_vec,npc_yaw = misc.inertial_to_frenet(npc_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw)

        ego_i_loc,_,_ = misc.frenet_to_inertial(ego_route,ego_f_loc[0]+a*(ego_x+1),ego_f_loc[1]+b*ego_y,ego_vec[0],ego_vec[1],ego_yaw)
        npc_i_loc,_,_ = misc.frenet_to_inertial(npc_route,npc_f_loc[0]+a*(npc_x+1),npc_f_loc[1]+b*npc_y,npc_vec[0],npc_vec[1],npc_yaw)

        ego_waypoint.location = carla.Location(x=ego_i_loc[0], y=ego_i_loc[1])
        npc_waypoint.location = carla.Location(x=npc_i_loc[0], y=npc_i_loc[1])

        self.world.debug.draw_point(location = ego_waypoint.location, color = ego_color, life_time = 1)
        self.world.debug.draw_point(location = npc_waypoint.location, color = npc_color, life_time = 1)

        ego_control = self.ego_controller.run_step(ego_speed,ego_waypoint)
        npc_control = self.npc_controller.run_step(npc_speed,npc_waypoint)
        self.ego_list[0].apply_control(ego_control)
        self.npc_list[0].apply_control(npc_control)

        print('ego:%f,%f,%f,npc:%f,%f,%f'%(self.ego_list[0].get_control().throttle,self.ego_list[0].get_control().steer,self.ego_list[0].get_control().brake,
                                        self.npc_list[0].get_control().throttle,self.npc_list[0].get_control().steer,self.npc_list[0].get_control().brake))
    
    
    # 车辆信息反馈
    def get_vehicle_step(self, ego_step, npc_step, step):

        ego_route = self.ego_route[ego_step]
        npc_route = self.npc_route[npc_step]
        ego_next_route = self.ego_route[ego_step + 1]
        npc_next_route = self.npc_route[npc_step + 1]

        ego_next_transform = self.ego_list[0].get_transform()
        npc_next_transform = self.npc_list[0].get_transform()
        # obstacle_next_transform = self.obstacle_list[0].get_transform()
        # 速度、加速度
        ego_velocity = self.ego_list[0].get_velocity()
        npc_velocity = self.npc_list[0].get_velocity()
        ego_yaw = ego_next_transform.rotation.yaw * np.pi/180
        npc_yaw = npc_next_transform.rotation.yaw * np.pi/180
        ego_acc = self.ego_list[0].get_acceleration()
        npc_acc = self.npc_list[0].get_acceleration()

        ego_f_loc,ego_vec,ego_yaw,ego_acc = misc.inertial_to_frenet(ego_route,ego_next_transform.location.x,ego_next_transform.location.y,ego_velocity.x,ego_velocity.y,ego_yaw,ego_acc.x,ego_acc.y)
        npc_f_loc,npc_vec,npc_yaw,npc_acc = misc.inertial_to_frenet(npc_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw,npc_acc.x,npc_acc.y)

        ego_npc_loc,ego_npc_vec,ego_npc_yaw = misc.inertial_to_frenet(ego_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw)
        npc_ego_loc,npc_ego_vec,npc_ego_yaw = misc.inertial_to_frenet(npc_route,ego_next_transform.location.x,ego_next_transform.location.y,ego_velocity.x,ego_velocity.y,npc_yaw)

        ego_next_loc,ego_next_vec,ego_next_yaw = misc.inertial_to_frenet(ego_next_route,ego_next_transform.location.x,ego_next_transform.location.y,ego_velocity.x,ego_velocity.y,ego_yaw)
        npc_next_loc,npc_next_vec,npc_next_yaw = misc.inertial_to_frenet(npc_next_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw)

        ego_target_disX = ego_f_loc[0]
        ego_target_disY = ego_f_loc[1]
        ego_npc_disX = ego_npc_loc[0]
        ego_npc_disY = ego_npc_loc[1]
        ego_next_disX = ego_next_loc[0]
        ego_next_disY = ego_next_loc[1]

        ego_dis_x = (npc_next_transform.location.x-ego_next_transform.location.x)
        ego_dis_y = (npc_next_transform.location.y-ego_next_transform.location.y)
        ego_dis = np.sqrt(ego_dis_x**2+ego_dis_y**2)
        # ego_ob_x = (obstacle_next_transform.location.x-ego_next_transform.location.x)
        # ego_ob_y = (obstacle_next_transform.location.y-ego_next_transform.location.y)
        # ego_ob = np.sqrt(ego_ob_x**2+ego_ob_y**2)

        ego_next_state = np.array([ego_target_disX/5,ego_target_disY/10,ego_next_disX/10,ego_next_disY/20,ego_vec[0]/30,ego_vec[1]/30,ego_next_vec[0]/30,ego_next_vec[1]/30,np.sin(ego_yaw/2),np.sin(ego_next_yaw/2),
                                    ego_npc_disX/5,ego_npc_disY/10,misc.get_speed(self.npc_list[0]),ego_npc_vec[0]/30,ego_npc_vec[1]/30,np.sin(ego_npc_yaw/2)]) # ego\npc\ob loc\vec\yaw

        npc_target_disX = npc_f_loc[0]
        npc_target_disY = npc_f_loc[1]
        npc_ego_disX = npc_ego_loc[0]
        npc_ego_disY = npc_ego_loc[1]
        npc_next_disX = npc_next_loc[0]
        npc_next_disY = npc_next_loc[1]

        npc_dis_x = (ego_next_transform.location.x-npc_next_transform.location.x)
        npc_dis_y = (ego_next_transform.location.y-npc_next_transform.location.y)
        npc_dis = np.sqrt(npc_dis_x**2+npc_dis_y**2)
        # npc_ob_x = (obstacle_next_transform.location.x-npc_next_transform.location.x)
        # npc_ob_y = (obstacle_next_transform.location.y-npc_next_transform.location.y)
        # npc_ob = np.sqrt(npc_ob_x**2+npc_ob_y**2)

        npc_next_state = np.array([npc_target_disX/5,npc_target_disY/10,npc_next_disX/10,npc_next_disY/20,npc_vec[0]/30,npc_vec[1]/30,npc_next_vec[0]/30,npc_next_vec[1]/30,np.sin(npc_yaw/2),np.sin(npc_next_yaw/2),
                                    npc_ego_disX/5,npc_ego_disY/10,misc.get_speed(self.ego_list[0]),npc_ego_vec[0]/30,npc_ego_vec[1]/30,np.sin(npc_ego_yaw/2)])
        
        # 碰撞、变道检测
        ego_col = self.sensor_list[0][0].get_collision_history()
        npc_col = self.sensor_list[1][0].get_collision_history()
        # ego_inv = ego_sensor[1].get_invasion_history()
        # npc_inv = npc_sensor[1].get_invasion_history()
        # ego_BEV = self.sensor_list[0][1].get_BEV()
        # npc_BEV = self.sensor_list[1][1].get_BEV()

        # 回报设置:碰撞惩罚、纵向奖励、最低速度惩罚、存活奖励 
        # ev=-1 if ego_velocity <= 2 else 0
        # nv=-1 if npc_velocity <= 2 else 0
        ego_bonus,npc_bonus = 0, 0
        
        if ego_target_disX > -0.5:
            ego_bonus = ego_step/25                    
        if npc_target_disX > -0.5:
            npc_bonus = npc_step/25

        ego_reward = ((-100)*ego_col[0] + (0.002)*(ego_dis) 
        + (-5)*(ego_target_disX/5)**2 + (-10)*(ego_target_disY/10)**2 + (-30)*np.abs(np.sin(ego_yaw/2)) 
        + (-2.5)*(ego_next_disX/10)**2 + (-5)*(ego_next_disY/20)**2 + (-15)*np.abs(np.sin(ego_next_yaw/2)) 
        + 50*ego_bonus - 0.0004*step*step
        - 1*abs(ego_acc[1]))
        
        npc_reward = ((-100)*npc_col[0] + (0.002)*(npc_dis) 
        + (-5)*(npc_target_disX/5)**2 + (-10)*(npc_target_disY/10)**2 + (-30)*np.abs(np.sin(npc_yaw/2)) 
        + (-2.5)*(npc_next_disX/10)**2 + (-5)*(npc_next_disY/20)**2 + (-15)*np.abs(np.sin(npc_next_yaw/2)) 
        + 50*npc_bonus - 0.0004*step*step
        - 1*abs(npc_acc[1]))

        # self.sensor_list[0][1].reset()
        # self.sensor_list[1][1].reset()

        # public evaluation
        ego_score = 0
        npc_score = 0

        # done结束状态判断
        if ego_step >= self.ego_num - 3:
            egocol_num = 0
            ego_finish = 1
        elif ego_col[0]==1: # ego结束条件ego_done
            egocol_num = 1
            ego_finish = 0
        else:
            egocol_num = 0
            ego_finish = 0

        if npc_step >= self.npc_num - 3:
            npccol_num = 0
            npc_finish = 1
        elif npc_col[0]==1: # npc结束条件npc_done
            npccol_num = 1
            npc_finish = 0
        else:
            npccol_num = 0
            npc_finish = 0  
        return [ego_next_state,ego_reward,npc_next_state,npc_reward,egocol_num,ego_finish,npccol_num,npc_finish]

    # 车辆动作空间
    def get_action_space(self):
        action_space = [0,0,0] # x,y,speed
        return action_space
    
    # 车辆状态空间
    def get_state_space(self):
        state_space = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
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

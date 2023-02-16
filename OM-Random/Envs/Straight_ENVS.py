from xml.etree.ElementTree import PI
import numpy as np
import math
import time
import carla
from carla import Transform, Location, Rotation

import Envs.Simple_Sensors as SS
from Lane_Decision.global_route_planner_dao import GlobalRoutePlannerDAO
from Lane_Decision.global_route_planner import GlobalRoutePlanner
from Track_Controller.controller import VehiclePIDController
import utils.misc as misc
from memory_profiler import profile
from Envs.carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from cv2 import cv2 as cv
from Path_Decision.frenet_optimal_trajectory import FrenetPlanner as PathPlanner

class Create_Envs(object):
    def __init__(self,args,directory):
        self.synchronous_mode = args.synchronous_mode
        self.no_rendering_mode = args.no_rendering_mode
        self.fixed_delta_seconds = args.fixed_delta_seconds
        self.sensor_list = []
        self.ego_list = []
        self.npc_list = []
        self.car_list = []
        self.obstacle_list = []
        self.ob_loc = []

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
        self.birdViewProducer = None
        self.directory = directory
        self.ego_controller = None
        self.npc_controller = None
        self.args = args
        # PATH
        # self.pathplanner = None
        # self.npc_pathplanner = None
        self.ego_wps_to_go = 0
        self.npc_wps_to_go = 0
        self.ego_path = 0
        self.npc_path = 0
        self.ego_f_idx = 0
        self.npc_f_idx = 0
        self.ego_last_idx = 0
        self.npc_last_idx = 0

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
        self.world = self.client.load_world('Town04')
        settings = self.world.get_settings()
        settings.synchronous_mode = self.synchronous_mode
        settings.no_rendering_mode = self.no_rendering_mode
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)

        # 蓝图
        self.blueprint_library = self.world.get_blueprint_library()
        self.birdViewProducer = BirdViewProducer(self.client, target_size=PixelDimensions(width=self.args.W, height=self.args.H), pixels_per_meter=self.args.res, crop_type=BirdViewCropType.FRONT_AND_REAR_AREA, render_lanes_on_junctions=False)
        return self.world

    def Create_actors(self): 
        self.ego_list = []
        self.npc_list = []
        self.obstacle_list = []
        self.sensor_list = []
        # ego车辆设置---------------------------------------------------------------
        ego_bp = self.blueprint_library.find(id='vehicle.lincoln.mkz2017')
        # 坐标建立
        self.ego_transform = Transform(Location(x=160.341522, y=-371.640472, z=0.281942), 
                    Rotation(pitch=0.000000, yaw=0.500910, roll=0.000000))
        # 车辆从蓝图定义以及坐标生成
        ego = self.world.spawn_actor(ego_bp, self.ego_transform)
        self.ego_list.append(ego)
        self.ego_controller = VehiclePIDController(self.ego_list[0],
                                                args_lateral=self.lateral_dict,
                                                args_longitudinal=self.longitudinal_dict
                                                )
        print('created %s' % ego.type_id)

        # 视角设置------------------------------------------------------------------
        spectator = self.world.get_spectator()
        # spec_transform = ego.get_transform()
        spec_transform = Transform(Location(x=140.341522, y=-375.140472, z=15.281942), 
                    Rotation(pitch=0.000000, yaw=0.500910, roll=0.000000))
        spec_transform.location += carla.Location(x=60,z=45)
        spec_transform.rotation = carla.Rotation(pitch=-90, yaw=90)
        spectator.set_transform(spec_transform)

        # npc设置--------------------------------------------------------------------
        self.npc_transform = Transform(Location(x=160.341522, y=-371.640472, z=0.281942), 
                    Rotation(pitch=0.000000, yaw=0.500910, roll=0.000000))
        for i in range(1):
            self.npc_transform.location += carla.Location(x=-35,y=-13.5)
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
        obstacle_transform = Transform(Location(x=160.341522, y=-371.640472, z=0.281942), 
                    Rotation(pitch=0.000000, yaw=0.500910, roll=0.000000))
        for i in range(27): #28
            if i == 0:
                obstacle_transform.location += carla.Location(x=95,y=3.8) #40,0.3
                obstacle_transform.location += carla.Location(x=0,y=-5.8)

            else:
                obsta_bp = self.blueprint_library.find(id='static.prop.streetbarrier')
                obstacle_transform.location += carla.Location(x=-3.5,y=4.4)
                obstacle1 = self.world.try_spawn_actor(obsta_bp, obstacle_transform)
                self.obstacle_list.append(obstacle1)
                obstacle_transform.location += carla.Location(y=-4.4)
                obstacle2 = self.world.try_spawn_actor(obsta_bp, obstacle_transform)
                self.obstacle_list.append(obstacle2)
                self.ob_loc.append([self.obstacle_list[i].get_location().x, self.obstacle_list[i].get_location().y, 
                                self.obstacle_list[i].get_location().z])


        # 传感器设置-------------------------------------------------------------------
        ego_collision = SS.CollisionSensor(ego)
        npc_collision = SS.CollisionSensor(npc)
        # ego_invasion = SS.LaneInvasionSensor(ego)
        # npc_invasion = SS.LaneInvasionSensor(npc)
        # ego_camera = SS.Camera(ego, 'ego', self.directory, self.H, self.W)
        # npc_camera = SS.Camera(npc, 'npc', self.directory, self.H, self.W)
        # self.sensor_list.extend([[ego_collision,ego_collision],[npc_collision,npc_collision]])

        self.sensor_list.extend([[ego_collision,ego_collision],[npc_collision,npc_collision]])
        
        # 车辆初始参数
        ego_target_speed = carla.Vector3D(20,0,0) # 16.5
        npc_target_speed = carla.Vector3D(0,0,0) # 20
        self.ego_list[0].set_target_velocity(ego_target_speed)
        self.npc_list[0].set_target_velocity(npc_target_speed)

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
        
        return np.array(positions)

    def get_route(self):
        # 全局路径
        ego_start_location = self.ego_transform.location
        ego_end_location = self.ego_transform.location + carla.Location(x=138)
        self.ego_route = self.route_positions_generate(ego_start_location,ego_end_location)
        self.ego_num = len(self.ego_route)

        npc_start_location = self.npc_transform.location
        npc_end_location = self.npc_transform.location + carla.Location(x=138)
        self.npc_route = self.route_positions_generate(npc_start_location,npc_end_location)
        self.npc_num = len(self.npc_route)

        return self.ego_route, self.npc_route, self.ego_num, self.npc_num
    
    def update_route(self,route):
        pathplanner = PathPlanner(self.args)
        # self.npc_pathplanner = PathPlanner(self.args)
        pathplanner.start(route)
        # self.npc_pathplanner.start(route)
        pathplanner.reset()
        self.ego_wps_to_go = 0
        self.npc_wps_to_go = 0
        self.ego_path = 0
        self.npc_path = 0
        self.ego_f_idx = 0
        self.npc_f_idx = 0
        return pathplanner

    def generate_path(self,vehicle,pathplanner,f_idx):
        # 局部规划
        temp = [vehicle.get_velocity(), vehicle.get_acceleration()]
        speed = misc.get_speed(vehicle)
        acc_vec = vehicle.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(vehicle.get_transform().rotation.yaw)
        state = [vehicle.get_location().x, vehicle.get_location().y, speed, acc, psi, temp, self.args.carla_max_s]
        # fpath = self.motionPlanner.run_step_single_path(state, self.f_idx, df_n=action[0], Tf=5, Vf_n=action[1])
        fpath, fplist, best_path_idx = pathplanner.run_step(state, f_idx, None, self.ob_loc, target_speed=30/3.6)
        # fpath, fplist, best_path_idx = self.pathplanner.run_step(state, self.f_idx, None, self.obstacle_list, target_speed=30/3.6)
        wps_to_go = len(fpath.t) - 3 if fpath!=0 else 0   # -2 bc len gives # of items not the idx of last item + 2wp controller is used
        return fpath, fplist, best_path_idx, wps_to_go

    def get_path(self,vehicle,pathplanner,f_idx):
        #局部路径
        # ego_fp, ego_fplist, ego_best_path_idx = self.generate_path('ego')
        fp, fplist, best_path_idx, wps_to_go = self.generate_path(vehicle,pathplanner,f_idx)
        path = fp
        positions = []
        # npc_positions = []
        if fp == 0:
            return 0,0,0
        for i in range((len(path.t))):
            position = [path.x[i],path.y[i]]
            positions.append(position)
        # for i in range((len(self.npc_path.t))):
        #     position = [self.npc_path[i].x,self.npc_path[i].y]
        #     ego_positions.append(position)
        return positions, wps_to_go, fp

    # 车辆控制
    def set_vehicle_control(self,ego_action,npc_action,ego_step,npc_step):
        if not self.args.direct_control:
            a,b,c,d = 0.5,0.1,0.2,-0.08
            ego_x,ego_y,ego_speed = ego_action
            npc_x,npc_y,npc_speed = npc_action
            if ego_speed >= 0:
                ego_color = carla.Color(0,int(154+100*ego_speed),0)
            else:
                ego_color = carla.Color(int(154-100*ego_speed),0,0)
            if npc_speed >= 0:
                npc_color = carla.Color(0,int(154+100*ego_speed),0)
            else:
                npc_color = carla.Color(int(154-100*ego_speed),0,0)

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

            ego_i_loc,_,_ = misc.frenet_to_inertial(ego_route,ego_f_loc[0]+a*(ego_x+1.001),ego_f_loc[1]+b*ego_y,ego_vec[0],ego_vec[1],ego_yaw)
            npc_i_loc,_,_ = misc.frenet_to_inertial(npc_route,npc_f_loc[0]+a*(npc_x+1.001),npc_f_loc[1]+b*npc_y,npc_vec[0],npc_vec[1],npc_yaw)

            ego_waypoint.location = carla.Location(x=ego_i_loc[0], y=ego_i_loc[1])
            npc_waypoint.location = carla.Location(x=npc_i_loc[0], y=npc_i_loc[1])

            # self.world.debug.draw_point(location = ego_waypoint.location, color = ego_color, life_time = 1)
            # self.world.debug.draw_point(location = npc_waypoint.location, color = npc_color, life_time = 1)

            ego_control = self.ego_controller.run_step(ego_speed,ego_waypoint)
            npc_control = self.npc_controller.run_step(npc_speed,npc_waypoint)
        else:
            ego_move,ego_steer = ego_action
            npc_move,npc_steer = npc_action

            ego_steer = self.args.c_tau*ego_steer + (1-self.args.c_tau)*self.ego_list[0].get_control().steer
            npc_steer = self.args.c_tau*npc_steer + (1-self.args.c_tau)*self.npc_list[0].get_control().steer
            if ego_move >= 0:
                ego_throttle = self.args.c_tau*ego_move + (1-self.args.c_tau)*self.ego_list[0].get_control().throttle
                ego_control = carla.VehicleControl(throttle = ego_throttle, steer = ego_steer, brake = 0)
            elif ego_move < 0:
                ego_brake = -self.args.c_tau*ego_move + (1-self.args.c_tau)*self.ego_list[0].get_control().brake
                ego_control = carla.VehicleControl(throttle = 0, steer = ego_steer, brake = ego_brake)
            if npc_move >= 0:
                npc_throttle = self.args.c_tau*npc_move + (1-self.args.c_tau)*self.npc_list[0].get_control().throttle
                npc_control = carla.VehicleControl(throttle = npc_throttle, steer = npc_steer, brake = 0)
            elif npc_move < 0:
                npc_brake = -self.args.c_tau*npc_move + (1-self.args.c_tau)*self.npc_list[0].get_control().brake
                npc_control = carla.VehicleControl(throttle = 0, steer = npc_steer, brake = npc_brake)

        self.ego_list[0].apply_control(ego_control)
        if self.args.npc_exist:
            self.npc_list[0].apply_control(npc_control)

        # print('ego:%f,%f,%f,npc:%f,%f,%f'%(self.ego_list[0].get_control().throttle,self.ego_list[0].get_control().steer,self.ego_list[0].get_control().brake,
        #                                 self.npc_list[0].get_control().throttle,self.npc_list[0].get_control().steer,self.npc_list[0].get_control().brake))
    

        # 车辆信息反馈
    def get_vehicle_step(self, ego_step, npc_step, step):

        ego = self.ego_list[0]
        npc = self.npc_list[0]
        ego_path = []
        npc_path = []
        ego_path_bonus = 0
        npc_path_bonus = 0

        if self.args.Start_Path:
            if step ==0:
                self.ego_pathplanner = self.update_route(self.ego_route)
                self.npc_pathplanner = self.update_route(self.npc_route)

            # print('index: ', self.ego_f_idx, self.ego_wps_to_go)
            if self.ego_f_idx >= self.ego_wps_to_go:
                a = time.time()                
                ego_path, self.ego_wps_to_go, ego_fp = self.get_path(ego,self.ego_pathplanner,self.ego_f_idx)
                # print('time: ',time.time()-a)
                if ego_path!=0:
                    self.ego_path = ego_path
                    for i in range((len(ego_fp.t))):
                        ego_loc = carla.Location(x=self.ego_path[i][0], y=self.ego_path[i][1])
                        self.world.debug.draw_point(location = ego_loc, life_time = 5)
                    self.ego_f_idx = 1
                
            if self.npc_f_idx >= self.npc_wps_to_go:                
                npc_path, self.npc_wps_to_go, npc_fp = self.get_path(npc,self.npc_pathplanner,self.npc_f_idx)
                if npc_path != 0:
                    self.npc_path = npc_path
                    for i in range((len(npc_fp.t))):
                        npc_loc = carla.Location(x=self.npc_path[i][0], y=self.npc_path[i][1])
                        self.world.debug.draw_point(location = npc_loc, life_time = 5)
                    self.npc_f_idx = 1

        
        ego_location = [ego.get_location().x, ego.get_location().y, math.radians(ego.get_transform().rotation.yaw)]
        npc_location = [npc.get_location().x, npc.get_location().y, math.radians(npc.get_transform().rotation.yaw)]

        if self.args.Start_Path:
            if ego_path!=0:
                self.ego_f_idx = misc.closest_wp_idx(ego_location, self.ego_path, self.ego_f_idx)
                ego_path_bonus = self.ego_f_idx - self.ego_last_idx
                self.ego_last_idx = self.ego_f_idx

            if npc_path != 0:
                self.npc_f_idx = misc.closest_wp_idx(npc_location, self.npc_path, self.npc_f_idx)
                npc_path_bonus = self.npc_f_idx - self.npc_last_idx
                self.npc_last_idx = self.npc_f_idx
            
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
        ego_acc = self.ego_list[0].get_acceleration()
        npc_acc = self.npc_list[0].get_acceleration()
        

        ego_f_loc,ego_vec,ego_yaw,ego_acc = misc.inertial_to_frenet(ego_route,ego_next_transform.location.x,ego_next_transform.location.y,ego_velocity.x,ego_velocity.y,ego_yaw,ego_acc.x,ego_acc.y)
        npc_f_loc,npc_vec,npc_yaw,npc_acc = misc.inertial_to_frenet(npc_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw,npc_acc.x,npc_acc.y)

        if self.args.Start_Path:
            if ego_path != 0:
                ego_f_path,_,_ = misc.inertial_to_frenet(ego_route,self.ego_path[self.ego_f_idx][0],self.ego_path[self.ego_f_idx][1],ego_velocity.x,ego_velocity.y,ego_yaw)
            if npc_path!=0:
                npc_f_path,_,_ = misc.inertial_to_frenet(npc_route,self.npc_path[self.npc_f_idx][0],self.npc_path[self.npc_f_idx][1],npc_velocity.x,npc_velocity.y,npc_yaw)

        ego_npc_loc,ego_npc_vec,ego_npc_yaw = misc.inertial_to_frenet(ego_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw)
        npc_ego_loc,npc_ego_vec,npc_ego_yaw = misc.inertial_to_frenet(npc_route,ego_next_transform.location.x,ego_next_transform.location.y,ego_velocity.x,ego_velocity.y,npc_yaw)

        ego_next_loc,ego_next_vec,ego_next_yaw = misc.inertial_to_frenet(ego_next_route,ego_next_transform.location.x,ego_next_transform.location.y,ego_velocity.x,ego_velocity.y,ego_yaw)
        npc_next_loc,npc_next_vec,npc_next_yaw = misc.inertial_to_frenet(npc_next_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw)

        ego_ob_loc,_,_ = misc.inertial_to_frenet(ego_route,obstacle_next_transform.location.x,obstacle_next_transform.location.y)
        npc_ob_loc,_,_ = misc.inertial_to_frenet(npc_route,obstacle_next_transform.location.x,obstacle_next_transform.location.y)

        ego_target_disX = ego_f_loc[0]
        ego_target_disY = ego_f_loc[1]
        ego_npc_disX = ego_npc_loc[0]
        ego_npc_disY = ego_npc_loc[1]
        ego_next_disX = ego_next_loc[0]
        ego_next_disY = ego_next_loc[1]

        ego_dis_x = (npc_next_transform.location.x-ego_next_transform.location.x)
        ego_dis_y = (npc_next_transform.location.y-ego_next_transform.location.y)
        ego_dis = np.sqrt(ego_dis_x**2+ego_dis_y**2)
        ego_ob_x = (obstacle_next_transform.location.x-ego_next_transform.location.x)
        ego_ob_y = (obstacle_next_transform.location.y-ego_next_transform.location.y)
        ego_ob = np.sqrt(ego_ob_x**2+ego_ob_y**2)

        ego_BEV_ = self.birdViewProducer.produce(agent_vehicle=self.ego_list[0])
        npc_BEV_ = self.birdViewProducer.produce(agent_vehicle=self.npc_list[0])
        # ego_BEV = ego_BEV_.swapaxes(0,2).swapaxes(1,2)
        # ego_BEV = npc_BEV_.swapaxes(0,2).swapaxes(1,2)

        ego_rgb = cv.cvtColor(BirdViewProducer.as_rgb(ego_BEV_), cv.COLOR_BGR2RGB)
        npc_rgb = cv.cvtColor(BirdViewProducer.as_rgb(npc_BEV_), cv.COLOR_BGR2RGB)
        # NOTE imshow requires BGR color model
        # cv.imshow("BirdView RGB", ego_rgb)
        # cv.imwrite(self.directory + '/save_1.png', ego_rgb)

        ego_BEV = ego_rgb.swapaxes(0,2).swapaxes(1,2)
        npc_BEV = npc_rgb.swapaxes(0,2).swapaxes(1,2)

        # ego_BEV = self.sensor_list[0][1].get_BEV()
        # npc_BEV = self.sensor_list[1][1].get_BEV()

        ego_next_state = np.array([ego_target_disX/5,ego_target_disY/10,ego_next_disX/10,ego_next_disY/10,ego_vec[0]/40,ego_vec[1]/40,ego_next_vec[0]/40,ego_next_vec[1]/40,np.sin(ego_yaw/2),np.sin(ego_next_yaw/2), # 自车
        ego_npc_disX/40,ego_npc_disY/10,misc.get_speed(self.npc_list[0])/40,ego_npc_vec[0]/30,ego_npc_vec[1]/30,np.sin(ego_npc_yaw/2), # 外车
        ego_ob_loc[0]/30,ego_ob_loc[1]/25]) # 障碍

        npc_target_disX = npc_f_loc[0]
        npc_target_disY = npc_f_loc[1]
        npc_ego_disX = npc_ego_loc[0]
        npc_ego_disY = npc_ego_loc[1]
        npc_next_disX = npc_next_loc[0]
        npc_next_disY = npc_next_loc[1]

        npc_dis_x = (ego_next_transform.location.x-npc_next_transform.location.x)
        npc_dis_y = (ego_next_transform.location.y-npc_next_transform.location.y)
        npc_dis = np.sqrt(npc_dis_x**2+npc_dis_y**2)
        npc_ob_x = (obstacle_next_transform.location.x-npc_next_transform.location.x)
        npc_ob_y = (obstacle_next_transform.location.y-npc_next_transform.location.y)
        npc_ob = np.sqrt(npc_ob_x**2+npc_ob_y**2)

        npc_next_state = np.array([npc_target_disX/5,npc_target_disY/10,npc_next_disX/10,npc_next_disY/10,npc_vec[0]/40,npc_vec[1]/40,npc_next_vec[0]/40,npc_next_vec[1]/40,np.sin(npc_yaw/2),np.sin(npc_next_yaw/2),
        npc_ego_disX/40,npc_ego_disY/10,misc.get_speed(self.ego_list[0])/40,npc_ego_vec[0]/30,npc_ego_vec[1]/30,np.sin(npc_ego_yaw/2),
        npc_ob_loc[0]/30,npc_ob_loc[1]/25])
        # print(ego_next_state,npc_next_state)
        # 碰撞、变道检测
        ego_col = self.sensor_list[0][0].get_collision_history()
        npc_col = self.sensor_list[1][0].get_collision_history()
        # ego_inv = ego_sensor[1].get_invasion_history()
        # npc_inv = npc_sensor[1].get_invasion_history()
        

        # 回报设置:碰撞惩罚、纵向奖励、最低速度惩罚、存活奖励 
        # ev=-1 if ego_velocity <= 2 else 0
        # nv=-1 if npc_velocity <= 2 else 0
        ego_bonus,npc_bonus,timeout = 0, 0, 0
        
        if ego_target_disX > -1:
            ego_bonus = 1                    
        if npc_target_disX > -1:
            npc_bonus = 1 
        if step >= self.args.max_length_of_trajectory - 1:
            timeout = 1

        # ego_reward = (-80)*ego_col[0] + (-5)*(ego_target_disX/5)**2 + (-10)*(ego_target_disY/10)**2 + (-30)*np.abs(np.sin(ego_yaw/2)) + (-2.5)*(ego_next_disX/10)**2 + (-5)*(ego_next_disY/20)**2 + (-15)*np.abs(np.sin(ego_next_yaw/2)) + (0.002)*(ego_dis) + 50*ego_bonus - 0.0005*step
        # npc_reward = (-80)*npc_col[0] + (-5)*(npc_target_disX/5)**2 + (-10)*(npc_target_disY/10)**2 + (-30)*np.abs(np.sin(npc_yaw/2)) + (-2.5)*(npc_next_disX/10)**2 + (-5)*(npc_next_disY/20)**2 + (-15)*np.abs(np.sin(npc_next_yaw/2)) + (0.002)*(npc_dis) + 50*npc_bonus - 0.0005*step
        
        # self.sensor_list[0][1].reset()
        # self.sensor_list[1][1].reset()
        # print(ego_reward,npc_reward,ego_bonus,npc_bonus)
        ego_score = 0
        npc_score = 0

        # done结束状态判断
        if ego_step >= self.ego_num - 3:
            egocol_num = 0
            ego_finish = 1
        elif ego_col[0]==1 or ego_path==0: # ego结束条件ego_done
            egocol_num = 1
            ego_finish = 0
        else:
            egocol_num = 0
            ego_finish = 0

        if npc_step >= self.npc_num - 3:
            npccol_num = 0
            npc_finish = 1
        elif npc_col[0]==1 or npc_path==0: # npc结束条件npc_done
            npccol_num = 1
            npc_finish = 0
        else:
            npccol_num = 0
            npc_finish = 0

        #simple reward
        # ego_reward = (-1)*ego_col[0] + (-0.6)*timeout + 1*ego_bonus
        # npc_reward = (-1)*npc_col[0] + (-0.6)*timeout + 1*npc_bonus

        #reward shaping
        ego_reward = ((-80)*ego_col[0] + (0.001)*(ego_dis + ego_ob) 
        + (-5)*(ego_target_disX/5)**2 + (-10)*(ego_target_disY/10)**2 + (-30)*np.abs(np.sin(ego_yaw/2)) 
        + (-2.5)*(ego_next_disX/10)**2 + (-5)*(ego_next_disY/10)**2 + (-15)*np.abs(np.sin(ego_next_yaw/2))
        + 40*ego_bonus - 0.002*step
        - 0.25*abs(ego_acc[1]))
        npc_reward = ((-80)*npc_col[0] + (0.001)*(npc_dis + npc_ob)
        + (-5)*(npc_target_disX/5)**2 + (-10)*(npc_target_disY/10)**2 + (-30)*np.abs(np.sin(npc_yaw/2))
        + (-2.5)*(npc_next_disX/10)**2 + (-5)*(npc_next_disY/10)**2 + (-15)*np.abs(np.sin(npc_next_yaw/2)) 
        + 40*npc_bonus - 0.002*step
        - 0.25*abs(npc_acc[1]))
          
        return [ego_next_state,ego_reward,npc_next_state,npc_reward,egocol_num,ego_finish,npccol_num,npc_finish,ego_BEV,npc_BEV]

    # 车辆动作空间
    def get_action_space(self):
        if self.args.direct_control:
            action_space = [0,0] # throttle-brake/steer
        else:
            action_space = [0,0,0] # x,y,speed
        return action_space
    
    # 车辆状态空间
    def get_state_space(self):
        state_space = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
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
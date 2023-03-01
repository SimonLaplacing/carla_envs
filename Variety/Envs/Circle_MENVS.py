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
        
        self.args = args
        self.agent_num = self.args.agent_num

        self.sensor_list = list(np.zeros(self.agent_num,dtype=int))
        self.ego_list = list(np.zeros(self.agent_num,dtype=int))
        self.obstacle_list = []
        self.ob_loc = list(np.zeros(self.agent_num,dtype=int))

        self.world = None
        self.client = None
        self.blueprint_library = None
        self.route = list(np.zeros(self.agent_num,dtype=int))
        # self.npc_route = []
        self.ego_num = list(999*np.ones(self.agent_num,dtype=int))
        # self.npc_num            
        self.c_tau = args.c_tau
        self.ego_transform = list(np.zeros(self.agent_num,dtype=int))
        # self.npc_transform = 0
        self.birdViewProducer = None
        self.directory = directory
        self.controller = list(np.zeros(self.agent_num,dtype=int))
        # self.npc_controller = None
        
        # PATH
        self.pathplanner = list(np.zeros(self.agent_num,dtype=int))
        # self.npc_pathplanner = None
        self.wps_to_go = list(np.zeros(self.agent_num,dtype=int))
        # self.npc_wps_to_go = 0
        self.path = list(np.zeros(self.agent_num,dtype=int))
        # self.npc_path = 0
        self.f_idx = list(np.zeros(self.agent_num,dtype=int))
        # self.npc_f_idx = 0
        self.last_idx = list(np.zeros(self.agent_num,dtype=int))
        # self.npc_last_idx = 0

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
        self.birdViewProducer = BirdViewProducer(self.client, target_size=PixelDimensions(width=self.args.W, height=self.args.H), pixels_per_meter=self.args.res, crop_type=BirdViewCropType.FRONT_AND_REAR_AREA, render_lanes_on_junctions=False)
        return self.world

    def Create_actors(self): 
        self.ego_list = list(np.zeros(self.agent_num,dtype=int))
        # self.npc_list = []
        self.obstacle_list = []
        self.sensor_list = list(np.zeros(self.agent_num,dtype=int))
        self.controller = list(np.zeros(self.agent_num,dtype=int))
        # ego车辆设置---------------------------------------------------------------
        ego_bp = self.blueprint_library.find(id='vehicle.lincoln.mkz2017')
        # 坐标建立
        self.ego_transform = Transform(Location(x=32, y=-4.350967, z=0.1), 
                    Rotation(pitch=-0.348271, yaw=180, roll=-0.000000))
        # 车辆从蓝图定义以及坐标生成
        ego = self.world.spawn_actor(ego_bp, self.ego_transform)
        self.ego_list[0] = ego
        self.controller[0] = VehiclePIDController(self.ego_list[0],
                                                args_lateral=self.lateral_dict,
                                                args_longitudinal=self.longitudinal_dict
                                                )
        print('created %s' % ego.type_id)

        # 视角设置------------------------------------------------------------------
        spectator = self.world.get_spectator()
        # spec_transform = ego.get_transform()
        spec_transform = Transform(Location(x=30, y=-4.350967, z=0), 
                    Rotation(pitch=-0.348271, yaw=180, roll=-0.000000))
        spec_transform.location += carla.Location(x=-15,z=50)
        spec_transform.rotation = carla.Rotation(pitch=-90,yaw=0,roll=-0.000000)
        spectator.set_transform(spec_transform)

        # ego序列设置--------------------------------------------------------------------
        self.ego_transform = Transform(Location(x=30, y=-4.350967, z=0.1), 
                    Rotation(pitch=-0.348271, yaw=180, roll=-0.000000))
        for i in range(self.agent_num-1):
            self.ego_transform.location += carla.Location(x=-11,y=6)
            self.ego_transform.transform.rotation = carla.Rotation(pitch=0.348271,yaw=275, roll=-0.000000)
            ego_bp = self.blueprint_library.find(id='vehicle.lincoln.mkz2017')
            # print(npc_bp.get_attribute('color').recommended_values)
            ego_bp.set_attribute('color', '229,28,0')
            ego = self.world.try_spawn_actor(ego_bp, self.ego_transform)
            if ego is None:
                print('%s npc created failed' % i)
            else:
                self.ego_list[i+1] = ego
                self.controller[i+1] = VehiclePIDController(self.ego_list[i+1],
                                                args_lateral=self.lateral_dict,
                                                args_longitudinal=self.longitudinal_dict,
                                                offset=self._offset,
                                                max_throttle=self._max_throt,
                                                max_brake=self._max_brake,
                                                max_steering=self._max_steer)
                print('created %s' % ego.type_id)

        # 障碍物设置------------------------------------------------------------------
        obsta_bp = self.blueprint_library.find(id='static.prop.streetbarrier')
        #障碍物1
        obstacle_transform1 =Transform(Location(x=30, y=-4.350967, z=0), 
                    Rotation(pitch=-0.348271, yaw=180, roll=-0.000000))
        obstacle_transform1.location += carla.Location(x=14,y=1.8)
        obstacle_transform1.rotation = carla.Rotation(pitch=0, yaw=0, roll=0.000000)
        for i in range(10):
            obstacle1 = self.world.try_spawn_actor(obsta_bp, obstacle_transform1)
            obstacle_transform1.location += carla.Location(x=-2.5,y=-0.04)
            self.obstacle_list.append(obstacle1)
        #障碍物2
        obstacle_transform2 =Transform(Location(x=30, y=-4.350967, z=0), 
                    Rotation(pitch=-0.348271, yaw=180, roll=-0.000000))
        obstacle_transform2.location += carla.Location(x=-8.5,y=1.82)
        obstacle_transform2.rotation = carla.Rotation(pitch=0, yaw=275, roll=0.000000)
        for i in range(4):
            obstacle2 = self.world.try_spawn_actor(obsta_bp, obstacle_transform2)
            obstacle_transform2.location += carla.Location(x=-0.1,y=2.5)
            self.obstacle_list.append(obstacle2)
        obstacle_transform3 =Transform(Location(x=30, y=-4.350967, z=0), 
                    Rotation(pitch=-0.348271, yaw=180, roll=-0.000000))
        obstacle_transform3.location += carla.Location(x=-10,y=-22)
        obstacle_transform3.rotation = carla.Rotation(pitch=0, yaw=0, roll=0.000000)
        for i in range(25):
            obstacle3 = self.world.try_spawn_actor(obsta_bp, obstacle_transform3)
            obstacle_transform3.location += carla.Location(x=-2.5)
            self.obstacle_list.append(obstacle3)
        for i in range(1):
            self.ob_loc.append([self.obstacle_list[i].get_location().x, self.obstacle_list[i].get_location().y, 
                                self.obstacle_list[i].get_location().z])


        # 传感器设置-------------------------------------------------------------------
        for i in range(self.agent_num):
            collision = SS.CollisionSensor(self.ego_list[i])
            self.sensor_list[i] = collision
        
        # 车辆初始参数
        target_speed = [carla.Vector3D(-10,0,0),carla.Vector3D(0,0,0),carla.Vector3D(0,0,0),carla.Vector3D(0,0,0),carla.Vector3D(0,0,0),carla.Vector3D(0,0,0)] # 16.5-20
        for i in range(self.agent_num):
            self.ego_list[i].set_target_velocity(target_speed[i])

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
        start_location = list(np.zeros(self.agent_num,dtype=int))
        delta = [carla.Location(x=-44.5,y=120),carla.Location(x=-33.5,y=120),carla.Location(x=-33.5,y=120),carla.Location(x=-33.5,y=120),carla.Location(x=-33.5,y=120)]
        for i in range(self.agent_num):
            start_location[i] = self.ego_list[i].get_location()
            self.route[i] = self.route_positions_generate(start_location[i],start_location[i]+delta[i])
            self.ego_num[i] = len(self.route[i])

        # npc_start_location = self.npc_transform.location
        # npc_end_location = self.npc_transform.location + carla.Location(x=138)
        # self.ego_route[1] = self.route_positions_generate(npc_start_location,npc_end_location)
        # self.npc_num = len(self.npc_route[1])
        return self.route, self.ego_num
    
    def update_route(self,route):
        pathplanner = PathPlanner(self.args)
        # self.npc_pathplanner = PathPlanner(self.args)
        pathplanner.start(route)
        # self.npc_pathplanner.start(route)
        pathplanner.reset()
        # self.wps_to_go = list(np.zeros(self.agent_num,dtype=int))
        # self.path = list(np.zeros(self.agent_num,dtype=int))
        # self.f_idx = list(np.zeros(self.agent_num,dtype=int)) 
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
        fpath, fplist, best_path_idx = pathplanner.run_step(state, f_idx, None, self.ob_loc, target_speed=70/3.6)
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
    def set_vehicle_control(self,action,step_list):
        control = list(np.zeros(self.agent_num,dtype=int))
        for i in range(self.agent_num):
            if not self.args.direct_control:
                a,b,c,d = 0.5,0.1,0.2,-0.08
                x,y,speed = [],[],[]
                x,y,speed = action[i]
                speed = c * speed + d + misc.get_speed(self.ego_list[0]) # more acceleration

                waypoint = carla.Transform()
                # npc_waypoint = carla.Transform()

                route = self.route[i][step_list[i]]
                # npc_route = self.npc_route[npc_step]

                next_transform = self.ego_list[0].get_transform()
                # npc_next_transform = self.npc_list[0].get_transform()
                # 速度、加速度
                velocity = self.ego_list[0].get_velocity()
                # npc_velocity = self.npc_list[0].get_velocity()
                yaw = next_transform.rotation.yaw * np.pi/180
                # npc_yaw = npc_next_transform.rotation.yaw * np.pi/180
                
                f_loc,vec,yaw = misc.inertial_to_frenet(route,next_transform.location.x,next_transform.location.y,velocity.x,velocity.y,yaw)
                # npc_f_loc,npc_vec,npc_yaw = misc.inertial_to_frenet(npc_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw)

                i_loc,_,_ = misc.frenet_to_inertial(route,f_loc[0]+a*(x+1.001),f_loc[1]+b*y,vec[0],vec[1],yaw)
                # npc_i_loc,_,_ = misc.frenet_to_inertial(npc_route,npc_f_loc[0]+a*(npc_x+1.001),npc_f_loc[1]+b*npc_y,npc_vec[0],npc_vec[1],npc_yaw)

                waypoint.location = carla.Location(x=i_loc[0], y=i_loc[1])
                # npc_waypoint.location = carla.Location(x=npc_i_loc[0], y=npc_i_loc[1])

                # self.world.debug.draw_point(location = ego_waypoint.location, color = ego_color, life_time = 1)
                # self.world.debug.draw_point(location = npc_waypoint.location, color = npc_color, life_time = 1)

                control[i] = self.controller[i].run_step(speed,waypoint)
                # npc_control = self.npc_controller.run_step(npc_speed,npc_waypoint)
            else:
                move,steer = action[i]
                # npc_move,npc_steer = npc_action

                steer = self.args.c_tau*steer + (1-self.args.c_tau)*self.ego_list[0].get_control().steer
                # npc_steer = self.args.c_tau*npc_steer + (1-self.args.c_tau)*self.npc_list[0].get_control().steer
                if move >= 0:
                    throttle = self.args.c_tau*move + (1-self.args.c_tau)*self.ego_list[0].get_control().throttle
                    control[i] = carla.VehicleControl(throttle = throttle, steer = steer, brake = 0)
                elif move < 0:
                    brake = -self.args.c_tau*move + (1-self.args.c_tau)*self.ego_list[0].get_control().brake
                    control[i] = carla.VehicleControl(throttle = 0, steer = steer, brake = brake)
                # if npc_move >= 0:
                #     npc_throttle = self.args.c_tau*npc_move + (1-self.args.c_tau)*self.npc_list[0].get_control().throttle
                #     npc_control = carla.VehicleControl(throttle = npc_throttle, steer = npc_steer, brake = 0)
                # elif npc_move < 0:
                #     npc_brake = -self.args.c_tau*npc_move + (1-self.args.c_tau)*self.npc_list[0].get_control().brake
                #     npc_control = carla.VehicleControl(throttle = 0, steer = npc_steer, brake = npc_brake)

        for i in range(self.agent_num):
            self.ego_list[i].apply_control(control[i])
        # self.npc_list[0].apply_control(npc_control)

        # print('ego:%f,%f,%f,npc:%f,%f,%f'%(self.ego_list[0].get_control().throttle,self.ego_list[0].get_control().steer,self.ego_list[0].get_control().brake,
        #                                 self.npc_list[0].get_control().throttle,self.npc_list[0].get_control().steer,self.npc_list[0].get_control().brake))
    

        # 车辆信息反馈
    def get_vehicle_step(self, step_list, step):
        data = list(np.zeros(self.agent_num,dtype=int))
        
        for i in range(self.agent_num):
            path = []
            path_bonus = 0
            if self.args.Start_Path:            
                if step ==0:                
                    self.pathplanner[i] = self.update_route(self.route[i])
                    self.wps_to_go[i] = 0
                    self.path[i] = 0
                    self.f_idx[i] = 0
            # print('index: ', self.ego_f_idx, self.ego_wps_to_go)
                if self.f_idx[i] >= self.wps_to_go[i]:
                    # a = time.time()                
                    path, self.wps_to_go[i], fp = self.get_path(self.ego_list[i],self.pathplanner[i],self.f_idx[i])
                    # print('time: ',time.time()-a)
                    if path!=0:
                        self.path[i] = path
                        for i in range((len(fp.t))):
                            loc = carla.Location(x=self.path[i][0], y=self.path[i][1])
                            self.world.debug.draw_point(location = loc, life_time = 5)
                        self.f_idx[i] = 1
                
            # if self.npc_f_idx >= self.npc_wps_to_go:                
            #     npc_path, self.npc_wps_to_go, npc_fp = self.get_path(npc,self.npc_pathplanner,self.npc_f_idx)
            #     if npc_path != 0:
            #         self.npc_path = npc_path
            #         for i in range((len(npc_fp.t))):
            #             npc_loc = carla.Location(x=self.npc_path[i][0], y=self.npc_path[i][1])
            #             self.world.debug.draw_point(location = npc_loc, life_time = 5)
            #         self.npc_f_idx = 1

        
            location = [self.ego_list[i].get_location().x, self.ego_list[i].get_location().y, math.radians(self.ego_list[i].get_transform().rotation.yaw)]
            # npc_location = [npc.get_location().x, npc.get_location().y, math.radians(npc.get_transform().rotation.yaw)]

            if self.args.Start_Path:
                if path!=0:
                    self.f_idx[i] = misc.closest_wp_idx(location, self.path[i], self.f_idx[i])
                    path_bonus = self.f_idx[i] - self.last_idx[i]
                    self.last_idx[i] = self.f_idx[i]

                # if npc_path != 0:
                #     self.npc_f_idx = misc.closest_wp_idx(npc_location, self.npc_path, self.npc_f_idx)
                #     npc_path_bonus = self.npc_f_idx - self.npc_last_idx
                #     self.npc_last_idx = self.npc_f_idx
                
            route = self.route[i][step_list[i]]
            # npc_route = self.npc_route[npc_step]
            # print('aaaaa: ',self.ego_num,step_list,i)
            next_route = self.route[i][step_list[i] + 1]
            # npc_next_route = self.npc_route[npc_step + 1]

            next_transform = self.ego_list[0].get_transform()
            # npc_next_transform = self.npc_list[0].get_transform()
            obstacle_next_transform = self.obstacle_list[0].get_transform()
            # 速度、加速度
            velocity = self.ego_list[0].get_velocity()
            # npc_velocity = self.npc_list[0].get_velocity()
            yaw = next_transform.rotation.yaw * np.pi/180
            # npc_yaw = npc_next_transform.rotation.yaw * np.pi/180
            acc = self.ego_list[0].get_acceleration()
            # npc_acc = self.npc_list[0].get_acceleration()
            

            f_loc,vec,yaw,acc = misc.inertial_to_frenet(route,next_transform.location.x,next_transform.location.y,velocity.x,velocity.y,yaw,acc.x,acc.y)
            # npc_f_loc,npc_vec,npc_yaw,npc_acc = misc.inertial_to_frenet(npc_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw,npc_acc.x,npc_acc.y)

            if self.args.Start_Path:
                if path != 0:
                    f_path,_,_ = misc.inertial_to_frenet(route,self.path[i][self.f_idx[i]][0],self.path[i][self.f_idx[i]][1],velocity.x,velocity.y,yaw)
                # if npc_path!=0:
                #     npc_f_path,_,_ = misc.inertial_to_frenet(npc_route,self.npc_path[self.npc_f_idx][0],self.npc_path[self.npc_f_idx][1],npc_velocity.x,npc_velocity.y,npc_yaw)
            # opponent_state = list(np.zeros(self.agent_num-1,dtype=int))
            
            # npc_ego_loc,npc_ego_vec,npc_ego_yaw = misc.inertial_to_frenet(npc_route,ego_next_transform.location.x,ego_next_transform.location.y,ego_velocity.x,ego_velocity.y,npc_yaw)

            next_loc,next_vec,next_yaw = misc.inertial_to_frenet(next_route,next_transform.location.x,next_transform.location.y,velocity.x,velocity.y,yaw)
            # npc_next_loc,npc_next_vec,npc_next_yaw = misc.inertial_to_frenet(npc_next_route,npc_next_transform.location.x,npc_next_transform.location.y,npc_velocity.x,npc_velocity.y,npc_yaw)

            ob_loc,_,_ = misc.inertial_to_frenet(route,obstacle_next_transform.location.x,obstacle_next_transform.location.y)
            # npc_ob_loc,_,_ = misc.inertial_to_frenet(npc_route,obstacle_next_transform.location.x,obstacle_next_transform.location.y)

            target_disX = f_loc[0]
            target_disY = f_loc[1]
            # ego_npc_disX = ego_npc_loc[0]
            # ego_npc_disY = ego_npc_loc[1]
            next_disX = next_loc[0]
            next_disY = next_loc[1]

            dis_x = (next_transform.location.x-next_transform.location.x)
            dis_y = (next_transform.location.y-next_transform.location.y)
            dis = np.sqrt(dis_x**2+dis_y**2)
            ob_x = (obstacle_next_transform.location.x-next_transform.location.x)
            ob_y = (obstacle_next_transform.location.y-next_transform.location.y)
            ob = np.sqrt(ob_x**2+ob_y**2)

            ego_BEV_ = self.birdViewProducer.produce(agent_vehicle=self.ego_list[0])
            # npc_BEV_ = self.birdViewProducer.produce(agent_vehicle=self.npc_list[0])
            # ego_BEV = ego_BEV_.swapaxes(0,2).swapaxes(1,2)
            # ego_BEV = npc_BEV_.swapaxes(0,2).swapaxes(1,2)

            ego_rgb = cv.cvtColor(BirdViewProducer.as_rgb(ego_BEV_), cv.COLOR_BGR2RGB)
            # npc_rgb = cv.cvtColor(BirdViewProducer.as_rgb(npc_BEV_), cv.COLOR_BGR2RGB)
            # NOTE imshow requires BGR color model
            # cv.imshow("BirdView RGB", ego_rgb)
            # cv.imwrite(self.directory + '/save_1.png', ego_rgb)

            ego_BEV = ego_rgb.swapaxes(0,2).swapaxes(1,2)
            # npc_BEV = npc_rgb.swapaxes(0,2).swapaxes(1,2)

            # ego_BEV = self.sensor_list[0][1].get_BEV()
            # npc_BEV = self.sensor_list[1][1].get_BEV()

            next_state = [target_disX/5,target_disY/10,next_disX/10,next_disY/10,vec[0]/40,vec[1]/40,next_vec[0]/40,next_vec[1]/40,np.sin(yaw/2),np.sin(next_yaw/2), # 自车10
            ob_loc[0]/30,ob_loc[1]/25] # 障碍2
            # ego_npc_loc[0]/40,ego_npc_loc[1]/10,misc.get_speed(self.npc_list[0])/40,ego_npc_vec[0]/30,ego_npc_vec[1]/30,np.sin(ego_npc_yaw/2) # 外车6
            for j in range(self.args.max_agent_num):
                if j<self.agent_num and j != i:
                    opponent_transform = self.ego_list[j].get_transform()
                    opponent_velocity = self.ego_list[0].get_velocity()
                    opponent_yaw = opponent_transform.rotation.yaw * np.pi/180
                    ego_npc_loc,ego_npc_vec,ego_npc_yaw = misc.inertial_to_frenet(route,opponent_transform.location.x,opponent_transform.location.y,opponent_velocity.x,opponent_velocity.y,opponent_yaw) 
                    next_state.extend([ego_npc_loc[0]/40,ego_npc_loc[1]/10,misc.get_speed(self.ego_list[j])/40,ego_npc_vec[0]/30,ego_npc_vec[1]/30,np.sin(ego_npc_yaw/2)])
                if j>=self.agent_num:
                    next_state.extend([1,1,0,0,0,0])
            next_state = np.array(next_state)
            # npc_target_disX = npc_f_loc[0]
            # npc_target_disY = npc_f_loc[1]
            # npc_ego_disX = npc_ego_loc[0]
            # npc_ego_disY = npc_ego_loc[1]
            # npc_next_disX = npc_next_loc[0]
            # npc_next_disY = npc_next_loc[1]

            # npc_dis_x = (ego_next_transform.location.x-npc_next_transform.location.x)
            # npc_dis_y = (ego_next_transform.location.y-npc_next_transform.location.y)
            # npc_dis = np.sqrt(npc_dis_x**2+npc_dis_y**2)
            # npc_ob_x = (obstacle_next_transform.location.x-npc_next_transform.location.x)
            # npc_ob_y = (obstacle_next_transform.location.y-npc_next_transform.location.y)
            # npc_ob = np.sqrt(npc_ob_x**2+npc_ob_y**2)

            # npc_next_state = np.array([npc_target_disX/5,npc_target_disY/10,npc_next_disX/10,npc_next_disY/10,npc_vec[0]/40,npc_vec[1]/40,npc_next_vec[0]/40,npc_next_vec[1]/40,np.sin(npc_yaw/2),np.sin(npc_next_yaw/2),
            # npc_ego_disX/40,npc_ego_disY/10,misc.get_speed(self.ego_list[0])/40,npc_ego_vec[0]/30,npc_ego_vec[1]/30,np.sin(npc_ego_yaw/2),
            # npc_ob_loc[0]/30,npc_ob_loc[1]/25])
            # print(ego_next_state,npc_next_state)
            # 碰撞、变道检测
            col = self.sensor_list[i].get_collision_history()
            # npc_col = self.sensor_list[1][0].get_collision_history()
            # ego_inv = ego_sensor[1].get_invasion_history()
            # npc_inv = npc_sensor[1].get_invasion_history()
            

            # 回报设置:碰撞惩罚、纵向奖励、最低速度惩罚、存活奖励 
            # ev=-1 if ego_velocity <= 2 else 0
            # nv=-1 if npc_velocity <= 2 else 0
            route_bonus,timeout = 0, 0
            
            if target_disX > -1:
                route_bonus = 1                    
            # if npc_target_disX > -1:
            #     npc_bonus = 1 
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
            if step_list[i] >= self.ego_num[i] - 3:
                col_num = 0
                finish = 1
            elif col[0]==1 or path==0: # ego结束条件ego_done
                col_num = 1
                finish = 0
            else:
                col_num = 0
                finish = 0

            # if npc_step >= self.npc_num - 3:
            #     npccol_num = 0
            #     npc_finish = 1
            # elif npc_col[0]==1 or npc_path==0: # npc结束条件npc_done
            #     npccol_num = 1
            #     npc_finish = 0
            # else:
            #     npccol_num = 0
            #     npc_finish = 0

            #simple reward
            # ego_reward = (-1)*ego_col[0] + (-0.6)*timeout + 1*ego_bonus
            # npc_reward = (-1)*npc_col[0] + (-0.6)*timeout + 1*npc_bonus

            #reward shaping
            reward = ((-40)*col[0] + (0.002)*(dis + ob) 
            + (-5)*(target_disX/5)**2 + (-10)*(target_disY/10)**2 + (-30)*np.abs(np.sin(yaw/2)) 
            + (-2.5)*(next_disX/10)**2 + (-5)*(next_disY/10)**2 + (-15)*np.abs(np.sin(next_yaw/2))
            + 10*route_bonus - 50*timeout + 10*path_bonus
            - 0.25*abs(acc[1]))
            # npc_reward = ((-80)*npc_col[0] + (0.002)*(npc_dis + npc_ob)
            # + (-5)*(npc_target_disX/5)**2 + (-10)*(npc_target_disY/10)**2 + (-30)*np.abs(np.sin(npc_yaw/2))
            # + (-2.5)*(npc_next_disX/10)**2 + (-5)*(npc_next_disY/10)**2 + (-15)*np.abs(np.sin(npc_next_yaw/2)) 
            # + 50*npc_bonus - 50*timeout + 10*npc_path_bonus
            # - 0.25*abs(npc_acc[1]))
            data[i] = [next_state,reward,col_num,finish,ego_BEV]
        return data

    # 车辆动作空间
    def get_action_space(self):
        if self.args.direct_control:
            action_space = [0,0] # throttle-brake/steer
        else:
            action_space = [0,0,0] # x,y,speed
        return action_space
    
    # 车辆状态空间
    def get_state_space(self):
        state_space = list(np.zeros(6*(self.args.max_agent_num+1),dtype=int))
        return state_space

    def clean(self):
        # 清洗环境
        print('Start Cleaning Envs')
        for x in self.sensor_list:
            if x.sensor.is_alive:
                x.sensor.destroy()
        for x in self.ego_list:
            # if x.is_alive:
            self.client.apply_batch([carla.command.DestroyActor(x)])
        # for x in self.npc_list:
        #     # if x.is_alive:
        #     self.client.apply_batch([carla.command.DestroyActor(x)])
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
        for x in self.sensor_list:
            if x.sensor.is_alive:
                x.sensor.destroy()
        for x in self.ego_list:
            # if x.is_alive:
            self.client.apply_batch([carla.command.DestroyActor(x)])
        # for x in self.npc_list:
        #     # if x.is_alive:
        #     self.client.apply_batch([carla.command.DestroyActor(x)])
        for x in self.obstacle_list:
            # if x.is_alive:
            self.client.apply_batch([carla.command.DestroyActor(x)])
        print('all clean!')
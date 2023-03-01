#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" Module with auxiliary functions. """

import math
import numpy as np
import sys
import glob
import os
import psutil
import carla


def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    for wpt in waypoints:
        # wpt_t = wpt
        begin = wpt.location + carla.Location(z=z)
        angle = math.radians(wpt.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, color = carla.Color(0,255,0), life_time=0)

def get_speed(vehicle):
    """
    Compute speed of a vehicle in m/s.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in m/s
    """
    vel = vehicle.get_velocity()

    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_transform: location of the target object
    :param current_transform: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0

def is_within_distance(target_location, current_location, orientation, max_distance, d_angle_th_up, d_angle_th_low=0):
    """
    Check if a target object is within a certain distance from a reference object.
    A vehicle in front would be something around 0 deg, while one behind around 180 deg.

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :param d_angle_th_up: upper thereshold for angle
        :param d_angle_th_low: low thereshold for angle (optional, default is 0)
        :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle_th_low < d_angle < d_angle_th_up


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    """
    Returns the 2D distance from a waypoint to a vehicle

        :param waypoint: actual waypoint
        :param vehicle_transform: transform of the target vehicle
    """
    loc = vehicle_transform.location
    x = waypoint.transform.location.x - loc.x
    y = waypoint.transform.location.y - loc.y

    return math.sqrt(x * x + y * y)


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]

def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm


def positive(num):
    """
    Return the given number if positive, else 0

        :param num: value to check
    """
    return num if num > 0.0 else 0.0

# 大地转自车坐标系
def inertial_to_SDV(veh,x=None,y=None,vx=None,vy=None,yaw=None,accx=None, accy=None):
    """
    Transform a point from the global coordinate system to the local one

        :param X,Y,VX,VY,YAW: global coordinates
    """
    veh_trans = veh.get_transform()
    X = veh_trans.location.x
    Y = veh_trans.location.y
    YAW = veh_trans.rotation.yaw * np.pi/180
    VX = veh.get_velocity().x
    VY = veh.get_velocity().y
    AX = veh.get_acceleration().x
    AY = veh.get_acceleration().y

    # Compute the rotation matrix
    R = np.array([[math.cos(YAW), math.sin(YAW)], [-math.sin(YAW), math.cos(YAW)]])

    # Compute the transformation
    T = np.array([x-X, y-Y]) if x is not None else None

    S = np.array([vx-VX, vy-VY]) if vx is not None else None

    A = np.array([accx-AX, accy-AY]) if accx is not None else None
    
    # Compute the transformed point
    loc = np.dot(R, T) if x is not None else None
    vec = np.dot(R, S) if vx is not None else None
    rot = np.array(yaw-YAW) if yaw is not None else None
    acc = np.dot(R, A) if accx is not None else None
    if accx == None:
        return loc, vec, rot
    else:
        return loc, vec, rot, acc

def SDV_to_inertial(veh,x=None,y=None,vx=None,vy=None,yaw=None):
    """
    Transform a point from the local coordinate system to the global one

        :param X,Y,VX,VY,YAW: local coordinates
    """
    veh_trans = veh.get_transform()
    X = veh_trans.location.x
    Y = veh_trans.location.y
    YAW = veh_trans.rotation.yaw * np.pi/180
    VX = veh.get_velocity().x
    VY = veh.get_velocity().y
    
    # Compute the rotation matrix
    R_reverse = np.array([[math.cos(YAW), -math.sin(YAW)], [math.sin(YAW), math.cos(YAW)]])
    
    # Compute the transformation
    T = np.array([x, y])
    
    S = np.array([vx, vy])
    
    # Compute the transformed point
    loc = np.dot(R_reverse, T) + np.array([X,Y])
    vec = np.dot(R_reverse, S) + np.array([VX,VY])
    rot = np.array(yaw+YAW)
    return loc, vec, rot

def inertial_to_frenet(route,x=None,y=None,vx=None,vy=None,yaw=None,accx=None,accy=None):
    X = route.location.x
    Y = route.location.y
    YAW = route.rotation.yaw * np.pi/180

    # Compute the rotation matrix
    R = np.array([[math.cos(YAW), math.sin(YAW)], [-math.sin(YAW), math.cos(YAW)]])

    # Compute the transformation

    T = np.array([x-X, y-Y]) if x is not None else None

    S = np.array([vx, vy])

    A = np.array([accx, accy])
    
    # Compute the transformed point
    loc = np.dot(R, T) if x is not None else None
    vec = np.dot(R, S) if vx is not None else None
    rot = np.array(yaw-YAW) if yaw is not None else None
    acc = np.dot(R, A) if accx is not None else None

    if accx == None:
        return loc, vec, rot
    else:
        return loc, vec, rot, acc

def frenet_to_inertial(route,s=None,d=None,vs=None,vd=None,rot=None,acc=None):
    X = route.location.x
    Y = route.location.y
    YAW = route.rotation.yaw * np.pi/180

    # Compute the rotation matrix
    R_reverse = np.array([[math.cos(YAW), -math.sin(YAW)], [math.sin(YAW), math.cos(YAW)]])

    # Compute the transformation

    T = np.array([s, d])

    S = np.array([vs, vd])
    
    # Compute the transformed point
    loc = np.dot(R_reverse, T) + np.array([X,Y])
    vec = np.dot(R_reverse, S)
    rot = np.array(rot+YAW)
    
    return loc, vec, rot

def judgeprocess(processname):
    pl = psutil.pids()
    for pid in pl:
        if psutil.Process(pid).name() == processname:
            return True
    else:
        return False

def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])  # inertial frame
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb

def closest_wp_idx(ego_state, fpath, f_idx, w_size=10):
    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(fpath) - 2 - f_idx else len(fpath) - 2 - f_idx
    for i in range(w_size):
        temp_wp = fpath[f_idx + i]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist \
                and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[2])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist
    return f_idx + closest_wp_index


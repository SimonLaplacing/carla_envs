# import argparse
# import collections
# from curses import raw
# import datetime
import glob
# import logging
# import math
import os
# import random
# import re
import sys
import weakref
import numpy as np

try:
    sys.path.append(glob.glob('D:/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from carla import Transform, Location, Rotation



# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        # self.sensor = None
        self.history = [0]
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        return self.history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        self.history[0] = 1

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        # self.sensor = None
        self.history = [0]
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    def get_invasion_history(self):
        return self.history
    
    def reset(self):
        self.history = [0]
        
    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        self.history[0] = 1

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class Camera(object):
    """ Class for camera sensors"""

    def __init__(self, parent_actor, name, directory, H, W):
        """Constructor method"""
        # self.sensor = None
        self._parent = parent_actor
        self.recording = True
        self.directory = directory
        self.name = name
        self.BEV = np.zeros((3, W, H), dtype=np.uint8)
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.camera.semantic_segmentation') #semantic_segmentation
        blueprint.set_attribute('image_size_x', str(W))
        blueprint.set_attribute('image_size_y', str(H))
        blueprint.set_attribute('fov', '90')
        # Set the time in seconds between sensor captures
        blueprint.set_attribute('sensor_tick', '0')
        transform = Transform(Location(z=30), Rotation(pitch=-90))
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: Camera._parse_image(weak_self, image))

    def get_BEV(self):
        return self.BEV

    @staticmethod
    def _parse_image(weak_self, image):
        """On camera method"""
        # raw_data = image.raw_data
        self = weak_self()
        if not self:
            return
        
        image.convert(cc.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8")) # BGRA_Binary TO BGRA_Seq
        array = np.reshape(array, (image.height, image.width, 4)) # BGRA_Seq TO BGRA_Array
        array = array[:, :, :3] # BGRA_Array TO BGR_Array
        array = array[:, :, ::-1] # BGR_Array TO RGB_Array
        array = array.swapaxes(0, 2) # [H, W, 3] TO [3, W, H]
        array = array.swapaxes(1, 2) # [3, W, H] TO [3, H, W]
        self.BEV = array
        # print('aaa: ',array.shape)
        if self.recording:
            image.save_to_disk(self.directory + self.name + '_image_%06d' % image.frame, cc.CityScapesPalette)
           

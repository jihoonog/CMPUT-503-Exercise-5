#!/usr/bin/env python3

"""
This is the number detection node for exercise 5
"""

import numpy as np
import os
import math
import rospy
import time
import message_filters
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from statistics import mode
from lane_controller import LaneController

import rospkg

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import (
    Pose2DStamped, 
    LanePose, 
    WheelEncoderStamped, 
    WheelsCmdStamped, 
    Twist2DStamped,
    BoolStamped,
    VehicleCorners,
    SegmentList,
    LEDPattern,
    )
from duckietown_msgs.srv import SetCustomLEDPattern
from std_msgs.msg import Header, Float32, String, Float64MultiArray, Float32MultiArray, Int32
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point32

import rosbag


# Change this before executing
VERBOSE = 0
SIM = False

class MLP(nn.Module):
    """
    From the Multilayer Perception (MLP) tutorial notebook
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        # x = [batch size, height * width]

        h_1 = F.relu(self.input_fc(x))
        h_1 = F.dropout(h_1, p=0.1)
        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))
        h_2 = F.dropout(h_2, p=0.1)
        # h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return y_pred, h_2


class NumberDetectionNode(DTROS):
    """
    The Number Detection Node will subscribe to the camera and use a ML model to determine the number from an image with an AprilTag. 
    """
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(NumberDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.DRIVER)
        if os.environ["VEHICLE_NAME"] is not None:
            self.veh_name = os.environ["VEHICLE_NAME"]
        else:
            self.veh_name = "csc22945"

        # Static parameters
        self.update_freq = 10
        self.rate = rospy.Rate(self.update_freq)
        self.rospack = rospkg.RosPack()

        # Publishers
        ## Publish commands to the motors
        self.pub_motor_commands = rospy.Publisher(f'/{self.veh_name}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.pub_car_cmd = rospy.Publisher(f'/{self.veh_name}/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL)
        
        ## LED emitter service
        service_name = f'/{self.veh_name}/led_emitter_node/set_custom_pattern'
        rospy.wait_for_service(service_name)
        self.LED_emitter_service = rospy.ServiceProxy(service_name, SetCustomLEDPattern, persistent=True)

        # Subscribers
        ## Subscribe to the lane_pose node
        self.sub_lane_reading = rospy.Subscriber(f"/{self.veh_name}/lane_filter_node/lane_pose", LanePose, self.cb_lane_pose, queue_size = 1)
        self.sub_segment_list = rospy.Subscriber(f"/{self.veh_name}/line_detector_node/segment_list", SegmentList, self.cb_segments, queue_size=1)
        self.sub_distance_to_robot_ahead = rospy.Subscriber(f"/{self.veh_name}/duckiebot_distance_node/distance", Float32, self.cb_vehicle_distance, queue_size=1)
        self.sub_centers = rospy.Subscriber(f"/{self.veh_name}/duckiebot_detection_node/centers", VehicleCorners, self.cb_vehicle_centers, queue_size=1)
        self.sub_circle_pattern_image = rospy.Subscriber(f"/{self.veh_name}/duckiebot_detection_node/detection_image/compressed", CompressedImage, queue_size=1)
        self.sub_detection = rospy.Subscriber(f"/{self.veh_name}/duckiebot_detection_node/detection", BoolStamped, self.cb_detection, queue_size=1)
        self.sub_tag_id = rospy.Subscriber(f"/{self.veh_name}/tag_id", Int32, self.cb_tag_id, queue_size=1)
        
        self.log("Initialized")

    def load_model(self):
        model_file_folder = self.rospack.get_path('number_detection') + '/config/tut1-model.pt'

    def load_intrinsics(self):
        # Find the intrinsic calibration parameters
        # cali_file_folder = '/data/config/calibrations/camera_intrinsic/'
        # self.frame_id = self.veh + '/camera_optical_frame'
        # self.cali_file = cali_file_folder + self.veh + ".yaml"

        self.cali_file = self.rospack.get_path('duckiebot_detection') + f"/config/calibrations/camera_intrinsic/{self.veh}.yaml"

        # Locate calibration yaml file or use the default otherwise
        rospy.loginfo(f'Looking for calibration {self.cali_file}')
        if not os.path.isfile(self.cali_file):
            self.logwarn("Calibration not found: %s.\n Using default instead." % self.cali_file)
            self.cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(self.cali_file):
            rospy.signal_shutdown("Found no calibration file ... aborting")

        # Load the calibration file
        calib_data = self.readYamlFile2(self.cali_file)
        self.log("Using calibration file: %s" % self.cali_file)

        return calib_data
    
    def get_extrinsic_filepath(self,name):
        #TODO: retrieve the calibration info from the right path.
        cali_file_folder = self.rospack.get_path('duckiebot_detection')+'/config/calibrations/camera_extrinsic/'

        cali_file = cali_file_folder + name + ".yaml"
        return cali_file

if __name__ == '__main__':
    node = NumberDetectionNode(node_name='robot_follower_node')
    # Keep it spinning to keep the node alive
    # main loop
    rospy.spin()
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
import cv2
import yaml
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from statistics import mode
from lane_controller import LaneController

import rospkg
from augmented_reality_basics import Augmenter
from duckietown_utils import load_homography, load_map
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
from sensor_msgs.msg import CompressedImage, CameraInfo
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

        self.model = MLP(28*28, 10)
    
        self.rospack = rospkg.RosPack()
        self.read_params_from_calibration_file()
        self.camera_info_dict = self.load_intrinsics()

        # extract parameters from camera_info_dict for apriltag detection
        f_x = self.camera_info_dict['camera_matrix']['data'][0]
        f_y = self.camera_info_dict['camera_matrix']['data'][4]
        c_x = self.camera_info_dict['camera_matrix']['data'][2]
        c_y = self.camera_info_dict['camera_matrix']['data'][5]
        self.camera_params = [f_x, f_y, c_x, c_y]
        K_list = self.camera_info_dict['camera_matrix']['data']
        self.K = np.array(K_list).reshape((3, 3))

        self.augmenter = Augmenter(self.homography, self.camera_info_msg)

        # Static parameters
        self.update_freq = 10
        self.rate = rospy.Rate(self.update_freq)

        # Publishers
        self.pub_number_bb = rospy.Publisher(f"/{self.veh_name}/number_detection_node/image/compressed", CompressedImage, queue_size=1)
        self.pub_cropped_number = rospy.Publisher(f"/{self.veh_name}/number_detection_node/cropped_number/image/compressed", CompressedImage, queue_size=1)
        # Subscribers
        ## Subscribe to the lane_pose node
        self.sub_images = rospy.Subscriber(f"/{self.veh_name}/camera_node/image/compressed", CompressedImage, self.cb_image, queue_size=1)
        
        self.load_model()

        self.log("Initialized")

    def cb_image(self, msg):
        br = CvBridge()
        # Convert image to cv2
        raw_image = br.compressed_imgmsg_to_cv2(msg)
        # processed_image = self.augmenter.process_image(raw_image)

        rangomax = np.array([255,175,50]) # B,G,R
        rangomin = np.array([60,60,0])
        mask = cv2.inRange(raw_image, rangomin, rangomax)
        # reduce the noise
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8), iterations = 2)
        x,y,w,h = cv2.boundingRect(opening)

        cv2.rectangle(raw_image, (x,y), (x + w, y + h), (0,255,0), 2)

        number = raw_image[y:y+h, x:x+w]
        if number is None or mask is None:
            return
        black_max = np.array([100,100,100])
        black_min = np.array([0,0,0])
        number_mask = cv2.inRange(number, black_min, black_max)
        number_mask = cv2.resize(number_mask, (28,28))
        self.pub_processed_image(raw_image, self.pub_number_bb)
        self.pub_processed_image(number_mask, self.pub_cropped_number)

        input_tensor = torch.from_numpy(number_mask).float()
        input_tensor = input_tensor[None, :]
        res_vector, _ = self.model(input_tensor)
        print(res_vector.argmax(1, keepdim=True).item())
        self.rate.sleep()

    def pub_processed_image(self, image, publisher):
        compressed_image = CompressedImage()
        compressed_image.header.stamp = rospy.Time.now()
        compressed_image.format = "jpeg"
        compressed_image.data = np.array(cv2.imencode('.jpg',image)[1]).tostring()

        publisher.publish(compressed_image)

    def read_params_from_calibration_file(self):
        # Get static parameters
        file_name_ex = self.get_extrinsic_filepath(self.veh_name)
        self.homography = self.readYamlFile(file_name_ex)
        self.camera_info_msg = rospy.wait_for_message(f'/{self.veh_name}/camera_node/camera_info', CameraInfo)


    def load_model(self):
        model_file_folder = self.rospack.get_path('number_detection') + '/config/MNIST_model.pt'
        self.model.load_state_dict(torch.load(model_file_folder, map_location=torch.device('cpu')))
        self.model.eval()

    def load_intrinsics(self):
        # Find the intrinsic calibration parameters
        # cali_file_folder = '/data/config/calibrations/camera_intrinsic/'
        # self.frame_id = self.veh_name + '/camera_optical_frame'
        # self.cali_file = cali_file_folder + self.veh_name + ".yaml"

        self.cali_file = self.rospack.get_path('duckiebot_detection') + f"/config/calibrations/camera_intrinsic/{self.veh_name}.yaml"

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

    def readYamlFile(self,fname):
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)["homography"]
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    def readYamlFile2(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file, Loader=yaml.Loader)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown('No calibration file found.')
                return

if __name__ == '__main__':
    node = NumberDetectionNode(node_name='robot_follower_node')
    # Keep it spinning to keep the node alive
    # main loop
    rospy.spin()

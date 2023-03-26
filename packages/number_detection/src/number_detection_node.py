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
from CNN import Net

# Change this before executing
VERBOSE = 0
SIM = False


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
            self.veh_name = "csc22935"

        #self.model = MLP(28*28, 10)
        self.model = Net()
        self.new_model = Net()
    
        self.rospack = rospkg.RosPack()
        self.read_params_from_calibration_file()
        self.camera_info_dict = self.load_intrinsics()
        self.figure_list = []
        self.figure_decision_queue = []

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
        self.sub_apriltag_id = rospy.Subscriber(f'/{self.veh_name}/tag_id', Int32, self.detect_apriltag_existance,queue_size=1)
        self.apriltag_exist = -1
        self.tag_figure_dict = {}
        self.tag_history_list = {}
        self.load_model()
                # Publishers
        ## Publish commands to the motors
        self.pub_motor_commands = rospy.Publisher(f'/{self.veh_name}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.pub_shutdown_commands = rospy.Publisher(f'/{self.veh_name}/number_detection_node/shutdown_cmd', String, queue_size=1)
        #self.pub_motor_commands = rospy.Publisher(f'/state_control_node/command', String, queue_size=1)
        self.pub_car_cmd = rospy.Publisher(f'/{self.veh_name}/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL)

        self.log("Initialized")
        self.apriltag_locations = {200:(0.17,0.17), 201:(1.65, 0.17), 
                                    94:(1.65, 2.84),93:(0.17, 2.84),153:(1.75, 1.252), 
                                    133:(1.253, 1.755),58:(0.574, 1.259),62: (0.075, 1.755),
                                    169:(0.574, 1.755), 162:(1.253, 1.253)}

    def most_common(self,lst):
        return max(set(lst), key=lst.count)

    def detect_apriltag_existance(self, msg):
        #print("msg",msg.data)
        data = msg.data
        if data != -1:
            self.apriltag_exist = msg.data
            if data not in list(self.tag_figure_dict.keys()):
                self.tag_figure_dict[data] = -1
        else:
            self.apriltag_exist = -1
        
        #self.rate.sleep()

    def shutdown(self):
        motor_cmd = WheelsCmdStamped()
        motor_cmd.header.stamp = rospy.Time.now()
        motor_cmd.vel_left = 0.0
        motor_cmd.vel_right = 0.0
        self.pub_motor_commands.publish(motor_cmd)
        car_control_msg = Twist2DStamped()
        car_control_msg.header.stamp = rospy.Time.now()
        car_control_msg.v - 0.0
        car_control_msg.omega = 0.0
        self.pub_car_cmd.publish(car_control_msg)

    def cb_image(self, msg):
        #print("in cb_image")
        br = CvBridge()
        # Convert image to cv2
        raw_image = br.compressed_imgmsg_to_cv2(msg)
        copy_raw = raw_image
        #raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        
        # processed_image = self.augmenter.process_image(raw_image)

        if len(self.figure_decision_queue) == 10:
            print(self.figure_decision_queue)
            self.pub_shutdown_commands.publish("shutdown")



            time.sleep(2)
            rospy.signal_shutdown("number_detection Node Shutdown command received")

            #self.shutdown()
            #time.sleep(2)
            #exit()
        

            

        
        #raw_image = br.compressed_imgmsg_to_cv2(msg)
        # processed_image = self.augmenter.process_image(raw_image)

        rangomax = np.array([255,175,50]) # B,G,R
        rangomin = np.array([60,60,0])
        mask = cv2.inRange(raw_image, rangomin, rangomax)
        # reduce the noise
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8), iterations = 2)
        x,y,w,h = cv2.boundingRect(opening)

        cv2.rectangle(raw_image, (x,y), (x + w, y + h), (0,255,0), 2)

        number = raw_image[y:y+h, x:x+w]
        if number is None or mask is None or w<70 or h < 70 or w>300 or h>300:
            self.pub_processed_image(copy_raw, self.pub_number_bb)
            return
        black_max = np.array([80,80,80])
        #black_max = np.array([100,100,100])
        black_min = np.array([0,0,0])
        number_mask = cv2.inRange(number, black_min, black_max)
        number_mask = cv2.resize(number_mask, (28,28))
        self.pub_processed_image(raw_image, self.pub_number_bb)
        self.pub_processed_image(number_mask, self.pub_cropped_number)


        input_tensor = torch.from_numpy(number_mask).float()
        input_tensor = input_tensor[None, :]
        #print("input_tensor",input_tensor.shape)
        #print("input_tensor",input_tensor)
        res_vector = self.model(input_tensor)
        #print("res_vector",res_vector)
        figure_decision = res_vector.argmax(1, keepdim=True).item()

        # The old model has some confusion in detecting 7 and 5, so we use a new model to do that.
        if figure_decision in [1,3,5,7]:
            figure_decision = self.new_model(input_tensor).argmax(1, keepdim=True).item()

        if self.apriltag_exist != -1:
            if self.apriltag_exist in list(self.tag_history_list.keys()):
                self.tag_history_list[self.apriltag_exist].append(figure_decision)
                frequent_detection = self.most_common(self.tag_history_list[self.apriltag_exist])
                self.tag_figure_dict[self.apriltag_exist] = frequent_detection

            else:
                self.tag_history_list[self.apriltag_exist] = [figure_decision]
                self.tag_figure_dict[self.apriltag_exist] = figure_decision

            if self.tag_figure_dict[self.apriltag_exist] not in self.figure_decision_queue:
                self.figure_decision_queue.append(self.tag_figure_dict[self.apriltag_exist])
            
            print(f"detection: tag {self.apriltag_exist}, figure {self.tag_figure_dict[self.apriltag_exist]} location:{self.apriltag_locations[self.apriltag_exist]}")
            print("current visited numbers",self.figure_decision_queue)
        # self.pub_processed_image(number_mask, self.pub_cropped_number)

        # self.pub_processed_image(raw_image, self.pub_number_bb)
        
            

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
        #model_file_folder = self.rospack.get_path('number_detection') + '/config/MNIST_model.pt'
        #model_file_folder = self.rospack.get_path('number_detection') + '/config/mnist_cnn0.pt'
        
        model_file_folder = self.rospack.get_path('number_detection') + '/config/mnist_cnn0.pt'

        self.model.load_state_dict(torch.load(model_file_folder))
        #self.model.load_state_dict(torch.load(model_file_folder, map_location=torch.device('cpu')))
        self.model.eval()

        model_file_folder = self.rospack.get_path('number_detection') + '/config/mnist_cnn_add_data_aug_5.pt'

        self.new_model.load_state_dict(torch.load(model_file_folder))
        #self.model.load_state_dict(torch.load(model_file_folder, map_location=torch.device('cpu')))
        self.new_model.eval()

    def load_intrinsics(self):
        # Find the intrinsic calibration parameters
        # cali_file_folder = '/data/config/calibrations/camera_intrinsic/'
        # self.frame_id = self.veh_name + '/camera_optical_frame'
        # self.cali_file = cali_file_folder + self.veh_name + ".yaml"

        self.cali_file = self.rospack.get_path('number_detection') + f"/config/calibrations/camera_intrinsic/{self.veh_name}.yaml"

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
        cali_file_folder = self.rospack.get_path('number_detection')+'/config/calibrations/camera_extrinsic/'

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

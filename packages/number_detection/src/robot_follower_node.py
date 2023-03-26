#!/usr/bin/env python3

"""
This is the robot follower node for exercise 4 
based on the lane controller node from dt-core here: https://github.com/duckietown/dt-core/blob/daffy/packages/lane_control/src/lane_controller_node.py
the stop lane filter code is from here https://raw.githubusercontent.com/duckietown/dt-core/daffy/packages/stop_line_filter/src/stop_line_filter_node.py

"""

import numpy as np
import os
import math
import rospy
import time
import message_filters
import typing
from statistics import mode
from lane_controller import LaneController

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


class RobotFollowerNode(DTROS):
    """
    Robot Follower Node is used to generate robot following commands based on the lane pose and the leader robot.
    """
    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """
        # Initialize the DTROS parent class
        super(RobotFollowerNode, self).__init__(node_name=node_name, node_type=NodeType.DRIVER)
        if os.environ["VEHICLE_NAME"] is not None:
            self.veh_name = os.environ["VEHICLE_NAME"]
        else:
            self.veh_name = "csc22945"

        # Static parameters
        self.update_freq = 10
        self.rate = rospy.Rate(self.update_freq)
        self.d_offset = 0.0
        self.lane_controller_parameters = {
            "Kp_d": 7.0,
            "Ki_d": 0.00,
            "Kd_d": 0.125,
            "Kp_theta": 6.5,
            "Ki_theta": 0.0,
            "Kd_theta": 0.125,
            "sample_time": 1.0 / self.update_freq,
            "d_bounds": (-2.0, 2.0),
            "theta_bounds": (-3.0,3.0),
        }
        ## for stop line detection
        self.stop_distance = 0.15 # distance from the stop line that we should stop
        self.min_segs = 20  # minimum number of red segments that we should detect to estimate a stop
        self.off_time = 2.0 # time to wait after we have passed the stop line
        self.max_y = 0.12   # If y value of detected red line is smaller than max_y we will not set at_stop_line true.
        self.stop_hist_len = 7
        self.stop_duration = 1.5
        self.stop_cooldown = 5 # The stop cooldown
        ## Vehicle detection
        self.safe_distance = 0.4
        self.camera_width = 640 # From the docs
        self.direction_threshold = 0.4 # The threshold for the direction of the vehicle as a fraction of the camera width
        self.direction_hist_len = 15

        # Initialize variables
        self.lane_pose = LanePose()
        self.lane_pid_controller = LaneController(self.lane_controller_parameters)
        ## For stop line detection
        self.stop_line_distance = None
        self.stop_line_detected = False
        self.at_stop_line = False
        self.cmd_stop = False
        self.stop_hist = []
        self.stop_time = 0.0
        self.process_intersection = False
        self.tag_id = -1
        ## For Vehicle detection
        self.vehicle_detected = False
        self.vehicle_centers = VehicleCorners()
        self.vehicle_x_mean = 0.0
        self.vehicle_direction = 0 # -1 left, 0 straight, 1 right
        self.direction_hist = [] # Stores the last direction of the vehicle
        self.current_led_pattern = "off"

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
        self.sub_shutdown_commands = rospy.Subscriber(f'/{self.veh_name}/number_detection_node/shutdown_cmd', String, self.shutdown, queue_size = 1)

        self.log("Initialized")

    # Start of callback functions
    def cb_segments(self, segment_list_msg):
        good_seg_count = 0
        stop_line_x_accumulator = 0.0
        stop_line_y_accumulator = 0.0
        for segment in segment_list_msg.segments:
            if segment.color != segment.RED:
                continue
            if segment.points[0].x < 0 or segment.points[1].x < 0: # The point is behind the robot
                continue
            p1_lane = self.to_lane_frame(segment.points[0])
            p2_lane = self.to_lane_frame(segment.points[1])
            avg_x = 0.5 * (p1_lane[0] + p2_lane[0])
            avg_y = 0.5 * (p1_lane[1] + p2_lane[1])
            stop_line_x_accumulator += avg_x
            stop_line_y_accumulator += avg_y
            good_seg_count += 1
        
        if good_seg_count < self.min_segs:
            self.stop_line_detected = False
            at_stop_line = False
            self.stop_line_distance = 99.9
        else:
            self.stop_line_detected = True
            stop_line_point_x = stop_line_x_accumulator / good_seg_count
            stop_line_point_y = stop_line_y_accumulator / good_seg_count
            self.stop_line_distance = np.sqrt(stop_line_point_x**2 + stop_line_point_y**2)
            # Only detect stop line if y is within max_y distance
            at_stop_line = (
                stop_line_point_x < self.stop_distance and np.abs(stop_line_point_y) < self.max_y
            )
        
        self.process_stop_line(at_stop_line)

    def cb_lane_pose(self, input_pose_msg):
        self.lane_pose = input_pose_msg
        self.get_control_action(self.lane_pose)

    def cb_vehicle_centers(self, centers_msg):
        self.vehicle_centers = centers_msg
        centers = self.vehicle_centers.corners
        
        # Only process lateral movement x
        x_total = 0.0
        if len(centers) > 0:
            for center in centers:
                x_total += center.x
            self.vehicle_x_mean = x_total /  len(centers)
            left_threshold = self.camera_width * self.direction_threshold
            right_threshold = self.camera_width * (1 - self.direction_threshold)
            if self.vehicle_x_mean < left_threshold:
                curr_vehicle_direction = int(-1)
            elif self.vehicle_x_mean > right_threshold:
                curr_vehicle_direction = int(1)
            else:
                curr_vehicle_direction = int(0)

            self.vehicle_direction = self.process_vehicle_direction_reading(curr_vehicle_direction)
            # print("Leader vehicle x mean: ", self.vehicle_x_mean, "Direction: ", self.vehicle_direction)
        else:
            self.vehicle_direction = self.process_vehicle_direction_reading(None)
            self.vehicle_x_mean = self.camera_width / 2
            

    def cb_detection(self, detection_msg):
        self.vehicle_detected = detection_msg.data

    def cb_vehicle_distance(self, distance_msg):
        self.vehicle_distance = distance_msg.data
    
    def cb_tag_id(self, tag_msg):
        if tag_msg.data != -1:
            self.tag_id = tag_msg.data
        
    def veh_leader_info(self):
        if self.vehicle_detected:
            print("Vehicle detected - distance:", self.vehicle_distance)
            print("Vehicle Direction: ", self.vehicle_direction)
        else:
            print("No vehicle detected")

    def process_stop_line(self, at_stop_line):
        """Storing the current distance to the next stop line, if one is detected.

        Args:
            msg (StopLineReading): The message containing the distance to the next stop line.
        """
        if len(self.stop_hist) > self.stop_hist_len:
            self.stop_hist.pop(0)

        if at_stop_line:
            self.stop_hist.append(True)
        else:
            self.stop_hist.append(False)

        if mode(self.stop_hist) == True:
            self.cmd_stop = True
        else:
            self.cmd_stop = False
    
    def process_vehicle_direction_reading(self, direction):
        """Store the current direction of the vehicle ahead. None if no vehicle is detected."""

        if len(self.direction_hist) > self.direction_hist_len:
            self.direction_hist.pop(0)
        
        self.direction_hist.append(direction)

        predicted_direction = mode(self.direction_hist)
        return predicted_direction

    def set_tail_lights(self, state_str):
        """Set the tail lights to the given state.

        Args:
            state (str): The state to have the tail lights:
            Can be, left, right, stop, or off
        """
        state = state_str.lower()
        if self.current_led_pattern == state:
            return
        self.current_led_pattern = state

        msg = LEDPattern()
        if state == "left":
            msg.color_list = ['switchedoff','switchedoff','switchedoff','switchedoff','yellow']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 2
            msg.frequency_mask = [0, 0, 0, 0, 1]
        elif state == "right":
            msg.color_list = ['switchedoff','switchedoff','switchedoff','yellow','switchedoff']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 2
            msg.frequency_mask = [0, 0, 0, 1, 0]
        elif state == "stop":
            msg.color_list = ['switchedoff','switchedoff','switchedoff','red','red']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 0
            msg.frequency_mask = [0, 0, 0, 0, 0]
        elif state == "off":
            msg.color_list = ['switchedoff','switchedoff','switchedoff','switchedoff','switchedoff']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 0
            msg.frequency_mask = [0, 0, 0, 0, 0]
        elif state == "white":
            msg.color_list = ['switchedoff','switchedoff','switchedoff','white','white']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 0
            msg.frequency_mask = [0, 0, 0, 0, 0]
        else:
            msg.color_list = ['switchedoff','switchedoff','switchedoff','switchedoff','switchedoff']
            msg.color_mask = [0, 0, 0, 0, 0]
            msg.frequency = 0
            msg.frequency_mask = [0, 0, 0, 0, 0]
            print("Invalid tail light state")
        
        self.LED_emitter_service(msg)



    def get_control_action(self, pose_msg):
        """
        Callback function that receives a pose message and updates the related control command
        """
        d_err = pose_msg.d - self.d_offset
        phi_err = pose_msg.phi

        # self.veh_leader_info()

        curr_time = rospy.get_time()

        stop_time_diff = curr_time - self.stop_time

        if (self.cmd_stop and stop_time_diff > self.stop_cooldown):
            self.stop_time = curr_time
            v = 0.0
            omega = 0.0

            self.process_intersection = True
            self.set_tail_lights("stop")
            self.car_cmd(v, omega, pose_msg)
            rospy.sleep(self.stop_duration)
        elif self.vehicle_ahead():
            v = 0.0
            omega = 0.0
            self.car_cmd(v, omega, pose_msg)
            self.set_tail_lights("stop")
        elif self.process_intersection:

            if self.vehicle_direction == 0:
                self.set_tail_lights("off")
                self.go_straight(pose_msg)
            elif self.vehicle_direction == 1:
                self.set_tail_lights("right")
                self.turn_right(pose_msg)
            elif self.vehicle_direction == -1:
                self.set_tail_lights("left")
                self.turn_left(pose_msg)
            else:
                # If None should drive autonomously
                # If it detects a stop sign then it will turn right onto the outer lane
                if self.tag_id in [169, 162]:
                    self.set_tail_lights("left")
                    self.turn_left(pose_msg)
                else:
                    self.set_tail_lights("white")
                    self.go_straight(pose_msg)
            
            self.process_intersection = False
            
        else:
            v, omega = self.lane_pid_controller.compute_control_actions(d_err, phi_err, None)
            self.set_tail_lights("white")
            self.car_cmd(v, omega, pose_msg)

        self.rate.sleep()

    def car_cmd(self, v, omega, lane_pose):
        car_control_msg = Twist2DStamped()
        car_control_msg.header = lane_pose.header

        car_control_msg.v = v
        car_control_msg.omega = omega * 2.05
        self.pub_car_cmd.publish(car_control_msg)
    
    def vehicle_ahead(self):
        if self.vehicle_detected and self.vehicle_distance < self.safe_distance:
            return True
        else:
            return False

    def turn_right(self, pose_msg):
        """Make a right turn at an intersection"""
        self.lane_pid_controller.disable_controller()
        
        self.car_cmd(v=0.4, omega=0, lane_pose=pose_msg)
        rospy.sleep(1.5)
        self.car_cmd(v=0.4, omega=-4, lane_pose=pose_msg)
        rospy.sleep(1.0)
        self.stop_hist = []
        self.cmd_stop = False
        self.lane_pid_controller.enable_controller()

    def turn_left(self, pose_msg):
        """Make a left turn at an intersection"""
        self.lane_pid_controller.disable_controller()
        self.car_cmd(v=0.5, omega = 1.5, lane_pose=pose_msg)
        rospy.sleep(2)
        self.stop_hist = []
        self.cmd_stop = False
        self.lane_pid_controller.enable_controller()

    def go_straight(self, pose_msg):
        """Go straight at an intersection"""
        self.lane_pid_controller.disable_controller()
        self.car_cmd(v = 0.4, omega = 0.0, lane_pose=pose_msg)
        rospy.sleep(2)
        self.stop_hist = []
        self.cmd_stop = False
        self.lane_pid_controller.enable_controller()

    def to_lane_frame(self, point):
        p_homo = np.array([point.x, point.y, 1])
        phi = self.lane_pose.phi
        d = self.lane_pose.d
        T = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), d], [0, 0, 1]])
        p_new_homo = T.dot(p_homo)
        p_new = p_new_homo[0:2]
        return p_new
    


    def on_shutdown(self):
        """Cleanup function."""
        while not rospy.is_shutdown():
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
        
    def shutdown(self, msg):
        if msg=="shutdown":
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
            time.sleep(2)

            rospy.signal_shutdown("Robot circuit Node Shutdown command received")
            exit()

if __name__ == '__main__':
    node = RobotFollowerNode(node_name='robot_follower_node')
    # Keep it spinning to keep the node alive
    # main loop
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()

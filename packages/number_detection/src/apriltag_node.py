#!/usr/bin/env python3
import os
import rospy
import cv2
import yaml
import numpy as np
from nav_msgs.msg import Odometry
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, CameraInfo
from augmented_reality_basics import Augmenter
from cv_bridge import CvBridge
from duckietown_utils import load_homography, load_map,get_duckiefleet_root
import rospkg 
from dt_apriltags import Detector, Detection
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, WheelsCmdStamped,LEDPattern
from duckietown_msgs.srv import SetCustomLEDPattern, ChangePattern
# Code from https://github.com/Coral79/exA-3/blob/44adf94bad728507608086b91fbf5645fc22555f/packages/augmented_reality_basics/include/augmented_reality_basics/augmented_reality_basics.py
# https://docs.photonvision.org/en/latest/docs/getting-started/pipeline-tuning/apriltag-tuning.html
import math
import geometry_msgs.msg
import tf
from std_msgs.msg import Header, Float32, String, Float64MultiArray, Float32MultiArray, Int32
import tf2_ros


def argmin(lst):
    tmp = lst.index(min(lst))

    return tmp



class AprilTagNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(AprilTagNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        if os.environ["VEHICLE_NAME"] is not None:
            self.veh = os.environ["VEHICLE_NAME"]
        else:
            self.veh = "csc22935"
        self.rospack = rospkg.RosPack()
        self.read_params_from_calibration_file()
        # Get parameters from config
        self.camera_info_dict = self.load_intrinsics()
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.tmp_broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.augmenter = Augmenter(self.homography,self.camera_info_msg)    
        # Subscribing 
        self.sub_image = rospy.Subscriber( f'/{self.veh}/camera_node/image/compressed',CompressedImage,self.project, queue_size=1)
        
        # Publisher
        # Keep this state so you don't need to reset the same color over and over again.
        self.pub_tag_id = rospy.Publisher(f'/{self.veh}/tag_id', Int32, queue_size=1)
        self.current_led_pattern = 4

        self.frequency_control = 0
        
        # extract parameters from camera_info_dict for apriltag detection
        f_x = self.camera_info_dict['camera_matrix']['data'][0]
        f_y = self.camera_info_dict['camera_matrix']['data'][4]
        c_x = self.camera_info_dict['camera_matrix']['data'][2]
        c_y = self.camera_info_dict['camera_matrix']['data'][5]
        self.camera_params = [f_x, f_y, c_x, c_y]
        K_list = self.camera_info_dict['camera_matrix']['data']
        self.K = np.array(K_list).reshape((3, 3))

        # initialise the apriltag detector
        self.at_detector = Detector(searchpath=['apriltags'],
                           families='tag36h11',
                           nthreads=4,
                           quad_decimate=2,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

        self.r = rospy.Rate(30) # 30hz

        #rospy.init_node('static_tf2_broadcaster_tag')
        self.buffer = tf2_ros.Buffer()

        self.buffer_listener = tf2_ros.TransformListener(self.buffer)
        


        self.fusion_x = 0
        self.fusion_y = 0
        self.fusion_z = 0

        self.fusion_rotation_z = 0
 
     
    def get_odom_location(self,req):
        data = req.pose.pose
        self.fusion_x = data.position.x
        self.fusion_y = data.position.y
        self.fusion_z = data.position.z
        self.fusion_rotation_z = data.orientation.z

    def draw_boundary(self,corners, image):
        """
        corners = [[375.08706665 209.4493866 ]
            [445.62255859 212.48104858]
            [453.69393921 136.50134277]
            [377.13027954 138.41127014]]
        
        """
        color = (0, 0, 255)
        x_list, y_list = [],[]
        for i in range(len(corners)-1):
            start = (int(corners[i][0]),int(corners[i][1]))
            end = (int(corners[i+1][0]),int(corners[i+1][1]))
            cv2.line(image, start, end, color, 5)
        
        cv2.line(image, (int(corners[0][0]),int(corners[0][1])), 
                        (int(corners[-1][0]),int(corners[-1][1])), color, 5)
        return image

    def add_text_to_img(self, tag_id, center, img):
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.5
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = 2

        text = "Others"
        if tag_id in [63, 143, 58, 62, 133, 153]:
            # T-intersection
            text = "T-intersection"

        elif tag_id in [162,169]:
            # Stop sign
            text = "Stop sign"

        elif tag_id in [93,94,200,201]:
            # U of A sign
            text = "U of A"


        cv2.putText(img,text, 
            (int(center[0]),int(center[1])), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        return img

    def transform_camera_view(self,pose_t,pose_R):
        static_transformStamped = geometry_msgs.msg.TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = f"{self.veh}/camera_optical_frame"
        static_transformStamped.child_frame_id = f"{self.veh}/new_location"

        static_transformStamped.transform.translation.x = float(pose_t[0][0])
        static_transformStamped.transform.translation.y = float(pose_t[1][0])
        static_transformStamped.transform.translation.z = float(pose_t[2][0])

        
        yaw = math.atan2(pose_R[1][0], pose_R[0][0])
        pitch = math.atan2(-pose_R[2][0], math.sqrt(pose_R[2][1]**2+pose_R[2][2]**2))
        roll = math.atan2(pose_R[2][1], pose_R[2][2])

        quat = tf.transformations.quaternion_from_euler(roll,pitch,yaw)

        
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]
        #print(static_transformStamped)

        self.broadcaster.sendTransform(static_transformStamped)

    def combine_trans(self, april_frame, trans):

        trans_x = trans.transform.translation.x
        trans_y = trans.transform.translation.y
        trans_z = trans.transform.translation.z
        quat = trans.transform.rotation
        #print(quat)
        static_transformStamped = geometry_msgs.msg.TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = april_frame
        static_transformStamped.child_frame_id = f"{self.veh}/calibrated_location"

        static_transformStamped.transform.translation.x = trans_x
        static_transformStamped.transform.translation.y = trans_y
        static_transformStamped.transform.translation.z = trans_z

        
        static_transformStamped.transform.rotation.x = quat.x
        static_transformStamped.transform.rotation.y = quat.y
        static_transformStamped.transform.rotation.z = quat.z
        static_transformStamped.transform.rotation.w = quat.w
        #print(static_transformStamped)

        self.tmp_broadcaster.sendTransform(static_transformStamped)



    def project(self, msg):

        
        br = CvBridge()
        # Convert image to cv2 image.
        self.raw_image = br.compressed_imgmsg_to_cv2(msg)
        # Convert to grey image and distort it.
        dis = self.augmenter.process_image(self.raw_image)
        new_img = dis
        if self.frequency_control % 10 == 0:

            tags = self.at_detector.detect(dis, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=0.065) # returns list of detection objects

            
            detection_threshold = 10 # The target margin need to be larger than this to get labelled.
            z_threshold = 1 # in meters
            
            if len(tags) == 0:
                # Means there's no tags present. Set the led to white.
                self.pub_tag_id.publish(-1)    
            else:
                distance_list = []
                tag_list = []

                for tag in tags:
                    
                    #print(tag)
                    z = tag.pose_t[2][0]
                    #print(tag.tag_id, tag.decision_margin)
                    if tag.decision_margin < detection_threshold or z > z_threshold:
                        continue
                    #print(tag.pose_t)
                    # Change led light according to the tag type.
                    
                    
                    distance_list.append(tag.pose_t[2][0])
                    tag_list.append(tag)

                if len(distance_list) != 0:
                    # If tag detected:
                    closest_tag_idx = argmin(distance_list)
                    #print("argmin",closest_tag_idx)
                    tag = tag_list[closest_tag_idx]

                    # Draw bounding box/
                    self.transform_camera_view(tag.pose_t,tag.pose_R)
                    # Now here you can get the ground truth location and yaw, use it to modify your odometry.
                    tag_name = tag.tag_id
                    self.pub_tag_id.publish(tag_name)

        self.frequency_control +=1
        # make new CompressedImage to publish
        #render = self.augmenter.render_segments(points=self._points, img=dis, segments=self._segments)
        #result = br.cv2_to_compressed_imgmsg(render,dst_format='jpg')
        self.r.sleep()

    def read_params_from_calibration_file(self):
        # Get static parameters
        file_name_ex = self.get_extrinsic_filepath(self.veh)
        self.homography = self.readYamlFile(file_name_ex)
        self.camera_info_msg = rospy.wait_for_message(f'/{self.veh}/camera_node/camera_info', CameraInfo)


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

    def load_intrinsics(self):
        # Find the intrinsic calibration parameters
        # cali_file_folder = '/data/config/calibrations/camera_intrinsic/'
        # self.frame_id = self.veh + '/camera_optical_frame'
        # self.cali_file = cali_file_folder + self.veh + ".yaml"

        self.cali_file = self.rospack.get_path('number_detection') + f"/config/calibrations/camera_intrinsic/{self.veh}.yaml"

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

if __name__ == '__main__':
    augmented_reality_basics_node = AprilTagNode(node_name='apriltag_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
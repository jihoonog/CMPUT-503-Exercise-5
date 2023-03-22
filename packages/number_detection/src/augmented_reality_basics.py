import cv2
import numpy as np

class Augmenter():
    def __init__(self, homography, camera_info_msg):
        self.H = [homography[0:3], homography[3:6], homography[6:9]]
        self.Hinv = np.linalg.inv(self.H)
        self.camera_info_msg = camera_info_msg
        self.K = np.array(self.camera_info_msg.K).reshape((3, 3))
        self.R = np.array(self.camera_info_msg.R).reshape((3, 3))
        self.D = np.array(self.camera_info_msg.D[0:4])
        self.P = np.array(self.camera_info_msg.P).reshape((3, 4))
        self.h = camera_info_msg.height
        self.w = camera_info_msg.width

    def process_image(self, cv_image_raw):
        ''' Undistort an image.
        '''
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.K,self.D,(self.w,self.h),0,(self.w,self.h))
        
        # Undistort the image
        res = cv2.undistort(cv_image_raw, self.K, self.D, None, newcameramtx)   

        # Convert it to black and white (greyscale)    
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        return res

    def ground2pixel(self, point):
        if point[2]!= 0:
            msg = 'This method assumes that the point is a ground point (z=0). '
            msg += 'However, the point is (%s,%s,%s)' % (point.x, point.y, point.z)
            raise ValueError(msg)

        ground_point = np.array([point[0], point[1], 1.0])
        image_point = np.dot(self.Hinv, ground_point)
        image_point = image_point / image_point[2]

        pixel = image_point[0:2]
        pixel = np.round(pixel).astype(int)
        print (pixel, image_point)
        return pixel

    def render_segments(self, points, img, segments):
        for i in range(len(segments)):
            point_x = points[segments[i]["points"][0]][1]
            point_y = points[segments[i]["points"][1]][1]
            point_x = self.ground2pixel(point_x)
            point_y = self.ground2pixel(point_y)
            color = segments[i]["color"]
            line_thickness = 2
            img = self.draw_segment(img, point_x, point_y, color)
        return img

    def draw_segment(self, image, pt_x, pt_y, color):
        defined_colors = {
            'red': ['rgb', [1, 0, 0]],
            'green': ['rgb', [0, 1, 0]],
            'blue': ['rgb', [0, 0, 1]],
            'yellow': ['rgb', [1, 1, 0]],
            'magenta': ['rgb', [1, 0, 1]],
            'cyan': ['rgb', [0, 1, 1]],
            'white': ['rgb', [1, 1, 1]],
            'black': ['rgb', [0, 0, 0]]}
        _color_type, [r, g, b] = defined_colors[color]
        cv2.line(image, (pt_x[0], pt_x[1]), (pt_y[0],
                                             pt_y[1]), (b * 255, g * 255, r * 255), 5)
        return image
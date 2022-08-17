from collections import deque

import matplotlib.pyplot as plt
from ROAR.agent_module.agent import Agent
from pathlib import Path
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.local_planner.loop_simple_waypoint_following_local_planner import \
    LoopSimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
import logging
import cv2
import numpy as np


class HarrisAgent(Agent):
    def __init__(self, target_speed=40, **kwargs):
        super().__init__(**kwargs)
        self.target_speed = target_speed
        self.logger = logging.getLogger("PID Agent")
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.pid_controller = PIDController(agent=self, steering_boundary=(-1, 1), throttle_boundary=(0, 1))
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan

        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = LoopSimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.pid_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1)
        self.logger.debug(
            f"Waypoint Following Agent Initiated. Reading f"
            f"rom {self.route_file_path.as_posix()}")
        self.debug = True
        self.error_queue = deque(maxlen=10)

        self.kP = 0.001
        self.kD = 0.01
        self.kI = 0.0001
        self.x_value = []
        self.y_value = []

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super().run_step(sensors_data=sensors_data, vehicle=vehicle)
        if sensors_data.location is not None:
            coords = sensors_data.location
            self.x_value.append(coords.x)
            self.y_value.append(coords.y)
            plt.plot(self.x_value,self.y_value)

        if self.front_rgb_camera.data is not None:
            try:
                # use vision to find line, and find the center point that we are supposed to follow
                img = self.clean_image(self.front_rgb_camera.data.copy())
                cv2.imshow("img", img)
                alineimg = np.uint8(img)
                cv2.imshow("alineimg", alineimg)

                line = cv2.HoughLinesP(alineimg, 1, np.pi/180, 50, None, 10, 0)
                lineimg = cv2.cvtColor(alineimg, cv2.COLOR_GRAY2BGR)

                #find longest line
                longest_line = 0
                index_line = 0
                if line is not None:
                    for i in range(0, len(line)):
                        l = line[i][0]
                        p1 = np.array((l[0], l[1]))
                        p2 = np.array((l[2], l[3]))
                        length = np.linalg.norm(p1 - p2)
                        if length > longest_line:
                            longest_line = length
                            index_line = i

                    l = line[index_line][0]
                    cv2.line(lineimg, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 20, cv2.LINE_AA)
                    Ly = l[3] - l[1]

                    Lx = l[2] - l[0]


                    print("Y: " + str(Ly) + "; X: " + str(Lx))
                    if (abs(Lx/Ly) > 10):
                        print("horiz")
                        control = VehicleControl(throttle = -1, steering = 0, brake = True)
                        return  control


                cv2.imshow("lineimg", lineimg)
                Xs, Ys = np.where(img == 255)
                next_point_in_pixel = (np.average(Ys).astype(int), img.shape[0] - np.average(Xs).astype(int))


                # now that we have the center point, declare robot's position as the mid, lower of the image
                robot_point_in_pixel = (img.shape[1] // 2, img.shape[0])

                # now execute a pid control on lat diff. Since we know that only the X axis will have difference
                robot_x = robot_point_in_pixel[0]
                next_point_x = next_point_in_pixel[0]

                error = robot_x - next_point_x
                self.error_queue.append(error)
                error_dt = 0 if len(self.error_queue) == 0 else error - self.error_queue[-1]
                error_it = sum(self.error_queue)

                e_p = self.kP * error
                e_d = self.kD * error_dt
                e_i = self.kI * error_it
                lat_control = np.clip(-1 * round((e_p + e_d + e_i), 3), -1, 1)

                if self.debug:
                    cv2.circle(img,
                               center=next_point_in_pixel,
                               radius=10,
                               color=(0.5, 0.5),
                               thickness=-1)
                    cv2.circle(img,
                               center=robot_point_in_pixel,
                               radius=10,
                               color=(1, 1),
                               thickness=-1)
                    #cv2.imshow("img", img)
                    cv2.waitKey(1)

                athrottle = np.clip(0.5 / (50* abs(lat_control)),0,1)
                control = VehicleControl(throttle=athrottle, steering=lat_control)

                print(control)
                return control
            except Exception as e:
                # self.logger.error("Unable to detect line")
                #return VehicleControl()
                print(e)

        return VehicleControl()

    def clean_image(self, orig):
        """
        Produce a cleaned image, with line marked as white
        :return:
        """
        shape = orig.shape
        img = orig[shape[0] // 2:, :, :]

        #finding yellow link
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([28, 60, 180])
        upper_yellow = np.array([255, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        #cv2.imshow("mask",mask_yellow)


        #cv2.imshow("ROI", img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray",gray)
        ret, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        # cv2.imshow("thresh", thresh)
        thresh = mask_yellow
        kernel = np.ones((10, 10), np.uint8)

        img = cv2.erode(thresh, kernel)
        img = cv2.dilate(img, kernel)

        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 50

        # your answer image
        img2 = np.zeros(output.shape)
        # for every component in the image, you keep it only if it's above min_size
        biggestsize = sizes[0]
        index = 0
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                if sizes[i] >= biggestsize:
                    index = i
                    biggestsize = sizes[i]
        img2[output == index + 1] = 255
        cv2.imshow("img2", img2)


        return img2

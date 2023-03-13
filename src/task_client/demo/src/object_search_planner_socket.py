#!/usr/bin/env python
"""
This module is used to plan paths for object search.
"""
import math

import cv2
import rospy
import numpy as np
import skfmm
import tf
import actionlib
from numpy import ma
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from arguments import get_args
from task_client_socket import task_client_socket
from task_client.srv import set_int
from task_client.msg import ObjectGoal
from task_client.msg import AskForSearchObjAction, AskForSearchObjResult


class ObjectSearchPlanner(object):
    """_summary_

    Args:
        object (_type_): _description_
    
    This class is used to plan path by given costmap.
    """

    def __init__(self, args):
        # Calculating global and local map sizes
        self.task_client = task_client_socket()
        self.cost_map_msg = self.get_costmap("global")
        self.map_resolution = args.map_resolution
        self.heading_angle_to_object = 0
        self.visited_map = None
        self.obstaclemap = None
        self.origin = None
        self.found_goal = 0
        self.object_in_map = None
        self.args = args
        self.fail_pub = rospy.Publisher("/search_faild", Bool, queue_size=1)
        self.score_map = None
        self.fail_flag = True
        self.last_dis = None
        return

    def get_costmap(self, range: str) -> OccupancyGrid:
        while True:
            try:
                global_map = self.task_client.request_info(
                    f"REQUEST_MAP --{range}")
                break
            except Exception:
                continue
        print("[CLIENT] Get", range, "Costmap")
        return global_map['global_map']['data']

    def plan(self, probability_map, probmap_origin, robot_pose):
        print("[CLIENT] ObjectSearchPlanner start to plan")

        # Check costmap exist
        if self.cost_map_msg is None:
            self.cost_map_msg = self.get_costmap("global")

        # Reshape costmap
        if (self.cost_map_msg is not None
                and self.cost_map_msg.info.width != 0):
            cost_map = np.asarray(self.cost_map_msg.data,
                                  dtype=np.uint8).reshape(
                                      self.cost_map_msg.info.height,
                                      self.cost_map_msg.info.width)
            cost_map[cost_map < 1] = 0
            cost_map[cost_map == 255] = 0
            cost_map[cost_map > 0] = 1
            obstaclemap = cost_map
        else:
            return [0, 0]

        # Initial visited map if not exist
        if self.visited_map is None:
            self.visited_map = np.zeros(
                (self.cost_map_msg.info.height, self.cost_map_msg.info.width))
            self.origin = self.cost_map_msg.info.origin

        self.obstaclemap = obstaclemap

        # Set visited map
        robot_locs = robot_pose[:2] - np.asarray(
            [self.origin.position.x, self.origin.position.y])
        x, y = robot_locs[1], robot_locs[0]

        loc_x, loc_y = [
            int(x * 100.0 / self.args.map_resolution),
            int(y * 100.0 / self.args.map_resolution)
        ]

        self.visited_map[loc_x - 1:loc_x + 2, loc_y - 1:loc_y + 2] = 1

        # calculate distance from robot pose
        print("[CLIENT] ObjectSearchPlanner starts checking traversable path")
        traversable = ma.masked_values(self.obstaclemap <= 0, 0)
        traversable[loc_x - 1:loc_x + 2, loc_y - 1:loc_y + 2] = 0
        if not (traversable == 1).all():
            dist = skfmm.distance(traversable, dx=1)
            dist_ori = dist * 1.0
        else:
            return None

        # get score map
        self.score_map = np.zeros(self.obstaclemap.shape)
        for y_pm in range(probability_map.shape[0]):
            for x_pm in range(probability_map.shape[1]):
                y_ob = y_pm + probmap_origin.position.y / self.map_resolution * 100 - self.origin.position.y / self.map_resolution * 100
                x_ob = x_pm + probmap_origin.position.x / self.map_resolution * 100 - self.origin.position.x / self.map_resolution * 100
                if (y_ob >= 0 and y_ob < self.obstaclemap.shape[0]
                        and x_ob >= 0 and x_ob < self.obstaclemap.shape[1]):
                    self.score_map[int(y_ob),
                                   int(x_ob)] = probability_map[int(y_pm),
                                                                int(x_pm)]

        # Dilated score map
        kernel19 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        score_map_dilated = cv2.dilate(self.score_map, kernel19)

        if self.score_map.max() >= 100:
            self.found_goal = 1

        if self.found_goal == 1:
            print("[CLIENT] ObjectSearchPlanner found goal...")
            # Calculate Goal accessible
            goal_traversable = ma.masked_values(self.obstaclemap > -9999, 0)
            goal_traversable[score_map_dilated >= 100] = 0
            try:
                goal_dist = skfmm.distance(goal_traversable, dx=1)
            except Exception:
                return self.global_goals

            goal_dist[obstaclemap > 0] += 10000

            index = np.argmin(goal_dist + dist_ori / 10.0)
            if goal_dist.min() < 10000:
                self.global_goals = [
                    index / self.obstaclemap.shape[1],
                    index % self.obstaclemap.shape[1]
                ]
            else:
                self.global_goals = [
                    index / self.obstaclemap.shape[1],
                    index % self.obstaclemap.shape[1]
                ]
                print("[CLIENT] Found goal but no accessible position!")

            # Calculate heading angle
            goal_indexs = np.nonzero(self.score_map >= 100)
            min_d = 9999
            for i in range(goal_indexs[0].shape[0]):
                delta_y = goal_indexs[0][i] - self.global_goals[0]
                delta_x = goal_indexs[1][i] - self.global_goals[1]
                self.object_in_map = [goal_indexs[0][i], goal_indexs[1][i]]

                if delta_x * delta_x + delta_y * delta_y < min_d:
                    min_d = delta_x * delta_x + delta_y * delta_y
                    self.heading_angle_to_object = np.arctan2(delta_y, delta_x)
        else:
            print("[CLIENT] ObjectSearchPlanner cannot found any goal...")

            # Goal not found, update visited map and dilate
            dist[self.obstaclemap > 0] += 10000
            kernel39 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 39))
            dist[cv2.dilate(self.visited_map, kernel39) > 0] += 10000

            dist = dist + 2 * (100 - score_map_dilated)

            # update global_goals by costmap
            index = np.argmin(dist)
            if dist.min() < 10000:
                self.global_goals = [
                    int(index / self.obstaclemap.shape[1]),
                    int(index % self.obstaclemap.shape[1])
                ]
            else:
                print(
                    "[CLIENT] ObjectSearchPlanner - There is no traversable path, Search again"
                )

                kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (19, 19))
                dilated = cv2.dilate(self.obstaclemap.astype('uint8'), kernel3)
                kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (13, 13))
                dilated2 = cv2.dilate(dilated, kernel4)
                curiosity = dilated2 - dilated

                dilated = cv2.dilate(self.visited_map, kernel3)

                curiosity[dilated > 0] = 0
                curiosity[self.score_map == 0] = 0
                dist_ori[curiosity <= 0] += 10000

                index = np.argmin(dist_ori)
                if dist_ori.min() < 10000:
                    self.global_goals = [
                        index / self.obstaclemap.shape[1],
                        index % self.obstaclemap.shape[1]
                    ]

                else:
                    print(
                        "[CLIENT] ObjectSearchPlanner - There is no traversable path, looking for traversable path failed"
                    )
                    action_lists = [[0, 0]]
                    _action = action_lists[0]
                    self.global_goals = [
                        int(_action[0] * self.obstaclemap.shape[0]),
                        int(_action[1] * self.obstaclemap.shape[1])
                    ]
                    self.fail_flag = True
                    fail_msg = Bool()
                    fail_msg.data = True
                    self.fail_pub.publish(fail_msg)
                    return None

            # update heading angle and visted map
            goal_indexs = np.nonzero(self.score_map > 20)
            min_d = 99999
            for i in range(goal_indexs[0].shape[0]):
                delta_y = goal_indexs[0][i] - self.global_goals[0]
                delta_x = goal_indexs[1][i] - self.global_goals[1]
                if delta_x * delta_x + delta_y * delta_y < min_d:
                    min_d = delta_x * delta_x + delta_y * delta_y
                    self.heading_angle_to_object = np.arctan2(delta_y, delta_x)

            vis_map = cv2.cvtColor(self.obstaclemap, cv2.COLOR_GRAY2RGB)
            vis_map[self.global_goals[0] - 4:self.global_goals[0] + 4,
                    self.global_goals[1] - 4:self.global_goals[1] +
                    4] = np.asarray([1, 0, 0])
            vis_map[loc_x - 4:loc_x + 4,
                    loc_y - 4:loc_y + 4] = np.asarray([0, 0, 1])

        return self.global_goals

    def map2real(self, goal):
        """Converts ground-truth 2D Map coordinates to absolute Habitat
        simulator position and rotation.
        """
        y, x = goal
        cont_x = x / 100. * self.map_resolution + self.origin.position.x
        cont_y = y / 100. * self.map_resolution + self.origin.position.y
        real_goal = [cont_x, cont_y]

        return real_goal

    def reset(self):
        self.heading_angle_to_object = 0
        self.visited_map = None
        self.obstaclemap = None
        self.origin = None
        self.found_goal = 0
        self.object_in_map = None
        self.score_map = None
        self.fail_flag = True
        self.last_dis = None


class ObjectSearchPlannerNode(object):
    """_summary_

    Args:
        object (_type_): _description_
        
    This class is used to call the semantic map construction process and use the generated semantic to search object by id.
    """

    def __init__(self, gt_path=''):
        self.task_client = task_client_socket()
        args = get_args()
        self.args = args
        self.obs_w, self.obs_h = args.env_frame_width, args.env_frame_height

        self.planner = ObjectSearchPlanner(args)

        self.rgb = None
        self.depth = None
        self.agent_pose = None
        self.lt_goal = None
        self.traj_lengths = 0
        self.explorable_map = cv2.imread(gt_path) if gt_path != '' else None

        self.map_frame = "map"
        self.camera_frame = "base_link"

        self.last_traj_pose = np.array([0., 0., 0.])
        self.filter_count = 0
        self.obs = None
        self.okk = False

        self.prob_map_sub = rospy.Subscriber("probability_map", OccupancyGrid,
                                             self.callback_probability_map)

        self.pub_goal = rospy.Publisher('/move_base_simple/goal',
                                        PoseStamped,
                                        queue_size=1)
        self.pub_object_goal = rospy.Publisher('object_goal',
                                               ObjectGoal,
                                               queue_size=1)

        self.occu_grid_pub = rospy.Publisher("map",
                                             OccupancyGrid,
                                             queue_size=1)

        self.bridge = CvBridge()
        self.listener = tf.TransformListener(rospy.Duration(10))

        self.ogrid = OccupancyGrid()
        self.ogrid.header.frame_id = 'map'
        self.msg_goal = PoseStamped()

        # Initial Object search action server
        self.obj_serach_server = actionlib.SimpleActionServer(
            "obj_search",
            AskForSearchObjAction,
            execute_cb=self.execute_cb,
            auto_start=False)
        self.obj_serach_server.start()
        self.search_state = None
        self.plan_state = False
        self.probability_map = None
        self.origin = None
        rospy.set_param("/is_get_semantic_map", False)

    def get_pose_change(self, pose1, pose2):
        x1, y1, o1 = pose1
        x2, y2, o2 = pose2

        theta = np.arctan2(y2 - y1, x2 - x1) - o1

        dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        dx = dist * np.cos(theta)
        dy = dist * np.sin(theta)

        do = o2 - o1
        return np.array([dx, dy, do])

    # Set probability map from ros message
    def callback_probability_map(self, map_msg):
        info = map_msg.info
        self.origin, _, w, h = info.origin, info.resolution, info.width, info.height
        self.probability_map = np.asarray(map_msg.data,
                                          dtype=np.uint8).reshape(h, w)
        pass

    def execute_cb(self, goal):
        print("[CLIENT] Object search action server is now serving")

        # Wait for target object id
        rospy.set_param("/is_get_semantic_map", True)
        rospy.wait_for_service('change_id')
        target_obj_id_client = rospy.ServiceProxy('change_id', set_int)
        target_obj_id_client(goal.obj_id)
        rospy.Rate(0.5).sleep()

        rospy.wait_for_service('change_id')
        target_obj_id_client = rospy.ServiceProxy('change_id', set_int)
        target_obj_id_client(goal.obj_id)
        rospy.Rate(0.5).sleep()

        print(f"[CLIENT] Get Object id: {goal.obj_id}")

        result_ = AskForSearchObjResult()
        success = True

        while self.search_state is None:
            print(
                '[CLIENT] Object search action server is now searching ......')

            # preemted check
            if self.obj_serach_server.is_preempt_requested():
                print("[CLIENT] Preempt occur, try to set_preempted")
                self.obj_serach_server.set_preempted()
                success = False
                break

            # request camera transform in map frame
            while not rospy.is_shutdown():
                try:
                    msg = self.task_client.request_info(
                        f"[TF]LOOKUP_TRANSFORM --source {self.map_frame} --target {self.camera_frame} --time [{str(rospy.Time(0).secs)},{str(rospy.Time(0).nsecs)}] "
                    )
                    (trans, rot) = (msg[:3], msg[3:])
                    break
                except Exception:
                    self.task_client.reconnect()
                    continue

            # Tranform rotion in quaternion
            (r, p, y) = tf.transformations.euler_from_quaternion(
                [rot[0], rot[1], rot[2], rot[3]])

            self.agent_pose = np.array([trans[0], trans[1], y])

            if self.last_traj_pose is not None:
                self.last_traj_pose = np.array([trans[0], trans[1], y])

                # Get probabilty map if not exist
                if self.probability_map is None:
                    rospy.wait_for_message("/probability_map", OccupancyGrid,
                                           rospy.Duration(10.0))
                    if self.probability_map is None:
                        continue

                print("[CLIENT] Get probability map, ready to plan trajectory")
                goals = self.planner.plan(self.probability_map, self.origin,
                                          self.agent_pose)

                if goals is not None:
                    print("[CLIENT] Goals found")
                    goals = self.planner.map2real(goals)

                    if self.filter_count % 16 == 0:
                        self.msg_goal = PoseStamped()
                        self.msg_goal.header.frame_id = "map"
                        self.msg_goal.pose.position.x = goals[0]
                        self.msg_goal.pose.position.y = goals[1]
                        self.msg_goal.pose.position.z = 0

                        quaternion = tf.transformations.quaternion_from_euler(
                            0, 0, self.planner.heading_angle_to_object)
                        self.msg_goal.pose.orientation.w = quaternion[3]
                        self.msg_goal.pose.orientation.x = quaternion[0]
                        self.msg_goal.pose.orientation.y = quaternion[1]
                        self.msg_goal.pose.orientation.z = quaternion[2]

                        self.msg_goal.header.stamp = rospy.Time.now()

                        if self.filter_count % 8 == 0:
                            while self.agent_pose[2] > np.pi:
                                self.agent_pose[2] -= 2 * np.pi
                            while self.agent_pose[2] < -np.pi:
                                self.agent_pose[2] += 2 * np.pi
                            yaw_offset = math.fmod(
                                ((self.agent_pose[2] -
                                  self.planner.heading_angle_to_object) +
                                 math.pi), 2 * math.pi) - math.pi

                            # Current pose is close enough
                            if self.okk or (
                                    self.planner.found_goal and
                                    abs(self.agent_pose[0] - goals[0]) < 0.2
                                    and abs(self.agent_pose[1] - goals[1]) <
                                    0.2) and abs(yaw_offset) < 0.17:
                                object_goal = ObjectGoal()
                                object_goal.header.frame_id = "map"
                                _object_pos = self.planner.map2real(
                                    self.planner.object_in_map)

                                object_goal.x = _object_pos[0]
                                object_goal.y = _object_pos[1]
                                object_goal.z = 0
                                object_goal.found_goal = 1

                                self.pub_object_goal.publish(object_goal)
                                self.okk = True
                                self.search_state = True
                                break
                            # Go to next goal
                            else:
                                goal_x = self.msg_goal.pose.position.x
                                goal_y = self.msg_goal.pose.position.y
                                goal_o_x = self.msg_goal.pose.orientation.x
                                goal_o_y = self.msg_goal.pose.orientation.y
                                goal_o_z = self.msg_goal.pose.orientation.z
                                goal_o_w = self.msg_goal.pose.orientation.w

                                while True:
                                    try:
                                        msg = self.task_client.send(
                                            f"[CONTROL]MOVE_TO_TARGET_POSE --trans [{goal_x},{goal_y},0] --rot [{goal_o_x},{goal_o_y},{goal_o_z},{goal_o_w}]"
                                        )
                                        break
                                    except Exception:
                                        self.task_client.reconnect()
                                        continue

                else:
                    self.search_state = False
                    break

            else:
                self.last_traj_pose = self.agent_pose
            self.filter_count += 1

        if success and self.search_state is not None:
            print("[CLIENT] Finish search")
            result_.result = self.search_state
            self.obj_serach_server.set_succeeded(result_)
        else:
            result_.result = False
            self.obj_serach_server.set_aborted(result_)

        rospy.set_param("/is_get_semantic_map", False)
        self.planner.reset()
        self.reset()

    def reset(self):
        self.rgb = None
        self.depth = None
        self.agent_pose = None
        self.lt_goal = None
        self.traj_lengths = 0

        self.map_frame = "map"
        self.camera_frame = "base_link"

        self.last_traj_pose = np.array([0., 0., 0.])
        self.filter_count = 0
        self.obs = None
        self.okk = False

        self.ogrid = OccupancyGrid()
        self.ogrid.header.frame_id = 'map'
        self.msg_goal = PoseStamped()

        self.search_state = None
        self.plan_state = False
        self.probability_map = None
        self.origin = None

        self.probability_map = None
        rospy.set_param("/is_get_semantic_map", False)


if __name__ == "__main__":
    rospy.init_node('active_slam_node', anonymous=True)
    look_active_node = ObjectSearchPlannerNode()
    rospy.spin()

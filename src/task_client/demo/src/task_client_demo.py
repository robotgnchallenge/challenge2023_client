import sys
import os
import math
import struct

import rospy
import pickle
import socket
import numpy as np
import actionlib
from socket_config import socket_conf
import tensorflow.compat.v1 as tensorflow

tensorflow.disable_eager_execution()
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir))
sys.path.append("../../scripts/contact_graspnet")
from task_client.srv import set_int
from task_client.msg import AskForSearchObjAction, AskForSearchObjGoal
from config_utils import load_config
from inference import inference

ADDR, HEADER, FORMAT, DISCONNECT_MSG = socket_conf()

ckpt_dir = "../../checkpoints/scene_test_2048_bs3_hor_sigma_001"
forward_passes = 5
arg_configs = []
z_range = [0.2, 1.1]
global_config = load_config(ckpt_dir,
                            batch_size=forward_passes,
                            arg_configs=arg_configs)
max_grasp = 10

room_nav_pose = {
    "Bedroom": [0.72, -0.09, 0.0, 0.0, 0.0, 0.9592, 0.2826],
    "Living room": [1.317, 2.542, 0.0, 0.0, 0.0, 0.8809, 0.7432],
    "Kitchen": [3.7093, 4.666, 0.0, 0.0, 0.0, -0.048, 0.9988],
    "Parlor": [2.46, 0, 0.0, 0.0, 0.0, -0.1943, 0.9809],
}

room_area = ["Bedroom", "Living room", "Parlor", "Kitchen"]

obj_id_map = {
    'Chair': 101,
    'Table': 102,
    'Bed': 103,
    'PottedTree': 104,
    'Cabinet': 105,
    'BAR': 106,
    'Armchair': 107,
    'Sofa': 108,
    'LandscapePainting': 109,
    'FloorLight': 110,
    'BunchOfSunFlowers': 111,
    "Crackers": 0,
    "Sugar": 1,
    "Can": 2,
    "Mustard": 3,
    "Spam": 4,
    "Banana": 5,
    "Bowl": 6,
    "Mug": 7,
    "Drill": 8,
    "Scissor": 9,
    "Strawberry": 11,
    "Apple": 12,
    "Lemon": 13,
    "Peach": 14,
    "Pear": 15,
    "Orange": 16,
    "Plum": 17,
    "Screwdriver": 19,
    "Ball": 21,
    "Toy": 25,
    "Wall": 100,
}


class TaskClient():

    def __init__(self) -> None:
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(ADDR)

    def reconnect(self) -> None:
        self.client.close()
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(ADDR)

    def send(self, msg_content):
        msg = msg_content.encode(FORMAT)
        msg_length = len(msg)
        header = struct.pack("i", msg_length)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' ' * (HEADER - len(send_length))
        self.client.sendall(header)
        self.client.sendall(msg)
        return self.client.recv(2048).decode(FORMAT)

    def request_info(self, msg_content):
        msg = msg_content.encode(FORMAT)
        msg_length = len(msg)
        header = struct.pack("i", msg_length)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' ' * (HEADER - len(send_length))
        self.client.sendall(header)
        self.client.sendall(msg)

        header = self.client.recv(4)

        data = b""
        while True:
            packet = self.client.recv(4096)
            if packet is not None:
                data += packet
            if packet is None or len(packet) < 4096:
                break

        data = pickle.loads(data)
        return data

    def check_valid_msg(self, msg):
        if (msg == DISCONNECT_MSG):
            self.client.sendall(DISCONNECT_MSG.encode(FORMAT))
            os._exit(0)
        else:
            return msg


task_state_client = TaskClient()
obj_search_action_client = actionlib.SimpleActionClient(
    'obj_search', AskForSearchObjAction)


def go_to_room(room: str):
    """
    Go to specific room by given name
    
    Input: room: string
    Output: None
    Example: go_to_room('Parlor')
    """
    pose_list = room_nav_pose[room]
    t_x = pose_list[0]
    t_y = pose_list[1]
    t_z = pose_list[2]
    t_r_x = pose_list[3]
    t_r_y = pose_list[4]
    t_r_z = pose_list[5]
    t_r_w = pose_list[6]

    while True:
        try:
            msg = task_state_client.send(
                f"[CONTROL]MOVE_TO_TARGET_POSE --trans [{t_x},{t_y},{t_z}] --rot [{t_r_x},{t_r_y},{t_r_z},{t_r_w}]"
            )
            print(msg)
            break
        except Exception:
            task_state_client.reconnect()
            continue


def obj_search(obj_id: int) -> bool:
    """
    Call Object search by given target object id

    Input: obj_id: int
    Output: string
    Example: obj_search(7)
    """

    # Set object search id to targe object id
    rospy.wait_for_service('change_id')
    change_obj_id_client = rospy.ServiceProxy('change_id', set_int)
    change_obj_id_client(obj_id)
    rospy.Rate(1.0).sleep()

    # Object search
    obj_search_action_client.wait_for_server()
    goal = AskForSearchObjGoal()
    goal.obj_id = obj_id

    obj_search_action_client.send_goal(goal)
    obj_search_action_client.wait_for_result()

    result = obj_search_action_client.get_result()
    if result.result:
        print("[The target item have been found.]")
    else:
        print("[The target item could not be found. Task failure !!!]")

    return result.result


def extract_bounding_box_from_segmap(obj_id: int, segmap: np.array) -> list:
    """
    Get target object bounding box by given segmentation map

    Input: obj_id(int), segmap(np array)
    Output: list [4]
    Example: extract_bounding_box_from_segmap(7, segmap)
    """
    class_ids = np.unique(segmap)
    for i in range(len(class_ids)):
        if class_ids[i] == obj_id:
            indexes = np.where(segmap == class_ids[i])
            min_x = np.min(indexes[0])
            min_y = np.min(indexes[1])
            max_x = np.max(indexes[0])
            max_y = np.max(indexes[1])
            return [min_x, min_y, max_x, max_y]
    return None


def go_to_pose(pose, angle_link4, angle_link5):
    """
    Set arm pose by given link4 and link5 angle for object search

    Input: pose: list[6], angle_link4: float, angle_link5: float
    Output: bool
    Example: go_to_pose([0,0,0,0,0,0],1.57,1.57)
    """

    # Move to target angle of link 4
    target_pose_link4 = [
        pose[0], pose[1], pose[2], pose[3] + angle_link4, pose[4], pose[5]
    ]
    movit_exec_flag = False
    while True:
        try:
            movit_exec_flag = True
            msg = task_state_client.send(
                "[MOVEIT]SET_ARM_JOINT_VALUE --pose " + str(target_pose_link4))
            rospy.Rate(1).sleep()
            if movit_exec_flag:
                break
            break
        except Exception:
            continue

    # Move to target angle of link 5
    target_pose_link5 = [
        pose[0], pose[1], pose[2], pose[3] + angle_link4,
        pose[4] + angle_link5, pose[5]
    ]
    movit_exec_flag = False

    while True:
        try:
            movit_exec_flag = True
            msg = task_state_client.send(
                "[MOVEIT]SET_ARM_JOINT_VALUE --pose " + str(target_pose_link5))
            rospy.Rate(1).sleep()
            if movit_exec_flag:
                break
            break
        except Exception:
            continue

    if msg and msg[-7:] == "FAILURE":
        return False

    return True


def obj_in_cam(bbox: list) -> bool:
    """
    Check is object in camera frame

    Input: list [4]
    Output: bool
    Example: obj_in_cam([x1,y1,x2,y2])
    """
    if bbox is not None:
        x = 1 / 2 * (bbox[0] + bbox[2])
        y = 1 / 2 * (bbox[1] + bbox[3])

        if (x > (0)) and \
           (x < (480)) and \
           (y > (0)) and (y < (640)):
            return True

    return False


def caculate_angle(bbox: list, height, width) -> list:
    """
    Calculate the camera matrix

    Input: bbox: list [4], height: int, width: int
    Output: list [2]
    Example: caculate_angle([x1,y1,x2,y2], 480, 640)
    """
    cx = width / 2
    cy = height / 2
    fx = width / (2 * math.tan(1.0 / 2.))
    fy = fx
    bbox_cy = 1 / 2 * (bbox[0] + bbox[2])
    bbox_cx = 1 / 2 * (bbox[1] + bbox[3])

    augular_x = math.atan(((bbox_cx - cx) / fx))
    augular_y = math.atan(((bbox_cy - cy) / fy))
    return [augular_x, -augular_y]


def get_seg_map():
    """
    Get segmentation map of hand camera

    Input: None
    Output: np.array [480,640]
    Example: get_seg_map()
    """
    i = 0
    segmap = None
    while True:
        try:
            i = i + 1
            segmap = task_state_client.request_info(
                "REQUEST_SENSOR_INFO --seg --hand")['seg_hand']['data']
            rospy.Rate(10.0).sleep()
            break
        except Exception:
            if i == 5:
                task_state_client.reconnect()
                i = 0
            continue
    return segmap


def obj_align_mid(bbox, obj_id):
    """
    Move robot arm to center the object in the camera image

    Input: bbox: list[4], obj_id: int
    Output: bool
    Example: obj_align_mid([x1,y1,x2,y2], 7)
    """
    print("[CLIENT]Start moving object to center")
    cur_jointval = request_info_to_server("[MOVEIT]GET_ARM_JOINT_VALUE")

    angle = caculate_angle(bbox, height=480.0, width=640.0)
    pose_suc = go_to_pose(cur_jointval, angle[0], angle[1])

    if pose_suc:
        return True
    else:
        seg_map = get_seg_map()
        if seg_map.all():
            bbox = extract_bounding_box_from_segmap(obj_id, seg_map)
            c_y = 1 / 2 * (bbox[0] + bbox[2])
            if c_y > 1 / 2 * 480.0:
                msg = task_state_client.send(
                    "[MOVEIT]SET_ARMBASE_JOINT_VALUE --pose " + str([0.10]))
            else:
                msg = task_state_client.send(
                    "[MOVEIT]SET_ARMBASE_JOINT_VALUE --pose " + str([0.35]))
            print(msg)
    return False


def arm_look_for_obj(obj_id: int) -> bool:
    """
    Move robot arm to look around and search object

    Input: obj_id: int
    Output: bool
    Example: arm_look_for_obj(7)
    """
    print("[CLIENT]Start looking for target object: ... ...")

    # Get object bbox in image
    seg_map = get_seg_map()
    bbox = extract_bounding_box_from_segmap(obj_id, seg_map)

    # Check is object in camera frame
    in_cam = obj_in_cam(bbox)
    if in_cam:
        obj_align_mid(bbox, obj_id)
        return True

    # If object not in camera frame, start object search
    arm_jiont_position = [-0.0174, 5.113, 4.85, 0.0523, 1.500, 0.0872]
    angle_list = [(0.0, 0.0), (0.0, 0.3), (0.0, -0.35), (0, 0.0), (0.8, 0.0),
                  (0.8, -0.2), (0.8, 0.3),
                  (-0.8, 0), (-0.8, 0.2), (-0.8, -0.35), (0.6, 0.0),
                  (0.6, -0.35), (0.6, 0.3), (-0.6, 0), (-0.6, -0.35),
                  (-0.6, 0.3), (0.4, 0), (0.4, -0.35), (0.4, 0.3), (-0.4, 0),
                  (-0.4, -0.35), (0.4, 0.3)]

    # Try different angles to search object, if object exist in camera frame, align the object in the middle of camera frame
    for (link_4_angle, link_5_angle) in angle_list:
        go_to_pose(arm_jiont_position, link_4_angle, link_5_angle)
        seg_map = get_seg_map()
        bbox = extract_bounding_box_from_segmap(obj_id, seg_map)
        if obj_in_cam(bbox):
            obj_align_mid(bbox, obj_id)
            return True
        else:
            continue

    return False


def set_arm_init_pose():
    """
    Move robot arm to initial pose

    Input: None
    Output: string
    Example: set_arm_init_pose()
    """
    init_pose = [0.052, 4.97, 3.16, -0.087, 0.93, -0.087]
    while True:
        try:
            msg = task_state_client.send(
                "[MOVEIT]SET_ARM_JOINT_VALUE --pose " + str(init_pose))
            rospy.Rate(1).sleep()
            break
        except Exception:
            continue

    msg = task_state_client.send("[MOVEIT]SET_ARMBASE_JOINT_VALUE --pose " +
                                 str([0.0]))
    return msg


def pc_obj_filter(pc_full, obj_id, seg_map):
    """
    Get object point cloud by segmentation map

    Input: pc_full : np.array [n,3], obj_id: int, seg_map: np.array [480,640]
    Output: np.array [n,3]
    Example: pc_obj_filter(pc, 7, seg_map)
    """
    width = 640
    height = 480

    cx = width / 2
    cy = height / 2
    fx = width / (2 * math.tan(1 / 2))
    fy = fx

    pc_filter = []
    for p in pc_full:
        u = int((fx * p[0]) / p[2] + cx)
        v = int((fy * p[1]) / p[2] + cy)
        if seg_map[v, u] == obj_id:
            pc_filter.append([p[0], p[1], p[2]])

    return np.array(pc_filter)


def send_state_msg_to_server(state_msg):
    """"
    State message publisher

    Input: state_msg : string
    Output: string
    Example: send_state_msg_to_server("REQUEST_FOR_INST")
    """
    while True:
        try:
            msg = task_state_client.send(state_msg)
            break
        except Exception:
            task_state_client.reconnect()
            continue

    msg = task_state_client.check_valid_msg(msg)
    return msg


def request_info_to_server(state_msg):
    """
    Sever Info reciever

    Input: state_msg : string
    Output: Depending on the request message
    Example: request_info_to_server("REQUEST_SENSOR_INFO --pc --laser --hand")
    """
    while True:
        try:
            msg = task_state_client.request_info(state_msg)
            break
        except Exception:
            task_state_client.reconnect()
            continue

    return msg


def send_grasp_pose_to_server(grasp_pose):
    """
    Pack the grasp data and send to server

    Input: grasp_pose : np.array [n,4,4]; n<=10
    Output: None
    Example: send_grasp_pose_to_server(grasp_pose)
    """
    grasp_num = grasp_pose.shape[0]
    msg = str(grasp_num).encode(FORMAT)
    header = struct.pack("i", len(msg))
    task_state_client.client.sendall(header)
    task_state_client.client.sendall(msg)

    for i in range(grasp_num):
        data = pickle.dumps(grasp_pose[i])

        size = sys.getsizeof(data)
        header = struct.pack("i", size)

        task_state_client.client.sendall(header)
        task_state_client.client.sendall(data)
        task_state_client.client.recv(2048)


def main():
    rospy.init_node('task_client_node', anonymous=True)

    # Initial mobile arm pose
    set_arm_init_pose()

    # Request task instruction
    inst = send_state_msg_to_server("REQUEST_FOR_INST")
    print("[SERVER] Task instruction: ", inst)

    # Instruction process, get the target obj and destination.
    obj_room = [ele for ele in room_area if (ele in inst.split(',')[0])]
    obj_room = obj_room[0]

    target_obj = inst.split('pick the')[1].split('and place')[0].replace(
        " ", "")
    obj_id = obj_id_map[target_obj]
    destination = [ele for ele in room_area if (ele in inst.split(',')[-1])]
    destination = destination[0]

    # Please Change to Your Object search Algorithm.......................
    # Go to the room of the target object
    go_to_room(obj_room)
    print("[CLIENT] Robot has arrived in the", obj_room)

    result = obj_search(obj_id)
    msg = send_state_msg_to_server("REQUEST_TO_CHECK_NAV_RESULT")
    print(result)
    print("[SERVER] ", msg)
    # ....................................................................

    # Check obj search result
    if (msg.split('][')[-1].split(']')[0] == "NAV_SUCCESS"):
        pass
    else:
        task_state_client.client.sendall(DISCONNECT_MSG.encode(FORMAT))
        os._exit(0)

    # Start align object into the object center
    msg = task_state_client.send("[MOVEIT]SET_ARMBASE_JOINT_VALUE --pose " +
                                 str([0.225]))
    look_obj_result = arm_look_for_obj(obj_id)
    print("[CLIENT]", look_obj_result)

    # Please Change to Your Grasp Detection Algorithm.......................
    # Get object point cloud and corresponing segmentation map
    cam_data = dict()
    while True:
        try:
            cam_data['pc_full'] = task_state_client.request_info(
                "REQUEST_SENSOR_INFO --pc --hand")['pc_full_hand']['data']
            seg_map = get_seg_map()
            cam_data["pc_full"] = pc_obj_filter(pc_full=cam_data["pc_full"],
                                                obj_id=obj_id,
                                                seg_map=seg_map)
            break
        except Exception as e:
            print(e)
            continue

    # Inference grasp pose by point cloud
    try:
        grasp_result = inference(cam_data,
                                 global_config,
                                 ckpt_dir,
                                 None,
                                 z_range=z_range,
                                 K=None,
                                 local_regions=False,
                                 filter_grasps=False,
                                 segmap_id=0,
                                 forward_passes=forward_passes,
                                 skip_border_objects=False,
                                 tf_pub=False)
    except Exception as e:
        print(e)
        task_state_client.client.sendall(DISCONNECT_MSG.encode(FORMAT))
        os._exit(0)

    # Grasp should be an np array with size [n, 4, 4], where n <= 10
    grasp_data = np.array(grasp_result[-1])
    if grasp_data.shape[0] >= max_grasp:
        grasp_data = grasp_data[0:max_grasp]
    # ......................................................................

    # Check garsp result
    send_grasp_pose_to_server(grasp_data)
    server_msg = send_state_msg_to_server("REQUEST_TO_CHECK_GSP_RESULT")
    print(server_msg)

    if ('[TASK_FAILED]' in server_msg):
        task_state_client.client.sendall(DISCONNECT_MSG.encode(FORMAT))
        os._exit(0)

    # Navigate to destination
    set_arm_init_pose()
    go_to_room(destination)

    table_search_res = obj_search(obj_id_map['Table'])
    print(table_search_res)

    # Check is valid to place
    server_msg = send_state_msg_to_server("REQUEST_TO_PLACE")
    print(server_msg)

    os._exit(0)


if __name__ == '__main__':
    main()

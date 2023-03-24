import sys
import os
import cv2
import time
import pickle
import socket
import struct
from select import select

import rospy
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir))
from socket_config import socket_conf
from constant import obj_idx_colour, obj_name

ADDR, HEADER, FORMAT, DISCONNECT_MSG = socket_conf()

obj_idx = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 21, 25
]


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
            break
        except Exception:
            continue

    msg = task_state_client.send("[MOVEIT]SET_ARMBASE_JOINT_VALUE --pose " +
                                 str([0.0]))
    return msg


def get_seg_info():
    data = task_state_client.request_info(
        "REQUEST_SENSOR_INFO --rgb --depth --seg --hand")
    masks = data['seg_hand']['data']

    data['bbox'] = []
    data['bbox_idx'] = []
    for idx in obj_idx:
        if (idx in masks):
            obj_pos = np.where(masks == idx)
            left = min(obj_pos[0])
            top = min(obj_pos[1])
            right = max(obj_pos[0])
            bottom = max(obj_pos[1])
            data['bbox'].append([left, top, right, bottom])
            data['bbox_idx'].append(idx)

    return data


def data_collection():
    if (not os.path.exists('../sim_data')):
        os.system('mkdir ../sim_data')

    set_arm_init_pose()
    print("Finish moving to init pose...")
    while (1):
        data = get_seg_info()
        np.save('../sim_data/' + data['seg_hand']['time'] + '.npy', data)
        time.sleep(5)


if __name__ == '__main__':
    home_pose = "[0.22878372358235885, 2.770545544239736, 3.0079523744660763, -0.2639381358204911, 0.934255714582961, 0.12304845181915694]"
    home_gripper_pose = "[0.0, 0.0]"

    # set_arm_init_pose()
    # msg  = task_state_client.request_info("REQUEST_SENSOR_INFO --pc --rgb --depth --seg --laser --hand")
    # msg = task_state_client.request_info(f"[TF]LOOKUP_TRANSFORM --source moma_base_link --target ArmBase --time [{str(time.secs)},{str(time.nsecs)}] ")
    # msg = task_state_client.send("[CONTROL]MOVE_TO_TARGET_POSE --trans [1,1,0] --rot [0,0,1,0]")
    # msg = task_state_client.request_info("REQUEST_MAP --map --global --local")

    data_collection()

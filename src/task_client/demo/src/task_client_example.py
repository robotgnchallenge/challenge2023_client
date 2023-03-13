import sys
import os
import pickle
import socket
import struct
from select import select

import tensorflow.compat.v1 as tensorflow

tensorflow.disable_eager_execution()
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir))
from socket_config import socket_conf

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

ADDR, HEADER, FORMAT, DISCONNECT_MSG = socket_conf()


class TaskClient():

    def __init__(self) -> None:
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

    def send_control_inst(self, msg_content):
        msg = msg_content.encode(FORMAT)
        msg_length = len(msg)
        header = struct.pack("i", msg_length)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' ' * (HEADER - len(send_length))
        self.client.sendall(header)
        self.client.sendall(msg)
        return

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
            self.send(DISCONNECT_MSG)
            sys.exit()
        else:
            return msg


def SaveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)


def RestoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed, turn)


def getKey(settings, timeout):
    if sys.platform == 'win32':
        key = msvcrt.getwch()
    else:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def KeyboardControl(task_state_client, state):
    key_timeout = 0.5
    settings = SaveTerminalSettings()
    msg = None

    nav_succ = False

    while (1):
        key = getKey(settings, key_timeout)
        if (key == 'i'):
            msg = task_state_client.send("[CONTROL]MOVE_FRONT")
        elif (key == 'u'):
            msg = task_state_client.send("[CONTROL]MOVE_LEFT_FRONT")
        elif (key == 'o'):
            msg = task_state_client.send("[CONTROL]MOVE_RIGHT_FRONT")
        elif (key == ','):
            msg = task_state_client.send("[CONTROL]MOVE_REAR")
        elif (key == 'm'):
            msg = task_state_client.send("[CONTROL]MOVE_LEFT_REAR")
        elif (key == '.'):
            msg = task_state_client.send("[CONTROL]MOVE_RIGHT_REAR")
        elif (key == 'j'):
            msg = task_state_client.send("[CONTROL]TURN_LEFT")
        elif (key == 'l'):
            msg = task_state_client.send("[CONTROL]TURN_RIGHT")
        elif (key == 'q'):
            msg = task_state_client.send("[CONTROL]SPEED_UP")
        elif (key == 'z'):
            msg = task_state_client.send("[CONTROL]SPEED_DOWN")
        elif (key == 'r'):
            break
        else:
            pass

        if (msg):
            print(msg)
            msg = None

    return nav_succ


if __name__ == '__main__':
    task_state_client = TaskClient()
    home_pose = "[0.22878372358235885, 2.770545544239736, 3.0079523744660763, -0.2639381358204911, 0.934255714582961, 0.12304845181915694]"
    home_gripper_pose = "[0.0, 0.0]"

    msg = task_state_client.request_info(
        "REQUEST_SENSOR_INFO --depth --seg --head")
    # msg  = task_state_client.request_info("REQUEST_SENSOR_INFO --pc --rgb --depth --seg --laser --hand")
    # msg = task_state_client.request_info(f"[TF]LOOKUP_TRANSFORM --source moma_base_link --target ArmBase --time [{str(time.secs)},{str(time.nsecs)}] ")
    # msg = task_state_client.send("[CONTROL]MOVE_TO_TARGET_POSE --trans [1,1,0] --rot [0,0,1,0]")
    # msg = task_state_client.request_info("REQUEST_MAP --map --global --local")
    print(msg)

import socket
import sys
import os
import pickle
import struct

from socket_config import socket_conf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(
    f"{os.path.expanduser('~')}/client_ws/src/task_client/scripts/contact_graspnet"
)

ADDR, HEADER, FORMAT, DISCONNECT_MSG = socket_conf()

forward_passes = 5
arg_configs = []
z_range = [0.2, 1.1]


class task_client_socket():

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
            self.send(DISCONNECT_MSG)
            sys.exit()
        else:
            return msg

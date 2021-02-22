"""camera_controller controller."""

from controller import Robot
from controller import RangeFinder
from controller import Camera
import numpy as np
import socket
import struct
import pickle
import threading
from PIL import Image as pimg
np.set_printoptions(precision=4, suppress=True)


def send_msg(msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    conn.sendall(msg)

def recv_msg():
    # Read message length and unpack it into an integer
    raw_msglen = recvall(4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(msglen)

def recvall(n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def respond(result, data = None):
    cmd = {}
    cmd["result"] = result
    cmd["data"] = data
    send_msg(pickle.dumps(cmd))

def continous_timestep():
    while robot.step(timestep) != -1:
        pass



robot = Robot()

cameraRGB = Camera("cameraRGB")
cameraDepth = RangeFinder("cameraDepth")
timestep = int(robot.getBasicTimeStep())

current_task = "idle"
args = None
command_is_executing = False
print_once_flag = True
rgb_enabled = False
depth_enabled = False

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 2001))
server.listen()
print("Waiting for connection")
robot.step(1) # webots won't print without a step
conn, addr = server.accept()
print("Connected")

x = threading.Thread(target=continous_timestep)
x.start()

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    if current_task == "idle":
        if print_once_flag:
            print("Waiting for command")
            robot.step(1)
            print_once_flag = False
        msg = recv_msg()
        if msg != None:
            cmd = pickle.loads(msg)
            print(cmd)
            current_task = cmd["name"]
            args = cmd["args"]
            print("Executing: " + current_task + " " + str(args))
            print_once_flag = True

    if current_task == "get_image":
        if not rgb_enabled:
            cameraRGB.enable(timestep)
            rgb_enabled = True
        else:
            np_img = np.array(cameraRGB.getImageArray(), dtype=np.uint8)
            respond("done", np_img)
            current_task = "idle"
            cameraRGB.disable()
            rgb_enabled = False

    if current_task == "get_depth":
        if not depth_enabled:
            cameraDepth.enable(timestep)
            depth_enabled = True
        else:
            np_dep = np.array(cameraDepth.getRangeImageArray())
            respond("done", np_dep)
            current_task = "idle"
            cameraDepth.disable()
            depth_enabled = False


conn.close()
print("Camera controller ended")
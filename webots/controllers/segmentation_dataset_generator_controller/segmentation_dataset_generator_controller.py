"""segmentation_dataset_generator_controller controller."""

from controller import Robot
from controller import Connector
from controller import RangeFinder
from controller import Camera
from controller import Supervisor
import random
import socket
import struct
import pickle
import numpy as np
import math
import cv2
import os

#robot = Robot()
supervisor = Supervisor()
connector = None
timestep = 100

conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

motors = [supervisor.getMotor("shoulder_pan_joint"), supervisor.getMotor("shoulder_lift_joint"), supervisor.getMotor("elbow_joint"),
          supervisor.getMotor("wrist_1_joint"), supervisor.getMotor("wrist_2_joint"), supervisor.getMotor("wrist_3_joint")]
#motor_sensors = [robot.getPositionSensor("shoulder_pan_joint_sensor"), robot.getPositionSensor("shoulder_lift_joint_sensor"), robot.getPositionSensor("elbow_joint_sensor"),
          #robot.getPositionSensor("wrist_1_joint_sensor"), robot.getPositionSensor("wrist_2_joint_sensor"), robot.getPositionSensor("wrist_3_joint_sensor")]

camera = supervisor.getCamera('cameraRGB')
depth_camera = supervisor.getRangeFinder('cameraDepth')
ur_node = supervisor.getFromDef('UR5')
camera_transform_node = ur_node.getField('children').getMFNode(0)
camera_node = camera_transform_node.getField('children').getMFNode(1)
#depth_camera.enable(35)

phone_part_objects = [
    supervisor.getFromDef('Bottom_Cover'),
    supervisor.getFromDef('Bottom_Cover_2'),
    supervisor.getFromDef('White_Cover'),
    supervisor.getFromDef('White_Cover_2'),
    supervisor.getFromDef('Black_Cover'),
    supervisor.getFromDef('Black_Cover_2'),
    supervisor.getFromDef('Blue_Cover'),
    supervisor.getFromDef('Blue_Cover_2'),
    supervisor.getFromDef('PCB'),
    supervisor.getFromDef('PCB_2')
]

index_to_class_name = [
    'BottomCover',
    'BottomCover',
    'WhiteCover',
    'WhiteCover',
    'BlackCover',
    'BlackCover',
    'BlueCover',
    'BlueCover',
    'PCB',
    'PCB',
]

pbr_apperance_nodes = []
original_colors = []
for part in phone_part_objects:
    children = part.getField('children')
    shape = children.getMFNode(1)
    pbr_apperance_node = shape.getField('appearance')
    color = pbr_apperance_node.getSFNode().getField('baseColor').getSFColor()
    pbr_apperance_nodes.append(pbr_apperance_node)
    original_colors.append(color)

translation_fields = [node.getField('translation') for node in phone_part_objects]
rotation_fields = [node.getField('rotation') for node in phone_part_objects]
front_cover_initial_pos = [-0.17, -0.16]
back_cover_initial_pos = [-0.16, -0.16]
pcb_initial_pos = [-0.14, -0.13]
default_rotation = [1, 0, 0, 1.57]
max_movement = [0.08, 0.08]


def randomize_phone_parts():
    height = 1.1
    height_step = 0.05

    for index, (translation_field, rotation_field) in enumerate(zip(translation_fields, rotation_fields)):
        current_position = translation_field.getSFVec3f()
        random_rotation = random.random() * 0.2 + 1.57
        rotation = [1, 0, 0, random_rotation]
        rotation_field.setSFRotation(rotation)
        random_x_shift = random.random() * (max_movement[0] * 2) - max_movement[0]
        random_z_shift = random.random() * (max_movement[1] * 2) - max_movement[1]
        if index < 2:
            # back cover
            translation_field.setSFVec3f(
                [back_cover_initial_pos[0] + random_x_shift, height, back_cover_initial_pos[1] + random_z_shift])
        elif 2 <= index < 8:
            # front cover
            translation_field.setSFVec3f(
                [front_cover_initial_pos[0] + random_x_shift, height, front_cover_initial_pos[1] + random_z_shift])
        else:
            translation_field.setSFVec3f(
                [pcb_initial_pos[0] + random_x_shift, height, pcb_initial_pos[1] + random_z_shift])
        height += height_step

    for part in phone_part_objects:
        part.resetPhysics()


def set_color_for_all_except_index(ii):
    for i, node in enumerate(pbr_apperance_nodes):
        if i == ii:
            node.getSFNode().getField('baseColor').setSFColor([1, 1, 0])
            continue
        node.getSFNode().getField('baseColor').setSFColor([0, 0, 0])


def restore_colors():
    for i, node in enumerate(pbr_apperance_nodes):
        node.getSFNode().getField('baseColor').setSFColor(original_colors[i])


def toggle_visibility_for_all_parts(visible):
    for part in phone_part_objects:
        part.setVisibility(camera_node, visible)


def transform_image(img_array):
    np_img = np.array(img_array, dtype=np.uint8)
    np_img = np_img.transpose((1, 0, 2))
    np_img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return np_img_bgr

randomize_phone_parts()

wait_time = 3.5 # seconds
last_run_time = supervisor.getTime()
take_image = True
is_first_run = True
images_to_take = 200
image_index = 0
camera.enable(1)
supervisor.step(1)
toggle_visibility_for_all_parts(False)
supervisor.step(1)
background_img = transform_image(camera.getImageArray())
cv2.imwrite('background.png', background_img)
toggle_visibility_for_all_parts(True)
restore_colors()
camera.disable()

dataset_path = 'dataset'
if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)

while supervisor.step(timestep) != -1:
    if supervisor.getTime() - last_run_time >= wait_time and image_index < images_to_take:
        camera.enable(1)
        supervisor.step(1)
        print(f'Taking image {image_index+1}/{images_to_take}')
        save_path = os.path.join(dataset_path, f'img{image_index}')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        full_img = transform_image(camera.getImageArray())
        cv2.imwrite(os.path.join(save_path, 'full_image.png'), full_img)
        for index in range(0, len(phone_part_objects)):
            set_color_for_all_except_index(index)
            supervisor.step(1)
            image = transform_image(camera.getImageArray())
            image_subtracted = cv2.subtract(image, background_img)
            image_grayscale = cv2.cvtColor(image_subtracted, cv2.COLOR_BGR2GRAY)
            _, image_binary = cv2.threshold(image_grayscale, 5, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel)
            image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel)
            save_name = f"mask{index}_{index_to_class_name[index]}.png"
            cv2.imwrite(os.path.join(save_path, save_name), image_binary)
            restore_colors()
        camera.disable()
        randomize_phone_parts()
        last_run_time = supervisor.getTime()
        image_index += 1
    elif image_index >= images_to_take:
        break

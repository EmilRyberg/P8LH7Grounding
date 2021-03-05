#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from ner.command_builder import CommandBuilder, PickUpTask, FindTask
from ner.ner import EntityType
from little_helper_interfaces.msg import Task, ObjectEntity, OuterObjectEntity, SpatialDescription, OuterTask
from ner.srv import NER, NERResponse, NERRequest
import argparse


def create_ros_task(task):
    ros_task = Task()
    object1_entity = None
    if isinstance(task, PickUpTask):
        ros_task.type = "pick_up"
        object1_entity = task.object_to_pick_up
    elif isinstance(task, FindTask):
        ros_task.type = "find"
        object1_entity = task.object_to_find
    ros_task.object1 = OuterObjectEntity(name=object1_entity.name, spatial_descriptions=[])
    for spatial_description in object1_entity.spatial_descriptions:
        ros_task.object1.spatial_descriptions.append(SpatialDescription(spatial_type=spatial_description.spatial_type.value,
                                                                    object_entity=ObjectEntity(
                                                                        name=spatial_description.object_entity.name)))
    return ros_task


class NERService:
    def __init__(self, model_path, tags_path):
        self.cmd_builder = CommandBuilder(model_path, tags_path)

    def handle_ner_request(self, request: NERRequest):
        task = self.cmd_builder.get_task(request.sentence)
        ros_task = create_ros_task(task)
        ros_outer_task = OuterTask(type=ros_task.type, object1=ros_task.object1, object2=ros_task.object2, child_tasks=[])

        if len(task.child_tasks) > 0:
            for child_task in task.child_tasks:
                ros_task = create_ros_task(child_task)
                ros_outer_task.child_tasks.append(ros_task)
        return NERResponse(task=ros_outer_task)

    def ner_server(self):
        rospy.init_node("ner_server")
        s = rospy.Service("ner", NER, self.handle_ner_request)
        rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_path", default="model.bin", help="The path to the weight file for the model")
    parser.add_argument("-t", "--tag_file", dest="tags_path", default="tags.txt",
                        help="The path to the tags file")
    print(rospy.myargv())
    args = parser.parse_args(rospy.myargv()[1:])
    service = NERService(args.model_path, args.tags_path)
    service.ner_server()


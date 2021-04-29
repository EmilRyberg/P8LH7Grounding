#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from little_helper_interfaces.msg import ObjectEntity, ObjectInfo, OuterObjectEntity, ROSGroundingReturn
from task_grounding.srv import ROSTaskGrounding, ROSTaskGroundingResponse, ROSTaskGroundingRequest
from task_grounding.task_grounding import TaskGrounding, TaskGroundingReturn, ErrorType


def create_ros_return(non_ros_return):
    object_info = non_ros_return.object_infos
    ros_info = ObjectInfo(mask=object_info.mask, cropped_rbg=object_info.cropped_rgb, bbox=object_info.bbox)
    ros_return = ROSGroundingReturn(is_success=non_ros_return.is_success, error_code=non_ros_return.error_code.value,
                                    object_info=ros_info)
    return ros_return


class TaskGroundingService:
    def __init__(self):
        self.grounding = TaskGrounding()
        self.returned = TaskGroundingReturn()

    def handle_grounding_request(self, request: ROSTaskGroundingRequest):
        return request

    def grounding_server(self):
        rospy.init_node("task_grounding_server")
        s = rospy.Service("task_grounding", ROSTaskGrounding, self.handle_grounding_request)
        rospy.spin()


if __name__ == '__main__':
    service = TaskGroundingService()
    service.grounding_server()

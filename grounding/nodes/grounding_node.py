#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from little_helper_interfaces.msg import ObjectEntity, ObjectInfo, OuterObjectEntity, ROSGroundingReturn
from grounding.srv import ROSGrounding, ROSGroundingResponse, ROSGroundingRequest
from grounding.grounding import Grounding, GroundingReturn, ErrorType


def create_ros_return(non_ros_return):
    object_info = non_ros_return.object_info
    ros_info = ObjectInfo(mask=object_info.mask, cropped_rbg=object_info.cropped_rgb, bbox=object_info.bbox)
    ros_return = ROSGroundingReturn(is_success=non_ros_return.is_success, error_code=non_ros_return.error_code.value,
                                    object_info=ros_info)
    return ros_return


class GroundingService:
    def __init__(self):
        self.grounding = Grounding()
        self.returned = GroundingReturn()

    def handle_grounding_request(self, request: ROSGroundingRequest):
        if request.command == "find":
            self.returned = self.grounding.find_object(request.entity)
            ros_return = create_ros_return(self.returned)
            return ROSGroundingResponse(grounding_return=ros_return)
        elif request.command == "update":
            self.returned = self.grounding.update_features(request.entity)
            ros_return = create_ros_return(self.returned)
            return ROSGroundingResponse(grounding_return=ros_return)
        elif request.command == "learn":
            self.returned = self.grounding.learn_new_object(request.entity)
            ros_return = create_ros_return(self.returned)
            return ROSGroundingResponse(grounding_return=ros_return)
        else:
            raise Exception("unknown command passed to grounding service, commands can be find, update, learn")

    def grounding_server(self):
        rospy.init_node("grounding_server")
        s = rospy.Service("grounding", ROSGrounding, self.handle_grounding_request)
        rospy.spin()


if __name__ == '__main__':
    service = GroundingService()
    service.grounding_server()

import rospy
import asyncio
import websockets
import cv2
import base64
from concurrent.futures import TimeoutError

timeout = 2


class UIInterface:
    def __init__(self, websocket_uri):
        self.uri = websocket_uri
        self.loop = asyncio.get_event_loop()
        self.is_connected = False
        self.websocket = None

    def connect(self):
        try:
            self.websocket = self.loop.run_until_complete(websockets.connect(self.uri))
            self.is_connected = True
            return True
        except ConnectionRefusedError as e:
            return False

    def send_as_robot(self, message):
        try:
            self.loop.run_until_complete(asyncio.wait_for(self.websocket.send(f"rob||{message}"), timeout))
        except TimeoutError as e:
            rospy.logerr("Websocket timed out", e)
        except Exception as e:
            rospy.logerr("Some other error occurred", e)

    def send_as_user(self, message):
        try:
            self.loop.run_until_complete(asyncio.wait_for(self.websocket.send(f"usr||{message}"), timeout))
        except TimeoutError as e:
            rospy.logerr("Websocket timed out", e)
        except Exception as e:
            rospy.logerr("Some other error occurred", e)

    def send_images(self, image_full, image_cutout):
        image_full_resized = cv2.resize(image_full, (960, 540))
        _, buffer_full = cv2.imencode(".jpg", image_full_resized, (cv2.IMWRITE_JPEG_QUALITY, 80))
        _, buffer_cutout = cv2.imencode(".jpg", image_cutout)
        try:
            self.loop.run_until_complete(asyncio.wait_for(self.websocket.send(f"img1||{base64.b64encode(buffer_cutout).decode()}"), timeout))
            self.loop.run_until_complete(asyncio.wait_for(self.websocket.send(f"img2||{base64.b64encode(buffer_full).decode()}"), timeout))
        except TimeoutError as e:
            rospy.logerr("Websocket timed out", e)
        except Exception as e:
            rospy.logerr("Some other error occurred", e)

    def __del__(self):
        if self.is_connected:
            self.loop.run_until_complete(self.websocket.close())


if __name__ == "__main__":
    ui = UIInterface(websocket_uri="ws://localhost:8765")
    success = ui.connect()
    print("Success?", success)
    if success:
        ui.send_as_user("hello")
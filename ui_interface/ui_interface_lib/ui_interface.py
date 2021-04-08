import asyncio
import websockets


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
        self.loop.run_until_complete(self.websocket.send(f"rob||{message}"))

    def send_as_user(self, message):
        self.loop.run_until_complete(self.websocket.send(f"usr||{message}"))

    def __del__(self):
        if self.is_connected:
            self.loop.run_until_complete(self.websocket.close())


if __name__ == "__main__":
    ui = UIInterface(websocket_uri="ws://localhost:8765")
    success = ui.connect()
    print("Success?", success)
    if success:
        ui.send_as_user("hello")
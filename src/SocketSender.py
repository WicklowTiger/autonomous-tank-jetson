import socket
import threading
from config import AUTH_ID
from time import sleep


class SocketSender:
    socket = None

    def __init__(self, udp_ip: str, udp_port: int):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.socket_setup()

    def socket_setup(self):
        pass
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def keep_alive(self):
        while True:
            self.send_message(AUTH_ID)
            sleep(0.5)

    def send_message(self, message: str):
        threading.Thread(target=lambda msg: (
            print(f"Sending msg to {self.udp_ip}:{self.udp_port} (msg=\"{msg}\")"),
            self.socket.sendto(bytes(f"{message},{AUTH_ID}", "utf-8"), (self.udp_ip, self.udp_port))
        ), args=(message, )).start()

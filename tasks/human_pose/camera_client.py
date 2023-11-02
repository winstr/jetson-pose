import socket
import pickle
import struct
import traceback

import cv2
import numpy as np


def split_frame(frame):
    # dummy, hardcoding.
    unit = 640
    picam_frame = frame[:, 0:unit, ...]
    ircam_frame = frame[:, unit:unit*2, ...]
    ocams_frame = frame[:, unit*2:unit*3, ...]
    return picam_frame, ircam_frame, ocams_frame


class CameraStreamingClient():

    def __init__(self):
        self.client_socket = None
        self.data = b''
        self.payload_size = struct.calcsize('L')

    def connect(self, server_ip: str, server_port: int):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((server_ip, server_port))
        print(f'Connection established: {server_ip}:{server_port}')

    def disconnect(self):
        if self.client_socket:
            self.client_socket.close()

    def stream(self) -> np.ndarray:
        try:
            while True:
                while len(self.data) < self.payload_size:
                    self.data += self.client_socket.recv(4096)
                packed_msg_size = self.data[:self.payload_size]
                self.data = self.data[self.payload_size:]
                msg_size = struct.unpack('L', packed_msg_size)[0]
                while len(self.data) < msg_size:
                    self.data += self.client_socket.recv(4096)
                frame_data = self.data[:msg_size]
                self.data = self.data[msg_size:]
                frame = pickle.loads(frame_data)
                yield frame
        except:
            traceback.print_exc()

    def display(self):
        try:
            for frame in self.stream():
                picam_frame, ircam_frame, ocams_frame = split_frame(frame)
                frame = cv2.hconcat([picam_frame, ircam_frame, ocams_frame])
                cv2.imshow('dst', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
        except:
            traceback.print_exc()


if __name__ == '__main__':
    # USAGE
    camera_streamer = CameraStreamingClient()
    camera_streamer.connect(server_ip='nano.local', server_port=5000)
    camera_streamer.display()
    camera_streamer.disconnect()

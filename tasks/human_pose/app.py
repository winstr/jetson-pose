from typing import List
import traceback

import cv2
import numpy as np
from flask import Flask, Response
from flask import redirect, url_for

from camera_client import CameraStreamingClient
from pose_estimator import PoseEstimatorDenseNet


bones = [[ 1,  2], [ 1,  3], [ 2,  4], [ 3,  5], [ 6, 18],
         [ 7, 18], [ 6,  8], [ 8, 10], [ 7,  9], [ 9, 11],
         [12,  6], [13,  7], [12, 14], [14, 16], [13, 15],
         [15, 17], [12, 13]]

app = Flask(__name__)

camera_streamer = CameraStreamingClient()
camera_streamer.connect(server_ip='nano.local', server_port=5000)
pose_estimator = PoseEstimatorDenseNet()


def crop_to_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]    # frame height and width
    s = min(w, h)             # short side
    gap_w = int((w - s) / 2)
    gap_h = int((h - s) / 2)
    frame = frame[gap_h:gap_h+s, gap_w:gap_w+s, ...]
    return frame


def resize_to_trtpose(frame: np.ndarray) -> np.ndarray:
    frame = cv2.resize(frame, dsize=(256, 256))  # densenet
    return frame


def draw_pose(frame: np.ndarray, pose: List):
    for start_i, end_i in bones:
        x_start, y_start = pose[start_i - 1]  # pt = point(x, y)
        x_end, y_end = pose[end_i - 1]
        if (x_start, y_start) == (0, 0) or (x_end, y_end) == (0, 0):
            continue
        cv2.line(frame,
                 (x_start.item(), y_start.item()),
                 (x_end.item(), y_end.item()),
                 (0, 255, 0),
                 1,
                 cv2.LINE_AA)


def split_frame(frame):
    # dummy, hardcoding.
    unit = 640
    picam_frame = frame[:, 0:unit, ...]
    ircam_frame = frame[:, unit:unit*2, ...]
    ocams_frame = frame[:, unit*2:unit*3, ...]
    return picam_frame, ircam_frame, ocams_frame

def crop_and_resize(frame):
    frame = crop_to_square(frame)
    frame = resize_to_trtpose(frame)
    return frame


def gen_frames():
    for frame in camera_streamer.stream():
        picam_frame, ircam_frame, ocams_frame = split_frame(frame)

        #frame = crop_to_square(frame)
        #frame = resize_to_trtpose(frame)

        picam_frame = crop_and_resize(picam_frame)
        ircam_frame = crop_and_resize(ircam_frame)
        ocams_frame = crop_and_resize(ocams_frame)

        for frame in [picam_frame, ircam_frame, ocams_frame]:
            poses = pose_estimator.parse_pose(frame)
            for pose in poses:
                draw_pose(frame, pose)

        #poses = pose_estimator.parse_pose(frame)
        #for pose in poses:
        #    draw_pose(frame, pose)

        frame = cv2.hconcat([picam_frame, ircam_frame, ocams_frame])

        _, jpeg_frame = cv2.imencode('.jpg', frame)
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')


@app.route('/camera')
def camera():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return redirect(url_for('camera'))


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5050)
    except:
        traceback.print_exc()
    finally:
        camera_streamer.disconnect()

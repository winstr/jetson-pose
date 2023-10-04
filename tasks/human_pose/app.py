import cv2
from flask import Flask, Response
from flask import redirect, url_for

from pose_estimator import ResNetPoseEstimator


app = Flask(__name__)

cap = cv2.VideoCapture('videos/pedestrians.mp4')
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
s = min(w, h)

w_gap = int((w - s) / 2)
h_gap = int((h - s) / 2)

estimator = ResNetPoseEstimator()


def gen_frames():
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = frame[h_gap:h_gap+s, w_gap:w_gap+s, :]
            estimator.estimate(frame)
            frame = cv2.resize(frame, (s, s))

            ret, jpeg_frame = cv2.imencode('.jpg', frame)
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')
    except:
        pass
    finally:
        cap.release()


@app.route('/video_feed0')
def video_feed0():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return redirect(url_for('video_feed0'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

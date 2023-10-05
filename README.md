# trt_pose

trt_pose는 NVIDIA Jetson에서 실시간으로 자세 추정(pose estimation)을 수행하게 하기 위해 작성되었습니다. 다른 NVIDIA 플랫폼에서도 유용하게 사용할 수 있을 것입니다. 공식 소스코드는 [여기](https://github.com/NVIDIA-AI-IOT/trt_pose)를 참고하세요. 이 프로젝트에는 아래와 같은 내용들이 포함되어 있습니다.

- Jetson Nano에서 실시간으로 사람의 자세를 추정하는, 사전 훈련된 모델들을 이용해 `left_eye`(왼쪽 눈), `left_elbow`(왼쪽 팔꿈치), `right_ankle`(오른쪽 발목) 등 여러 신체 부위들(keypoints)을 손쉽게 감지할 수 있습니다.

- [MSCOCO](https://cocodataset.org/#home) 포맷 기반의 keypoint 데이터와 훈련 스크립트를 통해 사람의 자세 뿐만 아니라, keypoints 감지와 관련된 다양한 작업을 위해 trt_pose를 훈련시키는 것을 실험해볼 수 있습니다.

## 시작하기

시작하려면 아래의 지침을 따르세요.

### 단계 1 - 의존성 설치

1. PyTorch와 Torchvision이 필요합니다. 만약 NVIDIA Jetson에서 일관성있는 실행 환경 구성을 원한다면, 아래와 같이 NGC(NVIDIA GPU Cloud)에서 제공하는 [도커 이미지](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml) 사용을 고려해 보세요.
   
   ```bash
   NAME="r32.7.1-py3"  # for Jetpack 4.6.1 (Jetson Nano)
   DIST="nvcr.io/nvidia/l4t-ml:$NAME"
   docker run -it --runtime nvidia \
       --ipc=host --net=host \
       -v $HOME/Docker/$NAME:/workspace \
       -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
       -e DISPLAY=$DISPLAY --name $NAME $DIST \
       /bin/bash
   ```

2. torch2trt 설치
   
   적당한 위치에 작업 디렉토리를 만들고, 그 아래에 torch2trt를 설치합니다. 여기서는 `/workspace/pose`를 작업 디렉토리로 가정하겠습니다. root 권한으로 실행되므로 필요하다면 sudo를 명령문 앞에 추가하세요.
   
   ```bash
   mkdir -p /workspace/pose
   cd /workspace/pose
   
   git clone https://github.com/NVIDIA-AI-IOT/torch2trt
   cd torch2trt
   python3 setup.py install --plugins
   ```

3. python packages 설치
   
   ```bash
   pip3 install tqdm cython pycocotools matplotlib
   ```

### 단계 2 - trt_pose 설치

```bash
cd /workspace/pose

git clone https://github.com/winstr/trt_pose.git
cd trt_pose
python3 setup.py install
```

### 단계 3 - 주피터 노트북 예제 실행하기

예제 실행을 위해 아래와 같이 MSCOCO 데이터세트로 사전 학습된 두 종류의 모델을 제공하고 있습니다. 이 모델들의 처리 성능은 Jetson Nano와 Jetson Xavier 플랫폼에서 아래와 같습니다.

| Model                              | Jetson Nano | Jetson Xavier | Weights                                                                               |
| ---------------------------------- | ----------- | ------------- | ------------------------------------------------------------------------------------- |
| resnet18_baseline_att_224x224_A    | 22          | 251           | [download (81MB)](https://drive.google.com/open?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd) |
| densenet121_baseline_att_256x256_B | 12          | 101           | [download (84MB)](https://drive.google.com/open?id=13FkJkx7evQ1WwP54UmdiDXWyFMY1OxDU) |

주피터 노트북에서 실시간 카메라 입력을 처리하려면, 아래 단계를 거쳐야 합니다.

1. 위 표에서 제공하는 모델 가중치(weights) 파일을 다운로드 합니다.

2. 다운로드한 파일을 `/workspace/pose/trt_pose/tasks/human_pose/weights`로 옮기세요.

3. [live_demo.ipynb](tasks/human_pose/live_demo.ipynb) 파일을 열어 작업 과정을 참고하세요.
   
   > 용도에 맞도록 커스터마이징이 필요할 수 있습니다.

## 기타 프로젝트

- [trt_pose_hand](http://github.com/NVIDIA-AI-IOT/trt_pose_hand) - trt_pose 기반의 실시간 손(hand) 자세 추정

- [torch2trt](http://github.com/NVIDIA-AI-IOT/torch2trt) - 사용하기 쉬운, Pytorch → TensorRT 변환기

- [JetBot](http://github.com/NVIDIA-AI-IOT/jetbot) - NVIDIA Jetson Nano 기반의 교육용 AI 로봇

- [JetRacer](http://github.com/NVIDIA-AI-IOT/jetracer) - NVIDIA Jetson Nano 기반의 교육용 AI 레이싱 자동차

- [JetCam](http://github.com/NVIDIA-AI-IOT/jetcam) - NVIDIA Jetson Nano 기반의, 사용하기 쉬운 Python 카메라 인터페이스

## 참고

trt_pose 모델은 아래 문서에 영감을 받았으나, 직접적인 복제품은 아닙니다. 모델 구조의 세부 사항에 대해서는 공식 [저장소](https://github.com/NVIDIA-AI-IOT/trt_pose)의 오픈소스 코드와 구성 파일을 검토해 주세요.

* _Cao, Zhe, et al. "Realtime multi-person 2d pose estimation using part affinity fields." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017._

* _Xiao, Bin, Haiping Wu, and Yichen Wei. "Simple baselines for human pose estimation and tracking." Proceedings of the European Conference on Computer Vision (ECCV). 2018._

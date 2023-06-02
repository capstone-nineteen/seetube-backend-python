# seetube-backend-python
시선 데이터, 표정 데이터 분석 및 영상 처리 알고리즘

#### 1. 장면 분할
scene_division.py

![image](https://github.com/capstone-nineteen/seetube-backend-python/assets/71063214/429481eb-cb64-4602-a87a-8b37a95bafa4)


Scene Transition Detection 라이브러리 PySceneDetect 사용
영상을 구성하는 프레임들 중 연속된 프레임의 색상, 채도, 명도 등을 비교. 그 차이가 임계값 초과인 경우 해당 지점을 장면의 전환점으로 식별
전환점을 기준으로 장면을 자동 분할하여 클립 영상 생성.

PySceneDetect : https://github.com/Breakthrough/PySceneDetect

#### 2. 집중도가 높았던 장면, 감정이 감지된 장면
generate_focus_emotion.py


- 집중도가 높았던 장면

시선 (x, y) 좌표와 동공 고정 여부 (Fixation/Saccade) 고려 영상 내부의 한 응시점에 동공이 고정된 경우 집중으로 판단 
장면별 집중률 계산, 평균 집중률이 70%이상인 장면 선정

- 감정이 감지된 장면

장면별로 많은 사람들이 공통으로 느낀 감정과 그 비율 계산, 감정 감지율이 20%이상인 장면 선정

#### 3. 씬스틸러 선정, 쇼츠 생성
sceneStealer_advanced.py

- 씬스틸러 선정

<img src = "https://assets.website-files.com/5f6bc60e665f54db361e52a9/5f6bc60e665f546a6b1e5400_logo_yolo.png" height="30%" width="30%">

딥러닝 기반 객체 탐지 모델, YOLOv5사용. 선정된 장면 영상에서 객체 데이터 수집
매 프레임마다 등장하는 객체들의 바운딩 박스 좌표 저장. 시선 클러스터링에는 DBSCAN 알고리즘 사용. 다량의 시선 데이터를 밀도 기반으로 클러스터링
가장 많은 시선을 포함하는 클러스터의 바운딩 박스 계산하고 '시선이 가장 집중된 구역'으로 선정
이후 시선 집중 구역과 객체의 바운딩 박스의 교차 정도 계산, 가장 많이 교차하는 객체를 씬스틸러로 선정

modified Yolov5 : https://github.com/capstone-nineteen/seeyoutube-backend-yolov5.git

- 씬스틸러 기반 쇼츠 생성

<img src="https://blog.kakaocdn.net/dn/uvneX/btqC4tkVgbD/jmfu1Z5MKiK0w45DPU2vwK/img.png" height="20%" width="20%">

OpenCV를 사용하여 씬스틸러를 확대한 9:16비의 쇼츠 영상 생성




#### 4. 하이라이트 생성
generate_highlights.py


장면별 집중률과 감정 감지율 점수화, 영상 장르와 일치하는 감정 점수에는 가중치 부여
모든 장면을 중요도 점수를 기준으로 정렬하고 상위 5개의 장면을 합쳐 하이라이트 영상으로 재편집.

ffmpeg, moviepy 라이브러리 사용. 

<img src="https://velog.velcdn.com/images/sangbooom/post/35a7bad1-637f-47bb-9bfd-6fab7532d0db/image.jpeg" width="30%" height="30%">

ffmpeg: https://github.com/kkroening/ffmpeg-python
moviepy : https://github.com/kkroening/ffmpeg-python







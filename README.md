# 🎬 Seetube의 영상 분석 알고리즘


#### 1. 장면 분할


<img src = "https://raw.githubusercontent.com/Breakthrough/PySceneDetect/master/docs/img/pyscenedetect_logo_small.png" height="30%" width="30%">



Scene Transition Detection 라이브러리 PySceneDetect 사용
영상을 구성하는 프레임들 중 연속된 프레임의 색상, 채도, 명도 등을 비교. 그 차이가 임계값 초과인 경우 해당 지점을 장면의 전환점으로 식별
전환점을 기준으로 장면을 자동 분할하여 클립 영상 생성.

PySceneDetect : https://github.com/Breakthrough/PySceneDetect

#### 2. 집중도가 높았던 장면, 감정이 감지된 장면



##### 집중도가 높았던 장면

시선 (x, y) 좌표와 동공 고정 여부 (Fixation/Saccade) 고려 영상 내부의 한 응시점에 동공이 고정된 경우 집중으로 판단 
장면별 집중률 계산, 평균 집중률이 70%이상인 장면 선정

##### 감정이 감지된 장면

장면별로 많은 사람들이 공통으로 느낀 감정과 그 비율 계산, 감정 감지율이 20%이상인 장면 선정

#### 3. 씬스틸러 선정, 쇼츠 생성


##### 씬스틸러

매 프레임마다 등장하는 객체들의 바운딩 박스 좌표 저장. 시선 클러스터링에는 DBSCAN 알고리즘 사용. 다량의 시선 데이터를 밀도 기반으로 클러스터링
가장 많은 시선을 포함하는 클러스터의 바운딩 박스 계산하고 '시선이 가장 집중된 구역'으로 선정
이후 시선 집중 구역과 객체의 바운딩 박스의 교차 정도 계산, 가장 많이 교차하는 객체를 씬스틸러로 선정

Yolov5 : https://github.com/capstone-nineteen/seeyoutube-backend-yolov5.git

##### 쇼츠 생성

OpenCV를 사용하여 씬스틸러를 확대한 9:16비의 쇼츠 영상 생성



<img src="https://github.com/capstone-nineteen/seetube-backend-python/assets/65602906/8fead283-baba-4551-9508-1f8133252c66" width = 70% height = 70%>



#### 4. 하이라이트 생성



장면별 집중률과 감정 감지율 점수화, 영상 장르와 일치하는 감정 점수에는 가중치 부여
모든 장면을 중요도 점수를 기준으로 정렬하고 상위 5개의 장면을 합쳐 하이라이트 영상으로 재편집.

ffmpeg, moviepy 라이브러리 사용. 

ffmpeg: https://github.com/kkroening/ffmpeg-python

moviepy : https://github.com/kkroening/ffmpeg-python



<img src="https://github.com/capstone-nineteen/seetube-backend-python/assets/65602906/3aa25028-a9ae-4d8f-91b9-598659445160" width="70%" height="70%">





# ✂ 영상 분석 결과 화면

##### 집중도& 감정이 감지된 장면
https://github.com/capstone-nineteen/seetube-backend-python/assets/71063214/d6f8b052-df0b-4ad6-b330-d6946ad1f756

##### 신스틸러
https://github.com/capstone-nineteen/seetube-backend-python/assets/71063214/a841a4e8-68f4-4a63-96a4-d034a6169615

##### 쇼츠
https://github.com/capstone-nineteen/seetube-backend-python/assets/71063214/9a7094c8-0734-48d4-bc79-8bda3257e02b

##### 하이라이트
https://github.com/capstone-nineteen/seetube-backend-python/assets/71063214/4a120b45-0c15-4886-b430-4d080a8ea6ed





















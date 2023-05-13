from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
from scenedetect.video_splitter import split_video_ffmpeg
from collections import Counter
import cv2
import boto3
import pymysql
import json

# s3연결
access_key = ''
secret_key = ''
bucket_name = 'seetube-videos'

# DB연결
conn = pymysql.connect( 
    user='root',
    password='',
    host='3.39.99.10',
    db='seetube',
    charset='utf8',
    cursorclass=pymysql.cursors.DictCursor)


# S3 리소스 생성
s3_resource = boto3.resource('s3', aws_access_key_id = access_key, aws_secret_access_key = secret_key)

# S3 객체 이름 설정
VIDEO_NAME = '캐논 파워샷 G7 X Mark II 안정환의 파워무비!'
object_name = 'video/' + VIDEO_NAME + '.mp4'

# S3 객체 가져오기
obj = s3_resource.Object(bucket_name, object_name)

# 객체에서 데이터 가져오기
video_data = obj.get()['Body'].read()

# 로컬 파일로 저장하기
with open('my-local-video.mp4', 'wb') as f:
    f.write(video_data)

# 영상 불러오기
VIDEO_PATH = 'my-local-video.mp4'
video = open_video(VIDEO_PATH)

# 디텍터 생성, 임계값 30, 장면 당 최소 5초
fps = cv2.VideoCapture(VIDEO_PATH).get(cv2.CAP_PROP_FPS)
content_detector = ContentDetector(threshold=27, min_scene_len=fps*5)

# Scene Manager 생성
scene_manager = SceneManager()
scene_manager.add_detector(content_detector)

# 디텍트 수행
scene_manager.detect_scenes(video, show_progress=True)
scene_list = scene_manager.get_scene_list()

# 장면 분할 결과 출력
sceneTime = [] #장면 시작하는 시간을 저장하는 리스트
for scene in scene_list:
  start, end = scene
  sceneTime.append(start.get_seconds())

# 썸네일 만들기 (jpg 파일로 저장)
save_images(
    scene_list, # 장면 리스트 [(시작, 끝)]
    video, # 영상
    num_images=1, # 각 장면 당 이미지 개수
    image_name_template='$SCENE_NUMBER', # 결과 이미지 파일 이름
    output_dir='thumbnails') # 결과 디렉토리 이름

# S3에 썸네일 업로드. 파일이름은 '[영상제목]-[시작초].jpg'
s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
for i in range(len(sceneTime)):
  int_time = int(sceneTime[i])
  local_thumbnail_path = 'thumbnails/' + str(i+1).zfill(3) + '.jpg'
  thumbnail_key = 'thumbnails/' + VIDEO_NAME +'-' + str(int_time) + '.jpg'
  with open(local_thumbnail_path, 'rb') as f:
    s3_client.upload_fileobj(f, bucket_name, thumbnail_key, ExtraArgs={'ACL': 'public-read'})


focusSceneAll = [] # 모든 리뷰어의 장면별 시간, 해당 집중도를 저장하는 리스트
emotionSceneAll = [] # 모든 리뷰어의 장면별 시간, 해당 장면에서 느낀 감정, 감정 비율을 저장하는 리스트
videoId = 3 # 리뷰가 완료된 비디오의 id

sql = "SELECT * FROM seetube.watchingInfos WHERE videoId = {}".format(videoId)

with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        output = cur.fetchall() # 딕셔너리로 저장

        l = 0
        while l < len(output):
            result = json.loads(output[l]['watchingInfos']) # json 데이터를 다시 한 번 딕셔너리로 변환
                
            
            c = 0
            i = 0
            while c < len(sceneTime)-1 :
            
                        focusTime = 0
                        emotionList = [] # 리뷰어 한 명의 장면별 시간, 해당 집중도를 저장하는 리스트
                        focuslist = [] # 리뷰어 한 명의 장면별 시간, 해당 장면에서 느낀 감정, 감정 비율을 저장하는 리스트
                        

                        while i < sceneTime[c+1] :

                            if(result[i]['gazeInfo']['eyeMovementState'] == "FIXATION" and result[i]['gazeInfo']['screenState'] == "INSIDE_OF_SCREEN"):
                                focusTime += 1

                            if((float)(result[i]['emotionInfo']['confidencePercentage'])> 40 or result[i]['emotionInfo']['confidencePercentage'] == 0):
                                emotionList.append(result[i]['emotionInfo']['classification'])


                            i += 1
            


                        focuslist.append(l+1) # 리뷰어 순서
                        focuslist.append(sceneTime[c]) # 장면 시작 시간 
                        focuslist.append(sceneTime[c+1]) # 장면 끝나는 시간  

                        # 집중도 시간을 최대 1로 맞춤
                        if focusTime > (sceneTime[c+1]-sceneTime[c]):
                            focusTime = sceneTime[c+1]-sceneTime[c]

                        focuslist.append(focusTime/(sceneTime[c+1]-sceneTime[c])) # 해당 장면 집중률 삽입
                        focusSceneAll.append(focuslist)
                        
                        emotionSceneOne = [] # 모든 리뷰어의 장면별 시간, 해당 감정상태를 저장하는 배열
                        emotionSceneOne.append(l+1) # 리뷰어 순서 
                        emotionSceneOne.append(sceneTime[c]) #장면  시작 시간 
                        emotionSceneOne.append(sceneTime[c+1]) #장면 구간 끝나는 시간 
                        emotionSceneOne.append(emotionList) # 리뷰어 한 명의 장면별 시간, 해당 집중도를 저장하는 리스트
                        emotionSceneAll.append(emotionSceneOne) # 모든 리뷰어의 장면별 시간, 해당 장면에서 느낀 감정, 감정 비율을 저장하는 리스트         


                        c += 1
            
            l += 1



    
        # 집중도 리스트를 딕셔너리로 변환
        focus_dict = {}
        for sub_lst in focusSceneAll:
            key = (sub_lst[1], sub_lst[2])
            if key in  focus_dict:
                 focus_dict[key].append(sub_lst[3])
            else:
                 focus_dict[key] = [sub_lst[3]]

        # 평균을 구하고 집중도 결과 리스트에 저장
        focus_result = [] 
        for key in focus_dict:
            avg = sum(focus_dict[key]) / len(focus_dict[key])
            focus_result.append([key[0], key[1], avg])


        # 감정 상태 8가지로 분류
        target_list = ['neutral', 'angry', 'happy', 'disgust', 'fear', 'sad', 'surprise', 'none']

        emotion_list = []

        # 리뷰어가 해당 장면에서 가장 많이 느낀 감정을 리스트로 저장
        for lst in emotionSceneAll:
            count_dict = {t: lst[3].count(t) for t in target_list} 
            max_count = max(count_dict.values())
            max_word = [t for t, c in count_dict.items() if c == max_count][0]
            emotion_list.append([lst[0], lst[1], lst[2], max_word])

        # 감정 리스트를 딕셔너리로 변환
        emotion_dict = {}
        for sub_lst in emotion_list:
            key = (sub_lst[1], sub_lst[2])
            if key in emotion_dict:
                emotion_dict[key].append(sub_lst[3])
            else:
                emotion_dict[key] = [sub_lst[3]]


           

        # 해당 장면 구간에서 가장 많이 느껴진 감정, 감정률을 계산하여 감정 결과 리스트에 저장
        emotion_result = [] 
        for key, value in emotion_dict.items():
            counter = Counter(value)
            most_common_word, most_common_count = counter.most_common(1)[0]
            emotion_result.append((key[0], key[1], most_common_word, most_common_count / len(value)))
        
        # 감정 상태가 중립인 것을 제외       
        # emotion_result = [r for r in emotion_result if r[2] != 'neutral']

        
        
        #DB에 결과값 INSERT
        for res in emotion_result:
            int_time = int(res[0])
            s3_thumbnail_path = 'thumbnails/' + VIDEO_NAME +'-' + str(int_time) + '.jpg'
            thumbnail_url = f"https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/{s3_thumbnail_path}"
            cur.execute("INSERT INTO emotions (emotionStartTime, emotionEndTime, emotion, emotionRate, videoId, thumbnailURL) VALUES (%s, %s, %s, %s, %s, %s)", (res[0], res[1], res[2], res[3], videoId, thumbnail_url))
        for res in focus_result:
            int_time = int(res[0])
            s3_thumbnail_path = 'thumbnails/' + VIDEO_NAME +'-' + str(int_time) + '.jpg'
            thumbnail_url = f"https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/{s3_thumbnail_path}"
            cur.execute("INSERT INTO focus (focusStartTime, focusEndTime, focusRate, videoId, thumbnailURL) VALUES (%s, %s, %s, %s, %s)", (res[0], res[1], res[2], videoId, thumbnail_url))

    
              

    conn.commit()

   
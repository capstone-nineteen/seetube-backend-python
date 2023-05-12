from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
from scenedetect.video_splitter import split_video_ffmpeg
import cv2
import boto3

#s3연결
access_key = '액세스 키 입력하세요'
secret_key = '비밀 액세스 키 입력하세요'
bucket_name = 'seetube-videos'

# S3 리소스 생성
s3_resource = boto3.resource('s3', aws_access_key_id = access_key, aws_secret_access_key = secret_key)

# S3 객체 이름 설정
VIDEO_NAME = '영상 제목 입력하세요'
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

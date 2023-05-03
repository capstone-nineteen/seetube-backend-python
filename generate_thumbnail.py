import boto3
import os
import subprocess

s3 = boto3.client('s3', aws_access_key_id='액세스키_아이디', aws_secret_access_key='비밀액세스키')

def generate_thumbnail(bucket_name, video_key, thumbnail_key):
    # 1. 원본 영상의 첫 청크만 임시로 다운로드
    chunk_size = 1024 * 1024  # 1 MB
    response = s3.get_object(Bucket=bucket_name, Key=video_key, Range=f'bytes=0-{chunk_size}')
    with open('/tmp/input.mp4', 'wb') as f:
        f.write(response['Body'].read())

    # 2. ffmpeg 사용하여 00:00:01초 지점에서 썸네일 이미지 생성 
    subprocess.call(['/usr/bin/ffmpeg', '-i', '/tmp/input.mp4', '-ss', '00:00:01', '-vframes', '1', '-vf', 'scale=360:-1', '/tmp/output.png']) # 해상도 가로 360픽셀, 원본 영상 비율 유지

    # 3. S3에 썸네일 이미지 업로드
    with open('/tmp/output.png', 'rb') as f:
        s3.upload_fileobj(f, bucket_name, thumbnail_key)
        # 퍼블릭으로 업로드하려면 추가인수 지정
        # s3.upload_fileobj(f, bucket_name, thumbnail_key, ExtraArgs={'ACL': 'public-read'})

##### 사용 예시 #####
generate_thumbnail('seetube-videos', '드라이하는 리트리버 이.것. 해줬더니 자꾸 꾸벅꾸벅 졸아요.mp4', '리트리버_썸네일.png')

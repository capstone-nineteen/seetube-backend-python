import pymysql
import json
import boto3
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import subprocess

# s3연결
access_key = ''
secret_key = ''
bucket_name = 'seetube-videos'

# S3 리소스 생성
s3_resource = boto3.resource('s3', aws_access_key_id = access_key, aws_secret_access_key = secret_key)

# DB연결
conn = pymysql.connect( 
    user='root',
    password='',
    host='3.39.99.10',
    db='seetube',
    charset='utf8',
    cursorclass=pymysql.cursors.DictCursor)

VIDEO_PATH = 'my-local-video.mp4'
VIDEO_NAME = '캐논 파워샷 G7 X Mark II 안정환의 파워무비!'
videoId = 3

with conn.cursor() as cur:
    # 비디오 테이블에서 중요 감정 가져오기
    cur.execute("SELECT importantEmotion FROM seetube.videos WHERE id = %s", (videoId,))
    importantEmotion = cur.fetchone()["importantEmotion"]

    # 집중도, 감정 테이블 join해서 가져오기
    cur.execute("SELECT DISTINCT f.focusStartTime, f.focusEndTime, f.focusRate, e.emotionRate, e.emotion FROM seetube.focus f LEFT OUTER JOIN seetube.emotions e ON f.videoId = e.videoId AND f.focusStartTime = e.emotionStartTime AND f.focusEndTime = e.emotionEndTime WHERE f.videoId = %s", (videoId),)
    result = cur.fetchall()

   
    # 하이라이트 결과 리스트 
    highlight_list = []

    # focus 정보를 이용하여 scene 정보를 추출하는 함수
    def extract_scene(focus):
    # focusRate와 emotionRate를 각각 0.5 비율로 환산한 후 합산
        focus_rate = focus["focusRate"] or 0
        emotion_rate = focus["emotionRate"] or 0
        scene_score = focus_rate * 0.5 + emotion_rate * 0.5
    # importantEmotion과 emotion이 같으면 1.5배 가중치를 곱함
        if focus['emotion'] == importantEmotion:
            scene_score *= 1.5
    # scene 정보를 딕셔너리로 저장하여 scene_list에 추가
        highlight_list.append({'startTime': focus['focusStartTime'], 'endTime': focus['focusEndTime'], 'sceneScore': scene_score})

    # focus 정보를 이용하여 scene 정보 추출
    for focus in result:
        extract_scene(focus)


    merged_scenes = []
    start_time = 0
    for i, scene in enumerate(highlight_list):
        if i == 0:
            end_time = scene['endTime']
        else:
            end_time = scene['endTime'] - highlight_list[i-1]['endTime'] + end_time
    
        merged_scenes.append({'startTime': start_time, 'endTime': end_time})
        start_time = end_time


    # 상위 5개 영상 하이라이트로 재편집
    for i, scene in enumerate(highlight_list[:5]):
            start_time = scene['startTime']
            end_time = scene['endTime']
            output_name = f'scene{i+1}.mp4'
            cmd = f'ffmpeg -i {VIDEO_PATH} -ss {start_time} -t {end_time - start_time} -b:v 3500k -c:v libx264 {output_name}'
            os.system(cmd)
            

    # s3에 하이라이트 영상 업로드
    input_files = "|".join([f"scene{i+1}.mp4" for i in range(5)])
    cmd = f'ffmpeg -i "concat:{input_files}" -c copy highlight_{videoId}.mp4'
    os.system(cmd)

    file_name = f'highlight_{videoId}.mp4'
    s3_path = 'highlight/' + file_name
    s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    
    s3_client.upload_file(file_name, bucket_name, s3_path, ExtraArgs={'ACL': 'public-read'})

    # 하이라이트 썸네일 가져오기
    int_time = int(merged_scenes[0]['startTime'])
    s3_thumbnail_path = 'thumbnails/' + VIDEO_NAME +'-' + str(int_time) + '.jpg'
    thumbnail_url = f"https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/{s3_thumbnail_path}"


    query = """
            INSERT INTO highlights (
                FirstSceneStartTimeInOriginalVideo,
                FirstSceneEndTimeInOriginalVideo,
                FirstSceneStartTimeInHighlight,
                FirstSceneEndTimeInHighlight,
                SecondSceneStartTimeInOriginalVideo,
                SecondSceneEndTimeInOriginalVideo,
                SecondSceneStartTimeInHighlight,
                SecondSceneEndTimeInHighlight,
                ThirdSceneStartTimeInOriginalVideo,
                ThirdSceneEndTimeInOriginalVideo,
                ThirdSceneStartTimeInHighlight,
                ThirdSceneEndTimeInHighlight,
                FourthSceneStartTimeInOriginalVideo,
                FourthSceneEndTimeInOriginalVideo,
                FourthSceneStartTimeInHighlight,
                FourthSceneEndTimeInHighlight,
                FifthSceneStartTimeInOriginalVideo,
                FifthSceneEndTimeInOriginalVideo,
                FifthSceneStartTimeInHighlight,
                FifthSceneEndTimeInHighlight,
                thumbnailURL,
                videoId
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
    
    values = (
        highlight_list[0]['startTime'],
        highlight_list[0]['endTime'],
        merged_scenes[0]['startTime'],
        merged_scenes[0]['endTime'],
        highlight_list[1]['startTime'],
        highlight_list[1]['endTime'],
        merged_scenes[1]['startTime'],
        merged_scenes[1]['endTime'],
        highlight_list[2]['startTime'],
        highlight_list[2]['endTime'],
        merged_scenes[2]['startTime'],
        merged_scenes[2]['endTime'],
        highlight_list[3]['startTime'],
        highlight_list[3]['endTime'],
        merged_scenes[3]['startTime'],
        merged_scenes[3]['endTime'],
        highlight_list[4]['startTime'],
        highlight_list[4]['endTime'],
        merged_scenes[4]['startTime'],
        merged_scenes[4]['endTime'],
        thumbnail_url,
        videoId
    )

    cur.execute(query, values)

    conn.commit()

    print("highlights record inserted.")
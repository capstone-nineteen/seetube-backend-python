import pymysql
import json
import boto3
from moviepy.editor import *
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
VIDEO_NAME = '2022 이화여대 남성교수중창단 입학식 공연 비하인드'
videoId = 5

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
        focus_rate = focus["focusRate"] or 0
        emotion_rate = focus["emotionRate"] or 0

        # importantEmotion과 emotion이 같으면 1.5배 가중치를 곱함
        if focus['emotion'] == importantEmotion:
            emotion_rate *= 1.5

        # scene_score를 계산
        scene_score = (focus_rate * 0.5 + emotion_rate * 0.5) 

        return scene_score
   
    
   

    # focus 정보를 이용하여 scene 정보 추출
    for focus in result:
        scene_score = extract_scene(focus)
         # scene 정보를 딕셔너리로 저장하여 scene_list에 추가
        highlight_list.append({'startTime': focus['focusStartTime'], 'endTime': focus['focusEndTime'], 'focusRate': focus['focusRate'], 'emotionRate': focus['emotionRate'], 'emotion': focus['emotion'], 'sceneScore': scene_score})


    merged_scenes = []
    start_time = 0
    for i, scene in enumerate(highlight_list):
        if i == 0:
            end_time = scene['endTime']
        else:
            end_time = scene['endTime'] - highlight_list[i-1]['startTime'] + end_time
        
        merged_scenes.append({'startTime': round(start_time, 4) , 'endTime': round(end_time, 4)})
        start_time = end_time

     

    # sceneScore를 기준으로 내림차순 정렬
    highlight_list.sort(key=lambda x: x['sceneScore'], reverse=True)

    # 상위 5개 영상 뽑아내기
    top_5_scenes = highlight_list[:5]

    # 시작시간을 기준으로 오름차순 정렬
    top_5_scenes.sort(key=lambda x: x['startTime'])

    # 상위 5개 영상 뽑아내기
    for i, scene in enumerate(top_5_scenes):
            start_time = scene['startTime']
            end_time = scene['endTime']
            output_name = f'scene{i+1}.mp4'
            cmd = f'ffmpeg -i {VIDEO_PATH} -ss {start_time} -t {end_time - start_time} -b:v 3500k -c:v libx264 {output_name}'
            os.system(cmd)

   
   # 영상들을 리스트로 저장
    clips = []
    for i in range(1, 6):
        clip = VideoFileClip(f"scene{i}.mp4")
        clips.append(clip)

    # 영상들을 리스트로 묶어서 concatenate_videoclips 함수를 이용하여 합칩니다.
    final_clip = concatenate_videoclips(clips)

    # 합쳐진 영상 파일을 저장합니다.
    final_clip.write_videofile(f"highlight_{videoId}.mp4",
                                audio_codec='aac',
                                temp_audiofile='temp-audio.m4a', 
                                remove_temp=True)

    file_name = f'highlight_{videoId}.mp4'
    s3_video_path = 'highlight/' + file_name
    s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    
    s3_client.upload_file(file_name, bucket_name, s3_video_path, ExtraArgs={'ACL': 'public-read'})

    thumbnail_url = []

    # 하이라이트 썸네일 가져오기
    for i in range(5):
        int_time = int(highlight_list[i]['startTime'])
        s3_thumbnail_path = 'thumbnails/' + VIDEO_NAME +'-' + str(int_time) + '.jpg'
        thumbnail_url.append(f"https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/{s3_thumbnail_path}")

    # 비디오 url 가져오기
    video_url = f"https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/{s3_video_path}"


    query = """
            INSERT INTO highlights (
                FirstSceneStartTimeInOriginalVideo,
                FirstSceneEndTimeInOriginalVideo,
                FirstSceneStartTimeInHighlight,
                FirstSceneEndTimeInHighlight,
                thumbnailURLInFirstScene,
                focusRateInFirstScene,
                emotionRateInFirstScene,
                emotionInFirstScene,
                SecondSceneStartTimeInOriginalVideo,
                SecondSceneEndTimeInOriginalVideo,
                SecondSceneStartTimeInHighlight,
                SecondSceneEndTimeInHighlight,
                thumbnailURLInSecondScene,
                focusRateInSecondScene,
                emotionRateInSecondScene,
                emotionInSecondScene,
                ThirdSceneStartTimeInOriginalVideo,
                ThirdSceneEndTimeInOriginalVideo,
                ThirdSceneStartTimeInHighlight,
                ThirdSceneEndTimeInHighlight,
                thumbnailURLInThirdScene,
                focusRateInThirdScene,
                emotionRateInThirdScene,
                emotionInThirdScene,
                FourthSceneStartTimeInOriginalVideo,
                FourthSceneEndTimeInOriginalVideo,
                FourthSceneStartTimeInHighlight,
                FourthSceneEndTimeInHighlight,
                thumbnailURLInFourthScene,
                focusRateInFourthScene,
                emotionRateInFourthScene,
                emotionInFourthScene,
                FifthSceneStartTimeInOriginalVideo,
                FifthSceneEndTimeInOriginalVideo,
                FifthSceneStartTimeInHighlight,
                FifthSceneEndTimeInHighlight,
                thumbnailURLInFifthScene,
                focusRateInFifthScene,
                emotionRateInFifthScene,
                emotionInFifthScene,
                videoURL,
                videoId
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
    
    values = (
        top_5_scenes[0]['startTime'],
        top_5_scenes[0]['endTime'],
        merged_scenes[0]['startTime'],
        merged_scenes[0]['endTime'],
        thumbnail_url[0],
        top_5_scenes[0]['focusRate'],
        top_5_scenes[0]['emotionRate'],
        top_5_scenes[0]['emotion'],
        top_5_scenes[1]['startTime'],
        top_5_scenes[1]['endTime'],
        merged_scenes[1]['startTime'],
        merged_scenes[1]['endTime'],
        thumbnail_url[1],
        top_5_scenes[1]['focusRate'],
        top_5_scenes[1]['emotionRate'],
        top_5_scenes[1]['emotion'],
        top_5_scenes[2]['startTime'],
        top_5_scenes[2]['endTime'],
        merged_scenes[2]['startTime'],
        merged_scenes[2]['endTime'],
        thumbnail_url[2],
        top_5_scenes[2]['focusRate'],
        top_5_scenes[2]['emotionRate'],
        top_5_scenes[2]['emotion'],
        top_5_scenes[3]['startTime'],
        top_5_scenes[3]['endTime'],
        merged_scenes[3]['startTime'],
        merged_scenes[3]['endTime'],
        thumbnail_url[3],
        top_5_scenes[3]['focusRate'],
        top_5_scenes[3]['emotionRate'],
        top_5_scenes[3]['emotion'],
        top_5_scenes[4]['startTime'],
        top_5_scenes[4]['endTime'],
        merged_scenes[4]['startTime'],
        merged_scenes[4]['endTime'],
        thumbnail_url[4],
        top_5_scenes[4]['focusRate'],
        top_5_scenes[4]['emotionRate'],
        top_5_scenes[4]['emotion'],
        video_url,
        videoId
    )

    cur.execute(query, values)

    conn.commit()

    

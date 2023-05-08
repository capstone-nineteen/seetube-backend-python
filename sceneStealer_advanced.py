import pymysql
import json
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import boto3
import os

overlappedProportion = []

#객체의 bounding box와 시선의 bounding box의 겹친 영역을 계산하는 함수
#rect1: object, boundingBox_points: 시선 좌표
def compute_intersect_area(rect1, boundingBox_points):
    
    
    x1, y1 = rect1[1], rect1[2] 
    x2, y2 = rect1[3], rect1[4]

    area_rect1 = (x2 - x1) * (y2 - y1)

    area_overlapped = 0

    for i in range(len(boundingBox_points)):

        x3, y3 = boundingBox_points[i][0], boundingBox_points[i][1] 
        x4, y4 = boundingBox_points[i][2], boundingBox_points[i][3]

    

        ## case1 오른쪽으로 벗어나 있는 경우

        if x2 < x3:
            area_overlapped += 0

        ## case2 왼쪽으로 벗어나 있는 경우
        elif x1 > x4:
            area_overlapped += 0

        ## case3 위쪽으로 벗어나 있는 경우
        elif  y2 < y3:
            area_overlapped += 0

        ## case4 아래쪽으로 벗어나 있는 경우
        elif  y1 > y4:
            area_overlapped += 0

        else:
            left_up_x = max(x1, x3)
            left_up_y = max(y1, y3)
            right_down_x = min(x2, x4)
            right_down_y = min(y2, y4)

            width = right_down_x - left_up_x
            height =  right_down_y - left_up_y
  
            area_overlapped += width * height

    return float(area_overlapped / area_rect1)

        
#s3연결
access_key = 'AKIARAMN2Z72HCN6ICFM'
secret_key = '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'
bucket_name = 'seetube-videos'

s3_client = boto3.client('s3', aws_access_key_id = access_key, 
                  aws_secret_access_key = secret_key)

dic = []
#DB연결
connect = pymysql.connect( 
    user='root',
    password='nineteen1919!',
    host='localhost',
    db='seetube',
    charset='utf8',
    cursorclass=pymysql.cursors.DictCursor)

cur = connect.cursor()


videoId = 3

#focus 테이블에서 가장 집중도 높은 장면 4개의 focusRate을 저장
getBestScenesQuery = "SELECT focusStartTime, focusEndTime FROM focus WHERE videoId = %s ORDER BY focusRate DESC LIMIT 4;"

cur.execute(getBestScenesQuery, (videoId,))

results = cur.fetchall()

startTime = [results[i]['focusStartTime'] for i in range(4)]
endTime = [results[i]['focusEndTime'] for i in range(4)]

print(startTime)
print(endTime)








 






#objectdetection 테이블에서 row 개수(=object 개수)를 불러와 count_obejcts 변수에 저장 
getCountOfObjectDetectionQuery = "SELECT COUNT(*) FROM objectdetection;"

cur.execute(getCountOfObjectDetectionQuery)

result = cur.fetchone()

count_objects = result['COUNT(*)']

objects = []

#objectdetection 테이블에서 각 object마다 장면의 time, x1, y1, x2, y2 좌표 값을 저장한 boundingBox_object 객체를
#boundingBox_objects 배열에 저장
#전체 영상을 colab에서 돌려야 할듯
getObjectBoundingBoxQuery = "SELECT * FROM objectdetection"

cur.execute(getObjectBoundingBoxQuery)

results = cur.fetchall()

objects.append(results)

boundingBox_objects = []

for i in range(count_objects):
    
    time = objects[0][i]['time']
    x1_tensor = objects[0][i]['x1']
    y1_tensor = objects[0][i]['y1']
    x2_tensor = objects[0][i]['x2']
    y2_tensor = objects[0][i]['y2']
    
    x1 = int(re.sub(r'[^0-9]','', x1_tensor))
    y1 = int(re.sub(r'[^0-9]','', y1_tensor))
    x2 = int(re.sub(r'[^0-9]','', x2_tensor))
    y2 = int(re.sub(r'[^0-9]','', y2_tensor))


    boundingBox_object = [time, x1, y1, x2, y2]

    boundingBox_objects.append(boundingBox_object)

    

#watchingInfos 테이블에서 해당 영상에 해당하는 리뷰(watchingInfo)개수를 count에 저장
watchingInfos = []

getCountOfWatchingInfoQuery = "SELECT COUNT(*) FROM watchingInfos WHERE videoId = %s;"

cur.execute(getCountOfWatchingInfoQuery, videoId)

result = cur.fetchone()

count = result['COUNT(*)']



#watchingInfos 테이블에서 video의 watchingInfos 가져오기
getWatchingInfoQuery = "SELECT watchingInfos FROM watchingInfos WHERE videoId = %s;"

cur.execute(getWatchingInfoQuery, videoId)

results = cur.fetchall()

watchingInfos = []

watchingInfos.append(results)



#비디오 불러오기

# S3 리소스 생성
s3_resource = boto3.resource('s3')

# S3 객체 이름 설정
object_name = 'video/캐논 파워샷 G7 X Mark II 안정환의 파워무비!.mp4'

# S3 객체 가져오기
obj = s3_resource.Object(bucket_name, object_name)

# 객체에서 데이터 가져오기
video_data = obj.get()['Body'].read()

# 로컬 파일로 저장하기
with open('my-local-video.mp4', 'wb') as f:
    f.write(video_data)

video = cv2.VideoCapture('my-local-video.mp4')

# 프레임 레이트, 프레임 사이즈, 총 프레임 수 가져오기
fps = video.get(cv2.CAP_PROP_FPS)
frame_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# 추출할 프레임 간격 계산 (1초에 해당하는 프레임 수)
interval = int(fps)

# 저장할 비디오 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, frame_size)

# 1초 간격으로 프레임 추출하여 저장
for i in range(0, frame_count, interval):
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = video.read()
    if ret:
        out.write(frame)

# 종료
video.release()
out.release()


i = 0

#4개의 씬스틸러, 쇼츠를 만들기 위해 4번 반복
for i in range(4):

    video = cv2.VideoCapture('my-local-video.mp4')

    if not video.isOpened():
        print("Could not Open")
        exit(0)

    # 시작 프레임 위치 계산
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    start_frame = int(startTime[i] * frame_rate)

    # 종료 프레임 위치 계산
    end_frame = int(endTime[i] * frame_rate)

    # 자를 프레임 수 계산
    frame_count = end_frame - start_frame

    
    # 프레임 스킵
    q = 0
    for q in range(start_frame):
        video.read()

    # 프레임 읽기 및 저장
    frames = []
    w = 0
    for w in range(frame_count):
        ret, frame = video.read()
        if ret:
            frames.append(frame)
        else:
            break

    

    # 자른 프레임을 새로운 동영상으로 저장하기
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('focused_video_'+ str(i+1) + '.mp4', fourcc, frame_rate, frames[0].shape[:2][::-1])
    for frame in frames:
        output_video.write(frame)

    output_video.release()

    video.release()

    video = cv2.VideoCapture('./focused_video_'+ str(i+1)+ '.mp4')

    if not video.isOpened():
        print("Could not Open")
        exit(0)




    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)




    #시선 좌표 클러스터링하여 매 초마다 객체의 bounding box와 시선 좌표의 bounding box 영역 비교
    #watchingInfo는 2초부터 시작
    #j = 0 -> playTime = 2
    #playTime = 13 -> j = 11
    for j in range(startTime[i], endTime[i]+1):

        xs = [] #해당 장면에서 모든 시청자들의 시선 좌표 중 x값들
        ys = [] #해당 장면에서 모든 시청자들의 시선 좌표 중 y값들

        boundingBox_points = []
    
        for k in range(count):
        
            json_watchingInfo = watchingInfos[0][k]

            watchingInfo = json.loads(json_watchingInfo['watchingInfos'])
        
            x = watchingInfo[j-2]['gazeInfo']['x']
            y = watchingInfo[j-2]['gazeInfo']['y']
        
            xs.append(x)
            ys.append(y)

        gaze_data = list(zip(xs, ys))
        print(gaze_data)
        dbscan = DBSCAN(eps=0.07, min_samples=3) #값은 추후에 조정
        labels = dbscan.fit_predict(gaze_data)


        # Plot 디버깅 그래프 그리기
        fig, ax = plt.subplots()

        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1: # 어느 클러스터에도 속하지 않는 점
                continue

            cluster_points = []
            
            for t in range(len(xs)):
                # label번 클러스터에 속한 점만 필터링
                if labels[t] == label:
                    # 정수형으로 넘겨주어야 바운딩박스 계산이 가능하기 때문에 형변환해준다
                    x = int(gaze_data[k][0]*10000000)
                    y = int(gaze_data[k][1]*10000000)
                    cluster_points.append((x, y))

            # 바운딩 박스 계산
            x, y, w, h = cv2.boundingRect(np.array(cluster_points))

            # 다시 float으로 형변환
            x = float(x) / 10000000
            y = float(y) / 10000000
            w = float(w) / 10000000
            h = float(h) / 10000000

            print("x: ", x)
            print("y: ", y)
            print("w: ", w)
            print("h: ", h)            
        
        # 클러스터 개수 출력
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Number of clusters found: {n_clusters}")



        rect = patches.Rectangle((x, y), w, h, linewidth=0.1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)


        # 시선점들 그리기
        ax.scatter(xs, ys, c=labels, cmap='rainbow')
        plt.show()

        box_x = x * width
        box_y = y * height
        box_w = w * width
        box_h = h * height

        gaze_box = [box_x, box_y, box_x+box_w, box_y+box_h]

        boundingBox_points.append(gaze_box)

        s = 0

        #객체의 bounding box와 시선 좌표의 bounding box 겹치는 영역 계산
        for s in range(count_objects):

            if boundingBox_objects[s][0] == j:
                overlapped_proportion = compute_intersect_area(boundingBox_objects[s], boundingBox_points)
                boundingBox_objects[s].append(overlapped_proportion)


    print(boundingBox_objects)

    #max는 중간 변수
    max_value = 0
    #overlapped_proportion이 가장 큰 boundingBox_object의 index
    max_index = 0

    p = 0

    for p in range(len(boundingBox_objects)):
        if boundingBox_objects[p][0] >= startTime[i] and boundingBox_objects[p][0] <= endTime[i]: 
            if boundingBox_objects[p][5] > max_value:
                max_value = boundingBox_objects[p][5]
                max_index = i

    print(boundingBox_objects[max_index])
    best_frame_time = boundingBox_objects[max_index][0]

    # best frame을 추출해 씬스틸러를 crop하는 코드
    best_frame_number = int(round(best_frame_time * fps))

    video.set(cv2.CAP_PROP_POS_FRAMES, best_frame_number)
    ret, frame = video.read()

    cv2.imwrite('./best_frame_'+str(i+1)+'.png', frame)

    # 비디오 위치를 첫 번째 프레임으로 설정
    # video.set(cv2.CAP_PROP_POS_FRAMES, 0)  






    image = Image.open("best_frame_"+str(i+1)+".png")

    box = [boundingBox_objects[max_index][1], boundingBox_objects[max_index][2], boundingBox_objects[max_index][3], boundingBox_objects[max_index][4]]
    region = image.crop(box)
    region.save('./scene_stealer_'+str(i+1)+'.png')

    #씬스틸러 이미지 s3에 업로드
    file_name = 'scene_stealer_'+str(i+1)+'.png'
    s3_path = 'sceneStealer/' + file_name

    s3_client.upload_file('./scene_stealer_'+str(i+1)+'.png', bucket_name, s3_path)


    #씬스틸러의 bounding box좌표를 중심으로 영상을 확대해서 crop
    x1 = boundingBox_objects[max_index][1]
    y1 = boundingBox_objects[max_index][2]
    x2 = boundingBox_objects[max_index][3]
    y2 = boundingBox_objects[max_index][4]

    xOfCenter = (x1 + x2) / 2
    yOfCenter = (y1 + y2) / 2

    print("scene stealer width:", region.width)
    print("scene stealer height:", region.height)

    object_area = (x2 - x1) * (y2 - y1)

    percentage = (float)(object_area / (width * height))
    print("percentage: ", percentage)

    #숫자 3을 조정
    a = (int)(((object_area * ((1/percentage)/3))/(9*16))**(1/2))

    # if object_area / (width * height) < (float)(1/10):
    #     a = (int)(((object_area * 15)/(9*16))**(1/2))
    # elif object_area / (width * height) > (float)(1/6):
    #     a = (int)(((object_area * 2)/(9*16))**(1/2))
    # else:
    #     a = (int)(((object_area * 4)/(9*16))**(1/2))
    

    print("a: ", a)
    x = (int)(xOfCenter - 4.5 * a)
    y = (int)(yOfCenter - 8 * a)

    if x < 0 : x = 0
    if y < 0 : y = 0

    w = (int)(9*a)
    h = (int)(16*a)

    if (x + w) > width : w = (int)(width - x)
    if (y + h) > height : h = (int)(height - y)

    print("width:", width)
    print("height:", height)
    print("x:", x)
    print("y:", y)
    print("w:", w)
    print("h:", h)


    roi = (x, y, w, h)

    fps = video.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter('./shorts_'+str(i+1)+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))



    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            cropped = frame[y:y+h, x:x+w]

            out.write(cropped)

            cv2.imshow('cropped', cropped)

            if cv2.waitKey(25) * 0xFF == ord('q'):
                break

        else:
            break

    video.release()
    out.release()

    #쇼츠 s3에 업로드
    file_name = 'shorts_'+str(i+1)+'.mp4'
    s3_path = 'shorts/' + file_name

    s3_client.upload_file('./shorts_'+str(i+1)+'.mp4', bucket_name, s3_path)



    








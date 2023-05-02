import pymysql
import json
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re

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

#focus 테이블에서 가장 높은 집중도를 뽑아내 max_focusRate 변수에 저장
getMaxFocusRateQuery = "SELECT MAX(focusRate) FROM focus WHERE videoId = %s ;"

cur.execute(getMaxFocusRateQuery, videoId)

result = cur.fetchone()

max_focusRate = result['MAX(focusRate)'] 

print(max_focusRate)

#focus 테이블에서 가장 집중도 높은 장면의 startTime과 endTime을 각각 변수에 저장
getBestSceneQuery = "SELECT * FROM focus WHERE videoId = %s AND focusRate >= %s;"

cur.execute(getBestSceneQuery, (videoId, max_focusRate))

results = cur.fetchall()

dic.append(results)



scene = dic[0][0]

startTime = scene['focusStartTime'] 

endTime = scene['focusEndTime'] 






#objectdetection 테이블에서 row 개수(=object 개수)를 불러와 count_obejcts 변수에 저장 
getCountOfObjectDetectionQuery = "SELECT COUNT(*) FROM objectdetection;"

cur.execute(getCountOfObjectDetectionQuery)

result = cur.fetchone()

count_objects = result['COUNT(*)']

objects = []


#objectdetection 테이블에서 각 object마다 장면의 time, x1, y1, x2, y2 좌표 값을 저장한 boundingBox_object 객체를
#boundingBox_objects 배열에 저장
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



#비디오 불러와서 비디오 length, 프레임의 width, height, fps 값 저장
filepath = './focused_scene.mp4'
video = cv2.VideoCapture(filepath)

if not video.isOpened():
    print("Could not Open : ", filepath)
    exit(0)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)




#시선 좌표 클러스터링하여 매 초마다 객체의 bounding box와 시선 좌표의 bounding box 영역 비교
for j in range(startTime, endTime + 1):

    xs = [] #해당 장면에서 모든 시청자들의 시선 좌표 중 x값들
    ys = [] #해당 장면에서 모든 시청자들의 시선 좌표 중 y값들

    boundingBox_points = []
    
    for i in range(count):
        
        json_watchingInfo = watchingInfos[0][i]

        watchingInfo = json.loads(json_watchingInfo['watchingInfos'])
        
        x = watchingInfo[j]['gazeInfo']['x']
        y = watchingInfo[j]['gazeInfo']['y']
        
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
            
        for k in range(len(xs)):
            # label번 클러스터에 속한 점만 필터링
            if labels[k] == label:
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

        

    #객체의 bounding box와 시선 좌표의 bounding box 겹치는 영역 계산
    for s in range(count_objects):

        if boundingBox_objects[s][0] == j - startTime:
            overlapped_proportion = compute_intersect_area(boundingBox_objects[s], boundingBox_points)
            boundingBox_objects[s].append(overlapped_proportion)


print(boundingBox_objects)

#max는 중간 변수
max = 0
#overlapped_proportion이 가장 큰 boundingBox_object의 index
max_index = 0

for i in range(len(boundingBox_objects)):
    if boundingBox_objects[i][5] > max:
            max = boundingBox_objects[i][5]
            max_index = i

print(boundingBox_objects[max_index])
best_frame_time = boundingBox_objects[max_index][0]

# best frame을 추출해 씬스틸러를 crop하는 코드
best_frame_number = int(round(best_frame_time * fps))

video.set(cv2.CAP_PROP_POS_FRAMES, best_frame_number)
ret, frame = video.read()

cv2.imwrite('./best_frame.png', frame)

video.release()

#time = 1 -> startTime + 1 = 13
# 13초 이미지

image = Image.open("best_frame.png")

box = [boundingBox_objects[max_index][1], boundingBox_objects[max_index][2], boundingBox_objects[max_index][3], boundingBox_objects[max_index][4]]
region = image.crop(box)
region.save('./scene_stealer.png')

    






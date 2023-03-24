import pymysql
import json
from collections import Counter

#DB연결
conn = pymysql.connect( 
    user='root',
    password='password',
    host='host',
    db='seetube',
    charset='utf8',
    cursorclass=pymysql.cursors.DictCursor)


videoId = 4
sceneTime = [0, 4, 10, 15, 18, 20] #장면 시작하는 시간을 저장하는 리스트
emotionSceneAll = [] #모든 리뷰어의 장면별 시간, 해당 장면에서 느낀 감정, 감정 비율을 저장하는 리스트

sql = "SELECT * FROM seetube.watchingInfos WHERE videoId = {}".format(videoId)


with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        output = cur.fetchall() #딕셔너리로 저장

        l = 0
        while l < len(output):
            result = json.loads(output[l]['watchingInfos']) #json 데이터를 다시 한 번 딕셔너리로 변환
                
            
            c = 0
            i = 0
            while c < len(sceneTime)-1 :
            
                        
                        emotionList = []
                        

                        while i < sceneTime[c+1] :


                            if((float)(result[i]['emotionInfo']['confidencePercentage'])> 40 or result[i]['emotionInfo']['confidencePercentage'] == 0):
                                emotionList.append(result[i]['emotionInfo']['classification'])


                            i += 1
            

                        
                        emotionSceneOne = [] #리뷰어 한 명의 데이터를 저장하는 리스트
                        emotionSceneOne.append(l+1)
                        emotionSceneOne.append(sceneTime[c]) #장면  시작 시간 
                        emotionSceneOne.append(sceneTime[c+1]) #장면 구간 끝나는 시간 
                        emotionSceneOne.append(emotionList)
                        emotionSceneAll.append(emotionSceneOne)          


                        c += 1
            
            l += 1



        target_list = ['neutral', 'angry', 'happy', 'disgust', 'fear', 'sad', 'surprise', 'none']


        emotion_list = []


        #리뷰어 한명당 해당 장면에서 가장 많이 느낀 감정을 리스트로 저장
        for lst in emotionSceneAll:
            count_dict = {t: lst[3].count(t) for t in target_list} 
            max_count = max(count_dict.values())
            max_word = [t for t, c in count_dict.items() if c == max_count][0]
            emotion_list.append([lst[0], lst[1], lst[2], max_word])
     

        emotion_dict = {}

        #모든 리뷰어들의 결과 값을 해당 장면 구간 별로 정렬해 딕셔너리로 저장
        for sub_lst in emotion_list:
            key = (sub_lst[1], sub_lst[2])
            if key in emotion_dict:
                emotion_dict[key].append(sub_lst[3])
            else:
                emotion_dict[key] = [sub_lst[3]]


       

        emotion_result = [] #장면별 제일 많이 느껴진 감정, 감정률을 저장하는 최종 결과값 리스트
           

        #해당 장면 구간에서 가장 많이 느껴진 감정, 감정률을 계산하여 저장
        for key, value in emotion_dict.items():
            counter = Counter(value)
            most_common_word, most_common_count = counter.most_common(1)[0]
            emotion_result.append((key[0], key[1], most_common_word, most_common_count / len(value)))

        
        emotion_result_list = [r for r in emotion_result if r[2] != 'neutral']

        
        
        #DB에 결과값 INSERT
        for res in emotion_result_list:
            cur.execute("INSERT INTO emotions (emotionStartTime, emotionEndTime, emotion, emotionRate, videoId) VALUES (%s, %s, %s, %s, %s)", (res[0], res[1], res[2], res[3], videoId))
           
            
    conn.commit()

    print("emotion record inserted.")

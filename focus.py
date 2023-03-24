import pymysql
import json

#DB연결
conn = pymysql.connect( 
    user='root',
    password='password',
    host='host',
    db='seetube',
    charset='utf8',
    cursorclass=pymysql.cursors.DictCursor)


videoId = 4
sceneTime = [0, 4, 14, 20, 22, 30] #장면 시작하는 시간을 저장하는 리스트
focusScene = [] #모든 리뷰어의 장면별 시간, 해당 집중도를 저장하는 리스트

sql = "SELECT * FROM seetube.watchingInfos WHERE videoId = {}".format(videoId)


with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        output = cur.fetchall() #딕셔너리로 저장

        l = 0
        while l < len(output):
            result = json.loads(output[l]['watchingInfo']) #json 데이터를 다시 한 번 딕셔너리로 변환
                
            
            c = 0
            i = 0
            while c < len(sceneTime)-1 :
            
                        focusTime = 0
                        emotionList = []
                        

                        while i < sceneTime[c+1] :

                            if(result['watchingInfos'][i]['gazeInfo']['eyeMovementState'] == "FIXATION" and result['watchingInfos'][i]['gazeInfo']['screenState'] == "INSIDE_OF_SCREEN"):
                                focusTime += 1


                            i += 1
            


                        focuslist = []


                        focuslist.append(l+1) #리뷰어 순서
                        focuslist.append(sceneTime[c]) #장면  시작 시간 
                        focuslist.append(sceneTime[c+1]) #장면 구간 끝나는 시간 
                        focuslist.append(focusTime/(sceneTime[c+1]-sceneTime[c])) #해당 장면 집중률
                        focusScene.append(focuslist)
                                 


                        c += 1
            
            l += 1

    
    
        #집중도 딕셔너리 생성
        dict_lst = {}
        for sub_lst in focusScene:
            key = (sub_lst[1], sub_lst[2])
            if key in dict_lst:
                dict_lst[key].append(sub_lst[3])
            else:
                dict_lst[key] = [sub_lst[3]]

        #집중도의 평균을 구하고 결과 리스트에 저장
        focus_result = []
        for key in dict_lst:
            avg = sum(dict_lst[key]) / len(dict_lst[key])
            if avg >= 0.7:
                focus_result.append([key[0], key[1], avg])


        for res in focus_result:
            cur.execute("INSERT INTO focus (focusStartTime, focusEndTime, focusRate) VALUES (%s, %s, %s)", (res[0], res[1], res[2]))
            


    conn.commit()

    print("focus record inserted.")

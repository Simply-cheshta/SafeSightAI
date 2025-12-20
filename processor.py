from ultralytics import YOLO
import cv2
import json
from datetime import datetime

model=YOLO("yolov8n.pt")

results=model.predict(source="datasets/crowd.mp4",stream=True,classes=[0])
all_logs=[]
for result in results:
    frame=result.orig_img
    frame_data={}
    frame_data['timestamp']=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    boxes=result.boxes
    frame_data['person_count']=len(boxes)
    frame_data['detections']=[]

    for box in boxes:
        cords=box.xyxy[0].tolist()
        class_id=box.cls[0].item()
        conf=box.conf[0].item()
        x1,y1,x2,y2=map(int,cords)
        detection={
            'coordinates':[x1,y1,x2,y2],
            'confidence':round(conf,2)
        }
        frame_data['detections'].append(detection)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        label=f'Person:{conf:.2f}'
        cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        print(f'Object:{class_id},Conf:{conf:.2f},Corners:{cords}')
    all_logs.append(frame_data)
    cv2.imshow("Detections",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
with open('detection_logs.json','w') as f:
    json.dump(all_logs,f,indent=4)

print(f"\nLogs saved,total frames processed:{len(all_logs)}")


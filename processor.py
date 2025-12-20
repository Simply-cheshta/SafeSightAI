from ultralytics import YOLO
import cv2
import json
from datetime import datetime

model=YOLO("yolov8n.pt")

results=model.predict(source="datasets/crowd.mp4",stream=True,classes=[0])
all_logs=[]
CROWD_LIMIT=10
for result in results:
    frame=result.orig_img
    frame_data={}
    frame_data['timestamp']=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    boxes=result.boxes
    person_count = len(boxes)
    frame_data['person_count']=len(boxes)
    if person_count>CROWD_LIMIT:
        status="DANGER"
        color=(0,0,255)
        alert_triggered=True
    else:
        status="SAFE"
        color=(0,255,0)
        alert_triggered=False
    frame_data['status']=status
    frame_data['alert_triggered']=alert_triggered
    frame_data['crowd_limit']=CROWD_LIMIT
    frame_data['detections']=[]

    banner_height=60
    overlay=frame.copy()
    cv2.rectangle(overlay,(0,0),(frame.shape[1],banner_height),color,-1)
    cv2.addWeighted(overlay,0.7,frame,0.3,0,frame)
    status_text = f"STATUS: {status}"
    count_text = f"People: {person_count}/{CROWD_LIMIT}"
    
    cv2.putText(frame, status_text, (20, 35), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, count_text, (frame.shape[1] - 280, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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

total_frames = len(all_logs)
danger_frames = sum(1 for log in all_logs if log['alert_triggered'])
safe_frames = total_frames - danger_frames
max_people = max(log['person_count'] for log in all_logs) if all_logs else 0

print(f"\n{'='*50}")
print(f"SECURITY LOGS SAVED")
print(f"{'='*50}")
print(f"Total frames processed: {total_frames}")
print(f"Safe frames: {safe_frames}")
print(f"Danger frames: {danger_frames}")
print(f"Crowd limit: {CROWD_LIMIT}")
print(f"Max people detected: {max_people}")
print(f"{'='*50}\n")

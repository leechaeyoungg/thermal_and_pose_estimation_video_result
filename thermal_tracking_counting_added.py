import os
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

# 모델 경로
detection_model_path = r"D:\IR_people_model_thermal_dataset3.pt"
pose_model_path = r"D:\yolo11l-pose.pt"

# 입력 영상 경로 및 출력 경로
input_video_path = r"C:\Users\dromii\Downloads\thermal_short.mp4"
output_video_path = r"C:\Users\dromii\Downloads\thermal_combined_tracking_6.mp4"

# COCO 포맷에 따른 키포인트 연결 정보 및 색상
skeleton = {
    "face": [(0, 1), (1, 3), (0, 2), (2, 4)],
    "arms": [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)],
    "legs": [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)],
    "torso": [(5, 11), (6, 12), (5, 6), (11, 12)]
}

colors = {
    "face": (0, 255, 0),
    "arms": (255, 0, 0),
    "legs": (0, 0, 255),
    "torso": (0, 255, 255)
}

# 모델 로드
detection_model = YOLO(detection_model_path)
pose_model = YOLO(pose_model_path)

# 객체 추적 정보 저장
trackers = {}  # {object_id: deque([(x, y), ...])}
object_id_counter = 0
tracked_ids = set()  # 이미 카운팅된 객체 ID 저장
alpha = 0.6  # 텍스트 배경 투명도 설정

# 유클리드 거리 계산
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# 영상 로드
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 결과 저장용 VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 원본 프레임 복사본 생성
    detection_frame = frame.copy()
    pose_frame = frame.copy()

    # 열영상 객체 탐지 수행
    detection_results = detection_model.predict(detection_frame, conf=0.5, iou=0.5, save=False, verbose=False)

    # 현재 프레임 내 탐지된 객체 수
    frame_object_count = len(detection_results[0].boxes)

    # 열영상 객체 탐지 및 경로 추적 처리
    new_trackers = {}
    for det_result in detection_results[0].boxes:
        x1, y1, x2, y2 = map(int, det_result.xyxy[0])
        confidence = det_result.conf[0]
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # 바운딩 박스와 라벨 표시
        label = f"Person {confidence:.2f}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x, text_y = x1, y1 - 10

        # 텍스트 배경 투명화 처리
        overlay = frame.copy()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(overlay, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y), (0, 255, 0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 기존 객체와 매칭
        matched = False
        for obj_id, path in trackers.items():
            if path and euclidean_distance(path[-1], (center_x, center_y)) < 50:
                new_trackers[obj_id] = path
                new_trackers[obj_id].append((center_x, center_y))
                matched = True
                break

        # 새로운 객체로 등록
        if not matched:
            object_id_counter += 1
            new_trackers[object_id_counter] = deque([(center_x, center_y)], maxlen=50)

    # 경로 그리기
    for obj_id, path in new_trackers.items():
        for i in range(1, len(path)):
            if path[i - 1] is None or path[i] is None:
                continue
            cv2.line(frame, path[i - 1], path[i], (255, 255, 0), 2)

    # 업데이트된 추적 정보 저장
    trackers = new_trackers

    # 화면 오른쪽 상단에 카운팅 결과 표시
    overlay = frame.copy()
    text = f"Count: {frame_object_count}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = width - text_size[0] - 20
    text_y = 30
    cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10), (width - 10, text_y + 10), (255, 255, 255), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # 포즈 추정 수행
    pose_results = pose_model.predict(pose_frame, conf=0.5, save=False, verbose=False)

    # 포즈 추정 결과 표시
    for result in pose_results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy() if hasattr(result.keypoints.xy, 'cpu') else np.array(result.keypoints.xy)

            # 키포인트 표시 및 연결
            for obj_keypoints in keypoints:
                for x, y in [(int(k[0]), int(k[1])) for k in obj_keypoints if k[0] != 0 and k[1] != 0]:
                    cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

                # 부위별 선 연결
                for part, connections in skeleton.items():
                    color = colors[part]
                    for start, end in connections:
                        if start < len(obj_keypoints) and end < len(obj_keypoints):
                            x1, y1 = obj_keypoints[start]
                            x2, y2 = obj_keypoints[end]
                            if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # 결과 프레임 저장
    out.write(frame)

# 리소스 정리
cap.release()
out.release()
print(f"Processed video saved at: {output_video_path}")

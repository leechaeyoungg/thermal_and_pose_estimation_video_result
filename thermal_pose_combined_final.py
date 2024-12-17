import os
from ultralytics import YOLO
import cv2
import numpy as np

# 모델 경로
detection_model_path = r"D:\IR_people_model_thermal_dataset3.pt"
pose_model_path = r"D:\yolo11l-pose.pt"

# 입력 영상 경로 및 출력 경로
input_video_path = r"C:\Users\dromii\Downloads\thermal_short.mp4"
output_video_path = r"C:\Users\dromii\Downloads\thermal_combined_result_upgrade.mp4"

# 모델 로드
detection_model = YOLO(detection_model_path)
pose_model = YOLO(pose_model_path)

# COCO 포맷에 따른 키포인트 연결 정보 및 색상
skeleton = {
    "face": [(0, 1), (1, 3), (0, 2), (2, 4)],
    "arms": [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)],
    "legs": [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)],
    "torso": [(5, 11), (6, 12), (5, 6), (11, 12)]
}

# 색상 정의 (BGR)
colors = {
    "face": (0, 255, 0),
    "arms": (255, 0, 0),
    "legs": (0, 0, 255),
    "torso": (0, 255, 255)
}

# 텍스트 배경 투명도 설정
alpha = 0.6

# 영상 로드
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 결과 저장용 VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 영상 프레임 순차 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 원본 프레임 복사본 생성 (간섭 방지)
    detection_frame = frame.copy()
    pose_frame = frame.copy()

    # 1단계: 열영상 탐지 수행
    detection_results = detection_model.predict(detection_frame, conf=0.5, iou=0.5, save=False, verbose=False)

    # 탐지 결과에서 바운딩 박스 표시
    for det_result in detection_results[0].boxes:
        x1, y1, x2, y2 = map(int, det_result.xyxy[0])
        confidence = det_result.conf[0]
        label = f"Person {confidence:.2f}"

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 텍스트 배경 투명화 처리
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x, text_y = x1, y1 - 10
        background_box = frame[text_y - text_size[1]:text_y, text_x:text_x + text_size[0]]

        # 텍스트 배경 생성
        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 255, 0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # 텍스트 그리기
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 2단계: 포즈 추정 수행
    pose_results = pose_model.predict(pose_frame, conf=0.5, save=False, verbose=False)

    # 포즈 추정 결과를 원본 프레임에 표시
    for result in pose_results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy() if hasattr(result.keypoints.xy, 'cpu') else np.array(result.keypoints.xy)

            # 다차원 배열로 구성된 키포인트를 객체별로 처리
            for obj_idx, obj_keypoints in enumerate(keypoints):
                valid_keypoints = [(int(x), int(y)) for x, y in obj_keypoints if not (x == 0 and y == 0)]

                # 키포인트 표시
                for x, y in valid_keypoints:
                    cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

                # 부위별로 선 연결 및 색상 적용
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

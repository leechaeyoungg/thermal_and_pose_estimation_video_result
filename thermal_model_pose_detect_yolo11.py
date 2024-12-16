import os
from ultralytics import YOLO
import cv2
import numpy as np

# 모델 경로
detection_model_path = r"D:\IR_people_model_thermal_dataset3.pt"
pose_model_path = r"D:\yolo11l-pose.pt"

# 입력 영상 경로 및 출력 경로
input_video_path = r"C:\Users\dromii\Downloads\thermal_train_result_1080.mp4"
output_video_path = r"C:\Users\dromii\Downloads\thermal_pose_result.mp4"

# 탐지 모델 로드
detection_model = YOLO(detection_model_path)

# 포즈 추정 모델 로드
pose_model = YOLO(pose_model_path)

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

    # 1단계: 사람 탐지
    detection_results = detection_model.predict(frame, conf=0.5, iou=0.5, save=False, verbose=False)

    # 탐지된 영역에서 모든 바운딩 박스 추출
    boxes = []
    for result in detection_results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # 바운딩 박스 좌표 추출
        boxes.append((x1, y1, x2, y2))

    # 2단계: 포즈 추정
    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = box
            cropped_object = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]  # 탐지된 영역 크롭

            # 크롭된 이미지가 유효한지 확인
            if cropped_object.size == 0 or cropped_object.shape[0] < 1 or cropped_object.shape[1] < 1:
                continue  # 유효하지 않은 경우 건너뜀

            # 흑백 이미지를 3채널로 변환 (열 영상일 경우 필요)
            if len(cropped_object.shape) == 2:  # 1채널 이미지인 경우
                cropped_object = cv2.cvtColor(cropped_object, cv2.COLOR_GRAY2BGR)

            # 크롭된 이미지를 모델에 전달
            pose_results = pose_model.predict(source=[cropped_object], conf=0.5, save=False, verbose=False)

            # 포즈 추정 결과를 원본 프레임에 표시
            if pose_results and hasattr(pose_results[0], 'keypoints') and len(pose_results[0].keypoints) > 0:
                for pose in pose_results[0].keypoints:
                    keypoints = pose.xy.cpu().numpy() if hasattr(pose.xy, 'cpu') else np.array(pose.xy)
                    if keypoints is not None and len(keypoints) > 0:  # 키포인트가 비어있지 않은지 확인
                        for keypoint in keypoints:
                            if len(keypoint) == 2:  # 키포인트가 올바른 형식인지 확인
                                kp_x, kp_y = int(keypoint[0] + x1), int(keypoint[1] + y1)  # 원본 좌표로 변환
                                cv2.circle(frame, (kp_x, kp_y), 3, (0, 255, 0), -1)

    # 결과 프레임 저장
    out.write(frame)

# 리소스 정리
cap.release()
out.release()
print(f"Processed video saved at: {output_video_path}")

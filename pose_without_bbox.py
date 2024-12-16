import os
from ultralytics import YOLO
import cv2
import numpy as np

# 모델 경로
pose_model_path = r"D:\yolo11l-pose.pt"

# 입력 영상 경로 및 출력 경로
input_video_path = r"C:\Users\dromii\Downloads\thermal_short.mp4"
output_video_path = r"C:\Users\dromii\Downloads\thermal_pose_result_edit6_final.mp4"

# 포즈 추정 모델 로드
pose_model = YOLO(pose_model_path)

# COCO 포맷에 따른 키포인트 연결 정보 및 색상
skeleton = {
    "face": [(0, 1), (1, 3), (0, 2), (2, 4)],  # 얼굴
    "arms": [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)],  # 팔
    "legs": [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)],  # 다리
    "torso": [(5, 11), (6, 12), (5, 6), (11, 12)]  # 몸통
}

# 색상 정의 (BGR)
colors = {
    "face": (0, 255, 0),  # 초록색
    "arms": (255, 0, 0),  # 파란색
    "legs": (0, 0, 255),  # 빨간색
    "torso": (0, 255, 255)  # 노란색
}

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

    # 포즈 추정 수행
    pose_results = pose_model.predict(frame, conf=0.5, save=False, verbose=False)

    # 포즈 추정 결과를 원본 프레임에 표시
    for result in pose_results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            # 키포인트 데이터 추출 및 유효성 확인
            keypoints = result.keypoints.xy.cpu().numpy() if hasattr(result.keypoints.xy, 'cpu') else np.array(result.keypoints.xy)

            # 다차원 배열로 구성된 키포인트를 객체별로 처리
            for obj_idx, obj_keypoints in enumerate(keypoints):
                valid_keypoints = [(int(x), int(y)) for x, y in obj_keypoints if not (x == 0 and y == 0)]

                # 키포인트 표시
                for x, y in valid_keypoints:
                    cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)  # 모든 키포인트 흰색으로 표시

                # 부위별로 선 연결 및 색상 적용
                for part, connections in skeleton.items():
                    color = colors[part]
                    for start, end in connections:
                        if start < len(obj_keypoints) and end < len(obj_keypoints):
                            x1, y1 = obj_keypoints[start]
                            x2, y2 = obj_keypoints[end]
                            # 유효한 키포인트 간에만 연결
                            if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # 결과 프레임 저장
    out.write(frame)

# 리소스 정리
cap.release()
out.release()
print(f"Processed video saved at: {output_video_path}")

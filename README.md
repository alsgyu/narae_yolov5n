# YOLOv5n 기반 Hailo NPU 객체 탐지 모델

본 프로젝트는 **드론 환경에서의 특수 마커(cross_marker, v_marker, tray) 인식**을 위해 YOLOv5n 모델을 학습하고, 최종적으로 **Hailo-8 NPU에 탑재 가능한 ONNX 모델로 변환**하는 전 과정을 포함합니다.

---

## 📂 프로젝트 구조
```
hailo_YOLOv5n/
├── notebook/                  # 코랩 학습 노트북
│   └── train_marker.ipynb
├── models/                    # 모델 가중치 파일
│   ├── best.pt
│   └── best.onnx
├── raspberry_pi/              # 라즈베리파이 + Hailo 실행 코드
│   └── run_inference.py
├── marker_augmentation/       # 증강 데이터셋 (marker.yaml 포함)
└── results/                   # 시각화된 결과
└── val_pred_vs_label.png
```
---

## 1. 학습 요약

- **모델:** YOLOv5n
- **프레임워크:** Ultralytics YOLOv5
- **클래스:** `cross_marker`, `v_marker`, `tray` 총 3개
- **입력 이미지 사이즈:** 640x640
- **데이터셋 구성:**  
  - Train: 210장  
  - Val: 60장  
  - Test: 30장

학습 결과 (tray 클래스 기준):
- Precision: 0.999  
- Recall: 1.000  
- mAP@0.5: 0.995  
- mAP@0.5:0.95: 0.937

---

## 2. 학습 및 변환 코드

- 학습은 `notebooks/train_marker.ipynb`에서 수행
- 변환된 ONNX 모델은 `weights/best.onnx`

```python
# PyTorch -> ONNX 변환 예시 코드
!python export.py \
  --weights weights/best.pt \
  --include onnx \
  --img 640
```

## 3. 라즈베리파이 + Hailo 실행

raspberry_pi/run_inference.py는 라즈베리파이 환경에서 Hailo 모델을 실행하는 스크립트입니다.

Hailo SDK 설치 및 hefs_compile 또는 hailo_model_zoo에서 best.onnx를 hef 파일로 변환해야 합니다.


## 4. 테스트 결과

 결과 이미지는 results/ 폴더에 저장되어 있으며, val_batchX_pred.jpeg, val_batchX_labels.jpeg 형태입니다.


## 📁 프로젝트 구성 파일 설명

| 파일/폴더명                        | 설명 |
|-----------------------------------|------|
| `data_train/`                     | LSTM 모델 학습에 사용되는 수어 데이터셋입니다. |
| `data_val/`                       | 모델 검증을 위한 수어 데이터셋입니다. |
| `label_mapping.json`              | 수어 단어 코드와 단어명을 매핑한 JSON 파일입니다. |
| `LSTM.py`                         | LSTM 모델의 학습 및 검증을 위한 파이썬 스크립트입니다. |
| `best_model.pt`                   | 검증 성능이 가장 뛰어난 시점의 학습된 LSTM 모델 가중치 파일입니다. |
| `realtime_translation_multiwindow.py` | 학습된 모델을 기반으로 웹캠 입력을 받아 실시간으로 수어 단어를 추론하는 코드입니다. |
| `Presentation.pdf` | Final report 입니다.|
| `demoVideo.mp4` | 실시간 수어 번역하는 데모영상입니다. |

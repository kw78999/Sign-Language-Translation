data_train : 학습용 데이터셋입니다.
data_val : 검증용 데이터셋입니다.
label_mapping.json : 수어단어코드와 단어명을 매핑하는 json 파일입니다.
LSTM.py : LSTM 모델의 학습 및 검증코드입니다.
best_model.pt : 학습된 LSTM 모델의 best 버전입니다.
realtime_translation_multiwindow.py : 학습된 LSTM 모델에 기반하여 실시간 웹캠을 통해 수어 단어를 추론합니다.

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Nadam
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import datetime

# 데이터 로드
input_data = pd.read_csv('tw_data_preprocessed.csv')

# 입력 변수와 타겟 변수 분리
x_data = input_data.drop(['인구수'], axis=1)
y_data = input_data[['인구수']]

# 입력 데이터의 차원
x_data_dim = len(x_data.columns)

# 데이터를 학습 세트와 테스트 세트로 분할
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777)

# 모델 구성
model = Sequential()
model.add(Dense(1024, input_dim=x_data_dim))
model.add(Activation('relu'))  # LeakyReLU를 ReLU로 변경
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))  # LeakyReLU를 ReLU로 변경
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))  # LeakyReLU를 ReLU로 변경
model.add(Dropout(0.2))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer=Nadam(), loss='mean_squared_error')

# 텐서보드 로그 디렉토리 설정
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 체크포인트 저장 디렉토리 설정
checkpoint_dir = os.path.join("checkpoints", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 체크포인트 콜백 설정
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5"),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='auto'
)

# 모델 학습
model.fit(
    x_train, 
    y_train, 
    epochs=512, 
    validation_data=(x_test, y_test), 
    callbacks=[tensorboard_callback, checkpoint_callback]
)

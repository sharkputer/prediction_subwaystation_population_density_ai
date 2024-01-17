from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from keras.initializers import he_normal
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import datetime
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 데이터 로드
input_data = pd.read_csv('tw_data_preprocessed.csv')



# 처음 100000개의 샘플만 선택
total_data_count = input_data.shape[0]
# 총 데이터 수 : 4739398
print(f'총 데이터 수 : {total_data_count}')
#input_data = input_data.head(1000000)

# 입력 변수와 타겟 변수 분리
x_data = input_data.drop(['인구수', '강수량_mm', '적설_cm'], axis=1)
y_data = input_data[['인구수']]

# VIF 계산
vif_data = pd.DataFrame()
vif_data["feature"] = x_data.columns
vif_data["VIF"] = [variance_inflation_factor(x_data.values, i) for i in range(len(x_data.columns))]

print(vif_data)

# 입력 데이터의 차원
x_data_dim = len(x_data.columns)

# 데이터를 학습 세트와 테스트 세트로 분할
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777)

# 모델 구성
model = Sequential()
model.add(Dense(512, input_dim=x_data_dim, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(64, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(1))  # 회귀 문제이므로 활성화 함수 없이 출력 레이어

# 모델 컴파일 - 최적화 알고리즘을 Adam으로 변경
model.compile(optimizer=Adam(), loss='mean_squared_error')

# 텐서보드 로그 디렉토리 설정
log_dir = os.path.join("logs_Adam", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 체크포인트 저장 디렉토리 설정
checkpoint_dir = os.path.join("checkpoints_Adam", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
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
    epochs=200,  # 에포크 수를 200으로 증가
    batch_size=64,  # 배치 크기를 64로 증가
    validation_data=(x_test, y_test), 
    callbacks=[tensorboard_callback, checkpoint_callback]
)

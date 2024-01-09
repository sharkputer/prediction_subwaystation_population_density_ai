from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd

input_data = pd.read_csv('tw_data_preprocessed.csv')
print(len(input_data.drop(['인구수'], axis=1).columns()))

# model = Sequential()
# model.add(Dense(128, input_dim=input_dim, activation='relu'))  # input_dim은 입력 특성의 수
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))  # 회귀 문제이므로 활성화 함수 없이 하나의 노드

# model.compile(optimizer='adam', loss='mean_squared_error')
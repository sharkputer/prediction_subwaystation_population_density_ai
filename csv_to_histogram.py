import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


plt.rcParams['axes.unicode_minus'] = False  # matplotlib 마이너스기호 표시
plt.rc('font', family='NanumGothic')  # matplotlib 한글폰트 표시

# pandas
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy import stats
from scipy.stats import yeojohnson
import os
import joblib

# 람다 값을 저장할 폴더 경로
folder_path = 'PreprocessingValues'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# data
df = pd.read_csv('transit_weather_data.csv')
df.fillna(0, inplace=True)

encoder = OneHotEncoder(sparse=False)

# 역명
one_hot_encoded = encoder.fit_transform(df[['역명']])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(['역명']))
#df = df.join(one_hot_encoded_df)
#df = df.drop('역명', axis=1)
joblib.dump(encoder, 'PreprocessingValues\\encoder_역명.joblib')

# # 요일
one_hot_encoded = encoder.fit_transform(df[['요일']])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(['요일']))
#df = df.join(one_hot_encoded_df)
#df = df.drop('요일', axis=1)
joblib.dump(encoder, 'PreprocessingValues\\encoder_요일.joblib')

# # 시간
one_hot_encoded = encoder.fit_transform(df[['시간']])
one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(['시간']))
#df = df.join(one_hot_encoded_df)
#df = df.drop('시간', axis=1)
joblib.dump(encoder, 'PreprocessingValues\\encoder_시간.joblib')

# 인구수
df.loc[df['인구수'] == 0, '인구수'] = 0.001
transformed_data, lambda_value = stats.boxcox(df['인구수'])
df['인구수'] = pd.Series(transformed_data)
with open('PreprocessingValues/boxcox_인구수_lambda.txt', 'w') as file:
    file.write(str(lambda_value))

# 강수량
df.loc[df['강수량_mm'] == 0, '강수량_mm'] = 0.001
df['강수량_mm'], lambda_value = yeojohnson(df['강수량_mm'])
# transformed_data, lambda_value = stats.boxcox(df['강수량_mm'])
#df['강수량_mm'] = pd.Series(transformed_data)
with open('PreprocessingValues/boxcox_강수량_mm_lambda.txt', 'w') as file:
    file.write(str(lambda_value))

# 습도
scaler = MinMaxScaler()
df['습도_퍼센트'] = scaler.fit_transform(df[['습도_퍼센트']])
joblib.dump(scaler, 'PreprocessingValues\\scaler_습도_퍼센트.joblib')

#df.to_csv('tw_data_preprocessing.csv')
# draw histogram graph
# df.hist(bins=100, figsize=(10,10))

# df.loc[df['인구수'] == 0, '인구수'] = 0.001

# import numpy as np
# df['인구수'] = np.log(df['인구수'])

#print(df['강수량_mm'].describe())

# df.hist(bins=100, figsize=(10,10))

# from scipy import stats
# df['인구수'] = pd.Series(stats.boxcox(df['인구수'])[0])


#print(encoder.categories_)

# df.hist(bins=100, figsize=(10,10))
# plt.show()
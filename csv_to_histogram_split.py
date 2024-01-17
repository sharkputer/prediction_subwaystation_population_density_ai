import pandas as pd
import joblib
from scipy import stats
import os

# 전체 행 수 계산
total_rows = sum(1 for _ in open('transit_weather_data.csv', 'r', encoding='utf-8')) - 1  # 첫 번째 행(헤더) 제외

# 전처리에 필요한 값들을 파일에서 불러오기
encoder_역명 = joblib.load('PreprocessingValues\\encoder_역명.joblib')
encoder_요일 = joblib.load('PreprocessingValues\\encoder_요일.joblib')
encoder_시간 = joblib.load('PreprocessingValues\\encoder_시간.joblib')
scaler_습도_퍼센트 = joblib.load('PreprocessingValues\\scaler_습도_퍼센트.joblib')

with open('PreprocessingValues/boxcox_인구수_lambda.txt', 'r') as file:
    lambda_인구수 = float(file.read())

with open('PreprocessingValues/boxcox_강수량_mm_lambda.txt', 'r') as file:
    lambda_강수량_mm = float(file.read())

# CSV 파일을 청크 단위로 읽기
chunk_size = 100000
chunks = pd.read_csv('transit_weather_data.csv', chunksize=chunk_size)

preprocessed_dfs = []  # 전처리된 데이터를 저장할 리스트
processed_rows = 0  # 처리된 행 수

for df in chunks:
    # 전처리 과정 (이전 코드와 동일)

    df.fillna(0, inplace=True)

    # 역명
    one_hot_encoded = encoder_역명.transform(df[['역명']])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder_역명.get_feature_names_out(['역명']))
    df = df.join(one_hot_encoded_df)
    df.drop('역명', axis=1, inplace=True)

    # 요일
    one_hot_encoded = encoder_요일.transform(df[['요일']])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder_요일.get_feature_names_out(['요일']))
    df = df.join(one_hot_encoded_df)
    df.drop('요일', axis=1, inplace=True)

    # 시간
    one_hot_encoded = encoder_시간.transform(df[['시간']])
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=encoder_시간.get_feature_names_out(['시간']))
    df = df.join(one_hot_encoded_df)
    df.drop('시간', axis=1, inplace=True)

    # 인구수
    df.loc[df['인구수'] == 0, '인구수'] = 0.001
    df['인구수'] = stats.boxcox(df['인구수'], lmbda=lambda_인구수)

    # 강수량
    df.loc[df['강수량_mm'] == 0, '강수량_mm'] = 0.001
    df['강수량_mm'] = stats.yeojohnson(df['강수량_mm'], lmbda=lambda_강수량_mm)

    # 습도
    df['습도_퍼센트'] = scaler_습도_퍼센트.transform(df[['습도_퍼센트']])

    preprocessed_dfs.append(df)
    processed_rows += len(df)

    # 진행 상황 출력
    progress = (processed_rows / total_rows) * 100
    print(f"Progress: {progress:.2f}%")

# 모든 청크를 하나의 DataFrame으로 결합
final_df = pd.concat(preprocessed_dfs, ignore_index=True)
final_df.fillna(0, inplace=True)

# 전처리된 데이터 저장
final_df.to_csv('tw_data_preprocessed.csv', index=False)

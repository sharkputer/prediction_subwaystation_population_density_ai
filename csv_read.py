import pandas as pd

df = pd.read_csv('transit_weather_data.csv')
df.fillna(0, inplace=True)

x_data = df.drop(['역명', '인구수'], axis=1)
y_data = df[['인구수']]

print(x_data)
print(y_data)
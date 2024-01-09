import pandas as pd
import pymysql

from datetime import datetime

print('START TIME : ',str(datetime.now())[10:19] )

conn=pymysql.connect(host='ictsk.myqnapcloud.com',port=9301,user='root',password='dhfpswl8282',db='subwaydb')

sql = "select DISTINCT "\
"transit_weather_data.역명 as 역명, "\
"DAYOFWEEK(transit_weather_data.날짜) AS 요일, "\
"DATE_FORMAT(transit_weather_data.시간, '%H') as 시간,"\
"transit_weather_data.승하차_인원 as 인구수, "\
"transit_weather_data.기온_섭씨 as 기온_섭씨, "\
"transit_weather_data.강수량_mm as 강수량_mm, "\
"transit_weather_data.풍속_ms as 풍속_ms, "\
"transit_weather_data.습도_퍼센트 as 습도_퍼센트, "\
"transit_weather_data.일조_hr as 일조_hr, "\
"transit_weather_data.적설_cm as 적설_cm "\
"from "\
"station_locations, transit_weather_data "\
"WHERE "\
"station_locations.역번호 = transit_weather_data.역번호; "

#print(sql)

df = pd.read_sql_query(sql, conn)
df.to_csv(r'transit_weather_data.csv', index=False)

print('END TIME : ',str(datetime.now())[10:19] )

conn.close()
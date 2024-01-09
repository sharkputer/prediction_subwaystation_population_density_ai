import os

# 람다 값을 저장할 폴더 경로
folder_path = 'boxcox'

# 폴더가 없으면 생성
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open('boxcox/강수량_mm_lambda.txt', 'w') as file:
    file.write(str(123))
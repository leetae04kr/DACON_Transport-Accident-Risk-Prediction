import pandas as pd

file_path = "data/train/B.csv"

# 파일의 처음 일부만 읽어서 구조 파악 (메모리 절약)
sample = pd.read_csv(file_path, nrows=5)

print("파일 샘플 미리보기:")
print(sample)

# 열 정보
print("\n열 이름 목록:")
print(sample.columns.tolist())

# 전체 행 수는 이렇게 추정 가능
with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    row_count = sum(1 for line in f) - 1  # 헤더 제외
print(f"\n예상 행 개수: {row_count}")

print(f"열 개수: {len(sample.columns)}")

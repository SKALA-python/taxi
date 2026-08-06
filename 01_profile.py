"""
yellow_tripdata_2026-05.csv 원본 진단 (전처리 전 EDA)
=====================================================

[목적]
  전처리 규칙을 "감으로" 정하지 않기 위해 먼저 원본의 실제 상태를 측정한다.
  이 스크립트는 데이터를 전혀 수정하지 않는다(read-only). 여기서 나온 수치가
  02_preprocess.py의 임계값(운행시간 상한, 거리 상한 등)과 결측 대체 전략의
  근거가 된다.

[확인 항목과 이유]
  [1] 규모        - 메모리 전략(dtype 축소 필요 여부)을 정하기 위해
  [2] 결측        - 어떤 컬럼이 얼마나 비었는지, 결측이 특정 행에 몰려 있는지
  [3] 기술통계    - min/max에서 물리적으로 불가능한 값을 찾기 위해
  [4] 기간        - 파일명(2026-05)과 실제 데이터 기간이 맞는지
  [5] 이상치 후보 - 각 제거 규칙이 몇 건에 해당하는지 사전 측정
                    (제거량을 모르고 규칙을 적용하면 데이터를 통째로
                     날릴 위험이 있다)
  [6] 범주형 분포 - 코드북에 없는 값(예: RatecodeID=99)이 있는지
"""
import pandas as pd
import numpy as np

SRC = "data/yellow_tripdata_2026-05.csv"

# dtype 명시: 409만 행을 기본 int64/float64로 읽으면 메모리가 수 GB로 커진다.
# 결측이 있는 정수 컬럼은 pandas nullable 정수형(Int16)이어야 결측을 float
# 변환 없이 유지할 수 있다(결측 규모를 정확히 세기 위해 중요).
dtypes = {
    "VendorID": "Int16", "passenger_count": "Int16", "trip_distance": "float32",
    "RatecodeID": "Int16", "store_and_fwd_flag": "object",
    "PULocationID": "Int16", "DOLocationID": "Int16", "payment_type": "Int16",
    "fare_amount": "float32", "extra": "float32", "mta_tax": "float32",
    "tip_amount": "float32", "tolls_amount": "float32",
    "improvement_surcharge": "float32", "total_amount": "float32",
    "congestion_surcharge": "float32", "Airport_fee": "float32",
    "cbd_congestion_fee": "float32",
}

df = pd.read_csv(SRC, dtype=dtypes,
                 parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])

pd.set_option("display.width", 200, "display.max_columns", 50)

print("=" * 78)
print(f"[1] SHAPE  rows={len(df):,}  cols={df.shape[1]}")
print("=" * 78)

# n_unique를 함께 보는 이유: 고유값이 2~7개면 수치형으로 저장돼 있어도
# 실제로는 범주형(코드값)이라는 뜻이다. 평균·표준편차를 계산해도 무의미하고,
# groupby 대상으로 봐야 한다.
print("\n[2] DTYPES / 결측치")
info = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "n_missing": df.isna().sum(),
    "pct_missing": (df.isna().mean() * 100).round(3),
    "n_unique": df.nunique(dropna=True),
})
print(info.to_string())

# min/max를 보는 것이 핵심 목적. 1%/99% 분위수를 함께 찍는 이유는
# "극단값이 소수의 오류인지, 아니면 꼬리가 원래 두꺼운지"를 구분하기 위해서다.
# 99%와 max의 격차가 크면 소수의 오류, 완만하면 실제 롱테일 분포다.
print("\n[3] 수치형 기술통계")
num = df.select_dtypes(include=["number"]).columns
# astype("float64"): float32/Int16이 섞이면 describe 출력 서식이 어긋난다.
print(df[num].astype("float64").describe(percentiles=[.01, .25, .5, .75, .99]).T
      .to_string(float_format="{:.3f}".format))

# 여러 컬럼의 결측 건수가 정확히 같다면(955,371) 우연이 아니라 같은 행에서
# 함께 비어 있다는 뜻이다. 원인이 특정 벤더의 수집 누락인지 확인한다.
# -> 벤더가 분산돼 있으면 벤더 오류가 아니라 리포팅 포맷 차이이며,
#    그 경우 삭제하면 특정 세그먼트가 통째로 빠지므로 대체 전략을 써야 한다.
print("\n[3-1] 결측 955,371행의 정체 (VendorID 교차표)")
miss = df.passenger_count.isna()
print(pd.crosstab(df.VendorID, miss, dropna=False).to_string())

# 파일명이 2026-05인데 실제 pickup이 그 밖에 있다면 미터기 시각 오류다.
print("\n[4] 기간 범위")
print(f"  pickup : {df.tpep_pickup_datetime.min()}  ~  {df.tpep_pickup_datetime.max()}")
print(f"  dropoff: {df.tpep_dropoff_datetime.min()}  ~  {df.tpep_dropoff_datetime.max()}")
out_of_month = ~df.tpep_pickup_datetime.between("2026-05-01", "2026-06-01")
print(f"  2026-05 밖 pickup: {out_of_month.sum():,}")

# 각 항목은 02_preprocess.py의 제거 규칙 후보와 1:1로 대응한다.
# 여기서 비율을 먼저 확인해 "이 규칙을 쓰면 몇 %가 사라지는가"를 판단한다.
# 비율이 크면(예: 20%+) 제거 대신 대체·플래그 전략으로 전환한다.
print("\n[5] 이상치 후보 카운트")
dur = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
checks = {
    "완전 중복 행":            df.duplicated().sum(),
    "dropoff <= pickup":       (dur <= 0).sum(),          # 시각 역전(불가능)
    "운행시간 > 24h":          (dur > 1440).sum(),        # 상한 후보 탐색용
    "운행시간 < 1분":          ((dur > 0) & (dur < 1)).sum(),   # 승차 직후 취소
    "trip_distance <= 0":      (df.trip_distance <= 0).sum(),   # 미터기 거리 미측정
    "trip_distance > 100mi":   (df.trip_distance > 100).sum(),  # GPS 오류
    "fare_amount < 0":         (df.fare_amount < 0).sum(),      # 환불/회계 조정
    "total_amount <= 0":       (df.total_amount <= 0).sum(),
    "tip_amount < 0":          (df.tip_amount < 0).sum(),
    "passenger_count == 0":    (df.passenger_count == 0).sum(), # 미기재로 추정
    "passenger_count > 6":     (df.passenger_count > 6).sum(),  # 법정 정원 초과
    "RatecodeID 비정상(1~6외)": (~df.RatecodeID.isin([1, 2, 3, 4, 5, 6])).sum(),
    "payment_type 비정상(1~6외)": (~df.payment_type.isin([1, 2, 3, 4, 5, 6])).sum(),
}
for k, v in checks.items():
    print(f"  {k:<28} {v:>10,}  ({v / len(df) * 100:6.3f}%)")

# 코드북에 정의되지 않은 값의 존재와 규모를 확인한다.
# (RatecodeID=99는 TLC 정의상 Unknown, payment_type=0은 신규 Flex Fare 코드)
print("\n[6] 범주형 분포")
for c in ["VendorID", "RatecodeID", "payment_type", "store_and_fwd_flag"]:
    print(f"\n-- {c}")
    print(df[c].value_counts(dropna=False).head(10).to_string())

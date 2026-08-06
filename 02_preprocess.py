"""
yellow_tripdata_2026-05.csv 전처리 파이프라인
=============================================

[목적]
  NYC TLC 옐로우 택시 원본 로그(409만 건)를 분석/모델링에 바로 쓸 수 있는
  형태로 정제한다. 원본은 미터기·결제단말에서 자동 수집된 raw 로그라
  (1) 물리적으로 불가능한 값, (2) 환불·분쟁 등 회계 조정 행,
  (3) 신규 리포팅 포맷으로 인한 대량 결측이 섞여 있다.

[3가지 처리 원칙]
  1) 제거 - "실제 운행이 아니거나 물리적으로 불가능한 행"만 제거한다.
     단순히 값이 크다는 이유(고액 요금, 장거리)로는 지우지 않는다.
     통계적 이상치 제거(예: IQR 3배)는 적용하지 않았다. 택시 수요는
     본래 롱테일 분포라 상위 꼬리를 자르면 공항/장거리 트립이라는
     실제 수요 세그먼트가 통째로 사라지기 때문이다.
  2) 보존 - 결측 23.4%는 삭제하지 않고 대체 + is_*_imputed 플래그를 남긴다.
     이 결측은 무작위(MCAR)가 아니라 payment_type=0(Flex Fare)이라는
     특정 세그먼트에 몰려 있어, 삭제하면 해당 세그먼트가 표본에서
     통째로 빠지는 선택 편향(selection bias)이 생긴다.
     플래그를 남기는 이유는 후속 분석에서 "대체값 때문에 나온 결과"인지
     검증(민감도 분석)할 수 있어야 하기 때문이다.
  3) 기록 - 모든 제거는 사유별 건수를 로그로 출력한다. 재현성과
     "얼마나 버렸는지"에 대한 설명 책임을 위해서다.

[입출력]
  IN : data/yellow_tripdata_2026-05.csv        (432MB, 4,090,836행)
  OUT: data/yellow_tripdata_2026-05_clean.parquet (117MB, 3,899,953행 x 32컬럼)
"""
import pandas as pd
import numpy as np

SRC = "data/yellow_tripdata_2026-05.csv"
# parquet 선택 이유: 컬럼 타입(dtype)이 파일에 함께 저장돼 재로딩 시
# 타입 추론이 불필요하고, 컬럼 단위 압축으로 용량이 1/4로 줄며,
# 후속 분석에서 필요한 컬럼만 읽을 수 있어 I/O가 빠르다.
OUT = "data/yellow_tripdata_2026-05_clean.parquet"

# ---------------------------------------------------------------------------
# 필터 임계값 - 각 값의 근거
# ---------------------------------------------------------------------------
# 파일명이 2026-05이므로 이 달에 시작한 운행만 유효하다. 원본에는 2008년
# pickup 등 미터기 시각 오류로 보이는 행이 14건 섞여 있다.
MONTH_START = pd.Timestamp("2026-05-01")
MONTH_END = pd.Timestamp("2026-06-01")

# DUR_MIN=1분: 1분 미만은 승객이 타자마자 취소했거나 미터기 오조작으로 본다.
#   (실제 이동으로 보기 어렵고, 거리 대비 시간이 0에 가까워 속도가 폭발한다)
# DUR_MAX=360분(6시간): 뉴욕 시내 운행에서 6시간은 비현실적이다.
#   흔히 쓰는 3시간 대신 6시간으로 느슨하게 잡은 이유는, 협상요금
#   (RatecodeID=5)으로 처리되는 교외 장거리 트립을 살리기 위해서다.
#   6시간 초과는 963건뿐이라 느슨하게 잡아도 노이즈 유입이 미미하다.
DUR_MIN, DUR_MAX = 1.0, 360.0

# DIST_MAX=100마일: 맨해튼에서 100마일이면 필라델피아 근처다. 그 이상은
#   GPS 오류로 판단. 원본 최대값이 307,491마일(지구 12바퀴)로 명백한 오류다.
DIST_MAX = 100.0

# SPEED_MAX=80mph: 뉴욕 지역 최고 제한속도(고속도로 65mph)를 여유 있게
#   상회하는 값. 이를 넘으면 거리 또는 시간 기록 중 하나가 잘못된 것이다.
#   거리·시간을 각각 검사해도 못 잡는 "둘 다 그럴듯하지만 조합이 불가능한"
#   행을 걸러내는 교차검증용 필터다.
SPEED_MAX = 80.0

# ---------------------------------------------------------------------------
# 로딩 - dtype을 명시하는 이유
#   (1) 409만 행에서 pandas 기본 int64/float64를 쓰면 메모리가 수 GB로 커진다.
#       Int16/float32/int8로 낮춰 절반 이하로 줄인다.
#   (2) passenger_count 등은 결측이 있어 numpy int로는 담기지 않는다.
#       pandas nullable 정수형(Int16)을 써야 결측을 NaN 변환 없이 유지한다.
#   (3) 타입 추론에 맡기면 청크마다 다른 타입이 추론될 위험이 있다.
# ---------------------------------------------------------------------------
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

print("[load] reading csv ...")
# parse_dates: 문자열로 읽은 뒤 변환하면 중간에 400만 개의 파이썬 문자열
# 객체가 생성돼 메모리·시간이 낭비된다. 읽는 단계에서 바로 datetime64로 변환.
df = pd.read_csv(SRC, dtype=dtypes,
                 parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
n0 = len(df)   # 제거율 계산의 기준이 되는 원본 행 수
print(f"[load] {n0:,} rows")

# 컬럼명 정규화: 원본은 VendorID(파스칼) / passenger_count(스네이크) /
# PULocationID(대문자 약어)가 뒤섞여 있어 코드에서 오타가 나기 쉽다.
# 전부 snake_case로 통일해 이후 df.컬럼명 접근을 일관되게 만든다.
df = df.rename(columns={
    "VendorID": "vendor_id", "RatecodeID": "ratecode_id",
    "PULocationID": "pu_location_id", "DOLocationID": "do_location_id",
    "Airport_fee": "airport_fee",
    "tpep_pickup_datetime": "pickup_dt", "tpep_dropoff_datetime": "dropoff_dt",
})

# ---------------------------------------------------------------------------
# 필터용 파생변수
#   운행시간·평균속도는 "최종 산출물"이자 동시에 "이상치 판정 기준"이다.
#   따라서 행 필터보다 먼저 계산해야 한다.
# ---------------------------------------------------------------------------
df["trip_duration_min"] = (
    (df.dropoff_dt - df.pickup_dt).dt.total_seconds() / 60).astype("float32")

# 이 시점의 trip_duration_min에는 아직 0 또는 음수가 남아 있어 0으로 나누는
# 경고가 발생한다. 결과 inf는 바로 아래에서 NaN으로 치환하므로 경고만 억제한다.
with np.errstate(divide="ignore", invalid="ignore"):
    df["avg_speed_mph"] = (
        df.trip_distance / (df.trip_duration_min / 60)).astype("float32")
# inf를 NaN으로 바꾸는 이유: inf가 남아 있으면 이후 describe()·mean() 등
# 모든 집계가 inf로 오염된다. NaN은 집계에서 자동 제외된다.
df["avg_speed_mph"] = df.avg_speed_mph.replace([np.inf, -np.inf], np.nan)

# ---------------------------------------------------------------------------
# 행 필터
# ---------------------------------------------------------------------------
log = []   # (사유, 제거건수, 원본대비%) - 마지막에 리포트 [A]로 출력


def drop(mask, reason):
    """mask=True 인 행을 제거하고 사유·건수를 log에 기록한다.

    제거를 이 함수로 일원화한 이유:
      - 어떤 규칙이 몇 건을 지웠는지 자동으로 남아 감사(audit)가 가능하다.
      - 규칙을 추가/삭제할 때 로그 코드를 따로 손댈 필요가 없다.

    fillna(False)를 쓰는 이유: avg_speed_mph 등에 NaN이 있으면 비교 결과가
    NA가 되고, NA로 인덱싱하면 예외가 난다. "판정 불가"는 보존(=제거 안 함)
    쪽으로 처리하는 것이 안전하다(의심스러우면 남긴다).
    """
    global df
    n = int(mask.fillna(False).sum())
    log.append((reason, n, n / n0 * 100))
    df = df[~mask.fillna(False)]
    return n


# 순서 주의: 중복 제거를 가장 먼저 한다. 다른 필터가 먼저 돌면 중복 쌍 중
# 한쪽만 사라져 중복 판정 자체가 왜곡될 수 있다.
drop(df.duplicated(), "완전 중복 행")

# 기간 외: inclusive="left" - 6월 1일 00:00:00 정각 pickup은 6월 데이터다.
drop(~df.pickup_dt.between(MONTH_START, MONTH_END, inclusive="left"),
     "2026-05 기간 외 pickup")

# 시각 역전: 하차가 승차보다 빠른 건 미터기 시계 오류. 물리적으로 불가능.
drop(df.trip_duration_min <= 0, "dropoff <= pickup (시각 역전)")
drop(df.trip_duration_min < DUR_MIN, f"운행시간 < {DUR_MIN:.0f}분")
drop(df.trip_duration_min > DUR_MAX, f"운행시간 > {DUR_MAX / 60:.0f}시간")

# 거리 0: 미터기가 거리를 못 잡은 행. 요금은 있으나 "어디서 어디로"라는
# 이동 정보가 없어 거리·속도·단가 분석 어디에도 쓸 수 없다.
drop(df.trip_distance <= 0, "trip_distance <= 0")
drop(df.trip_distance > DIST_MAX, f"trip_distance > {DIST_MAX:.0f}mi")

# 음수 요금: 실제 운행이 아니라 앞선 트립을 취소·환불한 회계 조정 행이다.
# 남겨두면 합계·평균이 실제보다 낮게 계산된다. (원본 최소 -950달러)
drop(df.fare_amount < 0, "fare_amount < 0 (환불/분쟁)")
drop(df.total_amount <= 0, "total_amount <= 0")
drop((df.tip_amount < 0) | (df.tolls_amount < 0), "tip/tolls < 0")

# 속도 교차검증 - 거리·시간이 각각은 정상 범위지만 조합이 불가능한 행.
drop(df.avg_speed_mph > SPEED_MAX, f"평균속도 > {SPEED_MAX:.0f}mph")

# 옐로우캡 법정 정원은 4인(대형 5~6인)이다. 7인 이상은 입력 오류.
drop(df.passenger_count > 6, "passenger_count > 6")

# ---------------------------------------------------------------------------
# 결측 / 코드값 정리
#   여기서부터는 행을 지우지 않는다. 위 원칙 (2) 보존에 해당.
# ---------------------------------------------------------------------------

# passenger_count
#   0명은 물리적으로 불가능하므로 "미기재"의 다른 표현으로 보고 NA로 통일한다.
#   대체값 1: 관측된 값의 약 76%가 1명인 최빈값이자 중앙값이다. 평균(1.24)
#   대신 최빈값을 쓴 이유는 정수형 범주에 가까운 변수라 1.24명이라는 값이
#   현실에 존재하지 않기 때문이다.
df.loc[df.passenger_count == 0, "passenger_count"] = pd.NA
df["is_passenger_imputed"] = df.passenger_count.isna()   # 대체 여부 추적용
df["passenger_count"] = df.passenger_count.fillna(1).astype("int8")

# ratecode_id
#   TLC 정의상 99는 "Unknown"이며 결측과 의미가 같다. 서로 다른 두 표현
#   (99 / NaN)을 0=Unknown 한 가지로 통일해야 groupby 시 같은 그룹으로 묶인다.
#   99를 그대로 두면 숫자로서 평균·정렬에 끼어들어 통계를 심하게 왜곡한다
#   (원본 RatecodeID 평균이 5.5로 나온 원인).
df["is_ratecode_unknown"] = df.ratecode_id.isna() | (df.ratecode_id == 99)
df["ratecode_id"] = (df.ratecode_id.where(df.ratecode_id.between(1, 6))
                     .fillna(0).astype("int8"))

# store_and_fwd_flag
#   Y/N 이진 플래그. 결측을 N으로 채우면 "통신 정상"이라는 없는 정보를
#   만들어내는 것이므로, 별도 범주 "U"(Unknown)로 둔다.
df["store_and_fwd_flag"] = df.store_and_fwd_flag.fillna("U").astype("category")

# 부가요금(혼잡통행료·공항료)
#   금액 컬럼의 결측은 "미보고"이지 "금액 미상"이 아니다. total_amount에는
#   이미 반영돼 있으므로 0으로 채워야 컬럼 간 합계가 맞는다.
#   다만 진짜 0원인지 미보고인지 구분이 필요할 수 있어 플래그를 남긴다.
df["is_fee_missing"] = df.congestion_surcharge.isna()
for c in ["congestion_surcharge", "airport_fee", "cbd_congestion_fee"]:
    df[c] = df[c].fillna(0).astype("float32")

# payment_type
#   0은 오류가 아니라 최근 도입된 Flex Fare(앱 선결제) 코드다. 전체의 22.5%를
#   차지하므로 유효 범주로 그대로 유지한다. 정의에 없는 값만 5(Unknown)로.
df["payment_type"] = df.payment_type.fillna(5).astype("int8")

# ---------------------------------------------------------------------------
# 파생변수 - 원본 컬럼만으로는 답할 수 없는 질문에 쓰기 위해 만든다.
# ---------------------------------------------------------------------------

# 수요의 시간 구조를 보기 위한 변수들. datetime 원본으로는 groupby가 불가능
# (초 단위까지 달라 185만 개 그룹이 생김)하므로 시/요일 단위로 이산화한다.
df["pickup_hour"] = df.pickup_dt.dt.hour.astype("int8")
df["pickup_dow"] = df.pickup_dt.dt.dayofweek.astype("int8")     # 0=월 ... 6=일
df["pickup_date"] = df.pickup_dt.dt.date   # 일별 집계용(저장 시에는 제외)
df["is_weekend"] = df.pickup_dow.isin([5, 6])

# 24개 시간대는 시각화에 너무 잘다. 통근·유흥 등 수요 성격이 비슷한
# 구간끼리 5개로 묶는다. 경계를 -1부터 시작하는 이유는 pd.cut이 왼쪽
# 경계를 배타적으로 처리해 0시가 어느 구간에도 안 들어가기 때문이다.
df["time_of_day"] = pd.cut(
    df.pickup_hour, bins=[-1, 5, 11, 17, 21, 23],
    labels=["Late Night", "Morning", "Afternoon", "Evening", "Night"]).astype("category")

# 마일당 요금: 거리가 다른 트립 간 요금 수준을 비교하기 위한 정규화 지표.
# 주의) 초단거리(0.01마일) 트립에서 값이 폭발하므로 분석 시 상하위 절사 권장.
df["fare_per_mile"] = (df.fare_amount / df.trip_distance).astype("float32")

# 팁 비율: 팁 금액(달러)은 요금 규모에 비례해 커지므로 그대로 비교하면
# 장거리 트립이 항상 후하게 보인다. 비율로 바꿔야 결제수단·요일 간 비교가
# 성립한다. fare_amount=0인 행은 정의 불가이므로 NaN(집계에서 자동 제외).
df["tip_pct"] = np.where(df.fare_amount > 0,
                         df.tip_amount / df.fare_amount * 100, np.nan).astype("float32")

# 공항 트립: 요금 체계·거리 분포가 시내 트립과 완전히 달라 분리 분석이 필요하다.
# airport_fee만으로 판단하면 위 결측 처리에서 0으로 채워진 22.5%를 놓치므로,
# 존 ID(1=Newark, 132=JFK, 138=LaGuardia)와 OR로 결합해 누락을 보완한다.
AIRPORT_ZONES = {1, 132, 138}
df["is_airport_trip"] = (df.airport_fee > 0) | \
                        df.pu_location_id.isin(AIRPORT_ZONES) | \
                        df.do_location_id.isin(AIRPORT_ZONES)

# 필터로 인덱스에 구멍이 뚫린 상태다. 0부터 재부여해 이후 위치 기반 접근과
# parquet 저장을 깔끔하게 만든다.
df = df.reset_index(drop=True)
n1 = len(df)

# ---------------------------------------------------------------------------
# 리포트 - 무엇을 얼마나 어떻게 바꿨는지 설명하기 위한 출력
# ---------------------------------------------------------------------------
pd.set_option("display.width", 200, "display.max_columns", 60)
print("\n" + "=" * 78)
print("[A] 제거 내역")
print("=" * 78)
rep = pd.DataFrame(log, columns=["사유", "제거행수", "원본대비%"])
rep = rep[rep.제거행수 > 0]   # 0건 규칙은 노이즈이므로 표에서 숨긴다
print(rep.to_string(index=False, float_format="{:.3f}".format))
print(f"\n  원본 {n0:,} -> 정제 {n1:,}  (제거 {n0 - n1:,}, {(n0 - n1) / n0 * 100:.2f}%)")

print("\n" + "=" * 78)
print("[B] 결측 처리 요약")
print("=" * 78)
# 대체 비율을 명시하는 목적: 후속 분석자가 "이 컬럼은 22%가 추정값"임을
# 알고 결과를 해석하도록 하기 위함.
for c, label in [("is_passenger_imputed", "passenger_count 대체(1명)"),
                 ("is_ratecode_unknown", "ratecode_id Unknown(0) 처리"),
                 ("is_fee_missing", "congestion/airport fee 0 대체")]:
    n = int(df[c].sum())
    print(f"  {label:<34} {n:>10,}  ({n / n1 * 100:5.2f}%)")
print(f"  잔여 결측 컬럼: "
      f"{df.columns[df.isna().any()].tolist() or '없음 (tip_pct 제외)'}")

print("\n" + "=" * 78)
print("[C] 정제 후 주요 지표")
print("=" * 78)
key = ["trip_distance", "trip_duration_min", "avg_speed_mph", "fare_amount",
       "tip_amount", "total_amount", "fare_per_mile", "passenger_count"]
# astype("float64"): float32/Int16이 섞이면 describe 출력에서 정밀도 표기가
# 어긋난다. 출력 목적으로만 승격한다(원본 df는 그대로 유지).
print(df[key].astype("float64").describe(percentiles=[.25, .5, .75, .95])
      .T.to_string(float_format="{:.2f}".format))

# [D] 시간대별 - 수요 피크와 정체 구간(속도 저하)을 동시에 확인하기 위함
print("\n[D] 시간대별 운행량 / 평균 지표")
by_h = df.groupby("pickup_hour").agg(
    trips=("total_amount", "size"), avg_dist=("trip_distance", "mean"),
    avg_dur=("trip_duration_min", "mean"), avg_speed=("avg_speed_mph", "mean"),
    avg_total=("total_amount", "mean"))
print(by_h.to_string(float_format="{:.2f}".format))

# [E] 요일별 - 평일/주말 수요 및 팁 성향 차이 확인
print("\n[E] 요일별 (0=월 ~ 6=일)")
print(df.groupby("pickup_dow").agg(
    trips=("total_amount", "size"), avg_total=("total_amount", "mean"),
    avg_tip_pct=("tip_pct", "mean")).to_string(float_format="{:.2f}".format))

# [F] 결제수단별 - 현금 팁은 미터기에 기록되지 않는다는 데이터 한계를
#     수치로 드러내기 위한 표(현금 평균 팁이 0으로 나오는지 확인).
print("\n[F] 결제수단별 (0=Flex,1=Card,2=Cash,3=No charge,4=Dispute)")
print(df.groupby("payment_type").agg(
    trips=("total_amount", "size"), avg_fare=("fare_amount", "mean"),
    avg_tip=("tip_amount", "mean"), avg_tip_pct=("tip_pct", "mean")
).to_string(float_format="{:.2f}".format))

# [G] 공항 트립 분리가 실제로 의미 있는지(분포가 다른지) 검증하는 표
print("\n[G] 공항 트립 vs 일반")
print(df.groupby("is_airport_trip").agg(
    trips=("total_amount", "size"), avg_dist=("trip_distance", "mean"),
    avg_total=("total_amount", "mean")).to_string(float_format="{:.2f}".format))

print("\n[H] 상위 승차 존 TOP10")
print(df.pu_location_id.value_counts().head(10).to_string())

# ---------------------------------------------------------------------------
# 저장
#   pickup_date는 파이썬 date 객체(dtype=object)라 parquet에서 비효율적이고
#   pickup_dt로 언제든 재생성 가능하므로 저장 대상에서 제외한다.
#   snappy: 압축률보다 읽기 속도를 우선하는 기본 코덱. 반복 로딩이 잦은
#   분석 워크플로에 적합하다.
# ---------------------------------------------------------------------------
df.drop(columns=["pickup_date"]).to_parquet(OUT, index=False, compression="snappy")
print(f"\n[save] {OUT}  ({n1:,} rows x {df.shape[1] - 1} cols)")

# Inhatc_3_1_AI_MiniProject
# 의료 보험료 예측 모델 (Insurance Charges Prediction)

본 프로젝트는 미국 의료 보험료 데이터(`insurance.csv`)를 활용하여, 개인의 나이, 성별, BMI, 자녀 수, 흡연 여부, 지역 등의 정보를 바탕으로 의료 보험료(`charges`)를 예측하는 회귀 모델을 구현한 것입니다.

---

## 문제 정의

- 문제 유형: 정형 데이터 기반 회귀(Regression)
- 목표 변수: `charges` (의료 보험료, 연속형 변수)
- 입력 변수:
  - `age`: 나이
  - `sex`: 성별
  - `bmi`: 체질량 지수
  - `children`: 자녀 수
  - `smoker`: 흡연 여부
  - `region`: 거주 지역

---

## 데이터 설명

- 데이터셋: `insurance.csv`
- 샘플 수: 1,338개
- 타깃 변수: `charges` (의료 보험료)
- 입력 변수:

| 변수명   | 설명               | 타입     |
|----------|--------------------|----------|
| age      | 나이               | 수치형   |
| sex      | 성별               | 범주형   |
| bmi      | 체질량지수         | 수치형   |
| children | 자녀 수            | 수치형   |
| smoker   | 흡연 여부          | 범주형   |
| region   | 거주 지역          | 범주형   |
| charges  | 의료 보험료 (타깃) | 수치형   |

---

## 전처리

- 범주형 변수(`sex`, `smoker`, `region`)는 `pd.get_dummies()`로 원-핫 인코딩 적용 (`drop_first=True`)
- 수치형 입력 변수는 `MinMaxScaler`를 이용해 0~1 범위로 정규화

---

## 하이퍼파라미터 실험 요약

| 실험 항목     | 설정 값       | 결과 (MSE)      | 해석                          |
|---------------|----------------|------------------|-------------------------------|
| Batch Size    | 64             | 0.1883           | 중간 크기 설정이 안정적 학습 유도 |
| Activation    | tanh           | 0.1771           | 가장 우수한 성능 |
| Activation    | sigmoid        | 0.1823           | 중간 성능 |
| Activation    | relu           | 0.2052           | 상대적으로 낮은 성능 |
| EarlyStopping | patience=5     | 0.1662           | 적절한 종료 조건으로 과적합 방지 |

- 활성화 함수 조합: relu(입력층) → tanh → tanh → relu (은닉층)
- 손실 함수: `mean_squared_error`
- 옵티마이저: Adam

---

## 모델 구성

- 총 5개 레이어 (입력층 + 은닉층 3개 + 출력층)
- 각 층 뉴런 수: 30 → 25 → 20 → 15 → 1
- 출력층 활성화 함수 없음 (회귀)

```python
model = Sequential([
    Dense(30, input_dim=입력변수수, activation='relu'),
    Dense(25, activation='tanh'),
    Dense(20, activation='tanh'),
    Dense(15, activation='relu'),
    Dense(1)
])

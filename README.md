# Inhatc_3_1_AI_MiniProject
# 의료 보험료 예측 모델 (Insurance Charges Prediction)

본 프로젝트는 미국 의료 보험료 데이터(`insurance.csv`)를 활용하여, 개인의 나이, 성별, BMI, 자녀 수, 흡연 여부, 지역 등의 정보를 바탕으로 의료 보험료(`charges`)를 예측하는 회귀 모델을 구현하였습니다.

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
- 출력 변수: `charges` (의료 보험료)
- 입력 변수:

| 변수명   | 설명               | 타입     |
|----------|--------------------|----------|
| age      | 나이               | 수치형   |
| sex      | 성별               | 범주형   |
| bmi      | 체질량지수         | 수치형   |
| children | 자녀 수            | 수치형   |
| smoker   | 흡연 여부          | 범주형   |
| region   | 거주 지역          | 범주형   |
| charges  | 의료 보험료 (출력) | 수치형   |

---

## 전처리

- 범주형 변수(`sex`, `smoker`, `region`)는 수치로 나타내기 위해 `pd.get_dummies()`로 원-핫 인코딩 적용 
- 수치형 입력 변수는 `MinMaxScaler`를 이용해 0~1 범위로 정규화

---

## 하이퍼파라미터 실험 및 결과 분석

모델 성능 개선을 위해 다음과 같은 하이퍼파라미터에 대해 실험을 진행하였고, 그 과정에서 유의미한 차이를 확인하였습니다.

### 1. 배치 크기 (Batch Size)

**실험 목적**: 학습 안정성과 속도 간의 균형을 찾기 위함  
**실험값**: [32, 64, 128]  
**설정 조건**: epochs=200, validation_split=0.25, EarlyStopping 적용

| Batch Size | MSE (val_loss 기준) |
|------------|---------------------|
| 32         | 0.1909              |
| 64         | 0.1883              |
| 128        | 0.1922              |

**관찰 및 해석**:  
- 64에서 가장 낮은 MSE를 기록하였고, 학습 시간과 손실 안정성 측면에서도 균형이 좋았습니다.  
- 데이터가 비교적 소규모(1338개)이므로 너무 큰 배치 크기는 오히려 성능 저하로 이어졌습니다.



### 2. 활성화 함수 (Activation Function)

**실험 목적**: 최적의 활성화 함수의 조합 탐색  
**실험값**: tanh, relu, sigmoid  
**공통 구조**: 4층 MLP, 각 층 뉴런 수 동일

| 활성화 함수 조합       | MSE     |
|------------------------|---------|
| tanh / tanh / tanh     | 0.1771  |
| relu / relu / relu     | 0.2052  |
| sigmoid / sigmoid / sigmoid | 0.1823 |

**관찰 및 해석**:  
- `tanh`는 중심이 0이라 학습 균형이 좋고, 데이터가 정규화된 경우 매우 잘 작동했습니다.  
- `relu`는 빠른 수렴은 가능했지만 과적합 경향이 보였으며, `sigmoid`는 수렴 속도가 느렸습니다.



### 3. 조기 종료 조건 (EarlyStopping - patience)

**실험 목적**: 과적합 방지 효과와 학습 지속성의 균형 찾기  
**실험값**: patience = 1, 5, 15

| Patience | MSE     |
|----------|---------|
| 1        | 0.2006  |
| 5        | 0.1662  |
| 15       | 0.1894  |

**관찰 및 해석**:  
- 너무 빠른 조기 종료(1)는 최적 성능 도달 전에 학습이 멈췄고,  
- 너무 큰 patience(15)는 과적합으로 이어졌습니다.  
- 5가 가장 이상적인 조기 종료 지점으로 판단되었습니다.

---

## 최종 결정된 하이퍼파라미터

- Batch Size: 64
- Epochs: 200
- Activation: relu (입력층) + tanh / tanh / relu (은닉층)
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- EarlyStopping: monitor='val_loss', patience=5
- 모델 구조: Dense 30 → 25 → 20 → 15 → 1



```python
model = Sequential([
    Dense(30, input_dim=입력변수, activation='relu'),
    Dense(25, activation='tanh'),
    Dense(20, activation='tanh'),
    Dense(15, activation='relu'),
    Dense(1)
])

---
layout: post
title:  "[일변량통계] Stationarity(정상성) 검증 및 SARIMA를 통한 예측"
categories: timeseries
comments: true
---
## ARIMA 모델의 가정 : 정상성(Stationarity)
ARIMA 예측모델은 **시계열 데이터의 정상성(stationarity)**를 가정합니다. 

아래 정상성에 대한 설명을 보면 알 수 있듯이 정상시계열은 **일정한 평균과 분산**을 가져서 분석하기 쉬우며, 장기적으로 예측가능한 패턴을 갖지 않습니다.

#### - 강한정상성(Strict Stationarity)
특정확률과정을 따르는 시계열 확률변수들의 **확률분포가 시간이 지나도 일정**하고 그 결합분포 또한 일정하다고 가정하는 것이 강한정상성(Strict Stationarity)입니다. 하지만 이러한 분포에 대한 가정하는 것은 비현실적입니다. 그래서 대안으로 나온 것이 약한정상성 가정입니다.

#### - 약한정상성(Weak Stationarity), 공분산정상성(Covariance Stationarity)
그래서 이러한 분포가정보다 약한 적률조건에 대한 가정을 한 것을 약정상성(Weak Stationarity)를 의미합니다. 이는 공분산정상성(Covariance Stationarity)라고도 부르는데요. 3가지 조건을 만족하는 시계열데이터를 의미합니다.

#### - 공분산정상성을 만족하는 3가지 조건
1) **평균 일정** 즉 시간의 흐름에 영향을 받지 않음

2) **분산 일정**, 유한 즉 시간의 흐름에 영향을 받지 않음

3) **공분산이 시간간격에만 의존** 즉 시간의 흐름에 영향을 받지 않음


(참고) 정상성을 만족해야하는 이유는 모델이 적절한 계수를 추정하기 위해서인데요. 즉 정상성을 만족하지 못하는 데이터를 모델에 넣어보아도 나오는 계수로는 어떠한 해석도 할 수 없습니다. 

보통 데이터분석에서 **OLS추정량**의 점근적 특성(일치성, 점근적 정규성)에는 **대수의 법칙**과 **중심극한정리** 성질이 만족한다는 가정이 있습니다. 또한, 대수의 법칙과 중심극한정리는 **표본의 iid조건**(independent identical distribution 조건)에 의존합니다.
하지만 시계열데이터는 표본 사이에 dependence가 존재하여 iid조건을 만족하지 못합니다. 그래서 통계학자들은 연구를 통해 시계열데이터의 경우, **정상성조건과 약한의존성 조건을 만족하면 대수의 법칙과 중심극한정리를 사용할 수 있음**을 밝혀냈습니다. 

# 1. Stationarity 확인
시계열데이터가 정상성을 지니면 외부 충격이 발생해도 해당 충격은 시간의 흐름에 따라 소멸하여 데이터의 움직임에 크게 영향을 끼치지 않습니다. 하지면 비정상성을 지닌다면 외부데이터는 향후 시계열데이터의 움직임에 영향을 끼치게 되고 시계열데이터의 속성 자체가 변하게 되는 것입니다.그래서 비정상성을 가지는 데이터는 예측을 해도 현실에 맞지 않을 확률이 큽니다. 

결국 **시각화와 검증을 통해 데이터가 정상성을 지니는가**를 파악해야 분석이 가능합니다.


```python
import warnings
warnings.filterwarnings('ignore')
```

### 데이터탐색 및 전처리


```python
import numpy as np
import pandas as pd

rawdata = pd.read_csv("경유월간소비.csv", encoding='cp949')
```


```python
rawdata = rawdata[::2]
rawdata.columns = ['date', 'diesel']

rawdata['date'] = pd.date_range('2000-01-01','2021-06-01', freq='MS')
rawdata.set_index('date', inplace=True)
rawdata.index.freq = 'MS'

rawdata['diesel'] = rawdata['diesel'].apply(pd.to_numeric)
```


```python
rawdata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diesel</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>10390</td>
    </tr>
    <tr>
      <th>2000-02-01</th>
      <td>10023</td>
    </tr>
    <tr>
      <th>2000-03-01</th>
      <td>11786</td>
    </tr>
    <tr>
      <th>2000-04-01</th>
      <td>11086</td>
    </tr>
    <tr>
      <th>2000-05-01</th>
      <td>12208</td>
    </tr>
  </tbody>
</table>
</div>



## 1) Plot for Stationarity
먼저 시각화를 통해 비정상시계열을 구분할 수 있습니다. 

#### 비정상성
명백한 **추세(trend)와 계절요인(seasonality)** 가 관찰된다면 이는 **비정상시계열데이터**라고 할 수 있습니다. 

그 이유는 위에서 말했듯이 **정상시계열은 시간의 흐름에 영향을 받지 않아야하는데**, 추세와 계절요인은 시간이 경과하면서 관측값에 영향을 미치기 때문입니다.

#### 정상성
정상시계열은 위에서 언급한 것처럼, 평균과 분선이 일정하여 향후 값을 예측할 수 없는 그래프를 보여줍니다.


```python
%matplotlib inline
import matplotlib.pyplot as plt
rawdata.plot(figsize=(12, 8)) # 추세가 관측되어 적어도 정상성을 만족하지 못한다는 것을 알 수 있습니다
```




    <AxesSubplot:xlabel='date'>




    
![png](output_7_1.png)
    


## 2) ADF Test for Stationarity
ADF(Augmented Dickey-Fuller Test)는 **DF검정(단위근검정)을 일반화한 검정**으로, **시계열정상성 검증**을 시행할 수 있습니다. 

DF Test는 **단위근검정(Unit root Test)** 라고도 불립니다. 
단위근이 존재한다는 것은 Random Walk모형을 의미하며, 이는 시계열변수가 Highly Persistent Process를 따르는 것을 말합니다. 즉 랜덤워크모형은 차분이 필요한 대표적인 비정상시계열 모형이죠. 

> 귀무가설 : 단위근존재 (정상성을 만족하지 못하는 데이터)

> 귀무가설 기각 (p-value 값이 0.05 미만) => 정상성 만족하는 시계열데이터라 가정

귀무가설이 특성방정식의 해에 단위근(1)이 포함되어 있는지를 검증하고 이는 '시계열데이터가 비정상적'임을 의미합니다. 

그래서 **p-value 값이 0.05 미만**이면, 귀무가설을 기각하고 주어진 **시계열 데이터는 정상성을 만족한다고 가정**할 수 있습니다.


```python
from statsmodels.tsa.stattools import adfuller
```


```python
adftest = adfuller(rawdata)
adfout = pd.Series(adftest[0:4], index=['ADF Test Statistic', 'p-value', '# Lags Used', '# Observations'])

for key, val in adftest[4].items():
    adfout[f'critical value ({key})'] = val   
adfout
```




    ADF Test Statistic       -1.100831
    p-value                   0.714740
    # Lags Used              15.000000
    # Observations          242.000000
    critical value (1%)      -3.457664
    critical value (5%)      -2.873559
    critical value (10%)     -2.573175
    dtype: float64



ADF 검증을 통해서도 시계열데이터가 정상성을 만족하지 못함을 확인할 수 있습니다. 매번 기각을 할 경우 정상성을 만족하는지 안하는지 헷갈리기 때문에 [참고코드](https://github.com/jeswingeorge/Learning-Time-Series-Python/blob/master/IMP-functions.md)를 활용하여 함수로 만들어 놓겠습니다. 

**ADF 검증을 편리하게 수행하기 위해 이를 함수로 만들어보겠습니다.**


```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series):

    result = adfuller(series.dropna())    
    labels = ['statistic','p-value','# lags used','# observations']
    
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())       
    
    if result[1] <= 0.05:
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
```


```python
adf_test(rawdata)
```

    statistic                -1.100831
    p-value                   0.714740
    # lags used              15.000000
    # observations          242.000000
    critical value (1%)      -3.457664
    critical value (5%)      -2.873559
    critical value (10%)     -2.573175
    Fail to reject the null hypothesis
    Data has a unit root and is non-stationary
    

## 3) ACF(Autocorrelation Function)
시계열 정상성을 평가할 수 있는 또 다른 방법이 ACF(Autocorrelation Function)입니다.

**자기상관계수(Autocorrelation)** 는 **시차(lag)에 따라 관측값의 상관관계**를 측정합니다. 자기상관함수(ACF)는 시차에 따라 측정된 일련의 자기상관계수를 모아놓은 것입니다. 

#### 비정상성
시차를 달리했을 때, **관측값이 천천히 감소한다면 trend가 있을 확률이 크겠죠. 그리고 특정 패턴을 보이면서 감소한다면 seasonality가 있을 가능성** 이 큽니다. 

#### 정상성
정상시계열의 경우는 ACF는 **상대적으로 빨리 0으로 접근**(95% 신뢰구간 내)하는지 확인해줘야합니다.


해당 데이터는 ACF를 통해서도 비정상임을 확인할 수 있습니다. 0으로 천천히 감소할 뿐만 아니라 특정한 패턴이 있는 것으로 보입니다.


```python
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(rawdata, lags=40);
```


    
![png](output_15_0.png)
    


# 2. Non-Stationary : 차분 및 로그변환 
**비정상시계열의 경우 정상시계열로 변환하는 과정**이 필요합니다. 

대표적으로 차분(Differencing)과 로그변환이 있습니다.
추세나 계절성이 있어 시계열데이터의 **평균이 일정하지 않을 경우**, **차분**을 통해 평균을 일정하게 변경할 수 있습니다. 그리고 **분산이 일정하지 않을 경우**, **로그변환**을 통해 분산을 일정하게 만들어줄 수 있습니다.


```python
from statsmodels.tsa.statespace.tools import diff

rawdata['diff_1'] = diff(rawdata, k_diff=1)
adf_test(rawdata['diff_1'])
```

    statistic                -4.713417
    p-value                   0.000079
    # lags used              14.000000
    # observations          242.000000
    critical value (1%)      -3.457664
    critical value (5%)      -2.873559
    critical value (10%)     -2.573175
    Reject the null hypothesis
    Data has no unit root and is stationary
    

**1차차분 이후 adf검증을 해보니, p-value가 0.05미만으로 정상시계열로 변환**된 것을 확인할 수 있습니다. 

현재 데이터의 경우 분산은 일정한 편으로 로그변환을 해주지 않을 것이지만 필요한 경우, ``np.log``를 사용하여 변환해주면 됩니다.

## (이론) 정상확률과정(Stationary Process) 모형
ARIMA모델을 본격적으로 사용하기 전에 **정상확률과정을 따르는 대표적인 모형**을 살펴보겠습니다. 

AR모형, MA모형 모두 위의 정상성 조건을 만족하는 시계열데이터입니다.

### 1) White Noise Process(백색잡음)

1) Constant Mean 

2) Constant Variance 

3) No autocorrelation

을 따르는 과정을 White Noise Process라고 합니다. 
그래서 Whit Noise Process는 약정상성 조건을 만족합니다. 

#### Gaussian white noise
가우시안 white noise의 경우, white noise 조건에 정규분포를 따른다는 가정이 추가된 것입니다. 

다른 분포와 달리 정규분포의 경우는 autocorrelation이 없다는 것이 해당 데이터가 독립이라는 의미를 함축하고 있습니다. 그리고 분포를 생각해보면 좌우대칭인 분포를 가지고 있습니다. 

그래서 각종 모형이 올바르게 구성되었는지 확인하기 위해 **잔차(교란항의 관측치)가 Gaussian White Noise를 따르는지 검증**하는 것입니다. 이를 통해 모형으로 설명할 수 없는 random shock만 교란항에 있는지 확인할 수 있습니다. 결국 설명가능한 모든 변수가 모델의 설명변수로 사용했는지 확인할 수 있습니다.

### 2) MA(q) 모델
**
해당 모형은 시계열 변수값이 **현재의 예측오차 White Nosie(random shock)** 와 **이전 시기의 예측오차 White Nosie(random shock)** 에 의해 결정되는 모형을 의미합니다. 즉 ***과거 예측오차***를 기반으로 모델을 구축합니다.
![image-2.png](attachment:image-2.png)

### 3) AR(p) 모델
해당 모형은 현재 시계열 변수값은 **과거 시계열변수값**과 **현재 White Noise(random shock)** 에 의해 결정되는 모형을 의미합니다. 즉 ***과거 관측값***을 이용하여 모델을 구축합니다. 단, 정주성을 만족하는 조건이 계수의 절대값이 1보다 작아야 (특성방정식 해가 단위근 외부에 존재) 합니다.
![image.png](attachment:image.png)

### 4) ARMA(p, q) 모델
**AR, MA 모델을 결합**한 것으로 미래값을 예측하기 위해 p개의 과거 관측값과 q개의 과거 오차값을 사용합니다.

### 5) ARIMA(p, d, q) 모델
ARMA모델에 **차분과정을 포함**시킨 것으로, 정상시계열로 만들기 위한 과정이 포함된 것입니다.
주의할 점은 해당 모델을 통해 관측된 예측값은 비차분화(un-differenced) 과정을 원래의 값으로 변환된 결과입니다.

### 6) SARIMA(p, d, q)(P, D, Q)m  모델
비계절성의 p, d, q의 동일한 과정을 **계절성**에 적용한 모델입니다.

# 3. 차수 및 모델선정


## 1) ACF 및 PACF
ACF와 PACF 도표는 **ARMA 모델의 차수를 결정**하는 데에 사용됩니다.
ACF는 위에서 설명했습니다. 

**PACF(Partial Autocorrelation Function)** 에 대해 알아보겠습니다.
먼저 **편자기상관(Partial Autocorrelation)** 은 **시차가 다른 시계열데이터간 순수한 상호연관성**을 나타냅니다. 
PACF는 두 시점 데이터 간 상관관계를 측정할 때, 두 시점 사이의 모든 시계열데이터의 영향을 제거한 다음에 상관관계를 측정한 것을 의미합니다. **편자기상관함수(PACF)** 는 이렇게 **시차에 따라 측정된 일련의 편자기상관계수**를 모아놓은 것입니다.

이렇게 **차분된 데이터의 ACF와 PACF를 그려서 차수를 선정**할 수 있습니다. 

ACF도표는 MA(q)모델의 q를 선정할 수 있습니다. 특정 시차 이후에 0으로 급격하게 감소한다면, 해당 시차를 q로 정하면 됩니다.
> ACF 도표 : q 결정 (MA)

PACF도표는 AR(p)모델의 p를 선정할 수 있습니다. 특정 시차 이후에 0으로 급격하게 감소한다면, 해당 시차를 p로 정하면 됩니다.
> PACF 도표 : p 결정 (AR)

AR모형의 경우 p시점까지는 직접적인 관련이 있지만, 그 이후에는 아무런 관련을 갖지 않으므로 즉 p시점을 넘어서게 되면 PACF의 값은 0이 되게 됩니다.

### 시각화


```python
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(rawdata['diff_1'].dropna(), lags=40, title='Autocorrelation');
```


    
![png](output_23_0.png)
    



```python
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(rawdata['diff_1'].dropna(), lags=40, title='Partial Autocorrelation');
```


    
![png](output_24_0.png)
    


해당 데이터의 경우, 계절성때문에 ARIMA모델을 사용할 경우 AR(12) 정도로 해줘야 될 것 같습니다. 

하지만 ACF 도표에서 보이듯이 계절성이 있는 것 같아, 계절성을 반영해주는 SARIMA 모델을 사용하는 것이 더 바람직해보입니다.

### 값으로 확인
참고로 도표로 보기 불편하여 값을 확인하고 싶으면 아래의 코드를 실행하면 됩니다.
상관계수(Correlation)을 계산하는 방식 자체가 다양해서 Partial Autocorrelation을 구하는 방법도 다양하여 편한 것으로 사용하면 됩니다.


```python
from statsmodels.tsa.stattools import acf, pacf_yw, pacf_ols
```


```python
acf(rawdata['diff_1'].dropna(), nlags=40)
```




    array([ 1.        , -0.448492  , -0.07108307,  0.03040533, -0.0953566 ,
            0.05366588,  0.02329903,  0.18238535, -0.24001238,  0.09934076,
           -0.09831666, -0.19413871,  0.53891366, -0.20049196, -0.10998774,
            0.04653065, -0.10532404,  0.07913436,  0.0130814 ,  0.10156845,
           -0.11440731,  0.03390377, -0.08514834, -0.14684195,  0.47397546,
           -0.22094501, -0.08013067,  0.07495509, -0.15687568,  0.11890325,
            0.01447273,  0.07172238, -0.09009184,  0.01473175, -0.04179905,
           -0.20784679,  0.51898832, -0.26801132, -0.02871159,  0.04783849,
           -0.10699782])




```python
pacf_yw(rawdata['diff_1'].dropna(), nlags=40, method='mle')
```




    array([ 1.        , -0.448492  , -0.34077295, -0.23392687, -0.32030303,
           -0.29196907, -0.28166319,  0.06834172, -0.09839745,  0.03834201,
           -0.09801561, -0.52746885,  0.04564765,  0.13551506,  0.02296841,
            0.14658472, -0.05037363,  0.06363819,  0.07018832, -0.12577457,
            0.04332602,  0.05958837, -0.01431013, -0.13055735,  0.09739667,
            0.05773655,  0.0234893 ,  0.0794986 , -0.08801107,  0.00781665,
            0.02891815, -0.06992021,  0.05490226, -0.02031514,  0.05078609,
           -0.17709106,  0.02661181, -0.02575504,  0.03074914,  0.05095436,
            0.09481825])




```python
pacf_ols(rawdata['diff_1'].dropna(), nlags=40)
```




    array([ 1.        , -0.44867551, -0.34064092, -0.2362108 , -0.32825098,
           -0.30418061, -0.29648532,  0.06596254, -0.09547256,  0.04816323,
           -0.09283178, -0.5506085 ,  0.01323592,  0.14745221,  0.0379658 ,
            0.16919794, -0.05544504,  0.0207201 ,  0.07188272, -0.13584794,
            0.0600996 ,  0.07773762, -0.00198667, -0.17015359,  0.12446963,
            0.09170698,  0.02051705,  0.11898891, -0.07349343,  0.06383518,
            0.08605935, -0.09330277,  0.05709738, -0.03421043,  0.05777611,
           -0.28495266,  0.08793702, -0.0282018 ,  0.0459933 ,  0.02445548,
            0.14278634])



## 2) auto_arima
위의 **'2. 차분'과정과 '3-1)ACF 및 PACF로 차수선정'을 동시에 할 수 있는 편리한 방법**이 있습니다. 

또한, 도표를 보고 결정할 경우, 주관이 반영되게 됩니다. 그래서 auto_arima 함수를 이용하여, 가능한 조합을 해본 이후 가장 적절한 모델을 선정할 수 있습니다. 

그럼 어떤 모델이 더 좋은지 선정하려면 모델 간 비교기준이 있어야합니다. 

### 모델선택기준(information criteria)
어떤 모델을 최종모델로 선정할 때, 아래의 두 가지를 고려합니다.

1) 잔차 (오차 최소화)

2) 파라미터 수 증가로 인한 자유도 손실에 대한 패널티 (파라미터 수 최소화)

위 두 가지를 고려한 대표적인 IC공식 3가지가 있습니다.
- AIC(Akaike's Infomation Criteria)
- SBIC(Schwarz's Bayesian Information Criteria)
- HQIC(Hannan-Quinn Information Criteria)

일반적으로 AIC(Akaike Information Criterion)를 많이 이용합니다.

**AIC의 값이 적을수록 더 좋은 모델**임을 의미합니다.


```python
from pmdarima import auto_arima
```


```python
# 계절성이 없을 경우
auto_arima(rawdata['diesel'], seasonal=False).summary()
```




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>    <td>258</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(0, 1, 1)</td> <th>  Log Likelihood     </th> <td>-2173.122</td>
</tr>
<tr>
  <th>Date:</th>            <td>Wed, 25 Aug 2021</td> <th>  AIC                </th> <td>4352.243</td> 
</tr>
<tr>
  <th>Time:</th>                <td>14:05:05</td>     <th>  BIC                </th> <td>4362.891</td> 
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>4356.525</td> 
</tr>
<tr>
  <th></th>                      <td> - 258</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>intercept</th> <td>   15.2217</td> <td>    8.503</td> <td>    1.790</td> <td> 0.073</td> <td>   -1.443</td> <td>   31.886</td>
</tr>
<tr>
  <th>ma.L1</th>     <td>   -0.8900</td> <td>    0.032</td> <td>  -27.956</td> <td> 0.000</td> <td>   -0.952</td> <td>   -0.828</td>
</tr>
<tr>
  <th>sigma2</th>    <td> 1.228e+06</td> <td> 1.02e+05</td> <td>   12.081</td> <td> 0.000</td> <td> 1.03e+06</td> <td> 1.43e+06</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.18</td> <th>  Jarque-Bera (JB):  </th> <td>15.19</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.67</td> <th>  Prob(JB):          </th> <td>0.00</td> 
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.62</td> <th>  Skew:              </th> <td>-0.46</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.03</td> <th>  Kurtosis:          </th> <td>3.77</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
# 계절성이 있을 경우
auto_arima(rawdata['diesel'], seasonal=True, m=12).summary()
```




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>                  <td>y</td>               <th>  No. Observations:  </th>    <td>258</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(0, 1, 1)x(1, 0, 1, 12)</td> <th>  Log Likelihood     </th> <td>-2118.435</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Wed, 25 Aug 2021</td>        <th>  AIC                </th> <td>4244.870</td> 
</tr>
<tr>
  <th>Time:</th>                       <td>14:05:21</td>            <th>  BIC                </th> <td>4259.066</td> 
</tr>
<tr>
  <th>Sample:</th>                         <td>0</td>               <th>  HQIC               </th> <td>4250.579</td> 
</tr>
<tr>
  <th></th>                             <td> - 258</td>             <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>               <td>opg</td>              <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ma.L1</th>    <td>   -0.8656</td> <td>    0.028</td> <td>  -31.456</td> <td> 0.000</td> <td>   -0.920</td> <td>   -0.812</td>
</tr>
<tr>
  <th>ar.S.L12</th> <td>    0.8939</td> <td>    0.035</td> <td>   25.213</td> <td> 0.000</td> <td>    0.824</td> <td>    0.963</td>
</tr>
<tr>
  <th>ma.S.L12</th> <td>   -0.6115</td> <td>    0.066</td> <td>   -9.228</td> <td> 0.000</td> <td>   -0.741</td> <td>   -0.482</td>
</tr>
<tr>
  <th>sigma2</th>   <td> 7.246e+05</td> <td> 4.48e+04</td> <td>   16.192</td> <td> 0.000</td> <td> 6.37e+05</td> <td> 8.12e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.01</td> <th>  Jarque-Bera (JB):  </th> <td>64.30</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.92</td> <th>  Prob(JB):          </th> <td>0.00</td> 
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.84</td> <th>  Skew:              </th> <td>-0.34</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.42</td> <th>  Kurtosis:          </th> <td>5.35</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



역시 계절성 때문인지 ARIMA로 선정한 최종모델의 AIC보다 **RIMA로 선정한 모델의 AIC가 더 낮은 것을 확인**수 있습니다. 

해당모델을 생성하여 예측과 정확도 평가까지 진행하겠습니다,

# 4. 모델 생성, 예측, 평가

## 1) train, test split


```python
split = len(rawdata['diesel']) - 24
```


```python
train = rawdata.iloc[:split]
test = rawdata.iloc[split:]
```


```python
train.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diesel</th>
      <th>diff_1</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-02-01</th>
      <td>12836</td>
      <td>-2562.0</td>
    </tr>
    <tr>
      <th>2019-03-01</th>
      <td>14746</td>
      <td>1910.0</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>15461</td>
      <td>715.0</td>
    </tr>
    <tr>
      <th>2019-05-01</th>
      <td>12809</td>
      <td>-2652.0</td>
    </tr>
    <tr>
      <th>2019-06-01</th>
      <td>14320</td>
      <td>1511.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diesel</th>
      <th>diff_1</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-07-01</th>
      <td>14174</td>
      <td>-146.0</td>
    </tr>
    <tr>
      <th>2019-08-01</th>
      <td>16465</td>
      <td>2291.0</td>
    </tr>
    <tr>
      <th>2019-09-01</th>
      <td>10857</td>
      <td>-5608.0</td>
    </tr>
    <tr>
      <th>2019-10-01</th>
      <td>14610</td>
      <td>3753.0</td>
    </tr>
    <tr>
      <th>2019-11-01</th>
      <td>14859</td>
      <td>249.0</td>
    </tr>
  </tbody>
</table>
</div>



## 2) create a model


```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train['diesel'], order = (0, 1, 1), seasonal_order = (1, 0, 1, 12))
results = model.fit()
results.summary()
```




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>               <td>diesel</td>             <th>  No. Observations:  </th>    <td>234</td>   
</tr>
<tr>
  <th>Model:</th>           <td>SARIMAX(0, 1, 1)x(1, 0, 1, 12)</td> <th>  Log Likelihood     </th> <td>-1903.966</td>
</tr>
<tr>
  <th>Date:</th>                   <td>Wed, 25 Aug 2021</td>        <th>  AIC                </th> <td>3815.931</td> 
</tr>
<tr>
  <th>Time:</th>                       <td>14:05:22</td>            <th>  BIC                </th> <td>3829.735</td> 
</tr>
<tr>
  <th>Sample:</th>                    <td>01-01-2000</td>           <th>  HQIC               </th> <td>3821.498</td> 
</tr>
<tr>
  <th></th>                          <td>- 06-01-2019</td>          <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>               <td>opg</td>              <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ma.L1</th>    <td>   -0.8650</td> <td>    0.027</td> <td>  -31.653</td> <td> 0.000</td> <td>   -0.919</td> <td>   -0.811</td>
</tr>
<tr>
  <th>ar.S.L12</th> <td>    0.8354</td> <td>    0.041</td> <td>   20.194</td> <td> 0.000</td> <td>    0.754</td> <td>    0.916</td>
</tr>
<tr>
  <th>ma.S.L12</th> <td>   -0.4373</td> <td>    0.074</td> <td>   -5.897</td> <td> 0.000</td> <td>   -0.583</td> <td>   -0.292</td>
</tr>
<tr>
  <th>sigma2</th>   <td> 6.075e+05</td> <td> 4.45e+04</td> <td>   13.647</td> <td> 0.000</td> <td>  5.2e+05</td> <td> 6.95e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.49</td> <th>  Jarque-Bera (JB):  </th> <td>47.58</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.48</td> <th>  Prob(JB):          </th> <td>0.00</td> 
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.51</td> <th>  Skew:              </th> <td>-0.19</td>
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td> <th>  Kurtosis:          </th> <td>5.18</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).



## 3) prediction

참고로 현재는 **SARIMAX**가 아닌 SARIMA를 이용하여 상관없지만 혹시 SARIMAX를 이용하여 외생변수를 추가해야하는 분들을 위해 설명드리자면, predict에 적용하는 **typ 파라미터**에 

``'levels'``를 입력할 경우, 분석대상 시계열데이터를 차분할 경우 **외생변수는 차분없이 해당 값을 그대로 이용**하고, 

``'linear'``을 입력할 경우, 분석대상 시계열데이터를 차분할 경우 **외생변수 함께 차분**합니다.


```python
start = len(train)
end = len(train) + len(test) - 1
```


```python
predictions = results.predict(start, end, typ='levels').rename('SARIMA Predictions')
```


```python
predictions.head()
```




    2019-07-01    14390.076315
    2019-08-01    14500.278449
    2019-09-01    14187.725017
    2019-10-01    13583.356704
    2019-11-01    14804.187175
    Freq: MS, Name: SARIMA Predictions, dtype: float64



## 4) Evaluation
시계열데이터에 대표적인 정확도 평가지표인 rmse를 이용하여 정확도를 판단하겠습니다. 

그리고 모델이 올바르게 구성되었는지 오차항의 정규성을 확인해주도록 하겠습니다.

#### RMSE
test data의 평균과 표준편차와 오차를 비교하여, 오차가 어느정도로 큰가 확인하여 줍니다.


```python
test['diesel'].plot(legend=True, figsize=(12,8))
predictions.plot(legend=True)
```




    <AxesSubplot:xlabel='date'>




    
![png](output_50_1.png)
    



```python
from statsmodels.tools.eval_measures import rmse
error = rmse(test['diesel'], predictions)
error
```




    1389.8916795544772




```python
test['diesel'].mean(), test['diesel'].std()
```




    (13826.041666666666, 1303.5906092210373)



#### 오차항의 정규성
오차항이 어느정도로 정규분포 형태를 띄고 있는지 확인하여, 올바른 모델이 구성되었는지 확인합니다.


```python
import scipy.stats as stats
import pylab

resid = test['diesel'] - predictions
stats.probplot(resid, dist="norm", plot=pylab)
pylab.show()
```


    
![png](output_54_0.png)
    


# 5. 최종예측
모델에 대한 성능평가까지 끝나서 우수한 모델을 선정했다면, 이제 최종예측단계만 남았습니다. 

train, test dataset 분리 없이 **전체 시계열데이터로 한번 더 fit해 준 이후 아직 관측되지 않은 미래에 대해 예측**해보겠습니다.


```python
# fit (total data)
final_model = SARIMAX(rawdata['diesel'], order = (0, 1, 1), seasonal_order = (1, 0, 1, 12))
final_result = final_model.fit()

# prediction
forecast = final_result.predict(len(rawdata), len(rawdata) + 11, typ='levels').rename('Final Forecast')

# plot
rawdata['diesel'].plot(legend=True, figsize=(12,8))
forecast.plot(legend=True)
```




    <AxesSubplot:xlabel='date'>




    
![png](output_56_1.png)
    


[(코드작성시 참고했던 자료)](https://github.com/jeswingeorge/Learning-Time-Series-Python/tree/master/06-General-Forecasting-Models)

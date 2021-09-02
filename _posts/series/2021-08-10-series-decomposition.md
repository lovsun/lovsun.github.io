---
layout: post
title: '[일변량통계] 시계열데이터분해 및 지수평활법을 통한 예측'
categories: timeseries
comments: true
published: true
---

### 데이터종류
시계열데이터에 대해 알아보기에 앞서, 데이터의 종류부터 살펴보겠습니다.
- **횡단면데이터(Cross Sectional data)** : 고정된 시점(t), 여러 자산(multiple assets)에 대한 데이터 (예, 오늘 수집된 S&P600 종목들의 일별종가와 EPS 데이터) 
- **시계열데이터(Time Series data)** : 일정기간(multiple time intervals), 단일 자산(one asset)에 대한 데이터( 예, 10년동안 수집된 삼성전자의 종가 데이터) 
- **패널데이터(Panel data)** : 일정기간(multiple time intervals), 여러 자산(multiple assets)의 다양한 속성(multidimensional)에 대한 데이터 (예, 10년동안 수집된 S&P500 종목의 일별종가와 EPS 데이터) = cross sectional + time series, x -> y (casuality)
- **다변량시계열(Multivariate Time Series data)** : 일정기간(multiple time intervals), 여러 자산(multiple assets)의 공통된 단일 속성(one-dimensional)에 대한 데이터 (예, 10년동안 수집된 S&P500 종목의 일별 종가), x <-> y (casuality)


## 구조모형(Structural Model)
구조모형은 종속변수(하나의 변수)의 변화를 설명하기 위해 **독립변수(다른 변수들)의 현재 및 과거값**을 이용하는 모델입니다. 주로 금융 혹은 경제이론을 기반으로 만들어진 모형으로, 장기예측에 적합하다는 특성이 있습니다. 

가령 무위험차익거래 조건을 생각해보아도, 단기간에는 성립하지 않지만 (가격 괴리 발생), 장기에는 시장가격으로 수렴할 것을 전제로 합니다. 그래서 **장기적으로는 이론을 바탕으로 만들어진 모형의 예측성능이 우수**하다고 할 수 있습니다. 

하지만 단기간에는 구조모형을 사용할 수 없는 경우가 많습니다. 가령 S&P500 주가지수를 설명하기 위해 PMI 지표를 사용할 경우, 주가지수는 매일 관측되지만 PMI지표는 한 달에 한번 발표되기 때문입니다.


## 일변량시계열모형(Univariate Time Series Model)
일변량시계열모형은 이름에서 알 수 있듯이, 하나의 변수만 사용되게 됩니다. 

그래서 한 변수의 **과거 자기자신의 값과 오차값**을 이용하여 모델링하는 방식입니다. 즉 시계열 변수가 움직이는 특성을 관찰하여 적합한 모형을 찾는 것에 핵심이 있습니다.

참고로 VAR모형은 구조모형과 시계열모형 속성이 합쳐진 다변량 시계열 모형을 의미합니다.

지금부터 **시계열데이터분해, 지수평활법 패키지**에 대해 살펴보겠습니다.
그리고 다음 글에서 지수평활법과 함께 많이 쓰이는 **ARIMA 예측모델**에 대해 살펴보도록 하겠습니다.

## 데이터탐색 및 전처리
실습에 이용할 데이터는 [한국석유공사에서 제공하는 경유 소비데이터](https://www.petronet.co.kr/v3/index.jsp)를 이용하겠습니다. 

시계열데이터에 대한 분석을 진행하기 전에 데이터를 불러오고 전처리부터 해주겠습니다.

> DATE freq info 추가하기

여기서 중요한 점은 향후 파이썬에서 제공하는 통계패키지로 시계열데이터를 분석하려면, 날짜의 frequency information을 넣어주어야합니다. 저희 예에서는 월초데이터이기 때문에 ``MS``를 넣어주었습니다.

frequency information을 넣어줄 땐 [Pands Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html) 에서 DateOffset objects 부분을 참고하면 됩니다.


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import numpy as np
import pandas as pd

rawdata = pd.read_csv("경유월간소비.csv", encoding='cp949')
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
      <th>날짜</th>
      <th>경유</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00년?01월</td>
      <td>10390</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>[68.34]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>02월</td>
      <td>10023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>[66.47]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>03월</td>
      <td>11786</td>
    </tr>
  </tbody>
</table>
</div>




```python
rawdata = rawdata[::2]
rawdata.columns = ['date', 'diesel']

#필요한 행만 추출
rawdata['date'] = pd.date_range('2000-01-01','2021-06-01', freq='MS')
rawdata.set_index('date', inplace=True)

#freq info 추가
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




```python
rawdata.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 258 entries, 2000-01-01 to 2021-06-01
    Freq: MS
    Data columns (total 1 columns):
     #   Column  Non-Null Count  Dtype
    ---  ------  --------------  -----
     0   diesel  258 non-null    int64
    dtypes: int64(1)
    memory usage: 4.0 KB
    

데이터를 간단히 시각화하여 특징을 확인하겠습니다. 

trend는 있는 것처럼 보이고, 계절산포 변동을 일정한 편이나 정확한 계절성을 파악하기 어렵습니다.


```python
%matplotlib inline
import matplotlib.pyplot as plt
rawdata.plot(figsize=(12, 8))
```




    <AxesSubplot:xlabel='date'>




![output_9_1](https://user-images.githubusercontent.com/68403764/130751676-7f8a71cf-6819-4567-92b2-33191f3485a5.png)
    


## 시계열데이터속성
본격적인 예측에 들어가기 전에, 시계열데이터의 성분부터 알아보겠습니다. 

시계열데이터의 변동에는 TCSI 성분이 있습니다.

> TCSI

1) **Trend(추세변동)** : 시간이 지남에 따라 방향으로는 증가(upward)하거나 감소(downward)하는 **장기트렌드**를 가질 수 있습니다. 그리고 크기로는 선형적(linear)으로 증가/감소할 수 있고 비선형적(nonlinear)으로 증가/감소할 수 있습니다.

2) **Cycle(순환변동)** : 시간이 지남에 따라 반복적이지 않고 **사전에 정해지지 않은(unknown) 주기**를 가질 수 있습니다. 비교적 cycle이 긴 경우에 해당합니다. 

3) **Seasonality(계절변동)** : 시계열데이터는 단위기간 내에서 **반복적이고 알려진(known) 주기**를 가질 수 있습니다. 비교적 cycle이 주별, 월별, 계절별 처럼 1년 이내인 경우가 이에 해당됩니다.

4) **Irregularity(불규칙변동)** : 시간에 의존적이지 않고 **랜덤한 원인**에 의해 나타나는 변동을 말합니다. 대표적인 예가 White Noise Process입니다. White Noise Process는 평균과 분산이 일정하며, 자기공분산(서로 다른 시점간 분산)도 0이 되어 자기상관이 없는 데이터를 의미합니다. 


# 1. 시계열데이터분해(Decomposition)
시계열데이터를 분해한다는 것은 시계열데이터의 관측값을 **추세, 계절, 불규칙 성분**에 따라 분해하는 과정을 의미합니다. 

statsmodels 통계패키지는 시계열데이터를 분해하는 [seasonal_decompose](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html) 함수를 제공합니다. 

해당 함수는 시계열데이터의 관측값을 추세, 계절, 불규칙 성분에 따라 분해해줍니다. **각 성분의 기여도** 즉 수치범위를 확인하여 전체 변동에 어떤 성분이 유의미한 영향을 끼치는지 확인할 수 있습니다.

저희가 결정할 요인은 오직 하나 **model이 additive인지 multiplicative인지 지정**해주면 됩니다.

> model = addictive vs multiplicative


## 1) 성분분해 및 방식

### - Additive Decomposition Model
y_t(t시점의 관측값) = Tt(추세) + St(계절성) + It(불규칙)

### - Multiplicative Decomposition Model
y_t(t시점의 관측값) = Tt(추세) * St(계절성) * It(불규칙)

두 가지 모두 추세, 계절성, 불규칙 성분이 있다는 공통점이 있습니다. 

그러면 두 방식을 구분하는 방법은 무엇일까요? 바로 데이터를 plot을 통해 그려보았을 때, 시간이 지나도 **변동의 높낮이**가 일정한 경우에는 addictive form을 사용하면 되고, 높낮이 즉 변동크기가 증폭되면 multiplicative form을 사용하면 됩니다.

#### (참고) Multiplicative -> Addictive
흔히 시계열데이터에서 계절성분과 불규칙성분은 제거해야한다는 말을 많이 하잖아요? 

해당 패키지의 편리한 점은 성분별로 기여도를 알 수 있기 때문에 계절효과를 제거하고 추세성분만 관찰할 수 있습니다. 

이때 가법모델의 경우는 단순히 뺄셈으로 계절성을 제거할 수 있지만, 승법모델은 곱셈으로 연결되어 있기 때문에 이에 어려움이 있습니다.

하지만 **승법모델 또한 적절한 변형을 통해(로그를 취하여) 가법모델로 변경**할 수 있습니다.
log(y_t) = log(Tt*St*It) = log(Tt) + log(St) + log(It) 즉 승법모델을 가법모델로 변환하였기 때문에, 로그를 취한 값에 시계열분해를 적용한 후, 얻어진 추세에 exp 함수만 취해준다면 추세만 관찰할 수 있게 됩니다.


```python
from statsmodels.tsa.seasonal import seasonal_decompose
```


```python
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
```


```python
decomposition = seasonal_decompose(rawdata, model='additive', extrapolate_trend='freq')  # extrapolate_trend 파라미터는 NaN값 피하기 위한 것

decomposition.plot()
```




    
![output_14_0](https://user-images.githubusercontent.com/68403764/130751678-12782bfe-e00d-4eb7-90cb-165cc93cb8da.png)
    



```python
# 계절성 확인을 위한 plot
decomposition.seasonal.plot(figsize=(12, 8))
```




    <AxesSubplot:xlabel='date'>




    
![output_15_1](https://user-images.githubusercontent.com/68403764/130751683-d28fb9ae-7a92-4cdd-b749-f191ec2e6616.png)
    



```python
df_decomposition = pd.DataFrame({'date' : rawdata.index.tolist(),
                                 'observed' : decomposition.observed.tolist(),
                                 'trend' : decomposition.trend.tolist(),
                                 'seasonal' : decomposition.seasonal.tolist(),
                                 'resid' :decomposition.resid.tolist()})
df_decomposition.head()
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
      <th>date</th>
      <th>observed</th>
      <th>trend</th>
      <th>seasonal</th>
      <th>resid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000-01-01</td>
      <td>10390.0</td>
      <td>10613.169823</td>
      <td>-517.092139</td>
      <td>293.922316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000-02-01</td>
      <td>10023.0</td>
      <td>10626.364899</td>
      <td>-1260.536235</td>
      <td>657.171336</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000-03-01</td>
      <td>11786.0</td>
      <td>10639.559975</td>
      <td>469.568912</td>
      <td>676.871114</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000-04-01</td>
      <td>11086.0</td>
      <td>10652.755051</td>
      <td>-75.784275</td>
      <td>509.029225</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000-05-01</td>
      <td>12208.0</td>
      <td>10665.950126</td>
      <td>326.252690</td>
      <td>1215.797184</td>
    </tr>
  </tbody>
</table>
</div>



## 2) 계절효과 그래프
이렇게 성분분해를 하다보면 계절효과를 자세히 살펴보고 싶어질 때가 생깁니다. 

statsmodels 패키지에 **month_plot()함수와 quarter_plot()함수**를 통해 **계절성을 유용하게 관찰**할 수 있습니다.

(주의사항) 해당 그래프를 사용할 때 한 가지 유의할 점은 데이터의 주기에 따라서 resample이 필요할 수 있습니다. 

#### month_plot() 
시계열데이터를 월별로 서브셋을 만들어 그래프를 생성합니다. 그리고 각 서브셋의 평균을 표시하여줍니다.
#### quarter_plot() 
시계열데이터를 분기별로 서브셋을 만들어 그래프를 생성합니다. 그리고 month_plot처럼 서브셋의 평균을 표시해 줍니다.


```python
from statsmodels.graphics.tsaplots import month_plot, quarter_plot
```


```python
month_plot(rawdata) #겨울철 난방용 경유수요가 증가한 이유때문인지 12월에 수요가 높아지는걸 확인할 수 있습니다.
```




    
![output_19_0](https://user-images.githubusercontent.com/68403764/130751684-fc57c287-762b-4624-bf5a-f940fa0c9edd.png)
    

    



```python
quarter_plot(rawdata['diesel'].resample(rule='Q').mean())
```




    
![output_20_0](https://user-images.githubusercontent.com/68403764/130751689-8b88f964-406c-4dd4-838c-5c9b1c71fda0.png)





# 2. Simple Moving Average(SMA)
본격적으로 많이 활용되는 예측모델을 살펴보기 전에, simple moving average부터 살펴보겠습니다. 

SMA은 단어에서 알 수 있듯이 **단순히 과거평균 값**을 이용하는 것입니다. 저희는 **윈도우 사이즈(window size)** 만 정해주면 됩니다. 윈도우사이즈가 길수록 일반적인 트렌드를 알 수 있고 윈도우 사이즈가 작을수록 계절요인같은 조금 더 디데틸한 트렌드를 파악할 수 있습니다. 

SMA는 불규칙변동을 완화하여 트렌드를 파악할 수 있는 효과가 있습니다.

> 파라미터 : 윈도우사이즈(window size)


```python
data = rawdata.copy()
```


```python
data['SMA'] = data['diesel'].rolling(window=12).mean()
data.tail(10)
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
      <th>SMA</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-01</th>
      <td>13783</td>
      <td>13735.166667</td>
    </tr>
    <tr>
      <th>2020-10-01</th>
      <td>13462</td>
      <td>13639.500000</td>
    </tr>
    <tr>
      <th>2020-11-01</th>
      <td>15703</td>
      <td>13709.833333</td>
    </tr>
    <tr>
      <th>2020-12-01</th>
      <td>14471</td>
      <td>13644.166667</td>
    </tr>
    <tr>
      <th>2021-01-01</th>
      <td>12961</td>
      <td>13741.083333</td>
    </tr>
    <tr>
      <th>2021-02-01</th>
      <td>12749</td>
      <td>13783.833333</td>
    </tr>
    <tr>
      <th>2021-03-01</th>
      <td>13049</td>
      <td>13788.166667</td>
    </tr>
    <tr>
      <th>2021-04-01</th>
      <td>14105</td>
      <td>13885.750000</td>
    </tr>
    <tr>
      <th>2021-05-01</th>
      <td>14253</td>
      <td>13785.750000</td>
    </tr>
    <tr>
      <th>2021-06-01</th>
      <td>14754</td>
      <td>13820.666667</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
data.plot(figsize=(12,8))
```




    <AxesSubplot:xlabel='date'>




    
![output_24_1](https://user-images.githubusercontent.com/68403764/130751695-7290f24b-afee-4c3a-b6dc-9ac5269db94a.png)
    


# 3. Exponential Smoothing(지수평활법)
Smoothing 방법론은 단순하면서 비교적 단기예측에 우수한 성능을 보여줘서, 실제로 현업에서 예측에 많이 사용되는 방법론입니다.  
이는 **과거값들의 선형결합**을 이용하여 미래값을 예측하는 방법입니다. 지수예측모델은 statsmodels 패키지에 포함된 함수를 이용하여 편리하게 모델링할 수 있습니다

SMA는 단순히 평균을 구해주기 때문에, 과거값과 현재값에 똑같은 가중치를 두게 됩니다. 이를 해결하기 위해 나온 것이 **Exponential Smoothing(지수평활법)** 입니다. 


지수평활볍의 **장점**은 p, q를 정해줘야하는 ARMA모형과 달리, smoothing factor만 정해주면 되기 때문에 사용하기 쉬우며 새로운 관측값이 생기면 바로 값을 업데이트할 수 있습니다. 

**단점**은 시계열의 최근값에 과도하게 영향을 받으므로 예측기간이 증가하여도 시계열 데이터가 시간이 지남에 따라 평균가격으로 수렴한다는 평균회귀(Mean Reversion) 속성을 만족하지 못합니다.

#### Exponential
여기서 지수(Exponential)의 의미는 exponentially 감소하는 그래프를 생각하면 쉬울 것입니다. 해당 그래프처럼 시계열의 **과거값들에 대해 감소하는 가중치를 부여**하겠다는 말입니다.

#### Smoothing
smoothing 용어에서 기억해야할 점은 **Smoothing Factor(smoothing constant)** 입니다. 이는 **최근 값에 얼마나 더 많은 가중치를 부여** 할지 결정해주는 파라미터라고 생각하면 됩니다. 값이 클수록 최근의 관츢값에 더 큰 가중치가 부여됩니다. 이렇게 smoothing factor을 이용하므로써 저희는 최근값에 더 큰 가중치를 둔 스무딩된 값을 구할 수 있습니다.


## 1) Simple Exponential Smoothing(단순지수평활법)

단순지수평활법은 수준, 트렌드, 계절 요인 중에서, **Level(수준)만을 추정**하여 줍니다. 그래서 트렌드나 계절적 요인이 없는 데이터 미래값을 예측하는 데에 적합합니다.

### $y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ...
+ (1 - \alpha)^t x_{0}}{1 + (1 - \alpha) + (1 - \alpha)^2 + ...
+ (1 - \alpha)^t}$

저희가 지정해줘야할 파라미터는 1개로 최근값에 얼마나 더 많은 가중치를 줄 지 결정을 해주면 됩니다. α는 최근 Level에 얼마나 더 많은 가중치를 부여할지 정하는 파라미터입니다. 

span은 가중치를 부여할 데이터의 개수를 의미하며,  α = 2 / (span + 1) 로 계산됩니다.

> 파라미터(1개) : α(smoothing_level)



```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

span = 12
alpha = 2 / (span + 1)

#해당 날짜에 예측값이 형성되기 때문에 time을 1씩 이동시켜야 해당 날짜에 예측한 다음값을 알 수 있다.
fitted = SimpleExpSmoothing(data['diesel']).fit(smoothing_level=alpha, optimized=False).fittedvalues
data['SES']=fitted.shift(-1)
```


```python
data.iloc[-48:].plot(figsize=(12,8));
```


    
![output_27_0](https://user-images.githubusercontent.com/68403764/130751697-25a463de-9f61-4f37-8c69-e3d68a56e437.png)
    


## 2) Holt's double Exponential Smoothing(홀트지수평활법)
홀트지수평활법은 추세를 갖는 시계열데이터에 대한 예측모델을 생성합니다. 추세는 관찰되지만 계절적 변동이 존재하지 않는 데이터에 적합합니다. 그래서 **Level(수준)뿐만 아니라 Slope(기울기)** 에 대한 smoothing factor을 지정해줘야합니다. 그래서 이를 **홀트지수평활법**이라고도 불리며 **이중지수평활법(Double Exponential Smoothing)**이라고도 불립니다.

(Initialization) L1 = Y1, T1 = 0

Level: Lt = α * Yt + (1 – α) * (Lt-1 + Tt-1)

Trend: Tt = β * (Lt – Lt-1) + (1 – β) * Tt-1

Forecast for period m: Ft+m = Lt + m*Tt


> 파라미터(2개) : α(smoothing_level) β(smoothing_trend)

참고로, ExponentialSmoothing함수에 trend 파라미터를 추가하면, 함수는 Double Exoponential Smoothing을 인식하여 α, β를 자동으로 선정해줍니다.


```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

fit_data = ExponentialSmoothing(data['diesel'], trend='add').fit()

parameters = fit_data.params
fitted = fit_data.fittedvalues
data['DES'] = fitted.shift(-1)
data.head()
```

    C:\Users\user\.conda\envs\nnet\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    




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
      <th>SMA</th>
      <th>SES</th>
      <th>DES</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>10390</td>
      <td>NaN</td>
      <td>10390.000000</td>
      <td>9717.608253</td>
    </tr>
    <tr>
      <th>2000-02-01</th>
      <td>10023</td>
      <td>NaN</td>
      <td>10333.538462</td>
      <td>9409.743301</td>
    </tr>
    <tr>
      <th>2000-03-01</th>
      <td>11786</td>
      <td>NaN</td>
      <td>10556.994083</td>
      <td>9456.062241</td>
    </tr>
    <tr>
      <th>2000-04-01</th>
      <td>11086</td>
      <td>NaN</td>
      <td>10638.379609</td>
      <td>9428.046794</td>
    </tr>
    <tr>
      <th>2000-05-01</th>
      <td>12208</td>
      <td>NaN</td>
      <td>10879.859669</td>
      <td>9628.032349</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(parameters)
```

    {'smoothing_level': 0.1464285714285714, 'smoothing_trend': 0.1464285714285714, 'smoothing_seasonal': nan, 'damping_trend': nan, 'initial_level': 10390.0, 'initial_trend': -367.0, 'initial_seasons': array([], dtype=float64), 'use_boxcox': False, 'lamda': None, 'remove_bias': False}
    


```python
data[['diesel', 'SES', 'DES']].iloc[-48:].plot(figsize=(12,8));
```


    
![output_31_0](https://user-images.githubusercontent.com/68403764/130751699-d5f463fc-94bb-4ef0-a2d7-4d4df2ee1616.png)
    


## 3) Holt-Winters Exponential Smoothing(홀트-윈터지수평활법)
홀트윈터지수평활법은 추세와 계절적 변동 모두 가지고 있는 시계열데이터에 대해 적합합니다. **Level(수준)뿐만 아니라 Slope(기울기), 계절 성분**에 대한 파라미터 모두 지정해줘야합니다. 

그래서 **삼중지수평활법(Triple Exponential Smoothing)** 이라고도 불립니다. 참고로 **홀트윈터지수평활법** 에서 계절성을 처리하는 방식은 시계열데이터분해처럼 2가지가 있습니다. 아래의 공식을 통해 살펴보겠습니다.

### 계절성처리방식
- additive seasonal form

![image.png](https://docs.oracle.com/cd/E12825_01/epm.111/cb_statistical/images/graphics/as_smoothing.gif)

- multiplicative seasonal form

![image-2.png](https://docs.oracle.com/cd/E12825_01/epm.111/cb_statistical/images/graphics/ms_smoothing.gif)

> 가법(additive) / 승법(multiplicative) 방식 중 선택

> 파라미터(3개) : α(smoothing_level), β(smoothing_trend), γ(smoothing_seasonal)


```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

fit_data = ExponentialSmoothing(data['diesel'], trend='add', seasonal='add', seasonal_periods=12).fit()

parameters = fit_data.params
fitted = fit_data.fittedvalues
data['TES'] = fitted.shift(-1)
data.head()
```

    C:\Users\user\.conda\envs\nnet\lib\site-packages\statsmodels\tsa\holtwinters\model.py:922: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      ConvergenceWarning,
    




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
      <th>SMA</th>
      <th>SES</th>
      <th>DES</th>
      <th>TES</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>10390</td>
      <td>NaN</td>
      <td>10390.000000</td>
      <td>9717.608253</td>
      <td>10004.377672</td>
    </tr>
    <tr>
      <th>2000-02-01</th>
      <td>10023</td>
      <td>NaN</td>
      <td>10333.538462</td>
      <td>9409.743301</td>
      <td>11838.033464</td>
    </tr>
    <tr>
      <th>2000-03-01</th>
      <td>11786</td>
      <td>NaN</td>
      <td>10556.994083</td>
      <td>9456.062241</td>
      <td>11096.779774</td>
    </tr>
    <tr>
      <th>2000-04-01</th>
      <td>11086</td>
      <td>NaN</td>
      <td>10638.379609</td>
      <td>9428.046794</td>
      <td>12178.967439</td>
    </tr>
    <tr>
      <th>2000-05-01</th>
      <td>12208</td>
      <td>NaN</td>
      <td>10879.859669</td>
      <td>9628.032349</td>
      <td>10667.927077</td>
    </tr>
  </tbody>
</table>
</div>




```python
#smoothing_level, smoothing_trend, smoothing_seasonal 모두 값이 들어가있는걸 확인할 수 있습니다.
print(parameters) 
```

    {'smoothing_level': 0.10739132563294301, 'smoothing_trend': 0.0037375152280230434, 'smoothing_seasonal': 0.3176991155703051, 'damping_trend': nan, 'initial_level': 11694.586850756077, 'initial_trend': 17.094292367735513, 'initial_seasons': array([-1277.98950905, -1719.68812377,    94.88356098,  -657.84553107,
             408.44076977, -1122.78810859, -2082.82581836,  -973.82152255,
           -1233.6759682 , -1589.42395718,  -823.2876392 ,   226.23204286]), 'use_boxcox': False, 'lamda': None, 'remove_bias': False}
    


```python
data[['diesel', 'SES', 'DES', 'TES']].iloc[-48:].plot(figsize=(12,8));
```


    
![output_35_0](https://user-images.githubusercontent.com/68403764/130751701-186ce4b9-c254-4ca7-873d-2558e2982186.png)
    


확실히 경유 수요의 경우, 위에서 봤던 것처럼 계절성이 있어서 Triple Exponential Smoothing 즉 수준, 추세, 계절성 모두 반영한 홀트윈터지수평활법의 예측성능이 좋은 것을 확인할 수 있습니다.

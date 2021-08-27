---
layout: post
title:  "[일변량통계] 페이스북 Prophet 예측 라이브러리"
categories: timeseries
comments: true
---
### Facebook's Prophet Forecasting Library
앞서서 시계열 예측방법론으로 **ARIMA 모델**에 대해 살펴보았습니다. 하지만 ARIMA 모델의 경우, **차분, 자기상관, 모형에 대한 가정 등 여러 이론에 대해 알아야한다는 단점**이 있습니다. 이를 위해 **시계열데이터 이론에 대해 잘모르더라도 사용할 수 있는 페이스북 Prophet 라이브러리**를 소개하겠습니다. 

페이스북이 소스코드까진 자세히 공개하지 않아 내부적으로 어떻게 작동하는지 확실히는 알 수 없지만, [Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start.html)에 사용법이 자세히 나와있습니다.

해당 라이브러리를 시계열데이터를 크게 3가지 성분을 이용하여 예측을 도와줍니다.
### 1) Trend
- piecewise linear growth curve trend
- piecewise logistic growth curve trend 

piecewise는 조각이라는 의미를 가지고 있습니다. 시간에 따라서 여러 방정식으로 묶일 수 있는걸 의미하는데요. 그 말은 **시간이 지나면서 데이터에 변화(changepoint)가 생기면, 모델이 이를 자동으로 감지**하여 예측해줍니다.

### 2) Seasonality
- yearly seasonal component 

계절성의 경우, **푸리에급수(Fourier series)**를 이용하여 포착합니다. 주기를 삼각함수 가중치로 분해합니다. 이때 주기를 모르더라도 상관없습니다.  모델이 **주기를 감지하고 자동으로 찾아내줍니다.** 

그리고, 지수평활법처럼 계절산포변화에 따라서 **addictive인지 multiplicative seasonality인지 선택**할 수 있습니다.
- weekly seasonal compnent

위에 푸리에급수를 이용하여 계절성을 포착한다고 했습니다.

다만 예외적으로 weekly seasonaliy일 경우는 **더미변수**를 이용하여 포착합니다.

### 3) holiday/event
위의 Trend, Seasonality는 앞서 작성한 지수평활법과 SARIMA 모델도 포착해줍니다. 페이스북 Prophet은 이 두 방법론과 달리 holiday를 넣어줄 수 있는데요. 아무래도 비즈니스 상황에서 휴일(명절, 크리스마스 등)에는 매출의 변동이 커질 수 밖에 없습니다. 그리고 기업의 업종에 따라서 매출의 변동이 큰 휴일은 다를 수 있습니다. 

그래서 **분석하는 데이터 종류에 따라서 외생(exogenous) 변수로 지정**해줄 수 있습니다.


## 1. 데이터수집 및 데이터전처리
해당 라이브러리를 이용하려면 사전에 무조건 전처리 해줘야하는 사항이 있습니다.


**날짜컬럼**의 컬럼명은 'ds'이어야하고 데이터유형은 datetime이어야합니다. 

그리고 시계열데이터가 들어간 **수치컬럼**의 컬럼명은 'y'이어야합니다.


> 날짜컬럼 : column name = 'ds' , dtype = datetime

> 수치컬럼 : column name = 'y'

> (주의) 날짜컬럼은 인덱스가 아니어야합니다.




```python
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

from fbprophet import Prophet
```


```python
rawdata = pd.read_csv("경유월간소비.csv", encoding='cp949')
rawdata = rawdata[::2]
rawdata.reset_index(inplace=True, drop=True)
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
      <td>02월</td>
      <td>10023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>03월</td>
      <td>11786</td>
    </tr>
    <tr>
      <th>3</th>
      <td>04월</td>
      <td>11086</td>
    </tr>
    <tr>
      <th>4</th>
      <td>05월</td>
      <td>12208</td>
    </tr>
  </tbody>
</table>
</div>



위의 데이터를 이용하여 전처리 작업을 해보겠습니다.


```python
##### column name = 'ds'.  column name = 'y'#####
rawdata.columns = ['ds', 'y']

##### datetime object 생성 #####
rawdata['ds'] = pd.date_range('2000-01-01','2021-06-01', freq='MS')

rawdata['y'] = rawdata['y'].apply(pd.to_numeric)
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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000-01-01</td>
      <td>10390</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000-02-01</td>
      <td>10023</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000-03-01</td>
      <td>11786</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000-04-01</td>
      <td>11086</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000-05-01</td>
      <td>12208</td>
    </tr>
  </tbody>
</table>
</div>



데이터준비가 끝났다면 info를 찍어서
**아래 두 가지 사항을 꼭 확인해주세요.**
- 컬럼명이 ds, y로 들어가 있는가
- ds컬럼의 dtype이 datetime인가


```python
rawdata.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 258 entries, 0 to 257
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype         
    ---  ------  --------------  -----         
     0   ds      258 non-null    datetime64[ns]
     1   y       258 non-null    int64         
    dtypes: datetime64[ns](1), int64(1)
    memory usage: 4.2 KB
    


```python
rawdata.plot(x='ds', y='y')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f19f5294610>




    
![prophet_9_1](https://user-images.githubusercontent.com/68403764/130752294-c5bcc228-a15e-49f8-a48e-ec546d85e9b6.png)
    


아래 코드는 이후 제가 growth = 'logistic'을 사용하기 위해 임의의 최대값과 최소값을 지정해두었습니다. 

즉 growth를 linear일 경우 필요없는 코드입니다.


```python
### growth='linear'일 경우 필요없는 코드 ###
rawdata['cap'] = round((rawdata['y'].max()) + (1*rawdata['y'].std()))
rawdata['floor'] = round((rawdata['y'].min()) - (1*rawdata['y'].std()))
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
      <th>ds</th>
      <th>y</th>
      <th>cap</th>
      <th>floor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000-01-01</td>
      <td>10390</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000-02-01</td>
      <td>10023</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000-03-01</td>
      <td>11786</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000-04-01</td>
      <td>11086</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000-05-01</td>
      <td>12208</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
  </tbody>
</table>
</div>



## 2. train-test split
예측을 얼마나 잘 수행하는지 알아보니 위해 데이터 split 먼저 해주겠습니다.

 12개월 즉 1년치 데이터를 test data로 사용하도록 하겠습니다.


```python
train = rawdata.iloc[:-12]
test = rawdata.iloc[-12:] 
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
      <th>ds</th>
      <th>y</th>
      <th>cap</th>
      <th>floor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>241</th>
      <td>2020-02-01</td>
      <td>12236</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>242</th>
      <td>2020-03-01</td>
      <td>12997</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2020-04-01</td>
      <td>12934</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2020-05-01</td>
      <td>15453</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>245</th>
      <td>2020-06-01</td>
      <td>14335</td>
      <td>17984</td>
      <td>6528</td>
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
      <th>ds</th>
      <th>y</th>
      <th>cap</th>
      <th>floor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>246</th>
      <td>2020-07-01</td>
      <td>13518</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>247</th>
      <td>2020-08-01</td>
      <td>13040</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>248</th>
      <td>2020-09-01</td>
      <td>13783</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>249</th>
      <td>2020-10-01</td>
      <td>13462</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>250</th>
      <td>2020-11-01</td>
      <td>15703</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
  </tbody>
</table>
</div>



## 3. 모델생성 및 학습, 예측

모델을 생성하고 학습시키기 위해 ``Prophet`` 의 ``init signature`` 중 일부의 기본값을 먼저 살펴보겠습니다.

### 1) Trend
- **growth**='linear' : logistic으로 변경가능
- **changepoints**=None
- **n_changepoints**=25 

현재 예제에서는 growth를 logsitic으로 사용해보겠습니다. logistic으로 사용하려면 dataframe에 미리 cap 혹은 floor 컬럼을 추가해놓아야합니다. cap은 해당 시기에 가능한 maximum 값이고 floor은 해당시기에 가능한 minimum 값입니다.

### 2) Seasonality
- **yearly_seasonality**='auto'
- **weekly_seasonality**='auto'
- **daily_seasonality**='auto'
- **seasonality_mode**='additive' : multiplicative로 변경가능

현재예제에서는 시간이 지나면서 산포변동이 크게 줄거나 증가하지않아서 additive로 사용하겠습니다.

### 3) holiday/event
- **holidays**=None 

이에 대해 참고사항으로 말하자면, 만약 holiday를 지정해줄 경우 

dataframe 형태(컬럼1 : holiday(string), 컬럼2 : ds(date type), 컬럼option : lower_window, upper_window)로 넣어주셔야 합니다. 

#### (참고) 예측범위 관련
- **interval_width**=0.80 

해당 값을 조정하여 예측값의 upper, lower 범위를 조정할 수 있습니다.

#### (참고) 과적합 방지를 위해, scale 조정 가능
- **changepoint_prior_scale**=0.05
- **seasonality_prior_scale**=10.0
- **holidays_prior_scale**=10.0

그러면 모델을 생성하고 fit하고 예측하는 코드를 확인해보겠습니다.

> (주의) daily data가 아니라면 예측할 날짜를 입력할때, freqency를 무조건 바꾸어줘야합니다.


```python
# 1) create instance
m = Prophet(growth='logistic')

# 2) fit
m.fit(train)

# 3) 예측할 기간 입력
pred_date = m.make_future_dataframe(periods=12, freq='MS')

### growth='linear'일 경우 필요없는 코드 ###
pred_date['cap'] = round((rawdata['y'].max()) + (1*rawdata['y'].std()))
pred_date['floor'] = round((rawdata['y'].min()) - (1*rawdata['y'].std()))

# 4) predict
predictions = m.predict(pred_date)
```

    INFO:numexpr.utils:NumExpr defaulting to 2 threads.
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    


```python
predictions.tail() 
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
      <th>ds</th>
      <th>trend</th>
      <th>cap</th>
      <th>floor</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>additive_terms</th>
      <th>additive_terms_lower</th>
      <th>additive_terms_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>253</th>
      <td>2021-02-01</td>
      <td>14025.555352</td>
      <td>17984</td>
      <td>6528</td>
      <td>11302.205071</td>
      <td>13683.657899</td>
      <td>14024.102194</td>
      <td>14027.081665</td>
      <td>-1499.940307</td>
      <td>-1499.940307</td>
      <td>-1499.940307</td>
      <td>-1499.940307</td>
      <td>-1499.940307</td>
      <td>-1499.940307</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12525.615045</td>
    </tr>
    <tr>
      <th>254</th>
      <td>2021-03-01</td>
      <td>14039.637255</td>
      <td>17984</td>
      <td>6528</td>
      <td>13363.945894</td>
      <td>15748.773784</td>
      <td>14037.811910</td>
      <td>14041.523134</td>
      <td>487.911758</td>
      <td>487.911758</td>
      <td>487.911758</td>
      <td>487.911758</td>
      <td>487.911758</td>
      <td>487.911758</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14527.549012</td>
    </tr>
    <tr>
      <th>255</th>
      <td>2021-04-01</td>
      <td>14055.200124</td>
      <td>17984</td>
      <td>6528</td>
      <td>12808.267769</td>
      <td>15095.343385</td>
      <td>14052.937841</td>
      <td>14057.584397</td>
      <td>-145.892899</td>
      <td>-145.892899</td>
      <td>-145.892899</td>
      <td>-145.892899</td>
      <td>-145.892899</td>
      <td>-145.892899</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13909.307225</td>
    </tr>
    <tr>
      <th>256</th>
      <td>2021-05-01</td>
      <td>14070.232961</td>
      <td>17984</td>
      <td>6528</td>
      <td>13247.477124</td>
      <td>15632.647260</td>
      <td>14067.620634</td>
      <td>14072.979804</td>
      <td>397.388069</td>
      <td>397.388069</td>
      <td>397.388069</td>
      <td>397.388069</td>
      <td>397.388069</td>
      <td>397.388069</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14467.621030</td>
    </tr>
    <tr>
      <th>257</th>
      <td>2021-06-01</td>
      <td>14085.737760</td>
      <td>17984</td>
      <td>6528</td>
      <td>13189.394416</td>
      <td>15516.275978</td>
      <td>14082.568914</td>
      <td>14088.951766</td>
      <td>297.662275</td>
      <td>297.662275</td>
      <td>297.662275</td>
      <td>297.662275</td>
      <td>297.662275</td>
      <td>297.662275</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14383.400035</td>
    </tr>
  </tbody>
</table>
</div>



보시면 정말 많은 정보를 제공하고 있습니다.
너무 많은 정보는 때로는 의사결정에 방해가 되기 때문에, 필요한 정보만 골라보겠습니다.

각자 분석task에 따라서 필요한 정보를 고르시면 될 것 같습니다.

- ds : 날짜
- yhat : 예측값


```python
predictions.columns 
```




    Index(['ds', 'trend', 'cap', 'floor', 'yhat_lower', 'yhat_upper',
           'trend_lower', 'trend_upper', 'additive_terms', 'additive_terms_lower',
           'additive_terms_upper', 'yearly', 'yearly_lower', 'yearly_upper',
           'multiplicative_terms', 'multiplicative_terms_lower',
           'multiplicative_terms_upper', 'yhat'],
          dtype='object')




```python
predictions[['ds', 'yhat_lower', 'yhat_upper','yhat']].tail(12)
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
      <th>ds</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>246</th>
      <td>2020-07-01</td>
      <td>11510.854291</td>
      <td>13835.498893</td>
      <td>12683.631736</td>
    </tr>
    <tr>
      <th>247</th>
      <td>2020-08-01</td>
      <td>12677.287512</td>
      <td>15030.351107</td>
      <td>13851.407527</td>
    </tr>
    <tr>
      <th>248</th>
      <td>2020-09-01</td>
      <td>12561.144330</td>
      <td>14924.772144</td>
      <td>13700.847752</td>
    </tr>
    <tr>
      <th>249</th>
      <td>2020-10-01</td>
      <td>12715.762791</td>
      <td>15010.603904</td>
      <td>13891.378142</td>
    </tr>
    <tr>
      <th>250</th>
      <td>2020-11-01</td>
      <td>13301.914188</td>
      <td>15656.930280</td>
      <td>14464.361477</td>
    </tr>
    <tr>
      <th>251</th>
      <td>2020-12-01</td>
      <td>14446.349469</td>
      <td>16887.689283</td>
      <td>15645.215180</td>
    </tr>
    <tr>
      <th>252</th>
      <td>2021-01-01</td>
      <td>12256.628392</td>
      <td>14604.092087</td>
      <td>13421.961381</td>
    </tr>
    <tr>
      <th>253</th>
      <td>2021-02-01</td>
      <td>11302.205071</td>
      <td>13683.657899</td>
      <td>12525.615045</td>
    </tr>
    <tr>
      <th>254</th>
      <td>2021-03-01</td>
      <td>13363.945894</td>
      <td>15748.773784</td>
      <td>14527.549012</td>
    </tr>
    <tr>
      <th>255</th>
      <td>2021-04-01</td>
      <td>12808.267769</td>
      <td>15095.343385</td>
      <td>13909.307225</td>
    </tr>
    <tr>
      <th>256</th>
      <td>2021-05-01</td>
      <td>13247.477124</td>
      <td>15632.647260</td>
      <td>14467.621030</td>
    </tr>
    <tr>
      <th>257</th>
      <td>2021-06-01</td>
      <td>13189.394416</td>
      <td>15516.275978</td>
      <td>14383.400035</td>
    </tr>
  </tbody>
</table>
</div>



## 정확도평가 및 시각화

### prophet object 이용한 시각화
- 참고로 아래의 코드에서 xlim을 지정해주는 데에 오류가 난다면, ``pd.plotting.register_matplotlib_converters()`` 해당 코드를 실행해주길 바랍니다.


```python
# 예측값 시각화
m.plot(predictions);
plt.xlim(test['ds'].iloc[0], test['ds'].iloc[-1])
```




    (737607.0, 737942.0)




    
![prophet_24_1](https://user-images.githubusercontent.com/68403764/130752295-7fb96a2b-6bce-44b9-afea-2501ec7cc916.png)
    



```python
# 각 성분 시각화
m.plot_components(predictions);
```


    
![prophet_25_0](https://user-images.githubusercontent.com/68403764/130752281-19293a82-0e9a-43a3-abe2-d702288ae5d8.png)
    


아래의 코드로 trend가 크게 변했던 시점을 시각화할 수 있습니다.


```python
# changepoint 시각화
from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(predictions)
a = add_changepoints_to_plot(fig.gca(), m, predictions) #fig의 get current axis, model, 예측 df
```


    
![prophet_27_0](https://user-images.githubusercontent.com/68403764/130752285-d43810a5-5afd-4775-8158-efb922cb9d78.png)
    


또한, 아래 코드를 통해 주요 변화로 선정한 25 points 중 최근 날짜와 값을 확인할 수 있습니다.


```python
rawdata.loc[rawdata["ds"].isin(m.changepoints)].tail(10)
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
      <th>ds</th>
      <th>y</th>
      <th>cap</th>
      <th>floor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>125</th>
      <td>2010-06-01</td>
      <td>10791</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>133</th>
      <td>2011-02-01</td>
      <td>10343</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>140</th>
      <td>2011-09-01</td>
      <td>11224</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>148</th>
      <td>2012-05-01</td>
      <td>11788</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>156</th>
      <td>2013-01-01</td>
      <td>11505</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>164</th>
      <td>2013-09-01</td>
      <td>10527</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>172</th>
      <td>2014-05-01</td>
      <td>12336</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2014-12-01</td>
      <td>13290</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>187</th>
      <td>2015-08-01</td>
      <td>13654</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
    <tr>
      <th>195</th>
      <td>2016-04-01</td>
      <td>13967</td>
      <td>17984</td>
      <td>6528</td>
    </tr>
  </tbody>
</table>
</div>



### 예측데이터프레임을 이용한 시각화
아래 그래프를 보시면 상당히 잘 추정한 것을 확인할 수 있습니다.


```python
predictions.plot(x='ds', y='yhat', figsize=(12,8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f19e4615d10>




    
![prophet_31_1](https://user-images.githubusercontent.com/68403764/130752289-080c4dcf-1117-4820-aaff-feac247a5fa9.png)
    



```python
ax = predictions.plot(x='ds', y='yhat', label='prediction value', legend=True, figsize=(12,8))
test.plot(x='ds', y='y', label='actual value', legend=True, ax=ax, xlim=(test['ds'].iloc[0], test['ds'].iloc[-1]))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f19e44e96d0>




    
![prophet_32_1](https://user-images.githubusercontent.com/68403764/130752291-e048006b-81b4-457a-b8ca-9a2f9c88c0ae.png)
    


### RMSE를 통한 성능평가
현재 1표준표차 보다도 작은 오차가 나왔습니다. 이정도면 상당히 잘 예측할 수 있다고 할 수 있겠죠.


```python
from statsmodels.tools.eval_measures import rmse
rmse(predictions.iloc[-12:]['yhat'], test['y'])
```

    /usr/local/lib/python3.7/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning:
    
    pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
    
    




    770.7609568946554




```python
test['y'].mean(), test['y'].std()
```




    (13820.666666666666, 874.85347690948)



## Cross Validation

위의 방식은 test data만을 가지고 정확도를 평가하고 있습니다. 이에 대한 대안으로 **시간의 흐름에 따라 train과 test**를 할 수 있는 cross validation 함수를 제공하고 있습니다. [자세한 내용](https://facebook.github.io/prophet/docs/diagnostics.html)을 알고 싶은 분들을 위해 링크를 걸어두겠습니다. 
간략하게 내용을 요약하자면 아래의 3가지를 알고 있으면 됩니다.

> initial : train을 수행할 기간

> horizon : 예측할 기간

> period : cutoff date 간 간격

(주의사항) 일별 string 형태로 넣어줘야합니다. 가령 1년이면 '365 days'로 넣어줘야합니다.

결국 initial로 준 기간만큼 train data로 활용하여, cutoff 후 horizon으로 정한 기간만큼 예측을 수행하는 것입니다. 그리고 cutoff 간 간격을 period로 지정하여, 몇 번 fold 할지 정할 수 있습니다.

### cross validation 시행


```python
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
```


```python
initial = 5 * 365 # 5년
initial = str(initial) + ' days'

horizon = 365 # 1년
horizon = str(horizon) + ' days'

period = 3 * 365 # 3년
period = str(period) + ' days'
```


```python
result_cv = cross_validation(m, initial=initial, period=period, horizon=horizon)
```

    INFO:fbprophet:Making 5 forecasts with cutoffs between 2007-06-05 00:00:00 and 2019-06-02 00:00:00
    


      0%|          | 0/5 [00:00<?, ?it/s]



```python
result_cv.tail()
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
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>y</th>
      <th>cutoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>55</th>
      <td>2020-02-01</td>
      <td>12841.716982</td>
      <td>11714.521645</td>
      <td>13918.128437</td>
      <td>12236</td>
      <td>2019-06-02</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2020-03-01</td>
      <td>14615.775686</td>
      <td>13488.943993</td>
      <td>15734.298355</td>
      <td>12997</td>
      <td>2019-06-02</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2020-04-01</td>
      <td>13874.051921</td>
      <td>12741.650672</td>
      <td>14933.858190</td>
      <td>12934</td>
      <td>2019-06-02</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2020-05-01</td>
      <td>14461.961995</td>
      <td>13271.681164</td>
      <td>15623.987344</td>
      <td>15453</td>
      <td>2019-06-02</td>
    </tr>
    <tr>
      <th>59</th>
      <td>2020-06-01</td>
      <td>14165.556871</td>
      <td>12964.181996</td>
      <td>15391.387318</td>
      <td>14335</td>
      <td>2019-06-02</td>
    </tr>
  </tbody>
</table>
</div>



### 성능평가
일별 기준으로 예측기간에 따라 다양한 정확도 지표를 제공합니다.

이를 활용하여 **얼마나 긴 기간을 예측하는 것이 적당**한지도 살펴보고, 예측기간을 선정하는 데에 도움이 될 것 같습니다.


```python
performance_metrics(result_cv).tail(10)
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
      <th>horizon</th>
      <th>mse</th>
      <th>rmse</th>
      <th>mae</th>
      <th>mape</th>
      <th>mdape</th>
      <th>coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>303 days</td>
      <td>1.776023e+06</td>
      <td>1332.675097</td>
      <td>1194.057989</td>
      <td>0.098426</td>
      <td>0.101993</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>35</th>
      <td>304 days</td>
      <td>1.627359e+06</td>
      <td>1275.679966</td>
      <td>1128.642309</td>
      <td>0.095062</td>
      <td>0.091900</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>331 days</td>
      <td>1.213532e+06</td>
      <td>1101.604288</td>
      <td>979.272296</td>
      <td>0.083311</td>
      <td>0.068418</td>
      <td>0.583333</td>
    </tr>
    <tr>
      <th>37</th>
      <td>332 days</td>
      <td>9.947461e+05</td>
      <td>997.369612</td>
      <td>918.493902</td>
      <td>0.074837</td>
      <td>0.068418</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>38</th>
      <td>333 days</td>
      <td>1.481513e+06</td>
      <td>1217.174321</td>
      <td>1102.907358</td>
      <td>0.085486</td>
      <td>0.091900</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>334 days</td>
      <td>1.254660e+06</td>
      <td>1120.115954</td>
      <td>1012.950882</td>
      <td>0.077655</td>
      <td>0.068407</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>40</th>
      <td>362 days</td>
      <td>3.806386e+06</td>
      <td>1950.996197</td>
      <td>1669.845572</td>
      <td>0.148390</td>
      <td>0.096092</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>41</th>
      <td>363 days</td>
      <td>3.660842e+06</td>
      <td>1913.332712</td>
      <td>1557.582293</td>
      <td>0.137917</td>
      <td>0.096092</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>364 days</td>
      <td>4.315090e+06</td>
      <td>2077.279502</td>
      <td>1803.413864</td>
      <td>0.153244</td>
      <td>0.134048</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>43</th>
      <td>365 days</td>
      <td>3.735180e+06</td>
      <td>1932.661330</td>
      <td>1519.485769</td>
      <td>0.133872</td>
      <td>0.102088</td>
      <td>0.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_cross_validation_metric(result_cv, metric='rmse');
```


![prophet_45_0](https://user-images.githubusercontent.com/68403764/130752293-b5641df0-9c8d-41d5-863f-a3a2580d9045.png)
    


해당 경유소비량은 이후 1년 보다는 6~9개월 정도 예측하는게 적당할 것 같네요.


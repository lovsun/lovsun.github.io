---
layout: post
title:  "[다변량통계] VAR 모델을 통한 다변량 예측"
categories: timeseries
comments: true
---
# VAR(p) 모델
앞에서는 일변량시계열 모형에 대해 알아보았습니다. 그리고 일반적으로 전통적인 통계는 계수의 올바른 추정을 위해 구조모형을 많이 사용합니다. 여기서는 인과관계가 불확실한 즉 **변수들 간에 서로 영향을 끼치는 경우를 위한 VAR 모델**에 대해 작성해보겠습니다.

VAR 모형은 **일변량시계열 모형**과 **연립방정식 구조화모형**이 결하여, 예측변수가 두 개 이상인 다변량 시계열모형입니다.

두 개 이상 변수 각각의 현재값은 과거 자기자신의 값과 과거 다른 변수의 값으로 선형결합되어 있습니다. 

![Image.png](https://wikimedia.org/api/rest_v1/media/math/render/svg/9ce3e0bc80793bad348d0cc6926585037f090b87)

### VAR(p) 모델의 장단점
해당 모형은 계수의 해석과 이론에 초점을 둔 **연립방정식구조화모형**의 문제점인 **내생변수와 외생변수를 식별해야한다는 단점을 보완**하고 있습니다. 구조화모형은 현실적으로 뚜렷한 인과관계 및 내생, 외생변수를 구분하는 것은 쉽지 않을 뿐더라 인과관계에 대해 올바르게 해석하기 위해 모형에 각종 제약을 가하고 있습니다. 

VAR 모형은 계수추정치를 어떻게 해석하느냐는 관점에서는 구조화모형보다 못하지만, **내/외생 변수 구별이 필요다는 장점**이 있습니다. 또한 **변수 간 종속되는걸 모델링한다는 점에서 일변량 AR모형보다 훨씬 더 유연**하다고 할 수 있습니다.

### 파라미터
VAR 모형에 자주 등장하는 파라미터는 아래와 같습니다.

> **p** : 시차

> **k** : 방정식 개수, 변수 개수

> **추정파라미터의 개수** : p * k * k + k (개)


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse

import warnings
warnings.filterwarnings("ignore")
```

## 1. 데이터수집 및 전처리
FRED의 ``M2 Money Stock 통화공급량 대용치``과 ``Personal Consumption Expenditure 개인소비지출`` 데이터를 이용하겠습니다.
해당 두 변수는 서로 영향을 끼쳐서 인과관계가 불분명합니다. 돈이 많으면 소비를 늘리기도 하고, 소비를 늘리면 이는 또 시중 통화량에 영향을 끼치게 됩니다.
FRED의 데이터수집방법을 모르는 분들은 Quant의 경제지표데이터수집 글을 참고하길 바랍니다.


```python
import fredapi
fred = fredapi.Fred(api_key='YOUR API KEY')
```

### M2 Money Stock
데이터를 불러온 이후 Personal Consumpotion Expenditure과 **주기를 맞춰주기 위해 날짜를 월별로 바꿔주는 전처리**를 수행하겠습니다.


```python
Money = pd.DataFrame(data=fred.get_series('WM2NS'), columns=['Money'])

# 연도와 월을 이용해 같은 날짜로 맞춰준 후 날짜 중복데이터 제거
Money.index = pd.to_datetime({'year':Money.index.year, 'month':Money.index.month, 'day':1})
Money = Money.reset_index().drop_duplicates(subset='index', keep='last')

Money.set_index('index', inplace=True)
Money.index.freq = 'MS'
```


```python
Money.head()
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
      <th>Money</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-11-01</th>
      <td>1594.8</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>1601.8</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>1599.8</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>1609.0</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>1637.5</td>
    </tr>
  </tbody>
</table>
</div>



### Personal Consumption Expenditure


```python
PCE = pd.DataFrame(data=fred.get_series('PCE'), columns=['Spending'])
PCE.index.freq = 'MS'
```


```python
PCE.head()
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
      <th>Spending</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1959-01-01</th>
      <td>306.1</td>
    </tr>
    <tr>
      <th>1959-02-01</th>
      <td>309.6</td>
    </tr>
    <tr>
      <th>1959-03-01</th>
      <td>312.7</td>
    </tr>
    <tr>
      <th>1959-04-01</th>
      <td>312.2</td>
    </tr>
    <tr>
      <th>1959-05-01</th>
      <td>316.1</td>
    </tr>
  </tbody>
</table>
</div>



### 최종 data


```python
data = Money.join(PCE)
data = data.dropna()
data.head()
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
      <th>Money</th>
      <th>Spending</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1980-11-01</th>
      <td>1594.8</td>
      <td>1826.8</td>
    </tr>
    <tr>
      <th>1980-12-01</th>
      <td>1601.8</td>
      <td>1851.7</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>1599.8</td>
      <td>1870.0</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>1609.0</td>
      <td>1884.2</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>1637.5</td>
      <td>1902.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (488, 2)



## 2. Statinarity Test

### 1) Plot for Stationarity
해당 데이터의 경우 명확한 trend가 관찰되며 정상성을 만족하지 못하는 시계열임을 알 수 있습니다.


```python
data.plot(figsize=(12,8))
```




    <AxesSubplot:xlabel='index'>




    
![output_15_1](https://user-images.githubusercontent.com/68403764/130752050-a15bded5-5914-44ad-8a73-6f46ea88465f.png)
    


### 2) ADF Test for Stationarity
ADF 테스트를 통해서도 M2, PCE 모두 정상성을 만족하지 못함을 알 수 있습니다. 

이에 대한 더 상세한 설명은 이전 ARIMA 관련 게시글 참고하길 바랍니다.


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
adf_test(data['Money'])
```

    statistic                 3.673275
    p-value                   1.000000
    # lags used              18.000000
    # observations          469.000000
    critical value (1%)      -3.444370
    critical value (5%)      -2.867722
    critical value (10%)     -2.570063
    Fail to reject the null hypothesis
    Data has a unit root and is non-stationary
    


```python
adf_test(data['Spending'])
```

    statistic                 2.250432
    p-value                   0.998922
    # lags used              17.000000
    # observations          470.000000
    critical value (1%)      -3.444340
    critical value (5%)      -2.867709
    critical value (10%)     -2.570056
    Fail to reject the null hypothesis
    Data has a unit root and is non-stationary
    

## 3. Differencing
만약 하나의 시계열데이터는 1차차분으로 정상성을 만족하지만, **다른 하나의 시계열이 2차차분까지 해줘야 정상성을 만족한다면 나중에 차분** 전으로 되돌려주기 위해 둘 다 2차차분을 하는 방법을 추천합니다.


```python
data_diff1 = data.diff()
```


```python
print('Money_diff1\n')
print(adf_test(data_diff1['Money']))

print('\n')

print('Spending_diff1\n')
print(adf_test(data_diff1['Spending']))
```

    Money_diff1
    
    statistic                -1.506180
    p-value                   0.530452
    # lags used              18.000000
    # observations          468.000000
    critical value (1%)      -3.444400
    critical value (5%)      -2.867736
    critical value (10%)     -2.570070
    Fail to reject the null hypothesis
    Data has a unit root and is non-stationary
    None
    
    
    Spending_diff1
    
    statistic                -4.917733
    p-value                   0.000032
    # lags used              16.000000
    # observations          470.000000
    critical value (1%)      -3.444340
    critical value (5%)      -2.867709
    critical value (10%)     -2.570056
    Reject the null hypothesis
    Data has no unit root and is stationary
    None
    

M2 Money Stock의 경우 1차 차분만으로 정상성을 만족하지 않아 **한번 더 차분을 진행하겠습니다.**


```python
data_diff2 = data_diff1.diff().dropna()
```


```python
print('Money_diff2\n')
print(adf_test(data_diff2['Money']))

print('\n')

print('Spending_diff2\n')
print(adf_test(data_diff2['Spending']))
```

    Money_diff2
    
    statistic              -6.662492e+00
    p-value                 4.807944e-09
    # lags used             1.800000e+01
    # observations          4.670000e+02
    critical value (1%)    -3.444431e+00
    critical value (5%)    -2.867749e+00
    critical value (10%)   -2.570077e+00
    Reject the null hypothesis
    Data has no unit root and is stationary
    None
    
    
    Spending_diff2
    
    statistic              -8.319302e+00
    p-value                 3.608756e-13
    # lags used             1.700000e+01
    # observations          4.680000e+02
    critical value (1%)    -3.444400e+00
    critical value (5%)    -2.867736e+00
    critical value (10%)   -2.570070e+00
    Reject the null hypothesis
    Data has no unit root and is stationary
    None
    

이제 두 시계열데이터 모두 정상성을 만족한걸 확인할 수 있습니다.

나중에 원래 데이터형태로 되돌려주기위해 변수명은 차분한걸 이용하여 지정해주는 것이 편리합니다.

## 4. train, test split
6개월 데이터를 테스트데이터로 사용하겠습니다. 이를 테스트데이터로 별도로 분리해놓겠습니다.


```python
test_set = 6

train = data_diff2[:-test_set]
test = data_diff2[-test_set:]
```

## 5. 차수(p) 선정

### - Cross-Equation Restrictions
차수를 선정하는 방법으로 특정 시자이후 변수의 계수가 0이라는 귀무가설을 검증하는 **교차방정식제약(cross-equation restrictions)** 이 있지만, 이는 **교란항이 정규분포**를 따라야한다는 가정이 있기 때문에 일반적으로 금융데이터에서 거의 충족되지 못합니다.

### - Information Criteria
그래서 차수를 선정하는 방법으로 ARIMA에서처럼 **정보기준(Information criteria)** 을 사용합니다. 즉 **모델 간 AIC 값을 비교하여 가장 낮은 AIC를 가지는 모델을 선택**하는 것입니다. 
참고로 다시 한번 설명드리자면 오차와 파라미터 수를 최소화하는 모델일수록 낮은 AIC 값을 가지게 됩니다.

하지만 ARIMA 때 활용했던 ``from pmdarima import auto_arima``는 **VAR 모형의 적절한 차수는 제공해주지 않기 때문에, for문을 사용하여서 여러 모델을 생성하여 AIC 값을 비교해주면 됩니다.** 


```python
model = VAR(train)

# 반복문을 통한 grid search 진행

for p in range(7):    
    results = model.fit(p)
    print(f'ORDER {p}')
    print(f'AIC: {results.aic}\n')
```

    ORDER 0
    AIC: 19.41424547875116
    
    ORDER 1
    AIC: 18.97330110240839
    
    ORDER 2
    AIC: 18.256850007457437
    
    ORDER 3
    AIC: 18.036559607818777
    
    ORDER 4
    AIC: 17.856431208815355
    
    ORDER 5
    AIC: 17.682702207526937
    
    ORDER 6
    AIC: 17.605159909604957
    
    

## 6. 모델생성 
AIC 값이 가장 작은 차수 6을 선정하겠습니다. 차수를 증가시켜 더 돌려보아도 되지만 

파라미터 수가 너무 증가하는 것을 방지하기 위해 일단 **차수 6으로 모델을 생성**해보겠습니다.


```python
results = model.fit(6)
results.summary()
```




      Summary of Regression Results   
    ==================================
    Model:                         VAR
    Method:                        OLS
    Date:           Wed, 25, Aug, 2021
    Time:                     16:30:55
    --------------------------------------------------------------------
    No. of Equations:         2.00000    BIC:                    17.8334
    Nobs:                     474.000    HQIC:                   17.6949
    Log likelihood:          -5491.58    FPE:                4.42421e+07
    AIC:                      17.6052    Det(Omega_mle):     4.19116e+07
    --------------------------------------------------------------------
    Results for equation Money
    ==============================================================================
                     coefficient       std. error           t-stat            prob
    ------------------------------------------------------------------------------
    const               2.649710         3.590922            0.738           0.461
    L1.Money           -1.075284         0.054150          -19.858           0.000
    L1.Spending        -0.400691         0.041910           -9.561           0.000
    L2.Money           -1.327268         0.081229          -16.340           0.000
    L2.Spending        -0.595694         0.055248          -10.782           0.000
    L3.Money           -0.811402         0.095044           -8.537           0.000
    L3.Spending        -0.555250         0.068705           -8.082           0.000
    L4.Money           -0.394453         0.093022           -4.240           0.000
    L4.Spending        -0.436322         0.065038           -6.709           0.000
    L5.Money           -0.036444         0.075247           -0.484           0.628
    L5.Spending        -0.190725         0.050293           -3.792           0.000
    L6.Money            0.047314         0.056981            0.830           0.406
    L6.Spending        -0.046662         0.039183           -1.191           0.234
    ==============================================================================
    
    Results for equation Spending
    ==============================================================================
                     coefficient       std. error           t-stat            prob
    ------------------------------------------------------------------------------
    const              -0.635732         4.528586           -0.140           0.888
    L1.Money           -0.322409         0.068290           -4.721           0.000
    L1.Spending        -0.677768         0.052854          -12.823           0.000
    L2.Money            0.466532         0.102440            4.554           0.000
    L2.Spending        -0.713001         0.069674          -10.233           0.000
    L3.Money            0.725146         0.119862            6.050           0.000
    L3.Spending        -0.435105         0.086645           -5.022           0.000
    L4.Money            0.841882         0.117312            7.176           0.000
    L4.Spending        -0.075991         0.082021           -0.926           0.354
    L5.Money            0.665601         0.094896            7.014           0.000
    L5.Spending        -0.014563         0.063425           -0.230           0.818
    L6.Money            0.328184         0.071860            4.567           0.000
    L6.Spending        -0.018742         0.049414           -0.379           0.704
    ==============================================================================
    
    Correlation matrix of residuals
                   Money  Spending
    Money       1.000000 -0.537668
    Spending   -0.537668  1.000000
    
    



## 7. 예측
생성한 모델을 이용하여 예측을 진행할 때 주의할 점이 있습니다.

> 파라미터 형식 : Numpy array (p행 * k열) - dataframe의 value 메소드 활용

> **p** : 시차

> **k** : 시계열데이터변수 개수

test data의 첫번째 예측값을 위해 train data의 마지막 5(시차)개 행을 이용하겠습니다. 

### 예측


```python
lagged_values = train.values[-6:] # 모델 선정시 선택한 시차 6
pred_diff2 = results.forecast(y=lagged_values, steps=6) # 6개월 뒤까지 예측하겠다는 의미로 위의 6과는 다릅니다.
```


```python
idx = pd.date_range(test.index[0], periods=6, freq='MS')
predictions = pd.DataFrame(data=pred_diff2, index=idx, columns=['Money_2d', 'Spending_2d'])
predictions
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
      <th>Money_2d</th>
      <th>Spending_2d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-01</th>
      <td>17.348643</td>
      <td>313.549687</td>
    </tr>
    <tr>
      <th>2021-02-01</th>
      <td>7.334989</td>
      <td>-2.411396</td>
    </tr>
    <tr>
      <th>2021-03-01</th>
      <td>-104.406082</td>
      <td>-133.101037</td>
    </tr>
    <tr>
      <th>2021-04-01</th>
      <td>63.221752</td>
      <td>55.360732</td>
    </tr>
    <tr>
      <th>2021-05-01</th>
      <td>6.960601</td>
      <td>-37.433021</td>
    </tr>
    <tr>
      <th>2021-06-01</th>
      <td>-18.272952</td>
      <td>-46.406169</td>
    </tr>
  </tbody>
</table>
</div>



### 차분 되될리기(Invert Defferencing)
2차차분을 해주었기 때문에 이를 원래대로 되될리는 과정이 필요합니다. 

먼저, test data의 **1차차분값**을 구하려면 **train data의 마지막 1차차분값**에 **2차차분 결과 누적으로 더해주면 됩니다.**
그리고 **원래 scale의 값**을 구하려면 **train data의 마지막 값**에 **1차차분 결과를 누적으로 더해주면 됩니다.**

아래의 코드가 이에 대한 내용입니다.

#### M2 Money Stock


```python
#1차차분값
predictions['Money_1d'] = (data['Money'].iloc[-test_set-1] - data['Money'].iloc[-test_set-2] ) + predictions['Money_2d'].cumsum() 
#origin sacle 값
predictions['Money_Prediction'] = data['Money'].iloc[-test_set-1] + predictions['Money_1d'].cumsum()
```

#### Personal Consumption Expenditure


```python
#1차차분값
predictions['Spending_1d'] = (data['Spending'].iloc[-test_set-1] - data['Spending'].iloc[-test_set-2] ) + predictions['Spending_2d'].cumsum() 
#origin sacle 값
predictions['Spending_Prediction'] = data['Spending'].iloc[-test_set-1] + predictions['Spending_1d'].cumsum()
```


```python
predictions
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
      <th>Money_2d</th>
      <th>Spending_2d</th>
      <th>Money_1d</th>
      <th>Money_Prediction</th>
      <th>Spending_1d</th>
      <th>Spending_Prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-01-01</th>
      <td>17.348643</td>
      <td>313.549687</td>
      <td>263.048643</td>
      <td>19593.448643</td>
      <td>235.749687</td>
      <td>14625.249687</td>
    </tr>
    <tr>
      <th>2021-02-01</th>
      <td>7.334989</td>
      <td>-2.411396</td>
      <td>270.383632</td>
      <td>19863.832275</td>
      <td>233.338291</td>
      <td>14858.587977</td>
    </tr>
    <tr>
      <th>2021-03-01</th>
      <td>-104.406082</td>
      <td>-133.101037</td>
      <td>165.977549</td>
      <td>20029.809825</td>
      <td>100.237253</td>
      <td>14958.825231</td>
    </tr>
    <tr>
      <th>2021-04-01</th>
      <td>63.221752</td>
      <td>55.360732</td>
      <td>229.199302</td>
      <td>20259.009126</td>
      <td>155.597985</td>
      <td>15114.423216</td>
    </tr>
    <tr>
      <th>2021-05-01</th>
      <td>6.960601</td>
      <td>-37.433021</td>
      <td>236.159902</td>
      <td>20495.169029</td>
      <td>118.164964</td>
      <td>15232.588180</td>
    </tr>
    <tr>
      <th>2021-06-01</th>
      <td>-18.272952</td>
      <td>-46.406169</td>
      <td>217.886951</td>
      <td>20713.055980</td>
      <td>71.758796</td>
      <td>15304.346976</td>
    </tr>
  </tbody>
</table>
</div>



## 8. 시각화 및 평가

### 1) 시각화


```python
test_originscale = data[-test_set:]
```

#### M2 Money Stock


```python
test_originscale['Money'].plot(legend=True, figsize=(12,8))
predictions['Money_Prediction'].plot(legend=True)
```




    <AxesSubplot:xlabel='index'>




![output_47_1](https://user-images.githubusercontent.com/68403764/130752058-107c99aa-c645-4910-9906-c10b0439d0e8.png)
    


#### Personal Consumption Expenditure


```python
test_originscale['Spending'].plot(legend=True, figsize=(12,8))
predictions['Spending_Prediction'].plot(legend=True)
```




    <AxesSubplot:xlabel='index'>




    
![output_49_1](https://user-images.githubusercontent.com/68403764/130752059-edd0e391-d176-4a82-ad03-69e7dba63ec0.png)
    


### 2) RMSE

#### M2 Money Stock


```python
rmse(test_originscale['Money'], predictions['Money_Prediction'])
```




    241.6066494205269




```python
test_originscale['Money'].mean(), test_originscale['Money'].std()
```




    (19978.516666666666, 421.2257941611211)



#### Personal Consumption Expenditure


```python
rmse(test_originscale['Spending'], predictions['Spending_Prediction'])
```




    400.1626664358737




```python
test_originscale['Spending'].mean(), test_originscale['Spending'].std()
```




    (15339.033333333335, 447.9574027367633)



Money Stock은 비교적 잘 예측하는 편이지만, PCE의 경우 실제값이 크게 증가한 2월을 잡아내지 못하였고 지속적으로 오차가 발생하는 것을 확인할 수 있습니다. 

그래도 두 가지 모두 RMSE가 1표준편차보다 작게 예측하는 것을 확인할 수 있습니다.

[(코드작성시 참고했던 자료)](https://github.com/jeswingeorge/Learning-Time-Series-Python/tree/master/06-General-Forecasting-Models)

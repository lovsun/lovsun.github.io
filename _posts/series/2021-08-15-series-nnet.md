---
layout: post
title:  "[일변량딥러닝] nnet-ts 신경망을 이용한 시계열 예측"
categories: timeseries
comments: true
---
# 딥러닝(nnet-ts)을 활용한 일변량 시계열 예측
``nnet-ts 라이브러리``에서 제공하는 신경망은 일변량 시걔열데이터 예측에특화된 라이브러리 입니다. 하지만 **파이썬 3버전**으로 넘어오면서 해당 라이브러리를 만든 분이 업데이트를 시켜지 않아 **소스코드를 변경하여서 라이브러리를 동작**시켜야합니다. 

방법은 [stackoverflow Q&A 답변](https://stackoverflow.com/questions/44673380/using-the-timeseriesnnet-method-from-the-nnet-ts-module-throws-nameerror) 을 참고하길 바랍니다.

nnet-ts는 **일변량데이터 예측에 상당히 우수한 성능**을 보이고 있습니다. 또한, 일변량시계열예측을 상당히 편리하고 간편하게 수행할 수 있습니다. 그 이유는 **시차를 지정해주면 모델 내부에서 자동적으로 데이터를 나누어 학습을 수행**합니다. 

그래서 결론적으로 말하자면 **train data와 test data의 split을 수행해줄 필요가 없습니다.**

## 01. 데이터수집 및 전처리


```python
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")
```


```python
rawdata = pd.read_csv("경유월간소비.csv", encoding='cp949')
```


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
import matplotlib.pyplot as plt
%matplotlib inline

rawdata.plot(figsize=(12, 8))
```




    <AxesSubplot:xlabel='date'>




    
![png](output_5_1.png)
    


## 02. 차수(p) 결정
과거 몇 개의 데이터를 이용하여 미래값을 예측할지 정해야합니다. **과거 자기자신의 값으로 예측한다는 점에서 AR 모형이 차수를 결정**하는 것과 비슷합니다. 

그래서 **PACF 도표**이용하여 몇 시점 전까지 사용할지 결정하겠습니다. 


```python
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(rawdata, lags=30);
```


    
![png](output_7_0.png)
    


PACF 도표 상 명확하진 않지만 **계절성이 있는 것으로 관찰**됩니다. 그래서 decomposition을 수행한 결과 계절성 값의 범위가 작진 않기 때문에 시차를 ``12``로 설정하겠습니다. 

즉 12달의 데이터를 이용하여 다음 달의 값을 예측하는 방법으로 모델을 구성해주겠습니다. 


```python
from statsmodels.tsa.seasonal import seasonal_decompose
```


```python
decomposition = seasonal_decompose(rawdata, model='additive', extrapolate_trend='freq')
decomposition.plot();
```


    
![png](output_10_0.png)
    


## 03. feature scaling
딥러닝 모델에 넣어주기 위해서는 데이터의 특성 그리고 분석목적에 따라서 **0 ~ 1 혹은 -1 ~ 1 사이로 데이터를 변환해주는 작업**이 필요합니다.

데이터프레임을 MinMaxScaler를 이용하여 fit, transform해주면 지정한 값의 범위로 변경된 2차원데이터가 형성됩니다.

그래서 이를 다시 1차원으로 바꿔주는 작업이 필요합니다.


```python
data = rawdata.copy()
```


```python
from sklearn.preprocessing import MinMaxScaler

#스케일러 object 생성
scaler = MinMaxScaler(feature_range = (0, 1))
scaled_data = scaler.fit_transform(data)
```


```python
# 스케일링과정에서 만들어진 2차원데이터
scaled_data.shape 
```




    (258, 1)




```python
# 1차원 데이터로 변형
scaled_data = scaled_data.reshape(-1)

print(scaled_data.shape)
print(scaled_data[:5])
```

    (258,)
    [0.27833215 0.23473509 0.44416726 0.36101212 0.49429793]
    

## 04. 모델 생성 및 예측
모델을 생성하고 예측하는 코드의 순서는 아래와 같습니다.

> train_end : **마지막 훈련데이터 지정**

> **모델 생성** : 뉴런 개수, 은닉층 활성화함수 선정 (이건 하이퍼파라미터이므로 성능을 높이기 위해 변경시켜볼 수 있습니다.)

> **학습** : lag를 설정해두면 모델이 내부에서 lag 만큼 데이터를 잘라서 학습시킵니다. 2번에서 결정한 차수를 넣어줍니다.

> **예측** : 예측할 시점을 지정


```python
from nnet_ts import *
```

    Using TensorFlow backend.
    


```python
count = 0
test_set = 12
```


```python
predictions = []

while(count < test_set):
    
    # 마지막 훈련데이터 지정
    train_end = len(scaled_data) - test_set + count
    
    # 모델 생성 및 학습
    np.random.seed(2022)
    model = TimeSeriesNnet(hidden_layers = [32, 16, 8], 
                           activation_functions = ['relu', 'tanh', 'tanh'])   
    model.fit(scaled_data[0:train_end], lag = 12, epochs = 25)
    
    # 예측
    predict = model.predict_ahead(n_ahead = 1)
    predictions.append(predict)
    
    # actual VS prediction
    print(f"({count + 1}) Actual Value is {round(scaled_data[train_end], 5)}")
    print(f"({count + 1}) Predicted Value is {round(pd.Series(predict)[0], 5)}")
    
    count += 1  
```


    (1) Actual Value is 0.64992
    (1) Predicted Value is 0.61196
    

    (2) Actual Value is 0.59313
    (2) Predicted Value is 0.59288
    

    (3) Actual Value is 0.6814
    (3) Predicted Value is 0.61195


    (4) Actual Value is 0.64326
    (4) Predicted Value is 0.61991
    

    (5) Actual Value is 0.90948
    (5) Predicted Value is 0.6871
    

    (6) Actual Value is 0.76313
    (6) Predicted Value is 0.88356
    

    (7) Actual Value is 0.58375
    (7) Predicted Value is 0.75988
    

    (8) Actual Value is 0.55856
    (8) Predicted Value is 0.51881
    

    (9) Actual Value is 0.5942
    (9) Predicted Value is 0.7435
    

    (10) Actual Value is 0.71965
    (10) Predicted Value is 0.78578
    

    (11) Actual Value is 0.73723
    (11) Predicted Value is 0.7455


    (12) Actual Value is 0.79675
    (12) Predicted Value is 0.8071
    

## 04. 원래 스케일로 변환 (Inverse Scaling)


```python
actual_predictions = scaler.inverse_transform(predictions)
actual_predictions
```




    array([[13198.44611251],
           [13037.88048708],
           [13198.40396535],
           [13265.40239477],
           [13831.01426578],
           [15484.78100097],
           [14443.67892468],
           [12414.32229269],
           [14305.81205344],
           [14661.71721971],
           [14322.58561945],
           [14841.20742154]])



## 05. 시각화 및 평가

### 시각화
실제값과 예측값의 시각화를 통해 값이 얼마나 많이 차이나는지 알아보겠습니다. 아래의 시각화한 그래프를 보면 상당히 **예측성능이 우수한 것을 확인할 수 있습니다.** 

만약 필요에 따라 모델의 **하이퍼파라미터를 조정한다면 더 우수한 성능을 낼 수 있겠죠.**


```python
actual = rawdata[-test_set:]
actual["prediction"] = actual_predictions
```


```python
actual.plot(figsize=(12,8))
```




    <AxesSubplot:xlabel='date'>




    
![png](output_25_1.png)
    


### RMSE
시각화만으로는 정확한 성능평가가 어렵기 때문에 정확성을 평가하는 지표 중 하나인 RMSE(Root Mean Square Error) 값을 구해보겠습니다.


```python
from statsmodels.tools.eval_measures import rmse

error = rmse(actual['diesel'], actual["prediction"])
error, actual['diesel'].mean(), actual['diesel'].std()
```




    (876.8745958985452, 13820.666666666666, 874.85347690948)



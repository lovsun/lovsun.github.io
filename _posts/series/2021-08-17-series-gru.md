---
layout: post
title:  "[다변량딥러닝] LSTM, GRU 신경망을 이용한 다변량 시계열 예측"
categories: timeseries
comments: true
---
# LSTM, GRU 활용한 다변량 시계열 예측
이번 글에서는 딥러닝으로 다변량시계열모형을 구축하여 예측을 시행해보겠습니다. 

다양한 딥러닝 네트워크 구조 중에서 순환신경망인 **LSTM을 활용하여 다변량 시계열 예측**을 코드에 대해 작성해보겠습니다.

### 사용시 주의점
딥러닝 모델의 장점은 관계를 사람이 파악하기 어려운 경우 딥러닝 네트워크를 이용하여 예측정확도를 올릴 수 있습니다. 하지만 단점으로는 흔히 블랙박스라고 하는데요. 왜 그렇게 예측했는지 해석하기 어렵습니다. 

그렇기 때문에, 앞서 소개한 **지수평활법, ARIMA, VAR 등 기존 통계모형과 함께 사용**하는 것이 좋습니다.

## 01. 데이터수집 및 전처리
활용할 데이터는 FRED에서 제공하는 **M2 Money Stock, Personal Consumption Expenditure**입니다. 이 두 변수 **변화율의 변동성**을 딥러닝모형을 통해 모델링해보겠습니다.

물론 PCE의 경우 월별로 데이터가 제공되어서, 딥러닝의 장점을 활용하기에 데이터가 많지는 않습니다. 그렇기 때문에 결과보다는 **모델링하는 방법에 대해 초점을 맞추어 작성**하겠습니다.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
```

### 데이터수집


```python
import fredapi
fred = fredapi.Fred(api_key='YOUR API KEY')
```


```python
Money = pd.DataFrame(data=fred.get_series('WM2NS'), columns=['Money'])

# 연도와 월을 이용해 같은 날짜로 맞춰준 후 날짜 중복데이터 제거
Money.index = pd.to_datetime({'year':Money.index.year, 'month':Money.index.month, 'day':1})
Money = Money.reset_index().drop_duplicates(subset='index', keep='last')

Money.set_index('index', inplace=True)
Money.index.freq = 'MS'

PCE = pd.DataFrame(data=fred.get_series('PCE'), columns=['Spending'])
PCE.index.freq = 'MS'

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
# 데이터 변화율 구하기
data_change  = data.pct_change(1)
data_change = data_change.dropna()
data_change.head()
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
      <th>1980-12-01</th>
      <td>0.004389</td>
      <td>0.013630</td>
    </tr>
    <tr>
      <th>1981-01-01</th>
      <td>-0.001249</td>
      <td>0.009883</td>
    </tr>
    <tr>
      <th>1981-02-01</th>
      <td>0.005751</td>
      <td>0.007594</td>
    </tr>
    <tr>
      <th>1981-03-01</th>
      <td>0.017713</td>
      <td>0.009925</td>
    </tr>
    <tr>
      <th>1981-04-01</th>
      <td>0.011969</td>
      <td>0.000788</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 변화율 변동성 구하기
window_size = 15
data = data_change.rolling(window = window_size, center = False).std() 
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
      <th>1982-02-01</th>
      <td>0.006362</td>
      <td>0.004844</td>
    </tr>
    <tr>
      <th>1982-03-01</th>
      <td>0.006425</td>
      <td>0.004630</td>
    </tr>
    <tr>
      <th>1982-04-01</th>
      <td>0.006037</td>
      <td>0.004655</td>
    </tr>
    <tr>
      <th>1982-05-01</th>
      <td>0.006052</td>
      <td>0.004716</td>
    </tr>
    <tr>
      <th>1982-06-01</th>
      <td>0.005534</td>
      <td>0.004635</td>
    </tr>
  </tbody>
</table>
</div>



## 02. Feature Engineering
먼저 피쳐엔지니어링에 대한 [위키피디아](https://en.wikipedia.org/wiki/Feature_engineering) 정의를 살펴보겠습니다. 

**Feature engineering** is the process of **using domain knowledge** to **extract features** (characteristics, properties, attributes) from raw data.

피쳐엔지니어링을 수행하는 코드는 **'도메인 지식'과 'TASK'** 에 따라서 달라지게 됩니다. 

가령 특정 날짜의 변동성이 궁금하다고 가정할 때, 해당 날짜의 변동성으로 이전 일주일 간 수익률의 표준편차를 사용할 수도 있고, 이전 한 달간 수익률의 표준편차를 사용할 수도 있습니다. 또한, 앞뒤 3일 간 수익률의 표준편차를 사용할 수도 있고, 앞뒤 15일간 수익률의 표준편차를 사용할 수도 있습니다.

그렇기 때문에 피쳐엔지니어링을 수행할 때는 도메인 지식을 가진 분들의 도움을 받고, 하고자하는 TASK를 명확히 정의하는 것이 중요합니다.


```python
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data['Money'], lags=30);
```


    
![output_8_0](https://user-images.githubusercontent.com/68403764/130808261-b29ecbf4-1d51-4f53-a8dc-840aebebce4b.png)
    



```python
plot_pacf(data['Spending'], lags=30);
```


    
![output_9_0](https://user-images.githubusercontent.com/68403764/130808265-f4ada46b-4c5a-4231-8b52-4babd239ef37.png)


pacf 도표를 참고하여 시차를 3으로 정해주겠습니다. 

그리고 최근 변동성이 과거에 비해 얼마나 커졌는지를 모델링해주도록 하겠습니다.


```python
x1 = np.log((data_change.shift(1) / data_change.shift(2)) * data_change.shift(1))
x2 = np.log((data_change.shift(1) / data_change.shift(3)) * data_change.shift(1))
x3 = np.log((data_change.shift(1) / data_change.shift(4)) * data_change.shift(1))

data = pd.concat([data_change, x1, x2, x3], axis =1)
data.columns = ['money_t','sp_t', 
                'money_t-1','sp_t-1', 
                'money_t-2','sp_t-2',
                'money_t-3','sp_t-3']

data = data.dropna()
```

## 03. train-test split
6개월 간 데이터셋을 테스트로 사용하겠습니다.

보통 딥러닝 모형을 모델링할 때 이보다 데이터가 훨씬 많아야하지만 방법론을 작성하는 것에 초점을 맞추도록 하겠습니다.


```python
test_start = len(data) - 6
train = data.iloc[:test_start]
test = data.iloc[test_start:]
```


```python
train_x = train.iloc[:,2:]
train_y = train.iloc[:,:2]

test_x = test.iloc[:,2:]
test_y = test.iloc[:,:2]
```


```python
train_x.tail()
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
      <th>money_t-1</th>
      <th>sp_t-1</th>
      <th>money_t-2</th>
      <th>sp_t-2</th>
      <th>money_t-3</th>
      <th>sp_t-3</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-02-01</th>
      <td>-4.713843</td>
      <td>-6.262246</td>
      <td>-4.702135</td>
      <td>-4.808814</td>
      <td>-4.996096</td>
      <td>-4.630504</td>
    </tr>
    <tr>
      <th>2018-02-01</th>
      <td>-3.940721</td>
      <td>-8.376368</td>
      <td>-3.599895</td>
      <td>-8.333621</td>
      <td>-3.660756</td>
      <td>-7.112858</td>
    </tr>
    <tr>
      <th>2018-09-01</th>
      <td>-5.729906</td>
      <td>-5.555071</td>
      <td>-4.662088</td>
      <td>-5.360685</td>
      <td>-5.091702</td>
      <td>-5.796495</td>
    </tr>
    <tr>
      <th>2018-10-01</th>
      <td>-8.074001</td>
      <td>-7.930854</td>
      <td>-8.369972</td>
      <td>-7.891106</td>
      <td>-7.302154</td>
      <td>-7.696720</td>
    </tr>
    <tr>
      <th>2019-09-01</th>
      <td>-7.010248</td>
      <td>-5.923946</td>
      <td>-6.688752</td>
      <td>-5.785059</td>
      <td>-6.640081</td>
      <td>-5.567239</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_x.tail()
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
      <th>money_t-1</th>
      <th>sp_t-1</th>
      <th>money_t-2</th>
      <th>sp_t-2</th>
      <th>money_t-3</th>
      <th>sp_t-3</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-11-01</th>
      <td>-6.473630</td>
      <td>-5.399482</td>
      <td>-5.007185</td>
      <td>-5.951817</td>
      <td>-6.149802</td>
      <td>-6.186548</td>
    </tr>
    <tr>
      <th>2019-12-01</th>
      <td>-3.746259</td>
      <td>-5.179456</td>
      <td>-4.782481</td>
      <td>-4.758422</td>
      <td>-3.316036</td>
      <td>-5.310757</td>
    </tr>
    <tr>
      <th>2020-01-01</th>
      <td>-3.871128</td>
      <td>-7.829760</td>
      <td>-3.025553</td>
      <td>-7.509230</td>
      <td>-4.061775</td>
      <td>-7.088196</td>
    </tr>
    <tr>
      <th>2020-02-01</th>
      <td>-4.924273</td>
      <td>-3.672381</td>
      <td>-4.563920</td>
      <td>-4.837268</td>
      <td>-3.718345</td>
      <td>-4.516738</td>
    </tr>
    <tr>
      <th>2020-12-01</th>
      <td>-2.820134</td>
      <td>-5.021775</td>
      <td>-2.541690</td>
      <td>-6.225712</td>
      <td>-3.424832</td>
      <td>-5.833743</td>
    </tr>
  </tbody>
</table>
</div>



## 04. Feature Scaling
위에 값을 보면 알 수 있는 것처럼 **x 데이터**의 경우, log를 이용하여 변화율을 사용했기 때문에 마이너스 값을 가지므로 **-1에서 1사이로 스케일링**을 해주겠습니다.
그리고 **y데이터**의 경우, 변동성이어서 항상 플러스의 값을 가지므로 **0에서 1 사이로 스케일링**을 해주겠습니다.

(주의사항) 참고로 주의할 점은 테스트 데이터의 경우 훈련하는 시점에 모른다고 가정하고 있습니다. 

그렇기때문에 **Train data로 MinMaxScaler를 fit해주고나서, 그 이후에 이를 이용하여 Train, Test data에 transform을 진행**해줘야합니다.
결국 스케일드된 test값은 -1 ~ 1 혹은 0 ~ 1 사이가 아닐 수도 있지만, 대체적으로 그 사이값을 가지고 큰 값의 차이가 나지 않는다면 딥러닝 모델에 넣는 데에 문제는 없습니다.


```python
from sklearn.preprocessing import MinMaxScaler

#스케일러 object 생성
scaler_x = MinMaxScaler(feature_range = (-1, 1))
scaler_x.fit(train_x) 
scaler_y = MinMaxScaler(feature_range = (0, 1))
scaler_y.fit(train_y) 
```




    MinMaxScaler()




```python
#train data scaling
scaled_train_x = scaler_x.transform(train_x)
scaled_train_y = scaler_y.transform(train_y) 

#test data scaling
scaled_test_x = scaler_x.transform(test_x)
scaled_test_y = scaler_y.transform(test_y)
```


```python
scaled_train_x[:5]
```




    array([[ 0.04978864,  0.72735081,  0.09678654,  0.25695917,  0.14649292,
             0.15935289],
           [-0.10217917, -0.60254741, -0.02133144, -0.36477211, -0.05584901,
            -0.64733966],
           [-0.99297853,  0.74998405, -1.        ,  0.59424827, -1.        ,
             0.60272974],
           [-0.04685872, -0.11124531, -0.03183183,  0.28419132, -0.101556  ,
             0.09164221],
           [ 0.40657774,  0.90157952,  0.37700417,  0.59015713,  0.17947144,
             0.79004768]])




```python
scaled_train_y[:5]
```




    array([[0.57270334, 0.56282769],
           [0.68222944, 0.28901189],
           [0.63153543, 0.34195725],
           [0.57335959, 0.58521345],
           [0.58297617, 0.3979338 ]])



## 05. X Dataset 재구성
항상 딥러닝 모델에 넣을 때는 **3차원(3D array)** (batch_size, time_steps, seq_len) 으로 데이터를 구성해줘야합니다. 

아래를 보면 앞선 글에서 데이터셋을 재구성 해주었던 **일변량과 달리 데이터셋, 모델 구성에서 두 부분**이 달라집니다.

1) 다변량이기 때문에 feature 수가 증가하며 2로 들어간 것을 확인할 수 있습니다.

2) 마지막 Dense layer neuron 또한 2으로 넣어줘야합니다. 즉 feature 혹은 변수의 개수만큼 neuron개수를 넣어줘야합니다.


```python
lag = 3
feature = 2
```


```python
scaled_train_x = np.reshape(scaled_train_x, (scaled_train_x.shape[0], lag, feature))
scaled_test_x = np.reshape(scaled_test_x, (scaled_test_x.shape[0], lag, feature))

print("train_x shape is ", scaled_train_x.shape)
print("test_x is ", scaled_test_x.shape)
```

    train_x shape is  (141, 3, 2)
    test_x is  (6, 3, 2)
    

## 06. 모델구성 및 학습

```python
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
```

    Using TensorFlow backend.
    

### LSTM
실제로 예측은 GRU를 통해 학습 및 예측을 진행하지만 LSTM으로도 모델이 어떻게 구성되어는지 한번 더 복습해보겠습니다. 

hidden state 벡터 크기를 8로 지정해서 모델을 돌린다고 할 때, 모델이 어떻게 구성되는지 알면 파라미터 수를 미리 예측할 수 있겠죠.

- LSTM 층

LSTM층의 파라미터의 개수는 (8 * 8 + 8 * 2 + 8) * 4 = **총 352개**가 나온다는 것을 알 수 있는데요. 일변량 LSTM 에서 이해를 제대로 하신 분이라면 바로 이해를 했을 것입니다. 혹시 이해하지 못했으면 해당 글 참고하길 바랍니다.

LSTM에서는 기본적으로 학습할 파라미터를 가진 **3개의 게이트**와 **Candidate Cell State**가 공식은 아래와 같습니다.

> **Input Gate** (현재 정보를 저장하기 위한 게이트) : sigmoid(W_hi * h_t-1 + W_xi * x_t + b)

> **Forget Gate** (기억을 삭제하기 위한 게이트) : sigmoid(W_hf * h_t-1 + W_xf * x_t + b)

> **Output Gate** (cell state와 함께 hidden state를 연산하기 위한 게이트) : sigmoid(W_ho * h_t-1 + W_xo * x_t + b)

> **Candidate Cell State** (현재 timestamp에서 기억할 값을 저장) : tanh(W_hc * h_t-1 + W_xc * x_t + b)


- 출력층

그리고 출력층은 18개 파라미터가 생성될 것을 미리 예측해볼 수 있습니다.

> **W_y** : 8 * 2 = 16개

> **b** : 2개



```python
seed = 0
np.random.seed(seed)
```


```python
model = Sequential()
model.add(LSTM(units=8, activation = 'relu', input_shape = (lag, feature)))
model.add(Dense(2))
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 8)                 352       
    _________________________________________________________________
    dense_1 (Dense)              (None, 2)                 18        
    =================================================================
    Total params: 370
    Trainable params: 370
    Non-trainable params: 0
    _________________________________________________________________
    

### GRU (Gated Recurrent Unit)
GRU는 뉴욕대학교 조경현 교수가 제안하였는데요. **LSTM을 간소화**시킨 신경망 구조로, **LSTM 과의 차이점**은 아래와 같습니다.

1) **출력값**은 Cell state는 없고, **Hidden state만** 있습니다.

2) 3개의 게이트를 **2개의 게이트(Update Gate, Reset Gate)**로 줄였습니다.

해당 모델구조차이가 파라미터 개수에 어떻게 영향을 미치는지 알아보겠습니다.

- GRU 층

GRU 층의 파라미터의 개수는 (8 * 8 + 8 * 2 + 8) * 3 = **총 264개**가 나온다는 것을 알 수 있는데요. 

GRU에서는 기본적으로 학습할 파라미터를 가진 **2개의 게이트**와 **Candidate Hidden State**가 공식은 아래와 같습니다.

> **Update Gate** (현재 정보를 저장 비율을 위한 게이트) : sigmoid(W_hu * h_t-1 + W_xu * x_t + b)

> **Reset Gate** (과거정보를 reset하기위한 게이트) : sigmoid(W_hr * h_t-1 + W_xr * x_t + b)

> **Candidate Hidden State** (현재 timestamp에서 기억할 값을 저장) : tanh(W_hc * ResetGate + W_xc * x_t + b)

출력값으로 내보내는 Hidden State는 ``(1 - Update Gate) * 이전 Hidden sate + Update Gate * Candidate Hidden state`` 공식에 의해 결정됩니다.

즉 위의 공식에 의해 벡터값들이 모두 정해지면 elementwise product와 sum으로 계산되기 때문에 이 부분은 파라미터 개수에 영향을 미치지 않습니다.


- 출력층

그리고 출력층은 18개 파라미터가 생성될 것을 미리 예측해볼 수 있습니다.

> **W_y** : 8 * 2 = 16개

> **b** : 2개

결론적으로 Simple RNN 부터 GRU 까지 **순환신경망의 파라미터 수**에 대해 간략하게 요약하자면, 

공통적으로 파라미터의 수는 **지정해준 hidden state 벡터크기**와 **각 timestamp에 들어가는 input data의 벡터크기**에 의해 결정됩니다.

그리고 모델구조 상 GRU층에서는 (Simple RNN 층의 파라미터수 * 3) 개 만큼, LSTM층에서는  (Simple RNN 층의 파라미터수 * 4) 개만큼 파라미터가 생성됩니다.

그래서 복잡도가 가장 높은 모델이 LSTM 입니다.


```python
seed = 0
np.random.seed(seed)
```


```python
model = Sequential()
model.add(GRU(units=8, activation = 'relu', input_shape = (lag, feature)))
model.add(Dense(2))
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    gru_1 (GRU)                  (None, 8)                 264       
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 18        
    =================================================================
    Total params: 282
    Trainable params: 282
    Non-trainable params: 0
    _________________________________________________________________
    

### Early Stopping


```python
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_loss', patience=1)
```


```python
model.compile(optimizer='adam', loss='mse')

history = model.fit(scaled_train_x, scaled_train_y, 
          validation_data = (scaled_test_x, scaled_test_y),
          callbacks=[early_stop],
          epochs = 25)
```

    Train on 141 samples, validate on 6 samples
    Epoch 1/25
    141/141 [==============================] - 0s 2ms/step - loss: 0.3137 - val_loss: 0.2965
    Epoch 2/25
    141/141 [==============================] - 0s 78us/step - loss: 0.2995 - val_loss: 0.2834
    Epoch 3/25
    141/141 [==============================] - 0s 78us/step - loss: 0.2860 - val_loss: 0.2706
    Epoch 4/25
    141/141 [==============================] - 0s 63us/step - loss: 0.2730 - val_loss: 0.2583
    Epoch 5/25
    141/141 [==============================] - 0s 81us/step - loss: 0.2605 - val_loss: 0.2464
    Epoch 6/25
    141/141 [==============================] - 0s 71us/step - loss: 0.2487 - val_loss: 0.2345
    Epoch 7/25
    141/141 [==============================] - 0s 77us/step - loss: 0.2372 - val_loss: 0.2231
    Epoch 8/25
    141/141 [==============================] - 0s 73us/step - loss: 0.2262 - val_loss: 0.2123
    Epoch 9/25
    141/141 [==============================] - 0s 64us/step - loss: 0.2155 - val_loss: 0.2016
    Epoch 10/25
    141/141 [==============================] - 0s 64us/step - loss: 0.2053 - val_loss: 0.1911
    Epoch 11/25
    141/141 [==============================] - 0s 64us/step - loss: 0.1952 - val_loss: 0.1808
    Epoch 12/25
    141/141 [==============================] - 0s 71us/step - loss: 0.1857 - val_loss: 0.1704
    Epoch 13/25
    141/141 [==============================] - 0s 70us/step - loss: 0.1763 - val_loss: 0.1602
    Epoch 14/25
    141/141 [==============================] - 0s 64us/step - loss: 0.1673 - val_loss: 0.1502
    Epoch 15/25
    141/141 [==============================] - 0s 71us/step - loss: 0.1583 - val_loss: 0.1405
    Epoch 16/25
    141/141 [==============================] - 0s 64us/step - loss: 0.1498 - val_loss: 0.1307
    Epoch 17/25
    141/141 [==============================] - 0s 64us/step - loss: 0.1414 - val_loss: 0.1212
    Epoch 18/25
    141/141 [==============================] - 0s 64us/step - loss: 0.1331 - val_loss: 0.1118
    Epoch 19/25
    141/141 [==============================] - 0s 64us/step - loss: 0.1252 - val_loss: 0.1024
    Epoch 20/25
    141/141 [==============================] - 0s 64us/step - loss: 0.1172 - val_loss: 0.0936
    Epoch 21/25
    141/141 [==============================] - 0s 71us/step - loss: 0.1095 - val_loss: 0.0851
    Epoch 22/25
    141/141 [==============================] - 0s 78us/step - loss: 0.1021 - val_loss: 0.0771
    Epoch 23/25
    141/141 [==============================] - 0s 71us/step - loss: 0.0951 - val_loss: 0.0694
    Epoch 24/25
    141/141 [==============================] - 0s 71us/step - loss: 0.0880 - val_loss: 0.0621
    Epoch 25/25
    141/141 [==============================] - 0s 64us/step - loss: 0.0814 - val_loss: 0.0550
    

### 손실그래프


```python
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
```




    [<matplotlib.lines.Line2D at 0x2015f7fd688>]




    
![output_37_1](https://user-images.githubusercontent.com/68403764/130808268-8338784e-5277-4d4c-affa-096ee1aab308.png)
    


### 과적합여부
과적합 여부를 파악하기 위해 train dataset에서 손실과 test dataset에서 손실을 비교해보겠습니다.


```python
score_train = model.evaluate(scaled_train_x, scaled_train_y, batch_size = 1)
score_test = model.evaluate(scaled_test_x, scaled_test_y, batch_size = 1)

print("Train data MSE = ", round(score_train, 5))
print("Test data MSE = ", round(score_test, 5))
```

    141/141 [==============================] - 0s 532us/step
    6/6 [==============================] - 0s 661us/step
    Train data MSE =  0.07689
    Test data MSE =  0.05504
    

## 07. 예측 및 평가

#### 예측


```python
prediction = model.predict(scaled_test_x)
```

#### 스케일변환


```python
true_prediction = scaler_y.inverse_transform(prediction)
```


```python
true_prediction
```




    array([[-1.3170096e-03, -6.1544293e-04],
           [-7.8336624e-03, -4.4890563e-03],
           [-7.9286620e-03, -3.5340418e-03],
           [-4.2351865e-05,  9.5426098e-05],
           [-1.0100403e-02, -3.8795229e-03],
           [-4.1168067e-03, -1.7954470e-03]], dtype=float32)




```python
true_prediction.shape
```




    (6, 2)




```python
actual = scaler_y.inverse_transform(scaled_test_y)
actual
```




    array([[ 0.00435075,  0.00296608],
           [ 0.01013426,  0.00408683],
           [ 0.01453086,  0.00127492],
           [-0.0102767 ,  0.00569238],
           [ 0.002253  ,  0.00102912],
           [ 0.01287419, -0.00537764]])



#### 데이터프레임으로 만들기


```python
pred_df = pd.DataFrame({'pred_m2_t' : true_prediction[:,0].tolist(),
                         'pred_sp_t' : true_prediction[:,1].tolist()}, 
                        index=test_y.index)
```


```python
result = test_y.copy()
result = result.join(pred_df)
result
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
      <th>money_t</th>
      <th>sp_t</th>
      <th>pred_m2_t</th>
      <th>pred_sp_t</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-10-01</th>
      <td>0.004351</td>
      <td>0.002966</td>
      <td>-0.001317</td>
      <td>-0.000615</td>
    </tr>
    <tr>
      <th>2019-11-01</th>
      <td>0.010134</td>
      <td>0.004087</td>
      <td>-0.007834</td>
      <td>-0.004489</td>
    </tr>
    <tr>
      <th>2019-12-01</th>
      <td>0.014531</td>
      <td>0.001275</td>
      <td>-0.007929</td>
      <td>-0.003534</td>
    </tr>
    <tr>
      <th>2020-01-01</th>
      <td>-0.010277</td>
      <td>0.005692</td>
      <td>-0.000042</td>
      <td>0.000095</td>
    </tr>
    <tr>
      <th>2020-02-01</th>
      <td>0.002253</td>
      <td>0.001029</td>
      <td>-0.010100</td>
      <td>-0.003880</td>
    </tr>
    <tr>
      <th>2020-12-01</th>
      <td>0.012874</td>
      <td>-0.005378</td>
      <td>-0.004117</td>
      <td>-0.001795</td>
    </tr>
  </tbody>
</table>
</div>



#### 시각화 (original scale)


```python
plt.plot(actual[:,0], marker = '^', label = "M2 Change")
plt.plot(true_prediction[:,0], marker = 'o', label = "M2 Change Prediction")
plt.legend()
```




    <matplotlib.legend.Legend at 0x2016086a148>





![output_52_1](https://user-images.githubusercontent.com/68403764/130808270-7eaf4101-f6a9-4109-bc92-9fed7b334004.png)
    



```python
plt.plot(actual[:,1], marker = '^', label = "Spending_Change")
plt.plot(true_prediction[:,1], marker = 'o', label = "Spending_Change Prediction")
plt.legend()
```




    <matplotlib.legend.Legend at 0x20160917588>




    
![output_53_1](https://user-images.githubusercontent.com/68403764/130808272-0e45d0a8-32a4-4935-a5d6-6a9832f0f8e4.png)

    


## 08. 모델저장 


```python
model.save('multi_gru_model.h5')
```

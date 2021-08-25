---
layout: post
title:  "[일변량딥러닝] RNN, LSTM 신경망을 이용한 시계열 예측"
categories: timeseries
comments: true
---
# 순환신경망(RNN, LSTM)을 활용한 시계열데이터 예측
이번 글에서는 **순환신경망을 활용하여 일변량 시계열데이터를 예측**해보겠습니다. 

### RNN
RNN(Recurrent Neural Network)는 기존 FFNN(Forward Feed Neural Network)와 달리 뉴런의 출력이 순환된다는 의미인데요. 자세히 말하자면, **은닉층의 출력값이 다시 은닉층의 입력으로 사용되는 신경망 구조**입니다. 그래서 timestamp가 있는 말뭉치, 오디오, 시계열 데이터를 예측하는 데에 우수한 성능을 보이고 있습니다.

### LSTM
하지만 이러한 RNN 구조는 시간이 지날수록 **경사를 소실하는 문제(vanishing gradients problem)** 이 있습니다. 쉽게 말하지면 시점이 길어지면 앞의 정보가 소실되는 **장기의존성 문제**를 가지고 있습니다. 

이를 해결하기 위해 나온 LSTM(Long Short Term Memory) 신경망은 **hidden state 값**뿐만 아니라 **cell state값도 다음 은닉층의 입력으로 사용**되면서 장기의존성 문제를 조금이나마 극복했습니다. 그래서 **단기, 장기 기억** 모두 높여 기억력을 개선하여 Long Short Term Memory라는 이름을 가지고 있습니다. 

구조는 아래에서 조금 더 자세히 살펴보겠습니다.

## 01. 데이터수집 및 전처리
데이터 수집 및 전처리에 대한 코드는 앞의 통계 일변량 분석과 같으므로 설명없이 넘어가겠습니다.


```python
import warnings
warnings.filterwarnings("ignore")
```


```python
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
```

### 데이터수집


```python
rawdata = pd.read_csv("경유월간소비.csv", encoding='cp949')
```

### 데이터전처리


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



## 02. train-test split
향후 12개월 데이터를 테스트셋으로 이용하겠습니다.


```python
test_start = len(rawdata) - 12

train = rawdata.iloc[:test_start]
test = rawdata.iloc[test_start:] 
```

## 03. feature scaling
(주의사항) 데이터를 훈련하는 시점에는 Test Dataset은 모른다고 가정해야합니다. 

그래서 전체 데이터가 아닌 **훈련데이터(Train Dataset)만을 이용하여 MinMaxScaler에 지정해준 값에 맞게 fit** 해주고, **이를 이용하고 Train Dataset, Test Dataset을 Transform 해줘야합니다.**
결국 스케일드된 Test Data 값은 -1 ~ 1 혹은 0 ~ 1 사이가 아닐 수도 있습니다. 하지만 대체적으로 그 사이값을 가지고 크게 지정범위를 벗어나지 않는다면 딥러닝 모델에 넣기에 큰 문제는 없습니다.


```python
from sklearn.preprocessing import MinMaxScaler

#스케일러 object 생성
scaler = MinMaxScaler(feature_range = (0, 1))
scaler.fit(train)
```




    MinMaxScaler()




```python
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
```


```python
scaled_train[:5]
```




    array([[0.27833215],
           [0.23473509],
           [0.44416726],
           [0.36101212],
           [0.49429793]])




```python
scaled_test[:5]
```




    array([[0.64991684],
           [0.59313376],
           [0.68139701],
           [0.64326443],
           [0.90947969]])



## 04. X Dataset 재구성
딥러닝 모델에 넣기 위해서는 입력데이터를 **3차원 형태**로 만들어줘야합니다.

만약 여러분이 시차를 5로 정했다면 (5시점 전까지의 데이터를 활용하여 예측을 진행), ``n_input``은 5가 되고 이는 샘플 각각의 행의 개수를 의미합니다. 

그리고 이번 글에서는 일변량 시계열 데이터를 예측하므로 예측변수는 한 개이기 때문에 나중에 RNN 모델에 들어갈 때 **(batch_size, 5, 1)**의 형태로 들어간다고 생각하면 됩니다.

일변량시계열데이터의 경우 훈련데이터를 테스트데이터로도 이용해야하고 위에서 말한 것처럼 X를 3차원으로 바꾸어줘야합니다.
이를 편리하게 할 수 있도록 도와주는 함수가 ``TimeseriesGenerator`` 입니다.


```python
from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 12 
n_features = 1
```

    Using TensorFlow backend.
    


```python
train_generator = TimeseriesGenerator(scaled_train,scaled_train, length=n_input, batch_size = 1)
```


```python
generated_x, generated_y = train_generator[0]
```


```python
generated_x.shape, generated_y.shape
```




    ((1, 12, 1), (1, 1))



## 05. 모델구성 및 학습


```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN, LSTM
```


```python
seed = 0
np.random.seed(seed)
```

### Simple RNN
먼저 간단하게 은닉층을 1개 Simple RNN을 쌓아주겠습니다.

그리고 데이터셋이 많지않기 때문에 hidden state의 크기는 16으로 지정해주겠습니다.

그러면 이제 **학습할 파라미터가 총 몇 개** 생성될지 미리 생각해보겠습니다. 모델의 구조를 알고 있다면 파라미터의 개수를 미리 생각해볼 수 있습니다. 

모델의 구조를 알고 있어야 정확도 성능을 향상시키기 위해 파라미터 튜닝도 잘할 수 있으므로 시간이 걸려도 미리 파라미터 개수를 생각해보는 연습은 장기적으로 좋을 것이라 생각합니다.


- Simple RNN 층

``h_t = activation function(W_h * h_t-1 + W_x * x_t + b)`` 이 공식이 Simple RNN의 은닉층 공식이 됩니다. 즉 이전 Hidden State값과 현재 timestamp의 x값을 입력으로 받아서 W_h, W_x, b 즉 가중치와 편향을 학습합니다. 결국 여러분이 지정해준 hidden state의 벡터크기과 각 timestamp마다 x input의 벡터크기에 따라서 학습할 파라미터의 개수가 결정됩니다.

> **W_h** : 16 * 16 = 256 개

> **W_x** : 1 * 16 = 16 개

> **b** : 16 개

이렇게 첫번째 은닉층에서 총 288개의 파라미터가 생성될 것을 미리 예측해볼 수 있습니다.


- 출력층

그리고 출력층은 ``y_t = linear(W_y * h_t + b)`` 공식에 따라서 아래처럼 총 17개 파라미터가 생성될 것을 미리 예측해볼 수 있습니다.

> **W_y** : 16 * 1 = 16개

> **b** : 1개


```python
model = Sequential()
model.add(SimpleRNN(units=16, activation = 'relu', input_shape = (n_input, n_features)))
model.add(Dense(1))
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    simple_rnn_1 (SimpleRNN)     (None, 16)                288       
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 305
    Trainable params: 305
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(optimizer='adam', loss='mse')

history = model.fit_generator(train_generator, epochs=10)
```

    Epoch 1/10
    234/234 [==============================] - 1s 2ms/step - loss: 0.0214
    Epoch 2/10
    234/234 [==============================] - 0s 2ms/step - loss: 0.0208
    Epoch 3/10
    234/234 [==============================] - 0s 2ms/step - loss: 0.0191
    Epoch 4/10
    234/234 [==============================] - 0s 2ms/step - loss: 0.0190
    Epoch 5/10
    234/234 [==============================] - 0s 2ms/step - loss: 0.0203
    Epoch 6/10
    234/234 [==============================] - 0s 2ms/step - loss: 0.0183
    Epoch 7/10
    234/234 [==============================] - 0s 2ms/step - loss: 0.0189
    Epoch 8/10
    234/234 [==============================] - 0s 2ms/step - loss: 0.0185
    Epoch 9/10
    234/234 [==============================] - 0s 2ms/step - loss: 0.0184
    Epoch 10/10
    234/234 [==============================] - 0s 2ms/step - loss: 0.0181
    

### LSTM

이번에는 은닉층으로 LSTM을 사용하여 학습을 진행해보겠습니다. 


- LSTM 층

LSTM층의 파라미터의 개수는 (위의 Simple RNN에서 파라미터의 개수 288 * 4)  = **총 1152개**가 나올 것을 예상해볼 수 있는데요, 그 이유에 대해 설명드리겠습니다. 

LSTM에서는 기본적으로 학습할 파라미터를 가진 **3개의 게이트**와 **Candidate Cell State**가 공식은 아래와 같습니다.

> **Input Gate** (현재 정보를 저장하기 위한 게이트) : sigmoid(W_hi * h_t-1 + W_xi * x_t + b)

> **Forget Gate** (기억을 삭제하기 위한 게이트) : sigmoid(W_hf * h_t-1 + W_xf * x_t + b)

> **Output Gate** (cell state와 함께 hidden state를 연산하기 위한 게이트) : sigmoid(W_ho * h_t-1 + W_xo * x_t + b)

> **Candidate Cell State** (현재 timestamp에서 기억할 값을 저장) : tanh(W_hc * h_t-1 + W_xc * x_t + b)

즉 Simple RNN에서 학습해야했던 공식이 4개로 많아졌습니다. 그래서 LSTM 층에서 288 * 4 개의 파라미터가 생성될 것으로 예상했는데요. LSTM은 위에서 말한 것처럼 단기와 장기기억 능력을 높이기 위해 총 2개의 출력 **Hidden state**와 **Cell state**를 다음 Timestamp로 내보냅니다.

**Cell State**는 ``(Forget Gate * 이전 Cell state) + (Input Gate * Candidate Cell State)`` 이렇게 계산됩니다. 결국 위에서 계산한 벡터를 활용하여 elementwise product와 elementwise sum에 의해 일어납니다. 

그리고 **Hidden State**는 ``Output Gate * tanh(Cell State)`` 이렇게 계산하여 벡터값이 결정되는데요. 즉 Cell State 값까지 정해지고 난 후, 맨 마지막에 일어나는 연산으로 Output Gate와  elementwise productn에 의해 계산됩니다.

이렇게 위의 4가지 공식만 확정되면 출력값이 모두 정해집니다.


- 출력층

그리고 출력층은 Simple RNN과 동일한 방식으로, 17개 파라미터가 생성될 것을 미리 예측해볼 수 있습니다.

> **W_y** : 16 * 1 = 16개

> **b** : 1개


```python
model = Sequential()
model.add(LSTM(units=16, activation = 'relu', input_shape = (n_input, n_features)))
model.add(Dense(1))
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 16)                1152      
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 1,169
    Trainable params: 1,169
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(optimizer='adam', loss='mse')

history = model.fit_generator(train_generator, epochs=10)
```

    Epoch 1/10
    234/234 [==============================] - 1s 4ms/step - loss: 0.0226
    Epoch 2/10
    234/234 [==============================] - 1s 3ms/step - loss: 0.0183
    Epoch 3/10
    234/234 [==============================] - 1s 3ms/step - loss: 0.0190
    Epoch 4/10
    234/234 [==============================] - 1s 3ms/step - loss: 0.0189
    Epoch 5/10
    234/234 [==============================] - 1s 3ms/step - loss: 0.0183
    Epoch 6/10
    234/234 [==============================] - 1s 3ms/step - loss: 0.0183
    Epoch 7/10
    234/234 [==============================] - 1s 3ms/step - loss: 0.0185
    Epoch 8/10
    234/234 [==============================] - 1s 3ms/step - loss: 0.0186
    Epoch 9/10
    234/234 [==============================] - 1s 3ms/step - loss: 0.0180
    Epoch 10/10
    234/234 [==============================] - 1s 3ms/step - loss: 0.0186
    

## 06. 예측 및 평가


```python
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
```




    [<matplotlib.lines.Line2D at 0x28a077e7348>]




    

![output_30_1](https://user-images.githubusercontent.com/68403764/130808492-4d71024c-bf35-43cc-85ff-7bc6c12aa524.png)
    


### 예측


```python
test_predictions = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    #one timestep ahead of historical 12points
    current_pred = model.predict(current_batch)[0]
    
    #prediction 저장
    test_predictions.append(current_pred)
    
    #drop frist data, append prediction
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)
```


```python
test_predictions
```




    [array([0.6419273], dtype=float32),
     array([0.6380137], dtype=float32),
     array([0.6255459], dtype=float32),
     array([0.63211024], dtype=float32),
     array([0.6246887], dtype=float32),
     array([0.61592597], dtype=float32),
     array([0.60596305], dtype=float32),
     array([0.6076634], dtype=float32),
     array([0.60715395], dtype=float32),
     array([0.6036662], dtype=float32),
     array([0.6001918], dtype=float32),
     array([0.5880745], dtype=float32)]



#### 스케일변환


```python
true_predictions = scaler.inverse_transform(test_predictions)
```

#### 데이터프레임만들기


```python
test['Predictions'] = true_predictions
```


```python
from statsmodels.tools.eval_measures import rmse
error = rmse(test['diesel'], test['Predictions'])
error, test['diesel'].mean()
```




    (1050.6838600424933, 13820.666666666666)




```python
test.plot()
```




    <AxesSubplot:xlabel='date'>




![output_39_1](https://user-images.githubusercontent.com/68403764/130808499-ef3416c7-e209-4520-b8fe-045f3fcff171.png)

    


### 08. 모델저장 및 로드
모델을 저장해놓았다가 나중에 불러와서 예측을 수행할 수 있습니다.

아래 불러온 모델을 확인하면 위와 똑같은 모델을 가져온 것을 알 수 있습니다.


```python
model.save('lstm_model.h5')
```


```python
from keras.models import load_model
saved_model = load_model('lstm_model.h5')
```


```python
saved_model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 16)                1152      
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 1,169
    Trainable params: 1,169
    Non-trainable params: 0
    _________________________________________________________________
    

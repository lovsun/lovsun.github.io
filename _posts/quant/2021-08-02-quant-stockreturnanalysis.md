---
layout: post
title:  "[데이터분석] 주가 및 수익률 분석"
categories : quant
comments: true
---

**주식 수익률 분석하는 방법** 에 대해 알아보겠습니다. 

yfinance 라이브러리를 이용하여 주가데이터를 가져와서 주식수익률을 분석하여 시각화까지 해보겠습니다


```python
import pandas as pd
import numpy as np
import yfinance as yf

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

# 데이터준비하기
제가 분석에 이용할 종목은 ``"GOOGL", "AAPL", "FB", "AMZN"``입니다.

yfinance의 download 메소드를 이용하여 주가를 다운로드 해오겠습니다.



```python
ticker = ["GOOGL", "AAPL", "FB", "AMZN"]
df = yf.download(ticker)['Adj Close']
df.dropna()
```

    [*********************100%***********************]  4 of 4 completed
    




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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>GOOGL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-05-18</th>
      <td>16.264032</td>
      <td>213.850006</td>
      <td>38.230000</td>
      <td>300.500488</td>
    </tr>
    <tr>
      <th>2012-05-21</th>
      <td>17.211582</td>
      <td>218.110001</td>
      <td>34.029999</td>
      <td>307.362366</td>
    </tr>
    <tr>
      <th>2012-05-22</th>
      <td>17.079418</td>
      <td>215.330002</td>
      <td>31.000000</td>
      <td>300.700714</td>
    </tr>
    <tr>
      <th>2012-05-23</th>
      <td>17.496151</td>
      <td>217.279999</td>
      <td>32.000000</td>
      <td>305.035034</td>
    </tr>
    <tr>
      <th>2012-05-24</th>
      <td>17.335464</td>
      <td>215.240005</td>
      <td>33.029999</td>
      <td>302.132141</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-08-09</th>
      <td>146.089996</td>
      <td>3341.870117</td>
      <td>361.609985</td>
      <td>2738.260010</td>
    </tr>
    <tr>
      <th>2021-08-10</th>
      <td>145.600006</td>
      <td>3320.679932</td>
      <td>361.130005</td>
      <td>2736.139893</td>
    </tr>
    <tr>
      <th>2021-08-11</th>
      <td>145.860001</td>
      <td>3292.110107</td>
      <td>359.959991</td>
      <td>2725.580078</td>
    </tr>
    <tr>
      <th>2021-08-12</th>
      <td>148.889999</td>
      <td>3303.500000</td>
      <td>362.649994</td>
      <td>2743.879883</td>
    </tr>
    <tr>
      <th>2021-08-13</th>
      <td>149.100006</td>
      <td>3293.969971</td>
      <td>363.179993</td>
      <td>2754.550049</td>
    </tr>
  </tbody>
</table>
<p>2325 rows × 4 columns</p>
</div>



# 주가 분석하기
일단 수익률을 분석하기 전에 먼저 주가데이터부터 분석해보겠습니다.

일별데이터는 변동성이 크기 때문에 ``월별로 모두 변환``한 이후 분석해보겠습니다. 

또한 페이스북이 2012년 2월 1일에 상장하였기때문에 2013년도부터 자료를 이용하여 분석을 진행하겠습니다.

asfreq() 메소드를 이용하여 resampling함으로써, 원하는 주기별로 분석할 수 있습니다.

> 'D' : Daily Frequency(upsampling)
    
> 'B' : Business day Frequency

> 'M' : Business day Frequency

> 'Y' : Business day Frequency


```python
df_daily = df.asfreq('D')
df_daily = df_daily.fillna(method = 'ffill')

df_monthly = df_daily.asfreq('M', how='end')
df_monthly = df_monthly[df_monthly.index >= '2013-01-01']
```

### 시각화
주가데이터를 정규화하여 시작점을 100으로 맞춰준 후 간단히 시각화를 통해서

어떤 종목의 주식이 가장 많이 올랐는지 확인해보겠습니다


```python
norm = df_monthly.div(df_monthly.iloc[0, :]).mul(100)
norm.plot(figsize=(12, 4))
```


    
![output_8_1](https://user-images.githubusercontent.com/68403764/129559809-3f0ff02b-7ec6-4d6e-b98c-5b6a2d50512e.png)
    


### CAGR
CAGR(compound annual growth rate, CAGR)은 __연평균 복리수익률__ 로, 

혹시 엑셀에 익숙하신 분들이라면 엑셀에서는 RATE()함수가 CAGR을 구해주고 있습니다. 

여기서는 수식을 직접 작성하여 분석대상 종목의 CAGR을 구해보도록하겠습니다.


```python
for i in df_monthly.columns :
    begin, end = df_monthly[i][0], df_monthly[i][-1]
    cagr = ((end/begin)**(1/len(set(df_monthly.index.year)))-1)*100
    print(f'{i} CAGR {cagr:.2f}%')
```

    AAPL CAGR 29.63%
    AMZN CAGR 32.44%
    FB CAGR 31.18%
    GOOGL CAGR 24.38%
    

# 수익률 분석하기
주가를 분석했다면 이제 수익률을 분석해보겠습니다. 일단 주가를 이용해 수익률 데이터를 구해준 뒤 순위를 알아보겠습니다.



```python
monthly_return = np.log(df_monthly/df_monthly.shift(1))
monthly_return = monthly_return.dropna()
monthly_return
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>FB</th>
      <th>GOOGL</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-02-28</th>
      <td>-0.025611</td>
      <td>-0.004644</td>
      <td>-0.128288</td>
      <td>0.058479</td>
    </tr>
    <tr>
      <th>2013-03-31</th>
      <td>0.002851</td>
      <td>0.008365</td>
      <td>-0.063243</td>
      <td>-0.008788</td>
    </tr>
    <tr>
      <th>2013-04-30</th>
      <td>0.000271</td>
      <td>-0.048751</td>
      <td>0.082146</td>
      <td>0.037539</td>
    </tr>
    <tr>
      <th>2013-05-31</th>
      <td>0.022172</td>
      <td>0.058869</td>
      <td>-0.131424</td>
      <td>0.055033</td>
    </tr>
    <tr>
      <th>2013-06-30</th>
      <td>-0.125896</td>
      <td>0.031051</td>
      <td>0.021532</td>
      <td>0.010448</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-03-31</th>
      <td>0.007313</td>
      <td>0.000372</td>
      <td>0.133895</td>
      <td>0.019886</td>
    </tr>
    <tr>
      <th>2021-04-30</th>
      <td>0.073453</td>
      <td>0.113920</td>
      <td>0.098690</td>
      <td>0.131975</td>
    </tr>
    <tr>
      <th>2021-05-31</th>
      <td>-0.051817</td>
      <td>-0.073076</td>
      <td>0.011166</td>
      <td>0.001422</td>
    </tr>
    <tr>
      <th>2021-06-30</th>
      <td>0.094500</td>
      <td>0.065184</td>
      <td>0.056132</td>
      <td>0.035405</td>
    </tr>
    <tr>
      <th>2021-07-31</th>
      <td>0.062958</td>
      <td>-0.033270</td>
      <td>0.024404</td>
      <td>0.098492</td>
    </tr>
  </tbody>
</table>
<p>102 rows × 4 columns</p>
</div>



### 수익률 순위분석
rank 메소드를 이용하여, 평균수익률이 가장 높았던 종목에 1등을 주도록 하겠습니다.

1등은 아마존이었네요.


```python
temp = monthly_return.mean()
temp.sort_index()
temp.rank(ascending=False)
```




    AAPL     3.0
    AMZN     1.0
    FB       2.0
    GOOGL    4.0
    dtype: float64



### 위험대비수익률
위험 대비 수익률이 좋았던 종목은 구글이 1등을 하였고 그 다음으로 아마존, 페이스북, 애플 순이었네요.

__수익률만 놓고 보았을 땐 아마존의 수익률이 좋았지만, 구글보다 변동성도 컸던 것을 확인할 수 있습니다.__


```python
np.mean(monthly_return) * 100
```




    AAPL     2.289777
    AMZN     2.478813
    FB       2.394540
    GOOGL    1.924995
    dtype: float64




```python
np.std(monthly_return) * 100
```




    AAPL     7.768388
    AMZN     7.909480
    FB       8.458852
    GOOGL    5.988558
    dtype: float64




```python
(np.mean(monthly_return) / np.std(monthly_return))*100
```




    AAPL     29.475571
    AMZN     31.339765
    FB       28.308101
    GOOGL    32.144540
    dtype: float64



### 수익률분포 시각화
과거 수익률분포를 시각화하여 분포가 __정규분포를 띄었는지 positive 혹은 negative로 skew__ 되었는지 확인할 수 있습니다.

그리고 어떤 종목이 매달수익률이 높을 때가 많은지도 파악할 수 있겠죠.


```python
plt.style.use('ggplot')

fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.hist(monthly_return['AAPL'], bins = 50, alpha = 0.4, color='r', label='AAPL')
ax2.hist(monthly_return['AMZN'], bins = 50, alpha = 0.4, color='g', label='AMZN')
ax3.hist(monthly_return['FB'], bins = 50, alpha = 0.4, color='b', label='FB')
ax4.hist(monthly_return['GOOGL'], bins = 50, alpha = 0.4, color='y', label='GOOGL')

ax1.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')
ax4.legend(loc='best')

plt.show()
```


    
![output_20_0](https://user-images.githubusercontent.com/68403764/129559813-1747b909-575e-4aea-9259-71c8c5f9d523.png)  


---
layout: post
title:  "[데이터수집] 주식시장"
categories : quant
comments: true
---

이전 게시글에서 경제지표 데이터를 가져오기 위해 제가 자주 사용하는 사이트를 소개해드렸습니다. 

이번에는 주식, ETF 관련 데이터 즉 주식시장에 대한 데이터를 가져오기 위해 제가 자주 사용하는 사이트를 소개해드리겠습니다. 
아래의 사이트 외에도 [EDO HISTORICAL DATA](https://eodhistoricaldata.com/), [FXCM](https://www.fxcm.com/uk/forex-trading-demo/) 등 더 다양한 사이트가 있지만 시행착오를 겪으며 가장 편리하게 이용하고 있는 사이트 위주로 설명드리겠습니다.

# yahoo finance
주가 데이터를 가져오고 싶은데 파이썬에 익숙하지 않은 분들에게 제가 가장 강력하게 추천하는 방식입니다. 많은 web data source중에서 야휴파이낸스는 __api key인증을 받을 필요가 없으며 무료__ 입니다. 그래서 초보자도 쉽게 데이터를 요청할 수 있습니다.
혹시 설치오류가 발생한다면 [yahoo finance 라이브러리 업데이트](https://pypi.org/project/yfinance/)를 참고하여 맞게 설치했는지 확인하길 바랍니다. 

(참고) 기존 설치자의 경우 코드를 실행시킬 때 ``JSONDecodeError``가 난다면 command창에 ``pip install yfinance --upgrade --no-cache-dir`` 을 이용하여 업그레이드한다면 정상작동될 것입니다.
> 무료, API KEY 필요없음

> 한국, 미국 주식시장데이터 모두 수집가능


```python
import yfinance as yf
import pandas as pd
pd.set_option('display.float_format', '{:,.2f}'.format)
```

### 1기업 주가데이터 사용예시
주가 데이터를 가져올 __기간을 설정__ 할 수 있을 뿐만 아니라 가져올 __시간간격__ 도 지정해줄 수 있습니다.


```python
ticker = "005930.KS"
Samsung = yf.download(ticker)
```

    [*********************100%***********************]  1 of 1 completed
    


```python
samsung_ytd = yf.download(ticker, period = "ytd") #올해주가데이터
samsung_5d = yf.download(ticker, period = "5d") #5일치주가데이터
samsung_30m = yf.download(ticker, period = "1d", interval = "30m") #5일치주가데이터 30분간격
```

    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    


```python
samsung_ytd
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
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
      <th>2021-01-04</th>
      <td>81,000.00</td>
      <td>84,400.00</td>
      <td>80,200.00</td>
      <td>83,000.00</td>
      <td>82,268.57</td>
      <td>38655276</td>
    </tr>
    <tr>
      <th>2021-01-05</th>
      <td>81,600.00</td>
      <td>83,900.00</td>
      <td>81,600.00</td>
      <td>83,900.00</td>
      <td>83,160.65</td>
      <td>35335669</td>
    </tr>
    <tr>
      <th>2021-01-06</th>
      <td>83,300.00</td>
      <td>84,500.00</td>
      <td>82,100.00</td>
      <td>82,200.00</td>
      <td>81,475.62</td>
      <td>42089013</td>
    </tr>
    <tr>
      <th>2021-01-07</th>
      <td>82,800.00</td>
      <td>84,200.00</td>
      <td>82,700.00</td>
      <td>82,900.00</td>
      <td>82,169.46</td>
      <td>32644642</td>
    </tr>
    <tr>
      <th>2021-01-08</th>
      <td>83,300.00</td>
      <td>90,000.00</td>
      <td>83,000.00</td>
      <td>88,800.00</td>
      <td>88,017.46</td>
      <td>59013307</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-08-09</th>
      <td>81,500.00</td>
      <td>82,300.00</td>
      <td>80,900.00</td>
      <td>81,500.00</td>
      <td>81,500.00</td>
      <td>15522581</td>
    </tr>
    <tr>
      <th>2021-08-10</th>
      <td>82,300.00</td>
      <td>82,400.00</td>
      <td>80,100.00</td>
      <td>80,200.00</td>
      <td>80,200.00</td>
      <td>20362639</td>
    </tr>
    <tr>
      <th>2021-08-11</th>
      <td>79,600.00</td>
      <td>79,800.00</td>
      <td>78,500.00</td>
      <td>78,500.00</td>
      <td>78,500.00</td>
      <td>30241137</td>
    </tr>
    <tr>
      <th>2021-08-12</th>
      <td>77,100.00</td>
      <td>78,200.00</td>
      <td>76,900.00</td>
      <td>77,000.00</td>
      <td>77,000.00</td>
      <td>42365223</td>
    </tr>
    <tr>
      <th>2021-08-13</th>
      <td>75,800.00</td>
      <td>76,000.00</td>
      <td>74,100.00</td>
      <td>74,400.00</td>
      <td>74,400.00</td>
      <td>61270643</td>
    </tr>
  </tbody>
</table>
<p>155 rows × 6 columns</p>
</div>




```python
samsung_30m
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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Datetime</th>
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
      <th>2021-08-13 09:00:00+09:00</th>
      <td>75,800.00</td>
      <td>76,000.00</td>
      <td>74,800.00</td>
      <td>75,000.00</td>
      <td>75,000.00</td>
      <td>11036476</td>
    </tr>
    <tr>
      <th>2021-08-13 09:30:00+09:00</th>
      <td>75,000.00</td>
      <td>75,100.00</td>
      <td>74,800.00</td>
      <td>74,800.00</td>
      <td>74,800.00</td>
      <td>4990255</td>
    </tr>
    <tr>
      <th>2021-08-13 10:00:00+09:00</th>
      <td>74,800.00</td>
      <td>74,900.00</td>
      <td>74,100.00</td>
      <td>74,600.00</td>
      <td>74,600.00</td>
      <td>8436050</td>
    </tr>
    <tr>
      <th>2021-08-13 10:30:00+09:00</th>
      <td>74,600.00</td>
      <td>74,700.00</td>
      <td>74,200.00</td>
      <td>74,300.00</td>
      <td>74,300.00</td>
      <td>3279841</td>
    </tr>
    <tr>
      <th>2021-08-13 11:00:00+09:00</th>
      <td>74,300.00</td>
      <td>74,800.00</td>
      <td>74,200.00</td>
      <td>74,700.00</td>
      <td>74,700.00</td>
      <td>3302087</td>
    </tr>
    <tr>
      <th>2021-08-13 11:30:00+09:00</th>
      <td>74,700.00</td>
      <td>74,700.00</td>
      <td>74,200.00</td>
      <td>74,300.00</td>
      <td>74,300.00</td>
      <td>2656681</td>
    </tr>
    <tr>
      <th>2021-08-13 12:00:00+09:00</th>
      <td>74,300.00</td>
      <td>74,300.00</td>
      <td>74,200.00</td>
      <td>74,300.00</td>
      <td>74,300.00</td>
      <td>1456138</td>
    </tr>
    <tr>
      <th>2021-08-13 12:30:00+09:00</th>
      <td>74,300.00</td>
      <td>74,500.00</td>
      <td>74,200.00</td>
      <td>74,400.00</td>
      <td>74,400.00</td>
      <td>2624967</td>
    </tr>
    <tr>
      <th>2021-08-13 13:00:00+09:00</th>
      <td>74,400.00</td>
      <td>74,400.00</td>
      <td>74,300.00</td>
      <td>74,400.00</td>
      <td>74,400.00</td>
      <td>1265310</td>
    </tr>
    <tr>
      <th>2021-08-13 13:30:00+09:00</th>
      <td>74,400.00</td>
      <td>74,400.00</td>
      <td>74,300.00</td>
      <td>74,400.00</td>
      <td>74,400.00</td>
      <td>2099723</td>
    </tr>
    <tr>
      <th>2021-08-13 14:00:00+09:00</th>
      <td>74,300.00</td>
      <td>74,400.00</td>
      <td>74,100.00</td>
      <td>74,200.00</td>
      <td>74,200.00</td>
      <td>3915728</td>
    </tr>
    <tr>
      <th>2021-08-13 14:30:00+09:00</th>
      <td>74,200.00</td>
      <td>74,300.00</td>
      <td>74,100.00</td>
      <td>74,300.00</td>
      <td>74,300.00</td>
      <td>3830241</td>
    </tr>
  </tbody>
</table>
</div>




```python
samsung_ytd.to_csv("samsungprice.csv") #해당코드를 이용하여 저장해줄 수 있습니다.
```

### 여러기업 주가데이터 사용예시
여러기업의 주가데이터를 비교하기 위해 이를 불러와서, 5년데이터 중 첫번째 주가를 100으로 맞춰준 후 시각화를 시켜보겠습니다,


```python
ticker = ['005930.KS', '035720.KS', '005380.KS'] #삼성전자, 카카오, 현대차
stocks = yf.download(ticker, period = '5y')['Adj Close']
```

    [*********************100%***********************]  3 of 3 completed
    


```python
norm = stocks.div(stocks.iloc[0, :]).mul(100)
```


```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot(stocks.index, stocks['005380.KS'], color = 'red') #현대차
plt.plot(stocks.index, stocks['005930.KS'], color = 'blue') #삼성전자
plt.plot(stocks.index, stocks['035720.KS'], color = 'green') #카카오

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('samsung vs hyundai vs kakao')

plt.show()
```


    
![png](output_12_0.png)
    


### 재무제표 데이터 사용예시


```python
ticker = "AAPL"
aapl = yf.Ticker(ticker)
aapl.balance_sheet
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
      <th>2020-09-26</th>
      <th>2019-09-28</th>
      <th>2018-09-29</th>
      <th>2017-09-30</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Total Liab</th>
      <td>258,549,000,000.00</td>
      <td>248,028,000,000.00</td>
      <td>258,578,000,000.00</td>
      <td>241,272,000,000.00</td>
    </tr>
    <tr>
      <th>Total Stockholder Equity</th>
      <td>65,339,000,000.00</td>
      <td>90,488,000,000.00</td>
      <td>107,147,000,000.00</td>
      <td>134,047,000,000.00</td>
    </tr>
    <tr>
      <th>Other Current Liab</th>
      <td>47,867,000,000.00</td>
      <td>43,242,000,000.00</td>
      <td>39,293,000,000.00</td>
      <td>38,099,000,000.00</td>
    </tr>
    <tr>
      <th>Total Assets</th>
      <td>323,888,000,000.00</td>
      <td>338,516,000,000.00</td>
      <td>365,725,000,000.00</td>
      <td>375,319,000,000.00</td>
    </tr>
    <tr>
      <th>Common Stock</th>
      <td>50,779,000,000.00</td>
      <td>45,174,000,000.00</td>
      <td>40,201,000,000.00</td>
      <td>35,867,000,000.00</td>
    </tr>
    <tr>
      <th>Other Current Assets</th>
      <td>11,264,000,000.00</td>
      <td>12,352,000,000.00</td>
      <td>12,087,000,000.00</td>
      <td>13,936,000,000.00</td>
    </tr>
    <tr>
      <th>Retained Earnings</th>
      <td>14,966,000,000.00</td>
      <td>45,898,000,000.00</td>
      <td>70,400,000,000.00</td>
      <td>98,330,000,000.00</td>
    </tr>
    <tr>
      <th>Other Liab</th>
      <td>46,108,000,000.00</td>
      <td>50,503,000,000.00</td>
      <td>48,914,000,000.00</td>
      <td>43,251,000,000.00</td>
    </tr>
    <tr>
      <th>Treasury Stock</th>
      <td>-406,000,000.00</td>
      <td>-584,000,000.00</td>
      <td>-3,454,000,000.00</td>
      <td>-150,000,000.00</td>
    </tr>
    <tr>
      <th>Other Assets</th>
      <td>33,952,000,000.00</td>
      <td>32,978,000,000.00</td>
      <td>22,283,000,000.00</td>
      <td>18,177,000,000.00</td>
    </tr>
    <tr>
      <th>Cash</th>
      <td>38,016,000,000.00</td>
      <td>48,844,000,000.00</td>
      <td>25,913,000,000.00</td>
      <td>20,289,000,000.00</td>
    </tr>
    <tr>
      <th>Total Current Liabilities</th>
      <td>105,392,000,000.00</td>
      <td>105,718,000,000.00</td>
      <td>115,929,000,000.00</td>
      <td>100,814,000,000.00</td>
    </tr>
    <tr>
      <th>Short Long Term Debt</th>
      <td>8,773,000,000.00</td>
      <td>10,260,000,000.00</td>
      <td>8,784,000,000.00</td>
      <td>6,496,000,000.00</td>
    </tr>
    <tr>
      <th>Other Stockholder Equity</th>
      <td>-406,000,000.00</td>
      <td>-584,000,000.00</td>
      <td>-3,454,000,000.00</td>
      <td>-150,000,000.00</td>
    </tr>
    <tr>
      <th>Property Plant Equipment</th>
      <td>45,336,000,000.00</td>
      <td>37,378,000,000.00</td>
      <td>41,304,000,000.00</td>
      <td>33,783,000,000.00</td>
    </tr>
    <tr>
      <th>Total Current Assets</th>
      <td>143,713,000,000.00</td>
      <td>162,819,000,000.00</td>
      <td>131,339,000,000.00</td>
      <td>128,645,000,000.00</td>
    </tr>
    <tr>
      <th>Long Term Investments</th>
      <td>100,887,000,000.00</td>
      <td>105,341,000,000.00</td>
      <td>170,799,000,000.00</td>
      <td>194,714,000,000.00</td>
    </tr>
    <tr>
      <th>Net Tangible Assets</th>
      <td>65,339,000,000.00</td>
      <td>90,488,000,000.00</td>
      <td>107,147,000,000.00</td>
      <td>134,047,000,000.00</td>
    </tr>
    <tr>
      <th>Short Term Investments</th>
      <td>52,927,000,000.00</td>
      <td>51,713,000,000.00</td>
      <td>40,388,000,000.00</td>
      <td>53,892,000,000.00</td>
    </tr>
    <tr>
      <th>Net Receivables</th>
      <td>37,445,000,000.00</td>
      <td>45,804,000,000.00</td>
      <td>48,995,000,000.00</td>
      <td>35,673,000,000.00</td>
    </tr>
    <tr>
      <th>Long Term Debt</th>
      <td>98,667,000,000.00</td>
      <td>91,807,000,000.00</td>
      <td>93,735,000,000.00</td>
      <td>97,207,000,000.00</td>
    </tr>
    <tr>
      <th>Inventory</th>
      <td>4,061,000,000.00</td>
      <td>4,106,000,000.00</td>
      <td>3,956,000,000.00</td>
      <td>4,855,000,000.00</td>
    </tr>
    <tr>
      <th>Accounts Payable</th>
      <td>42,296,000,000.00</td>
      <td>46,236,000,000.00</td>
      <td>55,888,000,000.00</td>
      <td>44,242,000,000.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
aapl.financials
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
      <th>2020-09-26</th>
      <th>2019-09-28</th>
      <th>2018-09-29</th>
      <th>2017-09-30</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Research Development</th>
      <td>18,752,000,000.00</td>
      <td>16,217,000,000.00</td>
      <td>14,236,000,000.00</td>
      <td>11,581,000,000.00</td>
    </tr>
    <tr>
      <th>Effect Of Accounting Charges</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>Income Before Tax</th>
      <td>67,091,000,000.00</td>
      <td>65,737,000,000.00</td>
      <td>72,903,000,000.00</td>
      <td>64,089,000,000.00</td>
    </tr>
    <tr>
      <th>Minority Interest</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>Net Income</th>
      <td>57,411,000,000.00</td>
      <td>55,256,000,000.00</td>
      <td>59,531,000,000.00</td>
      <td>48,351,000,000.00</td>
    </tr>
    <tr>
      <th>Selling General Administrative</th>
      <td>19,916,000,000.00</td>
      <td>18,245,000,000.00</td>
      <td>16,705,000,000.00</td>
      <td>15,261,000,000.00</td>
    </tr>
    <tr>
      <th>Gross Profit</th>
      <td>104,956,000,000.00</td>
      <td>98,392,000,000.00</td>
      <td>101,839,000,000.00</td>
      <td>88,186,000,000.00</td>
    </tr>
    <tr>
      <th>Ebit</th>
      <td>66,288,000,000.00</td>
      <td>63,930,000,000.00</td>
      <td>70,898,000,000.00</td>
      <td>61,344,000,000.00</td>
    </tr>
    <tr>
      <th>Operating Income</th>
      <td>66,288,000,000.00</td>
      <td>63,930,000,000.00</td>
      <td>70,898,000,000.00</td>
      <td>61,344,000,000.00</td>
    </tr>
    <tr>
      <th>Other Operating Expenses</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>Interest Expense</th>
      <td>-2,873,000,000.00</td>
      <td>-3,576,000,000.00</td>
      <td>-3,240,000,000.00</td>
      <td>-2,323,000,000.00</td>
    </tr>
    <tr>
      <th>Extraordinary Items</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>Non Recurring</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>Other Items</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>Income Tax Expense</th>
      <td>9,680,000,000.00</td>
      <td>10,481,000,000.00</td>
      <td>13,372,000,000.00</td>
      <td>15,738,000,000.00</td>
    </tr>
    <tr>
      <th>Total Revenue</th>
      <td>274,515,000,000.00</td>
      <td>260,174,000,000.00</td>
      <td>265,595,000,000.00</td>
      <td>229,234,000,000.00</td>
    </tr>
    <tr>
      <th>Total Operating Expenses</th>
      <td>208,227,000,000.00</td>
      <td>196,244,000,000.00</td>
      <td>194,697,000,000.00</td>
      <td>167,890,000,000.00</td>
    </tr>
    <tr>
      <th>Cost Of Revenue</th>
      <td>169,559,000,000.00</td>
      <td>161,782,000,000.00</td>
      <td>163,756,000,000.00</td>
      <td>141,048,000,000.00</td>
    </tr>
    <tr>
      <th>Total Other Income Expense Net</th>
      <td>803,000,000.00</td>
      <td>1,807,000,000.00</td>
      <td>2,005,000,000.00</td>
      <td>2,745,000,000.00</td>
    </tr>
    <tr>
      <th>Discontinued Operations</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>Net Income From Continuing Ops</th>
      <td>57,411,000,000.00</td>
      <td>55,256,000,000.00</td>
      <td>59,531,000,000.00</td>
      <td>48,351,000,000.00</td>
    </tr>
    <tr>
      <th>Net Income Applicable To Common Shares</th>
      <td>57,411,000,000.00</td>
      <td>55,256,000,000.00</td>
      <td>59,531,000,000.00</td>
      <td>48,351,000,000.00</td>
    </tr>
  </tbody>
</table>
</div>



# Financialmodelingprep
[FinancialModelingPrep](https://financialmodelingprep.com/developer) 사이트는 제가 미국주식을 이용하여 __백테스팅을 하거나 재무제표 혹은 밸류에이션 지표분석__ 을 할 때 가장 자주 사용하는 사이트입니다. 무료버전은 접근가능한 데이터나 요청횟수에 제한이 있지만 무료여도 가져올 수 있는 데이터가 상당히 많습니다. 


```python
import pandas as pd
import requests
import matplotlib.pyplot as plt
import datetime
```

1) 일단 해당사이트로 가서 ``API키를 발급`` 받은 후, key에 넣어주시면 됩니다.

2) 원하는 데이터를 찾아서 ``요청URL을 확인``하여 requests로 요청해줍니다.

3) 해당 데이터가 ``json형식이어서 이를 파싱``해줍니다.

4) (사용예시) 간단하게 제가 어떻게 활용하는지 보여드리자면, 
   AAPL 애플 주가데이터를 받아와서, 평균, 표준편차, 이동평균선을 함께 그어서 어떻게 변화해왔는지 확인할 수 있습니다. 


```python
key = 'YOUR_API_KEY'
######### 사용예시(SP500(^GSPC)) ######### 
SP500 = requests.get(f'https://financialmodelingprep.com/api/v3/historical-price-full/index/^GSPC?apikey={key}').json()
```


```python
SP500['historical']
```




    [{'date': '2021-08-13',
      'open': 4464.839844,
      'high': 4468.370117,
      'low': 4460.819824,
      'close': 4468.0,
      'adjClose': 4468.0,
      'volume': 2371630000.0,
      'unadjustedVolume': 2371630000.0,
      'change': 3.16016,
      'changePercent': 0.071,
      'vwap': 4465.72998,
      'label': 'August 13, 21',
      'changeOverTime': 0.00071},
     {'date': '2021-08-12',
      'open': 4446.080078,
      'high': 4461.77002,
      'low': 4435.959961,
      'close': 4460.830078,
      'adjClose': 4460.830078,
      'volume': 2543860000.0,
      'unadjustedVolume': 2543860000.0,
      'change': 14.75,
      'changePercent': 0.332,
      'vwap': 4452.85335,
      'label': 'August 12, 21',
      'changeOverTime': 0.00332},
     {'date': '2021-08-11',
      'open': 4442.180176,
      'high': 4449.439941,
      'low': 4436.419922,
      'close': 4442.410156,
      'adjClose': 4442.410156,
      'volume': 2803060000.0,
      'unadjustedVolume': 2803060000.0,
      'change': 0.22998,
      'changePercent': 0.005,
      'vwap': 4442.75667,
      'label': 'August 11, 21',
      'changeOverTime': 5e-05},
     ...]




```python
def stockanalysis(index):
    stock = requests.get(f'https://financialmodelingprep.com/api/v3/historical-price-full/{index}?apikey={key}')
    stock = stock.json()
    stock = stock['historical']

    hist_prices = {}

    for item in stock:
        date_stock = item['date']
        hist_prices[date_stock] = item

    stock = pd.DataFrame.from_dict(hist_prices, orient='index')
    stock.reset_index(inplace=True)
    stock['date'] = stock.loc[:,'date'].astype('datetime64[ns]')

    stock = stock[['date','adjClose']]
    stock.set_index('date',inplace=True)
    
    stock_Monthly_Prics = stock.resample('M').mean()
    stock_Monthly_Prics
    
    stock_Monthly_Prics['MA12M'] = stock_Monthly_Prics['adjClose'].rolling(12).mean()
    stock_Monthly_Prics['MA36M'] = stock_Monthly_Prics['adjClose'].rolling(36).mean()
    stock_Monthly_Prics['12MSTD'] = stock_Monthly_Prics['adjClose'].rolling(window=12).std()
    stock_Monthly_Prics['36MSTD'] = stock_Monthly_Prics['adjClose'].rolling(window=36).std()

    stock_Monthly_Prics['Upper'] = stock_Monthly_Prics['MA12M'] + (stock_Monthly_Prics['12MSTD'] * 1)
    stock_Monthly_Prics['Lower'] = stock_Monthly_Prics['MA12M'] - (stock_Monthly_Prics['12MSTD'] * 1)

    stock_Monthly_Prics[['adjClose','MA12M', 'MA36M', 'Upper','Lower']].plot(figsize=(10,4))
    plt.grid(True)
    plt.title(index + ' Bollinger Bands & Moving Averages')
    plt.axis('tight')
    plt.ylabel('Price')
```


```python
######### 사용예시(애플(AAPL)) ######### 
stockanalysis('AAPL')
```


    
![png](output_22_0.png)
    


## Fmp Cloud
[Fmp Cloud](https://fmpcloud.io/) 사이트에서는 무료버전에서 제한된 것이 많아서 상장종목 리스트를 받아오거나 할 때 사용하는데요. 저는 상대적으로 financialmodelingprep 보다는 적게 이용하는 편입니다. 바로 사용예시를 보여드리겠습니다.


```python
import pandas as pd
import requests
```

1) 일단 해당사이트로 가서 ``API키를 발급`` 받은 후, key에 넣어주시면 됩니다.

2) 원하는 데이터를 찾아서 ``요청URL을 확인``하여 requests로 요청해줍니다.

3) 해당 데이터가 ``json형식이어서 이를 파싱``해줍니다.

4) (사용예시) 나스닥 상장기업 리스트불러오기 


```python
key = 'YOUR_API_KEY'
```


```python
nasdaq_list = requests.get(f'https://fmpcloud.io/api/v3/search?query=&exchange=NASDAQ&limit=5000&apikey={key}')
nasdaq_list = nasdaq_list.json()

nasdaq_list
```




    [{'symbol': 'Z',
      'name': 'Zillow Group Inc.',
      'currency': 'USD',
      'stockExchange': 'Nasdaq Global Select',
      'exchangeShortName': 'NASDAQ'},
     {'symbol': 'ZG',
      'name': 'Zillow Group Inc',
      'currency': 'USD',
      'stockExchange': 'Nasdaq Global Select',
      'exchangeShortName': 'NASDAQ'},
     {'symbol': 'TC',
      'name': 'TuanChe Ltd',
      'currency': 'USD',
      'stockExchange': 'Nasdaq Capital Market',
      'exchangeShortName': 'NASDAQ'},
     {'symbol': 'VS',
      'name': 'Versus Systems Inc.',
      'currency': 'USD',
      'stockExchange': 'Nasdaq Capital Market',
      'exchangeShortName': 'NASDAQ'},
     {'symbol': 'FB',
      'name': 'Facebook Inc',
      'currency': 'USD',
      'stockExchange': 'Nasdaq Global Select',
      'exchangeShortName': 'NASDAQ'},
     {'symbol': 'MQ',
      'name': 'Marqeta, Inc.',
      'currency': 'USD',
      'stockExchange': 'Nasdaq Gloabl Select',
      'exchangeShortName': 'NASDAQ'},
     {'symbol': 'FA',
      'name': 'First Advantage Corporation',
      'currency': 'USD',
      'stockExchange': 'Nasdaq Gloabl Select',
      'exchangeShortName': 'NASDAQ'},
     {'symbol': 'EQ',
      'name': 'Equillium Inc',
      'currency': 'USD',
      'stockExchange': 'Nasdaq Global Select',
      'exchangeShortName': 'NASDAQ'},
     {'symbol': 'VG',
      'name': 'Vonage Holdings Corp',
      'currency': 'USD',
      'stockExchange': 'Nasdaq Global Select',
      'exchangeShortName': 'NASDAQ'},
     ...]



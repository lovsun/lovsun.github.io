---
layout: post
title:  "[데이터분석] 종목뉴스데이터를 수집하여 감성 분석"
categories : quant
comments: true
---


요즘엔 뉴스데이터가 너무 많이 올라와서, 종목에 대한 모든 기사를 보기 힘듭니다. 또한, 가끔은 내가 보고싶은 기사만 보게 되죠. 

그래서 이번 게시글에서는 __크롤링을 통해 주식뉴스를 수집하여 감정분석을 진행__ 해보겠습니다. 날짜별로 뉴스 제목을 가져와서 감정 상태를 분석하여 한눈에 파악할 수 있도록 시각화까지 시켜보겠습니다.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
import requests

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
```

# 데이터수집
미국투자하는 많은 분들이 [finviz 사이트](https://finviz.com/)를 참고할 텐데요. 크롤링을 통해 원하는 데이터먼저 수집해보겠습니다. 

제가 분석에 이용할 종목은 ``'GOOGL', 'AAPL', 'FB'``인데요. 각자 본인들이 투자하는 종목의 아래의 코드 전체를 함수로 만들어놓으면 편하겠죠.

종목별로 뉴스데이터에서 가져올 내용은 아래와 같습니다.

> 뉴스제목

> 뉴스발행 날짜 및 시간

### 종목별 뉴스데이터 가져오기


```python
tickers = ['GOOGL', 'AAPL', 'FB']
finviz_url = 'https://finviz.com/quote.ashx?t='

news_tables = {}
for ticker in tickers:
    # 데이터수집
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    r = requests.get(url,headers={'user-agent': 'my-app/0.0.1'})
    
    # 파싱
    soup = BeautifulSoup(r.text , 'html.parser')
    news_table = soup.find(id='news-table')
    
    # {종목 : 뉴스} 형태로 딕셔너리에 추가
    news_tables[ticker] = news_table
```

### 파싱을 통해 필요한 정보만 추출


```python
news_zip = []
for ticker, news in news_tables.items():

    for x in news.find_all('tr'):
        # 제목
        text = x.a.get_text() 
        # 날짜 및 시간
        date_scrape = x.td.get_text().split() #date
        if len(date_scrape) == 1:
            time = date_scrape[0]
        else: 
            date = date_scrape[0]
            time = date_scrape[1]
            
        news_zip.append([ticker, date, time, text])
```

### 최종 분석 대상 데이터프레임 생성


```python
columns = ['ticker', 'date', 'time', 'headline']
news_df = pd.DataFrame(news_zip, columns=columns)
news_df['date']= pd.to_datetime(news_df['date'])
news_df
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
      <th>ticker</th>
      <th>date</th>
      <th>time</th>
      <th>headline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOGL</td>
      <td>2021-08-16</td>
      <td>03:04AM</td>
      <td>Facebook to Expand Planned Undersea Cable Netw...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GOOGL</td>
      <td>2021-08-15</td>
      <td>11:53PM</td>
      <td>Dow Jones Futures: Resist This Urge Amid Chopp...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GOOGL</td>
      <td>2021-08-15</td>
      <td>10:16PM</td>
      <td>Google and Facebooks New Cable to Link Japan a...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GOOGL</td>
      <td>2021-08-15</td>
      <td>06:14PM</td>
      <td>The Metaverse Goes Beyond Facebook. Watch Thes...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GOOGL</td>
      <td>2021-08-15</td>
      <td>06:10AM</td>
      <td>3 Stocks You Can Buy and Hold Forever</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>FB</td>
      <td>2021-08-04</td>
      <td>01:34PM</td>
      <td>The metaverse is coming, but Big Techs latest ...</td>
    </tr>
    <tr>
      <th>296</th>
      <td>FB</td>
      <td>2021-08-04</td>
      <td>12:47PM</td>
      <td>U.S. lawmaker says Facebook move to cut off re...</td>
    </tr>
    <tr>
      <th>297</th>
      <td>FB</td>
      <td>2021-08-04</td>
      <td>12:44PM</td>
      <td>U.S. lawmaker says Facebook move to cut off re...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>FB</td>
      <td>2021-08-04</td>
      <td>11:01AM</td>
      <td>Instacart names former Facebook global ads chi...</td>
    </tr>
    <tr>
      <th>299</th>
      <td>FB</td>
      <td>2021-08-04</td>
      <td>11:00AM</td>
      <td>After Quitting Facebook, Carolyn Everson Is Jo...</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 4 columns</p>
</div>



# 감성분석
최종적으로 얻고싶은 결과물은 날짜별, 종목별 사람들의 감정상태입니다.

### VADER 소개
VADER(Valence Awarre Dictionary for Sentiment Reasoning)는 주로 소설미디어 감성 분석에 사용되는 패키지인데요. 

해당 모델은 __polarity__(긍정적인가 부정적인가) 뿐만 아니라 __intensity__(감정의 강도) 또한 잡아낼 수 있습니다. 감정이 __부정적, 중립, 긍정적일 확률__을 알려주며, 해당 확률의 합은 총 100%가 됩니다. 

또한 이 3가지 확률을 조합하여 -1에서 1 사이로 감정을 표현한 __compound score__ 또한 계산해줍니다.


```python
nltk.download('vader_lexicon')
```

    [nltk_data] Downloading package vader_lexicon to
    [nltk_data]     C:\Users\user\AppData\Roaming\nltk_data...
    [nltk_data]   Package vader_lexicon is already up-to-date!
    




    True




```python
vader = SentimentIntensityAnalyzer()

# 긍정적 문장의 예시
good = 'This was a good stock. I like the company'
print(vader.polarity_scores(good))

# 부정적 문장의 예시
bad = 'This is soso bad stock.'
print(vader.polarity_scores(bad))
```

    {'neg': 0.0, 'neu': 0.481, 'pos': 0.519, 'compound': 0.6597}
    {'neg': 0.467, 'neu': 0.533, 'pos': 0.0, 'compound': -0.5423}
    

### 뉴스타이틀에 VADER 적용
저희가 크롤링을 통해 가져온 뉴스에 해당 모델을 적용시켜줄게요. 그리고 이렇게 감정 상태를 분석한 결과를 데이터프레임에 추가해주겠습니다.


```python
sent_analysis = news_df['headline'].apply(lambda title: vader.polarity_scores(title)).tolist()
sent_analysis = pd.DataFrame(sent_analysis)
news_df = news_df.join(sent_analysis, rsuffix = 'right')
news_df
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
      <th>ticker</th>
      <th>date</th>
      <th>time</th>
      <th>headline</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
      <th>compound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GOOGL</td>
      <td>2021-08-16</td>
      <td>03:04AM</td>
      <td>Facebook to Expand Planned Undersea Cable Netw...</td>
      <td>0.000</td>
      <td>0.777</td>
      <td>0.223</td>
      <td>0.3182</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GOOGL</td>
      <td>2021-08-15</td>
      <td>11:53PM</td>
      <td>Dow Jones Futures: Resist This Urge Amid Chopp...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GOOGL</td>
      <td>2021-08-15</td>
      <td>10:16PM</td>
      <td>Google and Facebooks New Cable to Link Japan a...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GOOGL</td>
      <td>2021-08-15</td>
      <td>06:14PM</td>
      <td>The Metaverse Goes Beyond Facebook. Watch Thes...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GOOGL</td>
      <td>2021-08-15</td>
      <td>06:10AM</td>
      <td>3 Stocks You Can Buy and Hold Forever</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>FB</td>
      <td>2021-08-04</td>
      <td>01:34PM</td>
      <td>The metaverse is coming, but Big Techs latest ...</td>
      <td>0.204</td>
      <td>0.592</td>
      <td>0.204</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>296</th>
      <td>FB</td>
      <td>2021-08-04</td>
      <td>12:47PM</td>
      <td>U.S. lawmaker says Facebook move to cut off re...</td>
      <td>0.149</td>
      <td>0.851</td>
      <td>0.000</td>
      <td>-0.2732</td>
    </tr>
    <tr>
      <th>297</th>
      <td>FB</td>
      <td>2021-08-04</td>
      <td>12:44PM</td>
      <td>U.S. lawmaker says Facebook move to cut off re...</td>
      <td>0.149</td>
      <td>0.851</td>
      <td>0.000</td>
      <td>-0.2732</td>
    </tr>
    <tr>
      <th>298</th>
      <td>FB</td>
      <td>2021-08-04</td>
      <td>11:01AM</td>
      <td>Instacart names former Facebook global ads chi...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>299</th>
      <td>FB</td>
      <td>2021-08-04</td>
      <td>11:00AM</td>
      <td>After Quitting Facebook, Carolyn Everson Is Jo...</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 8 columns</p>
</div>



### Compound Score 요약 및 시각화
점수를 구했지만 __날짜와 종목별로 수치요약본__만 보면 편할 것 같다는 생각이 듭니다. 

그래서 날짜별, 종목별 Compound Score의 평균만 가져와보겠습니다.

또한, 최근 __일주치 감정을 시각화__ 시켜서 보기 편하게 만들어보겠습니다. 


```python
mean_scores = news_df.groupby(['ticker','date']).mean()
mean_scores
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
      <th></th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
      <th>compound</th>
    </tr>
    <tr>
      <th>ticker</th>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="7" valign="top">AAPL</th>
      <th>2021-08-09</th>
      <td>0.123143</td>
      <td>0.791429</td>
      <td>0.085286</td>
      <td>-0.164971</td>
    </tr>
    <tr>
      <th>2021-08-10</th>
      <td>0.029158</td>
      <td>0.891158</td>
      <td>0.079684</td>
      <td>0.096258</td>
    </tr>
    <tr>
      <th>2021-08-11</th>
      <td>0.047895</td>
      <td>0.864000</td>
      <td>0.088105</td>
      <td>0.035621</td>
    </tr>
    <tr>
      <th>2021-08-12</th>
      <td>0.066824</td>
      <td>0.840647</td>
      <td>0.092588</td>
      <td>0.056235</td>
    </tr>
    <tr>
      <th>2021-08-13</th>
      <td>0.086926</td>
      <td>0.824148</td>
      <td>0.088926</td>
      <td>0.018348</td>
    </tr>
    <tr>
      <th>2021-08-14</th>
      <td>0.074833</td>
      <td>0.891000</td>
      <td>0.034167</td>
      <td>-0.066650</td>
    </tr>
    <tr>
      <th>2021-08-15</th>
      <td>0.000000</td>
      <td>0.814400</td>
      <td>0.185600</td>
      <td>0.274780</td>
    </tr>
    <tr>
      <th rowspan="13" valign="top">FB</th>
      <th>2021-08-04</th>
      <td>0.092727</td>
      <td>0.862091</td>
      <td>0.045182</td>
      <td>-0.066273</td>
    </tr>
    <tr>
      <th>2021-08-05</th>
      <td>0.040000</td>
      <td>0.861467</td>
      <td>0.098533</td>
      <td>0.069827</td>
    </tr>
    <tr>
      <th>2021-08-06</th>
      <td>0.077667</td>
      <td>0.836933</td>
      <td>0.085467</td>
      <td>0.015307</td>
    </tr>
    <tr>
      <th>2021-08-07</th>
      <td>0.000000</td>
      <td>0.884500</td>
      <td>0.115500</td>
      <td>0.101150</td>
    </tr>
    <tr>
      <th>2021-08-08</th>
      <td>0.103000</td>
      <td>0.688500</td>
      <td>0.208500</td>
      <td>0.115300</td>
    </tr>
    <tr>
      <th>2021-08-09</th>
      <td>0.101778</td>
      <td>0.852111</td>
      <td>0.046111</td>
      <td>-0.116122</td>
    </tr>
    <tr>
      <th>2021-08-10</th>
      <td>0.026455</td>
      <td>0.836909</td>
      <td>0.136636</td>
      <td>0.184873</td>
    </tr>
    <tr>
      <th>2021-08-11</th>
      <td>0.053833</td>
      <td>0.813667</td>
      <td>0.132500</td>
      <td>0.093033</td>
    </tr>
    <tr>
      <th>2021-08-12</th>
      <td>0.081364</td>
      <td>0.877909</td>
      <td>0.040727</td>
      <td>-0.076109</td>
    </tr>
    <tr>
      <th>2021-08-13</th>
      <td>0.101615</td>
      <td>0.751538</td>
      <td>0.147000</td>
      <td>0.068462</td>
    </tr>
    <tr>
      <th>2021-08-14</th>
      <td>0.000000</td>
      <td>0.841000</td>
      <td>0.159000</td>
      <td>0.318450</td>
    </tr>
    <tr>
      <th>2021-08-15</th>
      <td>0.164000</td>
      <td>0.836000</td>
      <td>0.000000</td>
      <td>-0.299700</td>
    </tr>
    <tr>
      <th>2021-08-16</th>
      <td>0.000000</td>
      <td>0.777000</td>
      <td>0.223000</td>
      <td>0.318200</td>
    </tr>
    <tr>
      <th rowspan="11" valign="top">GOOGL</th>
      <th>2021-08-06</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2021-08-07</th>
      <td>0.000000</td>
      <td>0.866400</td>
      <td>0.133600</td>
      <td>0.208220</td>
    </tr>
    <tr>
      <th>2021-08-08</th>
      <td>0.087000</td>
      <td>0.808750</td>
      <td>0.104250</td>
      <td>0.006625</td>
    </tr>
    <tr>
      <th>2021-08-09</th>
      <td>0.064100</td>
      <td>0.806600</td>
      <td>0.129300</td>
      <td>0.087080</td>
    </tr>
    <tr>
      <th>2021-08-10</th>
      <td>0.119261</td>
      <td>0.794348</td>
      <td>0.086435</td>
      <td>-0.031135</td>
    </tr>
    <tr>
      <th>2021-08-11</th>
      <td>0.074200</td>
      <td>0.880667</td>
      <td>0.045133</td>
      <td>-0.094967</td>
    </tr>
    <tr>
      <th>2021-08-12</th>
      <td>0.052278</td>
      <td>0.862778</td>
      <td>0.084944</td>
      <td>0.046989</td>
    </tr>
    <tr>
      <th>2021-08-13</th>
      <td>0.096667</td>
      <td>0.830733</td>
      <td>0.072733</td>
      <td>-0.030833</td>
    </tr>
    <tr>
      <th>2021-08-14</th>
      <td>0.063000</td>
      <td>0.749000</td>
      <td>0.188000</td>
      <td>0.219900</td>
    </tr>
    <tr>
      <th>2021-08-15</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2021-08-16</th>
      <td>0.000000</td>
      <td>0.777000</td>
      <td>0.223000</td>
      <td>0.318200</td>
    </tr>
  </tbody>
</table>
</div>




```python
result = mean_scores.unstack().xs('compound', axis = 'columns').transpose()
result = result.tail(7).style.background_gradient(cmap='coolwarm')
result #참고로 nan은 뉴스 현재(2021-08-016 오후5시기준)까지 없었던 것을 의미합니다.
```




<style  type="text/css" >
#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow0_col0,#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow1_col0,#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow9_col0{
            background-color:  #000000;
            color:  #f1f1f1;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow0_col1{
            background-color:  #f6bda2;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow0_col2{
            background-color:  #f5a081;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow1_col1{
            background-color:  #f7b79b;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow1_col2{
            background-color:  #8badfd;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow2_col0,#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow4_col2,#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow8_col1{
            background-color:  #3b4cc0;
            color:  #f1f1f1;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow2_col1{
            background-color:  #9ebeff;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow2_col2{
            background-color:  #ccd9ed;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow3_col0,#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow6_col1{
            background-color:  #f1ccb8;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow3_col1{
            background-color:  #f08b6e;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow3_col2,#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow6_col2{
            background-color:  #6b8df0;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow4_col0{
            background-color:  #d1dae9;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow4_col1{
            background-color:  #f5c1a9;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow5_col0{
            background-color:  #dddcdc;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow5_col1{
            background-color:  #b3cdfb;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow5_col2{
            background-color:  #adc9fd;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow6_col0{
            background-color:  #c5d6f2;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow7_col0{
            background-color:  #84a7fc;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow7_col1,#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow8_col0,#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow9_col1,#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow9_col2{
            background-color:  #b40426;
            color:  #f1f1f1;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow7_col2{
            background-color:  #f39475;
            color:  #000000;
        }#T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow8_col2{
            background-color:  #85a8fc;
            color:  #000000;
        }</style><table id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8d" ><thead>    <tr>        <th class="index_name level0" >ticker</th>        <th class="col_heading level0 col0" >AAPL</th>        <th class="col_heading level0 col1" >FB</th>        <th class="col_heading level0 col2" >GOOGL</th>    </tr>    <tr>        <th class="index_name level0" >date</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8dlevel0_row0" class="row_heading level0 row0" >2021-08-07 00:00:00</th>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow0_col0" class="data row0 col0" >nan</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow0_col1" class="data row0 col1" >0.101150</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow0_col2" class="data row0 col2" >0.208220</td>
            </tr>
            <tr>
                        <th id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8dlevel0_row1" class="row_heading level0 row1" >2021-08-08 00:00:00</th>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow1_col0" class="data row1 col0" >nan</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow1_col1" class="data row1 col1" >0.115300</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow1_col2" class="data row1 col2" >0.006625</td>
            </tr>
            <tr>
                        <th id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8dlevel0_row2" class="row_heading level0 row2" >2021-08-09 00:00:00</th>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow2_col0" class="data row2 col0" >-0.164971</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow2_col1" class="data row2 col1" >-0.116122</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow2_col2" class="data row2 col2" >0.087080</td>
            </tr>
            <tr>
                        <th id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8dlevel0_row3" class="row_heading level0 row3" >2021-08-10 00:00:00</th>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow3_col0" class="data row3 col0" >0.096258</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow3_col1" class="data row3 col1" >0.184873</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow3_col2" class="data row3 col2" >-0.031135</td>
            </tr>
            <tr>
                        <th id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8dlevel0_row4" class="row_heading level0 row4" >2021-08-11 00:00:00</th>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow4_col0" class="data row4 col0" >0.035621</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow4_col1" class="data row4 col1" >0.093033</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow4_col2" class="data row4 col2" >-0.094967</td>
            </tr>
            <tr>
                        <th id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8dlevel0_row5" class="row_heading level0 row5" >2021-08-12 00:00:00</th>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow5_col0" class="data row5 col0" >0.056235</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow5_col1" class="data row5 col1" >-0.076109</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow5_col2" class="data row5 col2" >0.046989</td>
            </tr>
            <tr>
                        <th id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8dlevel0_row6" class="row_heading level0 row6" >2021-08-13 00:00:00</th>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow6_col0" class="data row6 col0" >0.018348</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow6_col1" class="data row6 col1" >0.068462</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow6_col2" class="data row6 col2" >-0.030833</td>
            </tr>
            <tr>
                        <th id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8dlevel0_row7" class="row_heading level0 row7" >2021-08-14 00:00:00</th>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow7_col0" class="data row7 col0" >-0.066650</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow7_col1" class="data row7 col1" >0.318450</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow7_col2" class="data row7 col2" >0.219900</td>
            </tr>
            <tr>
                        <th id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8dlevel0_row8" class="row_heading level0 row8" >2021-08-15 00:00:00</th>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow8_col0" class="data row8 col0" >0.274780</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow8_col1" class="data row8 col1" >-0.299700</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow8_col2" class="data row8 col2" >0.000000</td>
            </tr>
            <tr>
                        <th id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8dlevel0_row9" class="row_heading level0 row9" >2021-08-16 00:00:00</th>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow9_col0" class="data row9 col0" >nan</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow9_col1" class="data row9 col1" >0.318200</td>
                        <td id="T_86dd98e8_fe6d_11eb_a5ee_803253f6bb8drow9_col2" class="data row9 col2" >0.318200</td>
            </tr>
    </tbody></table>




```python

```

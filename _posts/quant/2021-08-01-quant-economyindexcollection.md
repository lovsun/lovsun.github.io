---
layout: post
title:  "[데이터수집] 경제지표 데이터수집 (FRED, Quandl, DBNOMICS)"
categories : quant
comments: true
---

요즘은 **API를 이용하여 쉽게 데이터를 가져올 수 있는 무료사이트**가 많습니다. 하지만 사이트마다 제공해주는 데이터, 데이터의 출처, 요청가능횟수 등이 다르다보니 다양한 사이트를 이용하게 됩니다. 

다양한 사이트 중에서 제가 경제지표 데이터를 불러올 때 제가 **가장 자주 사용하는 3가지 사이트**를 소개하고자합니다. 경제지표를 받아와서 데이터조작을 할 필요가 있는 분들에게 유용할 것입니다. 한국은행 OPEN API로 데이터 수집하는 내용은 따로 다루도록 하겠습니다.

> ``FRED``

> ``Quandl``

> ``DBNOMICS``


## FRED
[FRED](https://fred.stlouisfed.org/) 사이트는 제가 가장 자주 사용하는데요. 바로 미국 연방준비은행에서 관리하는 데이터베이스입니다. 이렇게 신뢰도가 높은 데이터를 ``API 키``만 있으면 받아올 수 있습니다.


```python
import pandas as pd
import plotly.graph_objects as go
import fredapi
```

간단하게 데이터를 받아와서 시각화를 해보는 코드입니다.

1) 일단 ``YOUR API KEY`` 라는 부분에 여러분의 API KEY를 적어줘야합니다. 
   FRED 사이트에 가셔서 API 키를 발급받을 수 있습니다.

2) ``원하는 경제지표를 다운로드``  받아오시면 됩니다. 
   예를 들어, 저는 장단기금리차(10년물국채, 3개월국채 이용)가 궁금하여 해당 지표를 받아왔습니다.

3) 간단하게 ``시각화`` 를 하여 확인할 수 있습니다. 
   값이 0인 라인을 추가하면 장단기금리차가 역전될 상황을 더 쉽게 파악할 수 있겠죠.


```python
fred = fredapi.Fred(api_key='YOUR_API_KEY')

######### 사용예시(장단기금리차) ######### 
T10Y_3M = pd.DataFrame(data=fred.get_series('T10Y3M'), columns=['T10Y3M'])
```


```python
T10Y_3M.head()
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
      <th>T10Y3M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1982-01-04</th>
      <td>2.32</td>
    </tr>
    <tr>
      <th>1982-01-05</th>
      <td>2.24</td>
    </tr>
    <tr>
      <th>1982-01-06</th>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1982-01-07</th>
      <td>2.46</td>
    </tr>
    <tr>
      <th>1982-01-08</th>
      <td>2.50</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = go.Figure()
fig.add_trace(go.Scatter(
                x=T10Y_3M.index,
                y=T10Y_3M.iloc[:,0],
                name="T10Y_3M",
                line_color='red',
                opacity=0.8
))

fig.add_trace(go.Scatter(
                x=T10Y_3M.index,
                y=[0] * len(T10Y_3M),
                mode='lines',
                line_color='orange'
))

fig.update_layout(title_text="T10Y_3M(18M)  (%, Daily)",
                  xaxis_rangeslider_visible=True)

fig.show()
```
![Economy_01](https://user-images.githubusercontent.com/68403764/129559777-6a3975c0-4207-4401-9f98-e17ac57c4415.PNG)

## Quandl
[Quandl](https://www.quandl.com/) 사이트도 API키를 발급받으면 손쉽게 경제지표를 가져올 수 있습니다. 제가 해당사이트를 알게 된 것은 ISM PMI지표를 FRED사이트에서는 제공해주지 않아서 찾다가 발견한 사이트입니다. 그럼 PMI지표를 이용하여 어떻게 활용할 수 있는지 알려드리겠습니다.


```python
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import quandl
```

1) Quandl 사이트로 가서, 회원가입하고 ``API 키``를 발급받으시길 바랍니다. 그리고 YOUR API KEY 부분에 키를 넣어주시면 됩니다.

2) 이후 ``데이터``를 받아오면 됩니다. 저는 대표적으로 경기를 확인할 때 보는 지표인 제조업 PMI, 미국 경기선행지수 데이터를 가져왔습니다. (경기선행지수는 FRED이용)

3) 이를 ``시각화``해봤는데요. PMI지표는 꺽였지만 여전히 50 위에 있으며, 경기선행지수는 상승 중인 것으로 미국 경기지표를 쉽게 파악할 수 있습니다.


```python
quandl.ApiConfig.api_key = 'YOUR_API_KEY'
######### 사용예시(제조업PMI, CLI) ######### 
pmi = quandl.get("ISM/MAN_PMI")
us_cli = pd.DataFrame(data=fred.get_series('USALOLITOAASTSAM'), columns=['US_CLI'])
```


```python
pmi.head()
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
      <th>PMI</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1948-01-01</th>
      <td>51.7</td>
    </tr>
    <tr>
      <th>1948-02-01</th>
      <td>50.2</td>
    </tr>
    <tr>
      <th>1948-03-01</th>
      <td>43.3</td>
    </tr>
    <tr>
      <th>1948-04-01</th>
      <td>45.4</td>
    </tr>
    <tr>
      <th>1948-05-01</th>
      <td>49.5</td>
    </tr>
  </tbody>
</table>
</div>




```python

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=us_cli.index,
                y=us_cli.iloc[:,0],
                name="US_CLI",
                line_color='deepskyblue',
                opacity=0.8),
                secondary_y=False, row=1, col=1
)

fig.add_trace(go.Scatter(
                x=pmi.index,
                y=pmi.iloc[:,0],
                name="M_pmi",
                line_color='red',
                opacity=0.8),
                secondary_y=True
)

fig.add_trace(go.Scatter(
                x=pmi.index,
                y=[50] * len(pmi),
                mode='lines',
                line_color='orange'),
                secondary_y=True
)

fig.update_layout(title_text="PMI, US_CLI (index, Monthly)",
                  xaxis_rangeslider_visible=True)

fig.show()
```
![Economy_02](https://user-images.githubusercontent.com/68403764/129559794-dc702339-6f15-4631-abc0-8fb691ff9089.PNG)

### DBNOMICS
[DBNOMICS](https://db.nomics.world) 사이트는 경제지표 데이터베이스로, 간단하게 경제지표를 검색해서 Download 버튼을 눌러 json형식 데이터를 받아올 수 있는 링크를 얻을 수 있습니다. 


```python
import pandas as pd
import requests
import json
```

사용할 때마다 코드를 짜는건 비효율적입니다. 그래서 함수로 만들어 놓았는데요.
    
1) ``json 형식의 데이터``를 불러와서 파싱해줍니다.

2) ``원하는 정보만 추출``합니다.

3) 데이터프레임 형태로 만들어 data ``변수에 저장``해줍니다.

4) 경제금융 분야에서 데이터의 출처는 중요합니다. 그래서 데이터출처를 출력해줍니다.

(사용예시) 사용예시를 보면 CLI(Composite Leading Indicator) 경기선행지수를 잘 받아온 것을 확인할 수 있습니다. 그리고 출처를 출력해주었는데요. 출처가 OECD로 매달 OECD에서 CLI지수를 발표하니까 제대로 출력되고 있는 것을 확인할 수 있습니다.


```python
def checkecodata(url):
    r= requests.get(url)
    r = r.json()
    
    periods = r['series']['docs'][0]['period']
    values = r['series']['docs'][0]['value']
    dataset = r['series']['docs'][0]['dataset_name']

    data = pd.DataFrame(values,index=periods)
    data.columns = [dataset]
    
    print(r['series']['docs'][0]['provider_code'])
    
    return data   
```


```python
######### 사용예시(CPI) ######### 
url = 'https://api.db.nomics.world/v22/series/OECD/MEI_CLI?dimensions=%7B%7D&observations=1'
data = checkecodata(url)
```

```python
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
      <th>Composite Leading Indicators (MEI)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-02</th>
      <td>99.82009</td>
    </tr>
    <tr>
      <th>2000-03</th>
      <td>99.87109</td>
    </tr>
    <tr>
      <th>2000-04</th>
      <td>99.91908</td>
    </tr>
    <tr>
      <th>2000-05</th>
      <td>100.25200</td>
    </tr>
    <tr>
      <th>2000-06</th>
      <td>100.57930</td>
    </tr>
  </tbody>
</table>
</div>


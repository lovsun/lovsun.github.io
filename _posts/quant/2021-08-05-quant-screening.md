---
layout: post
title:  "[데이터분석] 종목 스크리닝 및 필터링"
categories : quant
comments: true
---


이번 글에서는 조건에 맞는 기업만 추출해보는 코드를 작성할 것입니다. 흔히 스크리닝이라고도 불리는데요. 

예시로는 S&P500 종목 중에서 Industrials 섹터에 속하는 종목의 Valuation 지표(PE ratio, PEG ratio, P/B ratio)를 가져와서 필터링 및 스크리닝 작업을 해보겠습니다.

밸류에이션 지표 외에 재무비율 데이터 등 다양한 지표를 가져올 수 있으니까 본인의 분석목적에 맞게 활용해보세요.


```python
import pandas as pd
import requests
from tqdm import tqdm
```

# S&P500 종목 리스트
위키피디아에 [SP500종목리스트](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)가 나와있습니다. 이를 이용하여 **S&P500종목을 가져온 후, Industrials에 속하는 종목**만 리스트로 만들어주겠습니다


```python
sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
sp500 = sp500[0]
```


```python
# sp500종목 
tickers = sp500['Symbol'].values.tolist()
```


```python
# sp500종목 중 Industrials섹터인 종목 
industrial_tickers = sp500[sp500['GICS Sector'] == 'Industrials']['Symbol']
industrial_tickers = industrial_tickers.values.tolist()
```

# Finantial Ratio 수집하기
[FinancialModelingPrep](https://financialmodelingprep.com/developer/) 사이트를 통해 쉽게 재무비율을 수집할 수 있습니다. 위의 종목리스트에 대한 재무비율을 수집하여 데이터프레임형태로 만들어주겠습니다. 제가 올린 다른 글을 읽으면 나와있듯이, FinancialModelingPrep은 사전에 API 키를 발급받아야 이용할 수 있습니다.

1) 재무비율을 요청하는 ``url로 요청보내기``
   
   S&P500종목 전체에 대해 궁금하신 분들은 ``for ticker in tickers``를 사용하면 되고 
   
   industrials 섹터 종목에 대해 궁금하신 분들을 ``for ticker in industrial_tickers``를 사용하면 됩니다.

2) ``json 파일`` 파싱하기

3) 원하는 ``정보만 추출``하기

4) ``데이터프레임``으로 만들기


```python
YOUR_API_KEY = 'YOUR_API_KEY' #api키를 사전에 발급받으세요.

result = {}
for ticker in tqdm(industrial_tickers):
    ratio = requests.get(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey={YOUR_API_KEY}")
    ratio = ratio.json()
    
    pe_ratio = float(ratio[0]['peRatioTTM'])
    peg_ratio = float(ratio[0]['pegRatioTTM'])
    pb_ratio = float(ratio[0]['priceToBookRatioTTM'])
    
    result[ticker] = [pe_ratio, peg_ratio, pb_ratio]

result = pd.DataFrame(result)
result = result.transpose()

column_name = ['P/E', 'PEG', 'P/B']
result.columns = column_name
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 74/74 [01:06<00:00,  1.11it/s]
    


```python
result.head(15)
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
      <th>P/E</th>
      <th>PEG</th>
      <th>P/B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MMM</th>
      <td>19.856604</td>
      <td>1.123316</td>
      <td>8.036580</td>
    </tr>
    <tr>
      <th>ALK</th>
      <td>-11.564804</td>
      <td>0.042500</td>
      <td>2.129260</td>
    </tr>
    <tr>
      <th>ALLE</th>
      <td>26.914454</td>
      <td>-1.235912</td>
      <td>15.643129</td>
    </tr>
    <tr>
      <th>AAL</th>
      <td>-2.162049</td>
      <td>0.003446</td>
      <td>-1.637822</td>
    </tr>
    <tr>
      <th>AME</th>
      <td>36.212204</td>
      <td>27.086179</td>
      <td>4.967863</td>
    </tr>
    <tr>
      <th>AOS</th>
      <td>26.453875</td>
      <td>-4.053167</td>
      <td>6.324801</td>
    </tr>
    <tr>
      <th>BA</th>
      <td>-15.273919</td>
      <td>0.008645</td>
      <td>-8.193344</td>
    </tr>
    <tr>
      <th>CHRW</th>
      <td>19.206018</td>
      <td>-1.578376</td>
      <td>6.426834</td>
    </tr>
    <tr>
      <th>CARR</th>
      <td>19.921568</td>
      <td>1.701634</td>
      <td>6.817831</td>
    </tr>
    <tr>
      <th>CAT</th>
      <td>27.113080</td>
      <td>-0.533734</td>
      <td>7.011364</td>
    </tr>
    <tr>
      <th>CTAS</th>
      <td>37.914062</td>
      <td>1.353774</td>
      <td>11.068984</td>
    </tr>
    <tr>
      <th>CPRT</th>
      <td>39.795513</td>
      <td>2.378476</td>
      <td>10.193895</td>
    </tr>
    <tr>
      <th>CSX</th>
      <td>22.894556</td>
      <td>-1.692859</td>
      <td>5.686828</td>
    </tr>
    <tr>
      <th>CMI</th>
      <td>15.907593</td>
      <td>-0.936423</td>
      <td>3.903162</td>
    </tr>
    <tr>
      <th>DE</th>
      <td>26.148657</td>
      <td>-1.927357</td>
      <td>7.858543</td>
    </tr>
  </tbody>
</table>
</div>



(참고사항) 아래 딕셔너리에 key값은 여러분들이 수집할 수 있는 financial ratio 정보입니다.

배당비율, 유동비율, 당좌비율, 부채비율, Valuation 비율 등 다양한 정보를 얻을 수 있으니까

**투자전략에 따라서 적절한 ratio**를 가져오시면 됩니다.


```python
ratio[0]
```




    {'dividendYielTTM': 0.008249627620975444,
     'dividendYielPercentageTTM': 0.8249627620975444,
     'peRatioTTM': 61.752357,
     'pegRatioTTM': -1.6816415264450455,
     'payoutRatioTTM': 0.5052155844155845,
     'currentRatioTTM': 1.849412965798877,
     'quickRatioTTM': 1.4369576314446146,
     'cashRatioTTM': 0.9392547217968351,
     'daysOfSalesOutstandingTTM': 68.4375,
     'daysOfInventoryOutstandingTTM': 73.15953793318764,
     'operatingCycleTTM': 90.03453793318764,
     'daysOfPayablesOutstandingTTM': 68.25944427099594,
     'cashConversionCycleTTM': 18.21739384951607,
     'grossProfitMarginTTM': 0.38403846153846155,
     'operatingProfitMarginTTM': 0.09692307692307692,
     'pretaxProfitMarginTTM': 0.08846153846153847,
     'netProfitMarginTTM': 0.07403846153846154,
     'effectiveTaxRateTTM': 0.16304347826086957,
     'returnOnAssetsTTM': 0.043824701195219126,
     'returnOnEquityTTM': 0.1301775147928994,
     'returnOnCapitalEmployedTTM': 0.07383533548198067,
     'netIncomePerEBTTTM': 0.8369565217391305,
     'ebtPerEbitTTM': 0.9126984126984127,
     'ebitPerRevenueTTM': 0.09692307692307692,
     'debtRatioTTM': 0.6529311326124075,
     'debtEquityRatioTTM': 1.881272548376517,
     'longTermDebtToCapitalizationTTM': 0.4471441523118767,
     'totalDebtToCapitalizationTTM': 0.5070331447049313,
     'interestCoverageTTM': 5.929411764705883,
     'cashFlowToDebtRatioTTM': 0.2713647959183674,
     'companyEquityMultiplierTTM': 2.8812725483765167,
     'receivablesTurnoverTTM': 5.333333333333333,
     'payablesTurnoverTTM': 5.347245409015025,
     'inventoryTurnoverTTM': 4.989096573208723,
     'fixedAssetTurnoverTTM': 8.30670926517572,
     'assetTurnoverTTM': 0.5919180421172453,
     'operatingCashFlowPerShareTTM': 4.725152692948362,
     'freeCashFlowPerShareTTM': 4.808439755691283,
     'cashPerShareTTM': 10.216546363131593,
     'operatingCashFlowSalesRatioTTM': 0.16365384615384615,
     'freeCashFlowOperatingCashFlowRatioTTM': 1.0176263219741482,
     'cashFlowCoverageRatiosTTM': 0.2713647959183674,
     'shortTermCoverageRatiosTTM': 1.2701492537313432,
     'capitalExpenditureCoverageRatioTTM': 56.733333333333334,
     'dividendPaidAndCapexCoverageRatioTTM': 56.73332924853363,
     'priceBookValueRatioTTM': 7.732958838963594,
     'priceToBookRatioTTM': 7.732958838963594,
     'priceToSalesRatioTTM': 4.534190673076923,
     'priceEarningsRatioTTM': 61.241016883116885,
     'priceToFreeCashFlowsRatioTTM': 27.226087182448037,
     'priceToOperatingCashFlowsRatioTTM': 27.705982961222087,
     'priceCashFlowRatioTTM': 27.705982961222087,
     'priceEarningsToGrowthRatioTTM': -1.6816415264450455,
     'priceSalesRatioTTM': 4.534190673076923,
     'dividendYieldTTM': 0.008249627620975444,
     'enterpriseValueMultipleTTM': 37.68756287878788,
     'priceFairValueTTM': 7.732958838963594,
     'dividendPerShareTTM': 1.08}



# 스크리닝
각 **재무비율 별로 순위**를 매겨보는 task와 **조건에 맞는 종목만 필터링**을 해주는 task를 진행해보겠습니다.

참고로 **스크리닝과 필터링**은 원하는 종목으로 **basket을 구성**해줄 때도 사용되니까, 코드를 작성할 수 있으면 유용하게 사용할 수 있습니다. 가령 우량가치주로 바스킷을 구성해준다고 할 경우, P/E 기준으로 저평가되어있고 시가총액은 20억 보다 큰 주식으로 필터링하여 basket을 구성해줄 수 있겠죠.

### 순위
**P/E (PER) 비율이 높은 상위 15종목**을 확인해보겠습니다.


```python
screening = result.copy()
```


```python
screening.sort_values(by="P/E", ascending=False).head(15)
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
      <th>P/E</th>
      <th>PEG</th>
      <th>P/B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KSU</th>
      <td>206.060800</td>
      <td>14.092855</td>
      <td>6.424735</td>
    </tr>
    <tr>
      <th>INFO</th>
      <td>76.260420</td>
      <td>0.982318</td>
      <td>5.289038</td>
    </tr>
    <tr>
      <th>TDG</th>
      <td>72.475510</td>
      <td>-2.055453</td>
      <td>-11.153321</td>
    </tr>
    <tr>
      <th>IR</th>
      <td>68.309300</td>
      <td>-0.564866</td>
      <td>2.341563</td>
    </tr>
    <tr>
      <th>XYL</th>
      <td>61.752357</td>
      <td>-1.681642</td>
      <td>7.732959</td>
    </tr>
    <tr>
      <th>RTX</th>
      <td>59.066574</td>
      <td>-0.361230</td>
      <td>1.782715</td>
    </tr>
    <tr>
      <th>ROL</th>
      <td>54.410030</td>
      <td>1.927663</td>
      <td>17.274312</td>
    </tr>
    <tr>
      <th>GNRC</th>
      <td>50.844845</td>
      <td>1.340799</td>
      <td>15.585716</td>
    </tr>
    <tr>
      <th>ROP</th>
      <td>47.716137</td>
      <td>-1.022832</td>
      <td>4.549117</td>
    </tr>
    <tr>
      <th>TDY</th>
      <td>46.944996</td>
      <td>34.065145</td>
      <td>2.604319</td>
    </tr>
    <tr>
      <th>HWM</th>
      <td>46.637173</td>
      <td>-0.466633</td>
      <td>3.792938</td>
    </tr>
    <tr>
      <th>VRSK</th>
      <td>45.710167</td>
      <td>0.770485</td>
      <td>11.123341</td>
    </tr>
    <tr>
      <th>EFX</th>
      <td>44.071100</td>
      <td>-0.191267</td>
      <td>8.977119</td>
    </tr>
    <tr>
      <th>IEX</th>
      <td>40.605950</td>
      <td>-3.565710</td>
      <td>6.323315</td>
    </tr>
    <tr>
      <th>WM</th>
      <td>39.892440</td>
      <td>-4.011220</td>
      <td>8.622018</td>
    </tr>
  </tbody>
</table>
</div>



### 조건에 맞는 종목 선정
여기에서는 **PEG ratio 기준으로 20을 넘으면서 P/B ratio 기준으로는 5보다 작은 종목**을 필터링해보겠습니다.


```python
screening[(screening['PEG'] > 20) & (screening['P/B'] <= 5)]
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
      <th>P/E</th>
      <th>PEG</th>
      <th>P/B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AME</th>
      <td>36.212204</td>
      <td>27.086179</td>
      <td>4.967863</td>
    </tr>
    <tr>
      <th>TDY</th>
      <td>46.944996</td>
      <td>34.065145</td>
      <td>2.604319</td>
    </tr>
  </tbody>
</table>
</div>



---
layout: post
title:  "[데이터분석] 백테스팅 (정률투자법)"
categories : quant
comments: true
---

**ETF를 이용하여 백테스팅하는 방법**에 대해 포스팅하겠습니다. 

해당 글을 참고하여 여러분이 생각하는 아이디어를 직접 백테스팅으로 구현해보는 것도 좋은 생각일 것 같습니다.


```python
import pandas as pd
import numpy as np
import yfinance as yf

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

# 종목선정

[(참고사이트) ETF LIST 선정기준](https://allocatesmartly.com/aggressive-global-tactical-asset-allocation/)
A list of assets traded: US large-cap value (represented by IWD), US large-cap momentum (MTUM), US small-cap value (IWN), US small-cap equities (IWM), international equities (EFA), emerging market equities (EEM), intermediate-term US Treasuries (IEF), international treasuries (BWX), US corporate bonds (LQD), long-term US Treasuries (TLT), commodities (DBC), gold (GLD), and real estate (VNQ).

Ibbotson and Kaplan (2000) 자료에 따르면, 전술적 자산배분(Tactical asset allocation)은 주식 수익률에 가장 큰 비율(45%)를 차지합니다. 그만큼 거시경제 상황에 따라서 자산배분하는 것이 중요한데요. 이와 관련된 퀀트 자료를 찾다가 위 사이트를 알게 되었고, 해당 자료의 asset list를 활용하여 종목선정을 위한 수익률분석부터 진행해보겠습니다.



```python
ticker = ["IWD", "MTUM", "IWN", "IWM", "EFA", "EEM", "IEF", "BWX", "LQD", "TLT", "DBC", "GLD", "VNQ"]
df = yf.download(ticker)['Adj Close']
df.dropna()
```

    [*********************100%***********************]  13 of 13 completed
    




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
      <th>BWX</th>
      <th>DBC</th>
      <th>EEM</th>
      <th>EFA</th>
      <th>GLD</th>
      <th>IEF</th>
      <th>IWD</th>
      <th>IWM</th>
      <th>IWN</th>
      <th>LQD</th>
      <th>MTUM</th>
      <th>TLT</th>
      <th>VNQ</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>2013-04-18</th>
      <td>27.420938</td>
      <td>25.057907</td>
      <td>34.398304</td>
      <td>45.859001</td>
      <td>134.300003</td>
      <td>93.737366</td>
      <td>65.857666</td>
      <td>80.268036</td>
      <td>68.262016</td>
      <td>92.606995</td>
      <td>46.140614</td>
      <td>100.695847</td>
      <td>51.890358</td>
    </tr>
    <tr>
      <th>2013-04-19</th>
      <td>27.337278</td>
      <td>24.999609</td>
      <td>34.901211</td>
      <td>46.226624</td>
      <td>135.470001</td>
      <td>93.685593</td>
      <td>66.444443</td>
      <td>81.190933</td>
      <td>69.040253</td>
      <td>92.744171</td>
      <td>46.140614</td>
      <td>100.483101</td>
      <td>52.689987</td>
    </tr>
    <tr>
      <th>2013-04-23</th>
      <td>27.374464</td>
      <td>24.921879</td>
      <td>35.236477</td>
      <td>47.079182</td>
      <td>136.880005</td>
      <td>93.676964</td>
      <td>67.386559</td>
      <td>82.714218</td>
      <td>70.323059</td>
      <td>92.896561</td>
      <td>47.318592</td>
      <td>100.204994</td>
      <td>52.911316</td>
    </tr>
    <tr>
      <th>2013-04-24</th>
      <td>27.402346</td>
      <td>25.193933</td>
      <td>35.446026</td>
      <td>47.485928</td>
      <td>138.369995</td>
      <td>93.720123</td>
      <td>67.502251</td>
      <td>83.144341</td>
      <td>70.827629</td>
      <td>92.934669</td>
      <td>46.720558</td>
      <td>100.458580</td>
      <td>53.025539</td>
    </tr>
    <tr>
      <th>2013-04-25</th>
      <td>27.411638</td>
      <td>25.524281</td>
      <td>35.764523</td>
      <td>47.736221</td>
      <td>141.630005</td>
      <td>93.616600</td>
      <td>67.750198</td>
      <td>83.690941</td>
      <td>71.109810</td>
      <td>92.881264</td>
      <td>46.947090</td>
      <td>100.025032</td>
      <td>52.847054</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-08-09</th>
      <td>29.540001</td>
      <td>18.740000</td>
      <td>52.110001</td>
      <td>80.269997</td>
      <td>161.720001</td>
      <td>116.779999</td>
      <td>161.039993</td>
      <td>222.160004</td>
      <td>160.490005</td>
      <td>134.080002</td>
      <td>181.240005</td>
      <td>147.250000</td>
      <td>106.339996</td>
    </tr>
    <tr>
      <th>2021-08-10</th>
      <td>29.440001</td>
      <td>19.080000</td>
      <td>52.180000</td>
      <td>80.449997</td>
      <td>161.770004</td>
      <td>116.559998</td>
      <td>161.990005</td>
      <td>222.600006</td>
      <td>161.720001</td>
      <td>133.779999</td>
      <td>181.839996</td>
      <td>146.570007</td>
      <td>105.330002</td>
    </tr>
    <tr>
      <th>2021-08-11</th>
      <td>29.559999</td>
      <td>19.250000</td>
      <td>52.320000</td>
      <td>81.000000</td>
      <td>164.000000</td>
      <td>116.690002</td>
      <td>163.000000</td>
      <td>223.690002</td>
      <td>163.020004</td>
      <td>133.990005</td>
      <td>181.710007</td>
      <td>146.479996</td>
      <td>105.949997</td>
    </tr>
    <tr>
      <th>2021-08-12</th>
      <td>29.500000</td>
      <td>19.180000</td>
      <td>51.860001</td>
      <td>80.930000</td>
      <td>164.039993</td>
      <td>116.570000</td>
      <td>163.179993</td>
      <td>223.160004</td>
      <td>162.600006</td>
      <td>134.179993</td>
      <td>182.009995</td>
      <td>146.240005</td>
      <td>106.180000</td>
    </tr>
    <tr>
      <th>2021-08-13</th>
      <td>29.610001</td>
      <td>19.070000</td>
      <td>51.730000</td>
      <td>81.419998</td>
      <td>166.389999</td>
      <td>117.239998</td>
      <td>163.190002</td>
      <td>221.130005</td>
      <td>161.429993</td>
      <td>135.080002</td>
      <td>181.139999</td>
      <td>148.550003</td>
      <td>106.760002</td>
    </tr>
  </tbody>
</table>
<p>2096 rows × 13 columns</p>
</div>



데이터를 불러왔으면 여러분은 asfreq() 메소드를 이용하여 resampling함으로써, 원하는 주기별로 분석할 수 있습니다.

> 'D' : Daily Frequency, upsampling
    
> 'B' : Business day Frequency

> 'M' : Business day Frequency

> 'Y' : Business day Frequency

### 일별수익률


```python
df_daily = df.asfreq('D')
df_daily = df_daily.fillna(method = 'ffill')
daily_return = np.log(df_daily/df_daily.shift(1))
daily_return = daily_return.dropna()
daily_return*100
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
      <th>BWX</th>
      <th>DBC</th>
      <th>EEM</th>
      <th>EFA</th>
      <th>GLD</th>
      <th>IEF</th>
      <th>IWD</th>
      <th>IWM</th>
      <th>IWN</th>
      <th>LQD</th>
      <th>MTUM</th>
      <th>TLT</th>
      <th>VNQ</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>2013-04-19</th>
      <td>-0.305562</td>
      <td>-0.232925</td>
      <td>1.451426</td>
      <td>0.798440</td>
      <td>0.867410</td>
      <td>-0.055247</td>
      <td>0.887031</td>
      <td>1.143210</td>
      <td>1.133623</td>
      <td>0.148018</td>
      <td>0.000000</td>
      <td>-0.211499</td>
      <td>1.529245</td>
    </tr>
    <tr>
      <th>2013-04-20</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2013-04-21</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2013-04-22</th>
      <td>0.407191</td>
      <td>0.310443</td>
      <td>0.479157</td>
      <td>0.422106</td>
      <td>1.777852</td>
      <td>0.055247</td>
      <td>0.285640</td>
      <td>0.286588</td>
      <td>0.185611</td>
      <td>0.032850</td>
      <td>0.000000</td>
      <td>0.073281</td>
      <td>-0.135594</td>
    </tr>
    <tr>
      <th>2013-04-23</th>
      <td>-0.271257</td>
      <td>-0.621853</td>
      <td>0.476873</td>
      <td>1.405395</td>
      <td>-0.742408</td>
      <td>-0.064458</td>
      <td>1.122301</td>
      <td>1.572205</td>
      <td>1.655394</td>
      <td>0.131326</td>
      <td>2.520973</td>
      <td>-0.350434</td>
      <td>0.554772</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-08-09</th>
      <td>0.000000</td>
      <td>-1.430490</td>
      <td>0.326770</td>
      <td>-0.049821</td>
      <td>-1.789482</td>
      <td>-0.213849</td>
      <td>-0.186118</td>
      <td>-0.552126</td>
      <td>-0.713997</td>
      <td>-0.431646</td>
      <td>0.853337</td>
      <td>-0.359285</td>
      <td>-0.534592</td>
    </tr>
    <tr>
      <th>2021-08-10</th>
      <td>-0.339100</td>
      <td>1.798040</td>
      <td>0.134240</td>
      <td>0.223993</td>
      <td>0.030915</td>
      <td>-0.188567</td>
      <td>0.588190</td>
      <td>0.197861</td>
      <td>0.763478</td>
      <td>-0.224000</td>
      <td>0.330501</td>
      <td>-0.462864</td>
      <td>-0.954318</td>
    </tr>
    <tr>
      <th>2021-08-11</th>
      <td>0.406777</td>
      <td>0.887040</td>
      <td>0.267942</td>
      <td>0.681332</td>
      <td>1.369083</td>
      <td>0.111473</td>
      <td>0.621556</td>
      <td>0.488471</td>
      <td>0.800647</td>
      <td>0.156856</td>
      <td>-0.071511</td>
      <td>-0.061431</td>
      <td>0.586896</td>
    </tr>
    <tr>
      <th>2021-08-12</th>
      <td>-0.203181</td>
      <td>-0.364298</td>
      <td>-0.883091</td>
      <td>-0.086457</td>
      <td>0.024383</td>
      <td>-0.102892</td>
      <td>0.110364</td>
      <td>-0.237216</td>
      <td>-0.257968</td>
      <td>0.141692</td>
      <td>0.164955</td>
      <td>-0.163973</td>
      <td>0.216851</td>
    </tr>
    <tr>
      <th>2021-08-13</th>
      <td>0.372190</td>
      <td>-0.575168</td>
      <td>-0.250992</td>
      <td>0.603633</td>
      <td>1.422417</td>
      <td>0.573115</td>
      <td>0.006134</td>
      <td>-0.913823</td>
      <td>-0.722167</td>
      <td>0.668508</td>
      <td>-0.479139</td>
      <td>1.567248</td>
      <td>0.544757</td>
    </tr>
  </tbody>
</table>
<p>3039 rows × 13 columns</p>
</div>



### 연도별 수익률
연도별로 어떤 종목의 수익률이 가장 좋았는지 확인해볼 수 있습니다. 

작년은 확실히 GOLD의 수익률이 좋았으며 재작년은 우량주모멘텀 주식 수익률이 좋았던 것을 확인할 수 있습니다.


```python
df_yearly = df_daily.asfreq('Y', how='end')
yearly_return = np.log(df_yearly/df_yearly.shift(1))
yearly_return = yearly_return.dropna()
yearly_return*100
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
      <th>BWX</th>
      <th>DBC</th>
      <th>EEM</th>
      <th>EFA</th>
      <th>GLD</th>
      <th>IEF</th>
      <th>IWD</th>
      <th>IWM</th>
      <th>IWN</th>
      <th>LQD</th>
      <th>MTUM</th>
      <th>TLT</th>
      <th>VNQ</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>2014-12-31</th>
      <td>-2.556493</td>
      <td>-32.986895</td>
      <td>-3.976590</td>
      <td>-6.394183</td>
      <td>-2.211671</td>
      <td>8.680215</td>
      <td>12.376047</td>
      <td>4.913623</td>
      <td>4.057205</td>
      <td>7.890334</td>
      <td>13.646184</td>
      <td>24.139139</td>
      <td>26.542870</td>
    </tr>
    <tr>
      <th>2015-12-31</th>
      <td>-6.921233</td>
      <td>-32.279928</td>
      <td>-17.643680</td>
      <td>-0.997918</td>
      <td>-11.284283</td>
      <td>1.496588</td>
      <td>-4.053125</td>
      <td>-4.582335</td>
      <td>-8.049432</td>
      <td>-1.263445</td>
      <td>8.538985</td>
      <td>-1.805808</td>
      <td>2.403367</td>
    </tr>
    <tr>
      <th>2016-12-31</th>
      <td>0.637118</td>
      <td>17.027324</td>
      <td>10.317339</td>
      <td>1.363252</td>
      <td>7.726399</td>
      <td>0.996658</td>
      <td>15.918139</td>
      <td>19.549558</td>
      <td>27.738875</td>
      <td>6.019868</td>
      <td>4.877160</td>
      <td>1.164849</td>
      <td>8.225895</td>
    </tr>
    <tr>
      <th>2017-12-31</th>
      <td>9.471987</td>
      <td>4.746654</td>
      <td>31.672723</td>
      <td>22.374671</td>
      <td>12.052639</td>
      <td>2.520831</td>
      <td>12.617349</td>
      <td>13.612524</td>
      <td>7.409706</td>
      <td>6.817172</td>
      <td>31.842552</td>
      <td>8.785128</td>
      <td>4.782735</td>
    </tr>
    <tr>
      <th>2018-12-31</th>
      <td>-1.770114</td>
      <td>-12.364100</td>
      <td>-16.617622</td>
      <td>-14.874422</td>
      <td>-1.960048</td>
      <td>0.982894</td>
      <td>-8.824836</td>
      <td>-11.789696</td>
      <td>-13.936938</td>
      <td>-3.860748</td>
      <td>-1.687588</td>
      <td>-1.625028</td>
      <td>-6.216202</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>5.433689</td>
      <td>11.189232</td>
      <td>16.740194</td>
      <td>19.920720</td>
      <td>16.429051</td>
      <td>7.723429</td>
      <td>23.208918</td>
      <td>22.625648</td>
      <td>19.891393</td>
      <td>16.014600</td>
      <td>24.100396</td>
      <td>13.206665</td>
      <td>25.395019</td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>9.076849</td>
      <td>-8.161134</td>
      <td>15.717867</td>
      <td>7.320523</td>
      <td>22.165894</td>
      <td>9.537135</td>
      <td>2.696413</td>
      <td>18.257805</td>
      <td>4.552769</td>
      <td>10.409019</td>
      <td>26.130147</td>
      <td>16.680373</td>
      <td>-4.720119</td>
    </tr>
  </tbody>
</table>
</div>



### 월별수익률
그러면 올해 현재(2021-08-15)까지 월별수익률로 1등인 자산은 어떤 것일까요?


정답은 원자재(DBC)입니다. 우리 닥터쿠퍼에 더 투자했어야했네요...


```python
df_monthly = df_daily.asfreq('M', how='end')
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
      <th>BWX</th>
      <th>DBC</th>
      <th>EEM</th>
      <th>EFA</th>
      <th>GLD</th>
      <th>IEF</th>
      <th>IWD</th>
      <th>IWM</th>
      <th>IWN</th>
      <th>LQD</th>
      <th>MTUM</th>
      <th>TLT</th>
      <th>VNQ</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>2013-05-31</th>
      <td>-0.044388</td>
      <td>-0.015730</td>
      <td>-0.049483</td>
      <td>-0.030656</td>
      <td>-0.063992</td>
      <td>-0.031543</td>
      <td>0.026352</td>
      <td>0.038566</td>
      <td>0.029151</td>
      <td>-0.032704</td>
      <td>0.003082</td>
      <td>-0.070003</td>
      <td>-0.061638</td>
    </tr>
    <tr>
      <th>2013-06-30</th>
      <td>-0.017386</td>
      <td>-0.028635</td>
      <td>-0.054728</td>
      <td>-0.027145</td>
      <td>-0.117195</td>
      <td>-0.025791</td>
      <td>-0.009502</td>
      <td>-0.008214</td>
      <td>-0.004878</td>
      <td>-0.033222</td>
      <td>-0.008321</td>
      <td>-0.033260</td>
      <td>-0.020018</td>
    </tr>
    <tr>
      <th>2013-07-31</th>
      <td>0.023666</td>
      <td>0.031338</td>
      <td>0.013159</td>
      <td>0.051860</td>
      <td>0.071670</td>
      <td>-0.003533</td>
      <td>0.052699</td>
      <td>0.070777</td>
      <td>0.063333</td>
      <td>0.011000</td>
      <td>0.058045</td>
      <td>-0.022838</td>
      <td>0.008982</td>
    </tr>
    <tr>
      <th>2013-08-31</th>
      <td>-0.011595</td>
      <td>0.027764</td>
      <td>-0.025705</td>
      <td>-0.019746</td>
      <td>0.050738</td>
      <td>-0.014328</td>
      <td>-0.039007</td>
      <td>-0.032153</td>
      <td>-0.045536</td>
      <td>-0.010033</td>
      <td>-0.036286</td>
      <td>-0.013495</td>
      <td>-0.072356</td>
    </tr>
    <tr>
      <th>2013-09-30</th>
      <td>0.027530</td>
      <td>-0.034341</td>
      <td>0.069589</td>
      <td>0.075338</td>
      <td>-0.049020</td>
      <td>0.018294</td>
      <td>0.025076</td>
      <td>0.062873</td>
      <td>0.056118</td>
      <td>0.007391</td>
      <td>0.029683</td>
      <td>0.006497</td>
      <td>0.034226</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-03-31</th>
      <td>-0.028085</td>
      <td>-0.007199</td>
      <td>-0.007285</td>
      <td>0.024821</td>
      <td>-0.011499</td>
      <td>-0.024146</td>
      <td>0.058297</td>
      <td>0.013854</td>
      <td>0.053169</td>
      <td>-0.014890</td>
      <td>-0.012012</td>
      <td>-0.053878</td>
      <td>0.050237</td>
    </tr>
    <tr>
      <th>2021-04-30</th>
      <td>0.017093</td>
      <td>0.075354</td>
      <td>0.011927</td>
      <td>0.029097</td>
      <td>0.035014</td>
      <td>0.009963</td>
      <td>0.038827</td>
      <td>0.017720</td>
      <td>0.018084</td>
      <td>0.010378</td>
      <td>0.068180</td>
      <td>0.024645</td>
      <td>0.075662</td>
    </tr>
    <tr>
      <th>2021-05-31</th>
      <td>0.009783</td>
      <td>0.037802</td>
      <td>0.016353</td>
      <td>0.034230</td>
      <td>0.073979</td>
      <td>0.004240</td>
      <td>0.022469</td>
      <td>0.002709</td>
      <td>0.031223</td>
      <td>0.006207</td>
      <td>-0.010625</td>
      <td>0.000022</td>
      <td>0.008042</td>
    </tr>
    <tr>
      <th>2021-06-30</th>
      <td>-0.017462</td>
      <td>0.034349</td>
      <td>0.009459</td>
      <td>-0.010839</td>
      <td>-0.074160</td>
      <td>0.010174</td>
      <td>-0.012065</td>
      <td>0.018517</td>
      <td>-0.008302</td>
      <td>0.021873</td>
      <td>0.018424</td>
      <td>0.043242</td>
      <td>0.026033</td>
    </tr>
    <tr>
      <th>2021-07-31</th>
      <td>0.013916</td>
      <td>0.012903</td>
      <td>-0.066535</td>
      <td>0.007704</td>
      <td>0.024983</td>
      <td>0.019707</td>
      <td>0.008475</td>
      <td>-0.036947</td>
      <td>-0.035677</td>
      <td>0.014091</td>
      <td>0.009183</td>
      <td>0.036492</td>
      <td>0.043259</td>
    </tr>
  </tbody>
</table>
<p>99 rows × 13 columns</p>
</div>




```python
recent_monthly_return = monthly_return['2021']
a = recent_monthly_return.mean()
a.sort_index()
a.rank(ascending=False)
```




    BWX     11.0
    DBC      1.0
    EEM      8.0
    EFA      6.0
    GLD     13.0
    IEF     10.0
    IWD      4.0
    IWM      5.0
    IWN      3.0
    LQD      9.0
    MTUM     7.0
    TLT     12.0
    VNQ      2.0
    dtype: float64



### 종목 간 상관관계 분석


```python
corr = recent_monthly_return.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, cmap = 'RdYlBu_r', fmt='.1g')
```




    
![output_15_1](https://user-images.githubusercontent.com/68403764/129559819-a0dabccd-09da-4520-9682-9a74f14fa50b.png)
    


이러한 분석을 통하여 상관관계가 낮은 종목에 투자해서 위험을 줄일 수 있고, 상관관계를 높여 리스크를 더 키울수도 있습니다. 

이건 개인의 성향차이입니다.

여기서는 과거 전통적 자산배분으로 유명한 __채권과 주식에 나눠서 투자할 경우 수익률__ 이 얼마나 좋을지 백테스팅해보겠습니다. 

__(상황에 대한 가정)__ 주식(IWD)에만 모두 넣기에는 불안하여, 

__상관관계가 낮은 채권(IEF)에 각각 60%, 40% 나눠서 투자할 경우__ 채권에만 넣어놓았을 때보다 얼마나 수익률이 좋을까요?

# 백테스팅
여기서는 __정률투자법을 가정__하겠습니다. 즉 여러분이 2가지 자산에 일정비율로 투자했을경우를 가정하고 백테스팅을 진행할 것입니다.

참고로 꼭 정률투자법이 아니더라도 정액투자법 등 내부코드를 바꾸어가면서 여러분의 아이디어를 구현해보세요.

1) 첫 현금 $2000로 시작

2) 이를 채권(IEF) 40%, 주식(IWD) 60% 복리투자

3) 수익률 확인


```python
IEF_IWD = df_monthly[df_monthly.index >= '2011-01-01'][['IEF', 'IWD']]
```

### 결과를 담을 테이블생성
일단 백테스팅을 진행하기 전에 주식에만 투자할 경우와 채권에만 투자할 경우 테이블 먼저 만들어놓겠습니다


```python
IWD_price = IEF_IWD.IWD/IEF_IWD.IWD[0]*2000 #주식투자
IEF_price = IEF_IWD.IEF/IEF_IWD.IEF[0]*2000 #채권투자
result = pd.merge(IWD_price, IEF_price,left_index=True, right_index=True,how='left')
```


```python
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
      <th>IWD</th>
      <th>IEF</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-31</th>
      <td>2000.000000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>2011-02-28</th>
      <td>2075.739753</td>
      <td>1995.831493</td>
    </tr>
    <tr>
      <th>2011-03-31</th>
      <td>2081.620021</td>
      <td>1992.747693</td>
    </tr>
    <tr>
      <th>2011-04-30</th>
      <td>2135.577709</td>
      <td>2029.438993</td>
    </tr>
    <tr>
      <th>2011-05-31</th>
      <td>2114.055343</td>
      <td>2080.287732</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-03-31</th>
      <td>5804.567326</td>
      <td>2921.914856</td>
    </tr>
    <tr>
      <th>2021-04-30</th>
      <td>6034.375223</td>
      <td>2951.170061</td>
    </tr>
    <tr>
      <th>2021-05-31</th>
      <td>6171.494266</td>
      <td>2963.708854</td>
    </tr>
    <tr>
      <th>2021-06-30</th>
      <td>6097.482666</td>
      <td>2994.014306</td>
    </tr>
    <tr>
      <th>2021-07-31</th>
      <td>6149.378008</td>
      <td>3053.602690</td>
    </tr>
  </tbody>
</table>
<p>127 rows × 2 columns</p>
</div>



### Backtest 함수생성


```python
def backtesting(begin_asset = 2000, first_asset_pct = 0.5):
    first_asset_pct = 0.5
    second_asset_pct = 1- first_asset_pct
    portflio = [begin_asset]
    
    for i in range(len(IEF_IWD)-1):
        bond_shares = (begin_asset*first_asset_pct) / IEF_IWD.IEF[i]
        bond_ending = IEF_IWD.IEF[i+1]*bond_shares
        
        stock_shares = (begin_asset*second_asset_pct) / IEF_IWD.IWD[i]
        stock_ending = IEF_IWD.IWD[i+1]*stock_shares      
        
        total = bond_ending + stock_ending
        portflio.append(total)       
        begin_asset = total
    
    return portflio
```


```python
result['backtest'] = backtesting(2000, 0.4)
```


```python
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
      <th>IWD</th>
      <th>IEF</th>
      <th>backtest</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-31</th>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>2011-02-28</th>
      <td>2075.739753</td>
      <td>1995.831493</td>
      <td>2035.785623</td>
    </tr>
    <tr>
      <th>2011-03-31</th>
      <td>2081.620021</td>
      <td>1992.747693</td>
      <td>2037.096398</td>
    </tr>
    <tr>
      <th>2011-04-30</th>
      <td>2135.577709</td>
      <td>2029.438993</td>
      <td>2082.252127</td>
    </tr>
    <tr>
      <th>2011-05-31</th>
      <td>2114.055343</td>
      <td>2080.287732</td>
      <td>2097.845652</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-03-31</th>
      <td>5804.567326</td>
      <td>2921.914856</td>
      <td>4287.568113</td>
    </tr>
    <tr>
      <th>2021-04-30</th>
      <td>6034.375223</td>
      <td>2951.170061</td>
      <td>4393.906694</td>
    </tr>
    <tr>
      <th>2021-05-31</th>
      <td>6171.494266</td>
      <td>2963.708854</td>
      <td>4453.162354</td>
    </tr>
    <tr>
      <th>2021-06-30</th>
      <td>6097.482666</td>
      <td>2994.014306</td>
      <td>4449.228036</td>
    </tr>
    <tr>
      <th>2021-07-31</th>
      <td>6149.378008</td>
      <td>3053.602690</td>
      <td>4512.436996</td>
    </tr>
  </tbody>
</table>
<p>127 rows × 3 columns</p>
</div>



### 시각화를 통한 결과확인
확실히 채권에만 투자하는 것보다는 높은 수익률을 낼 수 있다는 것을 확인할 수 있습니다.


```python
plt.figure(figsize=(15, 4))
plt.plot(result.index, result['backtest'], label='Backtest', c='r')
plt.plot(result.index, result['IWD'], label='IWD', c = 'g')
plt.plot(result.index, result['IEF'], label='IEF', c = 'y')

plt.title("Backtest vs IWD vs IEF", size=18)
plt.legend(loc='best')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Portfolio', fontsize=15) 
plt.grid()
plt.show()
```


    
![output_27_0](https://user-images.githubusercontent.com/68403764/129559829-7d96f3d2-5f32-49c4-8c2b-2f6acf9fcb82.png)

    



```python
# 위험 대비 수익률 확인
(np.mean(result) / np.std(result))*100
```




    IWD         341.134864
    IEF         939.099623
    backtest    488.180618
    dtype: float64



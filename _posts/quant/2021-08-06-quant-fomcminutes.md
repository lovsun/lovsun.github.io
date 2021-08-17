---
layout: post
title:  "[데이터수집] FOMC Minutes 핵심문장 추출"
categories : quant
comments: true
---

**FOMC Statements는 간략**하고 매번 틀이 바뀌지 않아서 읽기 수월합니다. 

하지만 6월 FOMC 회의에 대해 공개된 [Minutes](https://www.federalreserve.gov/monetarypolicy/fomcminutes20210616.htm) 자료를 보면 알 수 있듯이, **미닛츠(Minutes)는 너무 길어서 전체를 읽어보기에는 시간상 어려움**이 있습니다.

그래서 **핵심문장만 수집**하여서 읽어보고 있습니다. 핵심문장만 수집하는 코드를 작성해보겠습니다.


```python
import pandas as pd

from bs4 import BeautifulSoup
import requests

import nltk
```

# URL 확인하고 데이터 수집하기
미닛츠가 공개되면 URL을 확인하고 넣어주면 됩니다. 

html 파일이기 때문에 lxml을 이용하여 파싱을 해주었습니다.


```python
url = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20210616.htm"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')
```

# 필요한 문장만 추출

### Price
일단 ``물가``에 대해 어떻게 생각하는지 문장을 수집해주겠습니다.

soup을 보면 알 수 있듯이 ``<p>태그`` 안에 대부분의 내용이 들어가 있습니다.

하지만 정말 price만 있는 문장만 가져온다면 맥락이 이해가 안되겠죠.

그래서 ``price를 언급한 단락``을 가져와서 읽기 편하게 문장토큰화를 이용하여 문장단위로 나눠주겠습니다.


> 원하는 **단어 선정**

> 단어를 언급한 **단락 수집**

> **문장토큰화**를 이용하여 문장 단위로 구분


```python
analyze_word = "price"

all_sentence = []
for content in soup.find_all('p'):
    content = content.get_text()
    if analyze_word in content:
        sentences = nltk.sent_tokenize(content)
        result = [sentence for sentence in sentences]
        all_sentence.append(result)
```


```python
all_sentence[:5]
```




    [['The manager provided an update on progress toward winding down a number of emergency facilities established under section 13(3) of the Federal Reserve Act.',
      "While market participants took note of the Federal Reserve Board's announcement on winding down the Secondary Market Corporate Credit Facility (SMCCF) holdings, it elicited little price response.",
      'The commencement of exchange-traded funds sales proceeded smoothly.',
      'All of the SMCCF assets are expected to be sold by the end of this year.'],
     ['Staff Review of the Economic Situation\nThe information available at the time of the June 15â\x80\x9316 meeting suggested that U.S. real gross domestic product (GDP) was expanding in the second quarter at a pace that was faster than in the first quarter of the year.',
      'Moreover, labor market conditions had improved further in April and May.',
      'Consumer price inflation through Aprilâ\x80\x94as measured by the 12-month percentage change in the PCE price indexâ\x80\x94had picked up notably, largely reflecting transitory factors.'],
     ['Recent 12-month change measures of inflation, using either PCE prices or the consumer price index (CPI), were boosted significantly by the base effects of the drop in prices from the spring of 2020 rolling out of the calculation.',
      'In addition, a surge in demand as the economy reopened further, combined with production bottlenecks and supply constraints, contributed to the large recent monthly price increases.',
      'Total PCE price inflation was 3.6 percent over the 12 months ending in April.',
      'Core PCE price inflation, which excludes changes in consumer energy prices and many consumer food prices, was 3.1 percent over the 12 months ending in April.',
      'In contrast, the trimmed mean measure of 12-month PCE inflation constructed by the Federal Reserve Bank of Dallas was 1.8 percent in April.',
      'In May, the 12-month change in the CPI was 5 percent, while core CPI inflation was 3.8 percent over the same period.',
      "In the second quarter, the staff's common inflation expectations index, which combines information from many indicators of inflation expectations and inflation compensation, had returned to the level that prevailed in 2014, a time when inflation was modest."],
     ['Housing demand continued to be robust, with construction of single-family homes and home sales remaining well above their pre-pandemic levels and house prices rising appreciably further.',
      'The incoming data for this sector indicated that residential investment spending was being temporarily held back in the second quarter by materials shortages and limited stocks of homes for sale.'],
     ['Available indicators suggested that equipment and intangibles investmentâ\x80\x94particularly in high-tech categoriesâ\x80\x94was increasing solidly in the second quarter.',
      'Rising orders for nondefense capital goods excluding aircraft were running well above the increases in shipments of those goods through April, which pointed to additional gains in business equipment spending in coming months.',
      'Moreover, business investment in the drilling and mining sector appeared to be increasing further, as crude oil and natural gas rigs in operationâ\x80\x94an indicator of drilling investmentâ\x80\x94continued to rise through early June, with oil prices moving higher.',
      'However, nominal nonresidential construction spending declined further in April, and investment in nonresidential structures outside of the drilling and mining sector looked to remain weak in the current quarter, likely reflecting continued hesitation by businesses to commit to building projects with lengthy times to completion and uncertain future returns.']]



그러면 이제부터는 쉽습니다. **price 부분에 여러분들이 원하는 단어만 바꾸어서 넣어주면 됩니다.**

### unemployment


```python
analyze_word = "unemployment"

all_sentence = []
for content in soup.find_all('p'):
    content = content.get_text()
    if analyze_word in content:
        sentences = nltk.sent_tokenize(content)
        result = [sentence for sentence in sentences]
        all_sentence.append(result)
```


```python
all_sentence
```




    [['Total nonfarm payroll employment increased solidly over April and May, though at a slower monthly pace than over February and March.',
      'As of May, total payroll employment had retraced twoâ\x80\x91thirds of the job losses seen at the onset of the pandemic, although employment in the leisure and hospitality sector and in the education sector (including both public and private education) had bounced back by less.',
      'Over April and May, the unemployment rate edged down and stood at 5.8 percent in May.',
      'The unemployment rates for African Americans, Asians, and Hispanics also moved down, although the rates for African Americans and Hispanics remained well above the national average.',
      'Both the labor force participation rate and the employment-to-population ratio moved up slightly, and both measures had recovered only partially from their lows during the pandemic.',
      'Initial claims for regular state unemployment insurance benefits had moved down further since midâ\x80\x91April and were at the lowest level since the beginning of the pandemic, though they remained high relative to their pre-pandemic level.',
      'Weekly estimates of private-sector payrolls constructed by Federal Reserve Board staff using data provided by the payroll processor ADP, which were available through May, suggested that the pace of private employment gains had stepped up late in that month.'],
     ['Staff Economic Outlook\nThe U.S. economic projection prepared by the staff for the June FOMC meeting was stronger than the April forecast.',
      'Real GDP growth was projected to increase substantially this year, with a correspondingly rapid decline in the unemployment rate.',
      'Further reductions in social distancing and favorable financial conditions were expected to support output growth, even though the effects of fiscal stimulus on economic growth were starting to unwind.',
      'With the boost to growth from continued reductions in social distancing assumed to fade after 2021 and the further unwinding of fiscal stimulus, GDP growth was expected to step down in 2022 and 2023.',
      'Nevertheless, with monetary policy assumed to remain highly accommodative, the staff continued to anticipate that real GDP growth would outpace that of potential over most of this period, leading to a decline in the unemployment rate to historically low levels.'],
     ["Participants' Views on Current Economic Conditions and the Economic Outlook\nIn conjunction with this FOMC meeting, participants submitted their projections of the most likely outcomes for real GDP growth, the unemployment rate, and inflation for each year from 2021 through 2023 and over the longer run, based on their individual assessments of appropriate monetary policy, including the path of the federal funds rate.",
      "The longer-run projections represented each participant's assessment of the rate to which each variable would be expected to converge, over time, under appropriate monetary policy and in the absence of further shocks to the economy.",
      'A Summary of Economic Projections was released to the public following the conclusion of the meeting.'],
     ['Participants commented on the continued improvement in labor market conditions in recent months.',
      'Job gains in April and May averaged more than 400,000, and the unemployment rate edged down, on net, to 5.8 percent over the period.',
      'Many participants pointed to the elevated number of job openings and high rates of job switching as further evidence of the improvement in labor market conditions.',
      "Many participants remarked, however, that the economy was still far from achieving the Committee's broad-based and inclusive maximum-employment goal, and some participants indicated that recent job gains, while strong, were weaker than they had expected.",
      'A number of participants noted that the labor market recovery continued to be uneven across demographic and income groups and across sectors.'],
     ['Participants noted that their District contacts had reported having trouble hiring workers to meet demand, likely reflecting factors such as early retirements, concerns about the virus, childcare responsibilities, and expanded unemployment insurance benefits.',
      'Some participants remarked that these factors were making people either less able or less inclined to work in the current environment.',
      'Citing recent wage data and reports from business contacts, many participants judged that labor shortages were putting upward pressure on wages or leading employers to provide additional financial incentives to attract and retain workers, particularly in lower-wage occupations.',
      'Participants expected labor market conditions to continue to improve, with labor shortages expected to ease throughout the summer and into the fall as progress on vaccinations continues, social distancing unwinds further, more schools reopen, and expanded unemployment insurance benefits expire.']]



### interest rates


```python
analyze_word = "interest rates"

all_sentence = []
for content in soup.find_all('p'):
    content = content.get_text()
    if analyze_word in content:
        sentences = nltk.sent_tokenize(content)
        result = [sentence for sentence in sentences]
        all_sentence.append(result)
```


```python
all_sentence
```




    [['Developments in Financial Markets and Open Market Operations\nThe manager turned first to a discussion of financial market developments over the intermeeting period.',
      'On net, U.S. financial conditions eased further, led by a decline in Treasury yields.',
      'Lower term premiums appeared to be a significant component of the declines, as reflected by lower implied volatility on longer-term interest rates.',
      'Equities rose slightly, the broad dollar weakened, and credit spreads were little changed at historically tight levels.'],
     ['Real PCE increased substantially in March and then was little changed from that high level in April.',
      'The components of the nominal retail sales data that are used to estimate PCE edged down in May, but the sales data for the previous two months were revised up markedly, pointing to stronger real PCE growth than had been initially estimated.',
      'Combined with reduced social distancing and more widespread vaccinations, key factors that influence consumer spendingâ\x80\x94including increasing job gains, the upward trend in real disposable income, high levels of household net worth, and low interest ratesâ\x80\x94pointed to strong real PCE growth over the rest of the year.'],
     ['Participants noted that overall financial conditions remained highly accommodative, in part reflecting the stance of monetary policy, which continued to deliver appropriate support to the economy.',
      'Several participants highlighted, however, that low interest rates were contributing to elevated house prices and that valuation pressures in housing markets might pose financial stability risks.'],
     ["Consistent with the Committee's decision to leave the target range for the federal funds rate unchanged, the Board of Governors voted unanimously to raise the interest rates on required and excess reserve balances to 0.15 percent.",
      "Setting the interest rate paid on required and excess reserve balances 15 basis points above the bottom of the target range for the federal funds rate is intended to foster trading in the federal funds market at rates well within the Federal Open Market Committee's target range and to support the smooth functioning of short-term funding markets.",
      'The Board of Governors also voted unanimously to approve establishment of the primary credit rate at the existing level of 0.25 percent, effective June 17, 2021.']]



### 전체 과정에 대한 함수생성
여러분이 주로 확인하는 단어들로 함수를 만들어 놓으면

매번 새로운 미닛츠가 올라올 때마다 url 주소만 바꾸어서 soup 데이터만 바꿔준 후

아래 함수코드를 실행해주면 됩니다.


```python
def summarized_fomc(analyze_word):
    
    summary = {}
    for word in analyze_word:
        
        all_sentence = []
        for content in soup.find_all('p'):
            content = content.get_text()
            if word in content:
                sentences = nltk.sent_tokenize(content)
                result = [sentence for sentence in sentences]
                
                all_sentence.append(result)
        
        summary[word] = all_sentence
                
    return summary
```


```python
################## 사용예시 ##################
data = summarized_fomc(['price', 'unemployment', 'interest rates'])
```


```python
data['price'][:5]
```




    [['The manager provided an update on progress toward winding down a number of emergency facilities established under section 13(3) of the Federal Reserve Act.',
      "While market participants took note of the Federal Reserve Board's announcement on winding down the Secondary Market Corporate Credit Facility (SMCCF) holdings, it elicited little price response.",
      'The commencement of exchange-traded funds sales proceeded smoothly.',
      'All of the SMCCF assets are expected to be sold by the end of this year.'],
     ['Staff Review of the Economic Situation\nThe information available at the time of the June 15â\x80\x9316 meeting suggested that U.S. real gross domestic product (GDP) was expanding in the second quarter at a pace that was faster than in the first quarter of the year.',
      'Moreover, labor market conditions had improved further in April and May.',
      'Consumer price inflation through Aprilâ\x80\x94as measured by the 12-month percentage change in the PCE price indexâ\x80\x94had picked up notably, largely reflecting transitory factors.'],
     ['Recent 12-month change measures of inflation, using either PCE prices or the consumer price index (CPI), were boosted significantly by the base effects of the drop in prices from the spring of 2020 rolling out of the calculation.',
      'In addition, a surge in demand as the economy reopened further, combined with production bottlenecks and supply constraints, contributed to the large recent monthly price increases.',
      'Total PCE price inflation was 3.6 percent over the 12 months ending in April.',
      'Core PCE price inflation, which excludes changes in consumer energy prices and many consumer food prices, was 3.1 percent over the 12 months ending in April.',
      'In contrast, the trimmed mean measure of 12-month PCE inflation constructed by the Federal Reserve Bank of Dallas was 1.8 percent in April.',
      'In May, the 12-month change in the CPI was 5 percent, while core CPI inflation was 3.8 percent over the same period.',
      "In the second quarter, the staff's common inflation expectations index, which combines information from many indicators of inflation expectations and inflation compensation, had returned to the level that prevailed in 2014, a time when inflation was modest."],
     ['Housing demand continued to be robust, with construction of single-family homes and home sales remaining well above their pre-pandemic levels and house prices rising appreciably further.',
      'The incoming data for this sector indicated that residential investment spending was being temporarily held back in the second quarter by materials shortages and limited stocks of homes for sale.'],
     ['Available indicators suggested that equipment and intangibles investmentâ\x80\x94particularly in high-tech categoriesâ\x80\x94was increasing solidly in the second quarter.',
      'Rising orders for nondefense capital goods excluding aircraft were running well above the increases in shipments of those goods through April, which pointed to additional gains in business equipment spending in coming months.',
      'Moreover, business investment in the drilling and mining sector appeared to be increasing further, as crude oil and natural gas rigs in operationâ\x80\x94an indicator of drilling investmentâ\x80\x94continued to rise through early June, with oil prices moving higher.',
      'However, nominal nonresidential construction spending declined further in April, and investment in nonresidential structures outside of the drilling and mining sector looked to remain weak in the current quarter, likely reflecting continued hesitation by businesses to commit to building projects with lengthy times to completion and uncertain future returns.']]




```python
data['unemployment']
```




    [['Total nonfarm payroll employment increased solidly over April and May, though at a slower monthly pace than over February and March.',
      'As of May, total payroll employment had retraced twoâ\x80\x91thirds of the job losses seen at the onset of the pandemic, although employment in the leisure and hospitality sector and in the education sector (including both public and private education) had bounced back by less.',
      'Over April and May, the unemployment rate edged down and stood at 5.8 percent in May.',
      'The unemployment rates for African Americans, Asians, and Hispanics also moved down, although the rates for African Americans and Hispanics remained well above the national average.',
      'Both the labor force participation rate and the employment-to-population ratio moved up slightly, and both measures had recovered only partially from their lows during the pandemic.',
      'Initial claims for regular state unemployment insurance benefits had moved down further since midâ\x80\x91April and were at the lowest level since the beginning of the pandemic, though they remained high relative to their pre-pandemic level.',
      'Weekly estimates of private-sector payrolls constructed by Federal Reserve Board staff using data provided by the payroll processor ADP, which were available through May, suggested that the pace of private employment gains had stepped up late in that month.'],
     ['Staff Economic Outlook\nThe U.S. economic projection prepared by the staff for the June FOMC meeting was stronger than the April forecast.',
      'Real GDP growth was projected to increase substantially this year, with a correspondingly rapid decline in the unemployment rate.',
      'Further reductions in social distancing and favorable financial conditions were expected to support output growth, even though the effects of fiscal stimulus on economic growth were starting to unwind.',
      'With the boost to growth from continued reductions in social distancing assumed to fade after 2021 and the further unwinding of fiscal stimulus, GDP growth was expected to step down in 2022 and 2023.',
      'Nevertheless, with monetary policy assumed to remain highly accommodative, the staff continued to anticipate that real GDP growth would outpace that of potential over most of this period, leading to a decline in the unemployment rate to historically low levels.'],
     ["Participants' Views on Current Economic Conditions and the Economic Outlook\nIn conjunction with this FOMC meeting, participants submitted their projections of the most likely outcomes for real GDP growth, the unemployment rate, and inflation for each year from 2021 through 2023 and over the longer run, based on their individual assessments of appropriate monetary policy, including the path of the federal funds rate.",
      "The longer-run projections represented each participant's assessment of the rate to which each variable would be expected to converge, over time, under appropriate monetary policy and in the absence of further shocks to the economy.",
      'A Summary of Economic Projections was released to the public following the conclusion of the meeting.'],
     ['Participants commented on the continued improvement in labor market conditions in recent months.',
      'Job gains in April and May averaged more than 400,000, and the unemployment rate edged down, on net, to 5.8 percent over the period.',
      'Many participants pointed to the elevated number of job openings and high rates of job switching as further evidence of the improvement in labor market conditions.',
      "Many participants remarked, however, that the economy was still far from achieving the Committee's broad-based and inclusive maximum-employment goal, and some participants indicated that recent job gains, while strong, were weaker than they had expected.",
      'A number of participants noted that the labor market recovery continued to be uneven across demographic and income groups and across sectors.'],
     ['Participants noted that their District contacts had reported having trouble hiring workers to meet demand, likely reflecting factors such as early retirements, concerns about the virus, childcare responsibilities, and expanded unemployment insurance benefits.',
      'Some participants remarked that these factors were making people either less able or less inclined to work in the current environment.',
      'Citing recent wage data and reports from business contacts, many participants judged that labor shortages were putting upward pressure on wages or leading employers to provide additional financial incentives to attract and retain workers, particularly in lower-wage occupations.',
      'Participants expected labor market conditions to continue to improve, with labor shortages expected to ease throughout the summer and into the fall as progress on vaccinations continues, social distancing unwinds further, more schools reopen, and expanded unemployment insurance benefits expire.']]




```python
data['interest rates']
```




    [['Developments in Financial Markets and Open Market Operations\nThe manager turned first to a discussion of financial market developments over the intermeeting period.',
      'On net, U.S. financial conditions eased further, led by a decline in Treasury yields.',
      'Lower term premiums appeared to be a significant component of the declines, as reflected by lower implied volatility on longer-term interest rates.',
      'Equities rose slightly, the broad dollar weakened, and credit spreads were little changed at historically tight levels.'],
     ['Real PCE increased substantially in March and then was little changed from that high level in April.',
      'The components of the nominal retail sales data that are used to estimate PCE edged down in May, but the sales data for the previous two months were revised up markedly, pointing to stronger real PCE growth than had been initially estimated.',
      'Combined with reduced social distancing and more widespread vaccinations, key factors that influence consumer spendingâ\x80\x94including increasing job gains, the upward trend in real disposable income, high levels of household net worth, and low interest ratesâ\x80\x94pointed to strong real PCE growth over the rest of the year.'],
     ['Participants noted that overall financial conditions remained highly accommodative, in part reflecting the stance of monetary policy, which continued to deliver appropriate support to the economy.',
      'Several participants highlighted, however, that low interest rates were contributing to elevated house prices and that valuation pressures in housing markets might pose financial stability risks.'],
     ["Consistent with the Committee's decision to leave the target range for the federal funds rate unchanged, the Board of Governors voted unanimously to raise the interest rates on required and excess reserve balances to 0.15 percent.",
      "Setting the interest rate paid on required and excess reserve balances 15 basis points above the bottom of the target range for the federal funds rate is intended to foster trading in the federal funds market at rates well within the Federal Open Market Committee's target range and to support the smooth functioning of short-term funding markets.",
      'The Board of Governors also voted unanimously to approve establishment of the primary credit rate at the existing level of 0.25 percent, effective June 17, 2021.']]



이렇게 함수로 가져온 결과가 개별적으로 가져온 것과 같음을 확인했습니다.

이제 시간이 부족할 경우, FOMC Minutes의 원하는 부분만 골라서 읽을 수 있습니다. 

### 텍스트파일로 저장
텍스트파일로 저장해놓으면, 길에 다니면서도 좀 더 간편하게 읽어볼 수 있겠죠.


```python
with open('fomc.txt', 'w', encoding='UTF-8') as f:
    for word, contents in data.items():
        f.write(f'{word} : {contents}\n\n')
```

---
layout: post
title:  "[데이터수집] 한국은행 OPEN API 경제데이터 엑셀 자동저장"
categories : quant
comments: true
---

최고의 경제전문가들이 모인 **한국은행에서는 OPEN API 서비스**를 제공하여서 누구나 손쉽게 통계지표를 수집할 수 있도록 해주는데요. 특히 한국은행 100대 통계지표 중 월별로 제공되는 데이터가 많습니다. 하지만 하나하나 들어가서 가져오기에는 번거로운 느낌이었습니다. 

그래서 **수집할 데이터에 대한 통계정보를 CSV파일**에 담아놓고 매달 가져와서 경제데이터를 확인하고 있는데요.

이번 글에서는 그 방법에 대해 공유하겠습니다.


```python
import datetime
import requests
import pandas as pd
```

# 수집할 통계목록 불러오기

일단 사전에 본인이 매달 수집하는 통계지표에 대한 정보를 CSV파일에 담아놓아야하는데요. 
통계지표의 통계표, 주기, 검색시작일자, 통계항목코드를 알고 싶으면 [통계코드검색](http://ecos.bok.or.kr/jsp/openapi/OpenApiController.jsp?t=guideStatCd&menuGroup=MENU000004&menuCode=MENU000024) 을 참고하길 바랍니다.

그래서 **사전에** 아래처럼 CSV 파일에서 불러와서 데이터프레임을 만들어줄 수 있도록 **통계지표 목록**을 만들어 놓아야 합니다.

또한 주의할 점은 dtype을 ``object``로 지정해주지않으면 나중에 url 생성할 때 에러가 발생할 수 있으니까 이 점 유의해주세요.


```python
data_list = pd.read_csv('ECOS_DATA.csv',index_col=0, dtype=object, encoding='CP949')
data_list
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
      <th>통계표코드</th>
      <th>주기</th>
      <th>검색시작일자</th>
      <th>통계항목1코드</th>
      <th>통계항목2코드</th>
      <th>통계항목3코드</th>
    </tr>
    <tr>
      <th>통계명</th>
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
      <th>소비자물가지수(총지수)</th>
      <td>021Y126</td>
      <td>MM</td>
      <td>198508</td>
      <td>00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>소비자물가지수(농산물 및 석유류 제외)</th>
      <td>021Y126</td>
      <td>MM</td>
      <td>198508</td>
      <td>QB</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>생산자제품출하지수(계절조정)</th>
      <td>080Y101</td>
      <td>MM</td>
      <td>198001</td>
      <td>I11A</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>생산자제품재고지수(계절조정)</th>
      <td>080Y101</td>
      <td>MM</td>
      <td>198001</td>
      <td>I11A</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>본원통화(말잔)</th>
      <td>010Y002</td>
      <td>MM</td>
      <td>198601</td>
      <td>AAAA13</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>M2(광의통화, 말잔)</th>
      <td>010Y002</td>
      <td>MM</td>
      <td>198601</td>
      <td>AAAA17</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# OPEN API 이용하기
[한국은행 OPEN API서비스](http://ecos.bok.or.kr/jsp/openapi/OpenApiController.jsp?t=main)에서 사전에 API 키를 먼저 발급받으시길 바랍니다. 받는 방법은 다른 사이트에도 많고 어렵지 않으니까 생략하도록 하겠습니다.


```python
key = 'YOUR API KEY'
```

그러면 데이터를 수집하여 엑셀로 출력해주는 함수를 만들어주기 전에 API가 어떻게 작동하는지 확인해보겠습니다.

한국은행에서 제공하는 총 6가지 개발명세서 중에서 제가 이용할 개발명세서는 [통계조회조건설정](http://ecos.bok.or.kr/jsp/openapi/OpenApiController.jsp?t=guideServiceDtl&apiCode=OA-1040&menuGroup=MENU000004)입니다.

**간단한 예시와 함께 요청을 보내고 응답을 확인해보겠습니다.**

### 요청보내고 응답 확인하기
현재 ``소비자물가지수(농산물 및 석유류 제외)``가 궁금하다고 가정해볼게요. 

아까 위에서 통계코드검색 링크를 따라가면 알 수 있듯이, 소비자물가지수(농산물 및 석유류 제외)의 통계정보는 아래와 같습니다.


```python
STAT_CODE = '021Y126' #통계표코드

Freq = 'MM' #주기(년:YY, 분기:QQ, 월:MM, 일:DD)

Begdate = '198501' #검색시작일자(주기에 맞는 형식으로 입력: 2011, 20111, 201101, 20110101 등)

Enddate = '202108' #검색종료일자(주기에 맞는 형식으로 입력: 2020, 20204, 202012, 20201231 등)

Subcode1 = 'QB'#통계항목1코드
```

requests 라이브러리의 get메소드를 이용하여 요청을 보내고 응답을 확인해보겠습니다.

json 형식의 응답을 확인하여 **하나의 데이터포인트**(``raw['StatisticSearch']['row'][0]``)를 가져와보면,

``TIME``에 날짜에 대한 정보가 ``DATA_VALUE``에 지표에 대한 수치가 나와있는 것을 확인할 수 있습니다.


```python
url = f'http://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/100000/{STAT_CODE}/{Freq}/{Begdate}/{Enddate}/{Subcode1}'
raw = requests.get(url).json()
```


```python
raw['StatisticSearch']['row'][0]
```




    {'UNIT_NAME': '2015=100',
     'STAT_NAME': '7.4.2 소비자물가지수(2015=100)(전국,특수분류)',
     'ITEM_CODE1': 'QB',
     'STAT_CODE': '021Y126',
     'ITEM_CODE2': ' ',
     'ITEM_CODE3': ' ',
     'ITEM_NAME1': '농산물및석유류제외지수',
     'ITEM_NAME2': ' ',
     'DATA_VALUE': '32.735',
     'ITEM_NAME3': ' ',
     'TIME': '198501'}



그리고 날짜와 수치에 대한 정보만 가지고 온다면, **단위가 얼만이고 언제를 기준점**으로 하는지 알 수 없습니다.
이에 대한 정보는 ``UNIT_NAME``에 담겨있는 것을 확인할 수 있습니다.

제가 한국은행 OPEN API 서비스를 이용하면서 보아왔던 통계지표들 중에서

단순 금액데이터이면, UNIT_NAME에 금액단위 (ex, 십억원)
지수데이터이면, 기준시점 (ex, 2015=100)이 나와있었습니다.


```python
raw['StatisticSearch']['row'][0]['UNIT_NAME']
```




    '2015=100'



### 함수로 생성
만들 최종 결과물은 엑셀파일을 하나 생성할 것입니다.
그리고 분석하길 원하는 통계지표에 대한 정보와 날짜별 수치를 시트별로 나눠서 담아줄 것입니다.
그러면 최종적으로 **분석하길 원하는 모든 통계지표가 하나의 엑셀파일에 시트별로 담겨져 있겠죠.**

> 오늘 날짜로 파일명을 생성해주고, enddate를 url 요청으로 보내주기 위해  **날짜변수 생성**

> **엑셀파일을 생성**하고 열어주기

> **통계목록(csv파일)에서 한 행의 정보씩 가져오기**

> 하나의 시트에 **통계정보와 날짜별 수치 정보를 담아주기**

> 이를 **csv 목록 전체에 대해 반복**하기

> 전체 과정이 끝나면 엑셀파일 저장하고 닫기


```python
def ECOSDATA(key, data_list):
    
    # 오늘 연도와 월 string 생성
    today = datetime.datetime.now()
    ym = today.strftime("%Y%m")
    
    # 엑셀파일 생성 및 열기
    writer = pd.ExcelWriter(f'ECOSDATA_{ym}.xlsx', engine='xlsxwriter')
    
    # csv파일 한 행씩 아래의 과정 진행
    for i in data_list.index.tolist():
        
        data = data_list[data_list.index == i]
        
        # url에 넣은 정보생성
        STAT_CODE = data['통계표코드'][0]
        Freq = data['주기'][0]
        Begdate = data['검색시작일자'][0]
        Enddate = ym
        Subcode1 = data['통계항목1코드'][0]
        Subcode2 = data['통계항목2코드'][0]
        Subcode3 = data['통계항목3코드'][0]
        
        # open api 서비스 url요청
        url = f'http://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/100000/{STAT_CODE}/{Freq}/{Begdate}/{Enddate}/{Subcode1}/{Subcode2}/{Subcode3}'
        raw = requests.get(url).json()
        
        # 응답을 데이터프레임 형태로 만들어서 엑셀 파일에 저장
        try:
            info = pd.DataFrame(columns=['STAT_NAME', 'ITEM_NAME1', 'UNIT_NAME'])
            info.loc[0] = [raw['StatisticSearch']['row'][0]['STAT_NAME'],
                           raw['StatisticSearch']['row'][0]['ITEM_NAME1'],
                           raw['StatisticSearch']['row'][0]['UNIT_NAME']]
            info.to_excel(writer, sheet_name=i, index=False)

            df = pd.DataFrame(raw['StatisticSearch']['row'])[['TIME', 'DATA_VALUE']]
            df.to_excel(writer, sheet_name=i, startrow=4, index=False)   

        except Exception as e:
            print(i, e)
            pass

    writer.save()  
    print("Completed")
```


```python
############# 사용예시 #############
ECOSDATA(key, data_list)
```

    Completed
    

전체 과정이 끝나면 여러분의 working directory에 2021년 8월일 경우, ``ECOSDATA_202108.xlsx`` 파일이 생성된 것을 확인할 수 있습니다.

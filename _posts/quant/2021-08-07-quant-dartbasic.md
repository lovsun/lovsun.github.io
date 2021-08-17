---
layout: post
title:  "[데이터수집] DART API 고유번호 및 접수번호 수집하기"
categories : quant
comments: true
---

[DART고유번호개발가이드](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019018)

이번 게시물에서는 DART API를 이용하여 기업을 분석하기에 앞서 **기업의 고유번호를 알아내보도록 하겠습니다.** 이것을 알아야하는 이유는 DART 통해 기업 재무정보를 받아오고 싶으면 **기업 고유번호를 알아야합니다.** 

여기서 주의할 것은 종목코드와 기업 고유번호는 다르다는 것입니다. 

- **종목코드** : 6자리로서 네이버금융 같은 사이트에 기업을 입력하면 바로 확인가능한 상장사의 식별코드입니다. 기업의 주민등록번호 같은 것입니다.
- **고유번호** : 이것은 8자리로서, 다트에서 기업정보를 알아내기위해 알아야하는데요. 다트에서는 상장기업 뿐 아니라 일정요건을 충족하는 비상장기업도 공시의무가 있기 때문에, 공시의무가 있는 모든 기업이 가지고 있는 각자의 번호입니다.


```python
import requests, zipfile, io 
from bs4 import BeautifulSoup 
```

# 고유번호 정보

DART API 또한 이용하기에 앞서 API KEY를 받아야 합니다. 

키만 받으면 바로 url요청으로 고유번호 정보가 담긴 파일을 받아올 수 있습니다. 위의 고유번호 개발가이드를 가보면 알 수 있듯이 **출력포맷이 Zip FILE (binary)** 이기 때문에,

저희는 **zipfile, io 라이브러리**를 이용할 것입니다.
그래서 zipfile의 압축을 풀어서 data폴더에 저장한 xml 파일만 읽어오도록 하겠습니다.


```python
key = "YOUR API KEY"
```


```python
url = f"http://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={key}"
res_url = requests.get(url)


z = zipfile.ZipFile(io.BytesIO(res_url.content))
z.extractall("./data")

xml = "./data/CORPCODE.xml"
read_xml = open(xml, "r", encoding='utf-8').read()
soup = BeautifulSoup(read_xml, 'lxml')
```

### 기업명으로 고유번호 얻기
xml을 파싱하여 기업에 대한 정보를 얻을 수 있습니다.

가령 아래처럼 **삼성전자를 포함하고 있는 기업명과 기업고유번호**를 파악할 수 있습니다.


```python
items = soup.find_all('list')
for item in items: 
    #기업명 이용
    if '삼성전자' in item.find('corp_name').text:
        company_number = item.find('corp_code').text
        company_name = item.find('corp_name').text
        print(f'company name: {company_name} \ncompany number: {company_number}' )
```

    company name: 삼성전자서비스씨에스 
    company number: 01345812
    company name: 삼성전자서비스 
    company number: 00258999
    company name: 삼성전자판매 
    company number: 00252074
    company name: 삼성전자로지텍 
    company number: 00366997
    company name: 삼성전자 
    company number: 00126380
    

### 종목코드로 고유번호 얻기
구체적인 종목코드를 넣어줘서 기업 고유번호를 얻을 수 있습니다. 

**종목코드는 모든 상장기업이 다르기 때문에, 정확한 고유번호를 빨리 파악할 수 있습니다.** 

하지만 단점으로는 비상장기업은 상장종목코드가 없기 때문에 해당 방법으로는 고유번호를 알 수 없습니다.


```python
items = soup.find_all('list')   
for item in items:
    stockcode = item.find('stock_code').text
    #상장종목코드이용
    if stockcode == "035720":
        company_number = item.find('corp_code').text
        company_name = item.find('corp_name').text
        print(f'company name: {company_name} \ncompany number: {company_number}' )
```

    company name: 카카오 
    company number: 00258801
    

### 함수로 생성
위의 전과정을 함수로 생성해주겠습니다. 저는 주로 상장종목 위주로 분석을 합니다.

그래서 **종목코드를 넣을 경우, 기업명과 기업고유번호를 알려주는 함수**를 만들겠습니다.

어떤 방식으로 만들든 코드를 수정하여 여러분이 편한 방식을 선택하면 됩니다.


```python
def companycode(key, inputcode):
    url = f"http://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={key}"
    res_url = requests.get(url)
    
    z = zipfile.ZipFile(io.BytesIO(res_url.content))
    z.extractall("./data")

    xml = "./data/CORPCODE.xml"
    read_xml = open(xml, "r", encoding='utf-8').read()
    soup = BeautifulSoup(read_xml, 'lxml')

    items = soup.find_all('list')   
    for item in items:
        stockcode = item.find('stock_code').text
        if stockcode == str(inputcode):
            company_number = item.find('corp_code').text
            company_name = item.find('corp_name').text
            print(f'company name: {company_name} \ncompany number: {company_number}' )

    return company_name, company_number
```


```python
############ 사용예시 ############ 
companycode(key, '035720')
```

    company name: 카카오 
    company number: 00258801
    




    ('카카오', '00258801')



# 접수번호 정보
이번 게시물에서는 Dart의 공시검색개발가이드를 이용하여서, 기업의 접수번호를 알아내볼 것입니다. 향후 Dart API를 이용하여 기업 [재무제표](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019019) 개발가이드를 이용하기 위해서는 **접수번호(rcept_no)를 알고 있어야 합니다.** 

이를 알아낼 방법을 찾다가 [공시검색 개발가이드](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019001)를 이용하면 **기업의 최근 공시자료**에 대한 정보를 주기되고, 응답결과로 **접수번호(rcept_no)도 포함**되어 있다는 것을 알게 되었습니다.


```python
import requests 
import pandas as pd 
import datetime  
```

### 요청보내고 응답 확인하기
현재 ``카카오`` 의 공시자료가 궁금하다고 가정해볼게요. 

위의 기업고유번호 함수를 통해 카카오의 고유번호를 알아냈습니다. 

그러면 **카카오가 가장 최근에 공시한 내용 중 정기공시(A)에 대해 알아오는 함수**를 만들어주겠습니다.

참고로 여러분이 발행공시, 지분공시, 외부감사관련 공시 등 다른 내용이 궁금하다면

개발가이드 요청인자의 공시유형을 참고하여 ``pblntf_ty``의 ``value 값``을 변경하여 주면 됩니다. 현재 저는 정기공시(사업보고서, 반기보고서, 분기보고서 등)가 궁금하다고 가정하였기 때문에, A여서 ``pblntf_ty=A``로 고정해두겠습니다.


```python
corp_number = "00258801" #고유번호

now = datetime.datetime.now() 
search_period = datetime.timedelta(days=365*5)

begin_date = (now - search_period).strftime('%Y%m%d') #시작일
end_date = now.strftime('%Y%m%d') #종료일
page = 20

r = requests.get(f'https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&corp_code={corp_number}&bgn_de={begin_date}&end_de={end_date}&pblntf_ty=A&page_count={page}')
report = r.json()['list']
report
```




    [{'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '분기보고서 (2021.03)',
      'rcept_no': '20210517002106',
      'flr_nm': '카카오',
      'rcept_dt': '20210517',
      'rm': ''},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '사업보고서 (2020.12)',
      'rcept_no': '20210318001373',
      'flr_nm': '카카오',
      'rcept_dt': '20210318',
      'rm': '연'},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '분기보고서 (2020.09)',
      'rcept_no': '20201116001873',
      'flr_nm': '카카오',
      'rcept_dt': '20201116',
      'rm': ''},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '반기보고서 (2020.06)',
      'rcept_no': '20200814002188',
      'flr_nm': '카카오',
      'rcept_dt': '20200814',
      'rm': ''},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '분기보고서 (2020.03)',
      'rcept_no': '20200515002518',
      'flr_nm': '카카오',
      'rcept_dt': '20200515',
      'rm': ''},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '사업보고서 (2019.12)',
      'rcept_no': '20200330004659',
      'flr_nm': '카카오',
      'rcept_dt': '20200330',
      'rm': '연'},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '분기보고서 (2019.09)',
      'rcept_no': '20191114002583',
      'flr_nm': '카카오',
      'rcept_dt': '20191114',
      'rm': ''},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '반기보고서 (2019.06)',
      'rcept_no': '20190814002715',
      'flr_nm': '카카오',
      'rcept_dt': '20190814',
      'rm': ''},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '분기보고서 (2019.03)',
      'rcept_no': '20190515002549',
      'flr_nm': '카카오',
      'rcept_dt': '20190515',
      'rm': ''},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '사업보고서 (2018.12)',
      'rcept_no': '20190401005013',
      'flr_nm': '카카오',
      'rcept_dt': '20190401',
      'rm': '연'},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '[기재정정]분기보고서 (2018.09)',
      'rcept_no': '20181204000182',
      'flr_nm': '카카오',
      'rcept_dt': '20181204',
      'rm': ''},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '[기재정정]반기보고서 (2018.06)',
      'rcept_no': '20181204000180',
      'flr_nm': '카카오',
      'rcept_dt': '20181204',
      'rm': ''},
     {'corp_code': '00258801',
      'corp_name': '카카오',
      'stock_code': '035720',
      'corp_cls': 'Y',
      'report_nm': '분기보고서 (2018.09)',
      'rcept_no': '20181114002663',
      'flr_nm': '카카오',
      'rcept_dt': '20181114',
      'rm': '정'}]



### 필요한 정보만 추출
데이터포인트가 딕셔너리 형태로 들어가 있는데, 원하는 정보만 추출해줘야겠죠.

저희는 향후 재무제표를 분석할 것이기 때문에, 보고서명('report_nm')별로 접수코드('rcept_no')를 파악해보겠습니다.

> **보고서명별로 접수코드 데이터프레임** 생성

> 접수코드를 이용하여 **보고서 코드 추가** _이건 꼭 필요한 작업은 아니지만 보고서코드를 알면 3, 6, 9, 12월 중 언제의 보고서인지 알 수 있어서 추가해주었습니다._


```python
#보고서명과 접수코드 딕셔너리 형태로 담아주기
overview = {}
for i in range(len(report)):
    overview[report[i]['report_nm']] = report[i]['rcept_no']
overview

#데이터프레임 형태로 바꾼 후 보고서코드 추가하기
recept = pd.DataFrame.from_dict(overview, orient='index', columns=['Recept_No'])
recept['Report_Code'] = ""
for i in range(len(recept.index)):
    if '03' in recept.index[i]:
        recept['Report_Code'][i] = '11013'
    if '06' in recept.index[i]:
        recept['Report_Code'][i] = '11012'
    if '09' in recept.index[i]:
        recept['Report_Code'][i] = '11014'
    if '12' in recept.index[i]:
        recept['Report_Code'][i]= '11011'
```


```python
recept
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
      <th>Recept_No</th>
      <th>Report_Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>분기보고서 (2021.03)</th>
      <td>20210517002106</td>
      <td>11013</td>
    </tr>
    <tr>
      <th>사업보고서 (2020.12)</th>
      <td>20210318001373</td>
      <td>11011</td>
    </tr>
    <tr>
      <th>분기보고서 (2020.09)</th>
      <td>20201116001873</td>
      <td>11014</td>
    </tr>
    <tr>
      <th>반기보고서 (2020.06)</th>
      <td>20200814002188</td>
      <td>11012</td>
    </tr>
    <tr>
      <th>분기보고서 (2020.03)</th>
      <td>20200515002518</td>
      <td>11013</td>
    </tr>
    <tr>
      <th>사업보고서 (2019.12)</th>
      <td>20200330004659</td>
      <td>11011</td>
    </tr>
    <tr>
      <th>분기보고서 (2019.09)</th>
      <td>20191114002583</td>
      <td>11014</td>
    </tr>
    <tr>
      <th>반기보고서 (2019.06)</th>
      <td>20190814002715</td>
      <td>11012</td>
    </tr>
    <tr>
      <th>분기보고서 (2019.03)</th>
      <td>20190515002549</td>
      <td>11013</td>
    </tr>
    <tr>
      <th>사업보고서 (2018.12)</th>
      <td>20190401005013</td>
      <td>11011</td>
    </tr>
    <tr>
      <th>[기재정정]분기보고서 (2018.09)</th>
      <td>20181204000182</td>
      <td>11014</td>
    </tr>
    <tr>
      <th>[기재정정]반기보고서 (2018.06)</th>
      <td>20181204000180</td>
      <td>11012</td>
    </tr>
    <tr>
      <th>분기보고서 (2018.09)</th>
      <td>20181114002663</td>
      <td>11014</td>
    </tr>
    <tr>
      <th>반기보고서 (2018.06)</th>
      <td>20180814002960</td>
      <td>11012</td>
    </tr>
    <tr>
      <th>분기보고서 (2018.03)</th>
      <td>20180515002511</td>
      <td>11013</td>
    </tr>
    <tr>
      <th>사업보고서 (2017.12)</th>
      <td>20180330002462</td>
      <td>11011</td>
    </tr>
    <tr>
      <th>분기보고서 (2017.09)</th>
      <td>20171114002088</td>
      <td>11014</td>
    </tr>
    <tr>
      <th>반기보고서 (2017.06)</th>
      <td>20170814002090</td>
      <td>11012</td>
    </tr>
    <tr>
      <th>분기보고서 (2017.03)</th>
      <td>20170515003759</td>
      <td>11013</td>
    </tr>
    <tr>
      <th>사업보고서 (2016.12)</th>
      <td>20170331005003</td>
      <td>11011</td>
    </tr>
  </tbody>
</table>
</div>



### 함수로 생성
위의 전과정을 함수로 생성해주고 사용예시를 함께 보여주겠습니다.


```python
def recept_overview(key, corp_number):
    #요청인자
    now = datetime.datetime.now()
    search_period = datetime.timedelta(days=365*5)

    begin_date = (now - search_period).strftime('%Y%m%d')
    end_date = now.strftime('%Y%m%d')
    page = 20

    #웹에 데이터 요청하기 (url완성)
    r = requests.get(f'https://opendart.fss.or.kr/api/list.json?crtfc_key={key}&corp_code={corp_number}&bgn_de={begin_date}&end_de={end_date}&pblntf_ty=A&page_count={page}')
    report = r.json()['list']

    #필요한 정보만 골라내기(접수코드)
    overview = {}
    for i in range(len(report)):
        overview[report[i]['report_nm']] = report[i]['rcept_no']
    overview

    #최근 5개년 정기공시에 대한 접수코드와 보고서코드 테이블 생성
    recept = pd.DataFrame.from_dict(overview, orient='index', columns=['Recept_No'])
    recept['Report_Code'] = ""
    for i in range(len(recept.index)):
        if '03' in recept.index[i]:
            recept['Report_Code'][i] = '11013'
        if '06' in recept.index[i]:
            recept['Report_Code'][i] = '11012'
        if '09' in recept.index[i]:
            recept['Report_Code'][i] = '11014'
        if '12' in recept.index[i]:
            recept['Report_Code'][i]= '11011'

    return recept
```


```python
############ 사용예시 ############ 
recept_overview(key, "00258801")
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
      <th>Recept_No</th>
      <th>Report_Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>분기보고서 (2021.03)</th>
      <td>20210517002106</td>
      <td>11013</td>
    </tr>
    <tr>
      <th>사업보고서 (2020.12)</th>
      <td>20210318001373</td>
      <td>11011</td>
    </tr>
    <tr>
      <th>분기보고서 (2020.09)</th>
      <td>20201116001873</td>
      <td>11014</td>
    </tr>
    <tr>
      <th>반기보고서 (2020.06)</th>
      <td>20200814002188</td>
      <td>11012</td>
    </tr>
    <tr>
      <th>분기보고서 (2020.03)</th>
      <td>20200515002518</td>
      <td>11013</td>
    </tr>
    <tr>
      <th>사업보고서 (2019.12)</th>
      <td>20200330004659</td>
      <td>11011</td>
    </tr>
    <tr>
      <th>분기보고서 (2019.09)</th>
      <td>20191114002583</td>
      <td>11014</td>
    </tr>
    <tr>
      <th>반기보고서 (2019.06)</th>
      <td>20190814002715</td>
      <td>11012</td>
    </tr>
    <tr>
      <th>분기보고서 (2019.03)</th>
      <td>20190515002549</td>
      <td>11013</td>
    </tr>
    <tr>
      <th>사업보고서 (2018.12)</th>
      <td>20190401005013</td>
      <td>11011</td>
    </tr>
    <tr>
      <th>[기재정정]분기보고서 (2018.09)</th>
      <td>20181204000182</td>
      <td>11014</td>
    </tr>
    <tr>
      <th>[기재정정]반기보고서 (2018.06)</th>
      <td>20181204000180</td>
      <td>11012</td>
    </tr>
    <tr>
      <th>분기보고서 (2018.09)</th>
      <td>20181114002663</td>
      <td>11014</td>
    </tr>
    <tr>
      <th>반기보고서 (2018.06)</th>
      <td>20180814002960</td>
      <td>11012</td>
    </tr>
    <tr>
      <th>분기보고서 (2018.03)</th>
      <td>20180515002511</td>
      <td>11013</td>
    </tr>
    <tr>
      <th>사업보고서 (2017.12)</th>
      <td>20180330002462</td>
      <td>11011</td>
    </tr>
    <tr>
      <th>분기보고서 (2017.09)</th>
      <td>20171114002088</td>
      <td>11014</td>
    </tr>
    <tr>
      <th>반기보고서 (2017.06)</th>
      <td>20170814002090</td>
      <td>11012</td>
    </tr>
    <tr>
      <th>분기보고서 (2017.03)</th>
      <td>20170515003759</td>
      <td>11013</td>
    </tr>
    <tr>
      <th>사업보고서 (2016.12)</th>
      <td>20170331005003</td>
      <td>11011</td>
    </tr>
  </tbody>
</table>
</div>



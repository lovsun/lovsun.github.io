---
layout: post
title:  "[데이터수집] DART API 3개년 재무제표 자동 수집 및 저장"
categories : quant
comments: true
---

이번 게시물에서는 Dart의 [단일회사 전체 재무제표 개발가이드](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019020)를 이용하여, 기업의 재무제표를 가져오겠습니다. 

사업보고서를 기준으로, **가장 최근 사업보고서 3개년 사업보고서 데이터**를 불러올 것입니다. 
이렇게 불러와서 엑셀로 저장까지 해준다면 조금 더 손쉽게 기업을 분석할 수 있겠죠.

참고로 이전 글에 이어서 작성하는 것이기 때문에, 

기업고유번호 및 접수번호 글을 아직 따라하지 않으신 분들은 이 글 먼저 따라하셔야 아래의 코드를 실행할 수 있습니다.

# 단일회사재무제표 개발가이드

### 요청보내고 응답받아보기

개발가이드의 요청인자에 따라서 요청을 보내면 
json 응답에서 list 키 값 안에 **계정과목별로 수치데이터**가 담겨있는 것을 확인할 수 있습니다.

요청을 보내는 **파라미터**는 아래와 같습니다.

> ``crtfc_key`` = 개인 API키

> ``corp_code`` = 기업고유번호 (이전 글에 기업고유번호 알아내는 방법을 올려놓았습니다.)

> ``bsns_year`` = 사업연도 (저는 이전 글에서 만든 접수코드를 이용하여 공시된 가장 최근 사업보고서 연도를 추출하겠습니다. 이렇게 하면 매년 연도를 바꿔줄 필요가 없겠죠)

> ``reprt_code`` = 보고서코드(3, 6, 9, 12월 보고서 중 본인이 분석하길 원하는 보고서 선택)

> ``fs_div`` = 연결/개별 재무제표 선택


```python
corp_code = '00258801' #기업고유번호

year_fs = recept[recept['Report_Code'] == '11011'].index[0][-8:-4] #bsns_year, 가장 최근 사업보고서 년도추출

reprt_code = '11011' #보고서 코드

fs_div = 'CFS' #CFS:연결재무제표, OFS:재무제표

#데이터 받아오기
Year_FS = requests.get(f"https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json?crtfc_key={key}&corp_code={corp_code}&bsns_year={year_fs}&reprt_code={reprt_code}&fs_div={fs_div}")
Year_FS = Year_FS.json()
Year_FS = Year_FS['list']
Year_FS
```




    [{'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_CurrentAssets',
      'account_nm': '유동자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '4462924201049',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '2829694454045',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '2858950414592',
      'ord': '1'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_CashAndCashEquivalents',
      'account_nm': '현금및현금성자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '2877513939692',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '1918225198949',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '1247013249518',
      'ord': '2'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'dart_ShortTermDepositsNotClassifiedAsCashEquivalents',
      'account_nm': '단기금융상품',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '694068762001',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '207766855476',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '924215115420',
      'ord': '3'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_CurrentFinancialAssetsAtFairValueThroughProfitOrLossDesignatedUponInitialRecognition',
      'account_nm': '유동 당기손익-공정가치 측정 지정 금융자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '134502747309',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '49512474331',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '25780947342',
      'ord': '4'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'dart_CurrentDerivativeAsset',
      'account_nm': '파생상품자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '311605977',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '1111591977',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '1174289977',
      'ord': '5'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'dart_ShortTermTradeReceivable',
      'account_nm': '매출채권',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '247374452328',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '157220905352',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '193669497516',
      'ord': '6'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_OtherCurrentFinancialAssets',
      'account_nm': '기타유동금융자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '225055175264',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '252486445359',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '207655813171',
      'ord': '7'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_Inventories',
      'account_nm': '재고자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '45813596549',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '49449770906',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '35836145752',
      'ord': '8'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_NoncurrentAssetsOrDisposalGroupsClassifiedAsHeldForSaleOrAsHeldForDistributionToOwners',
      'account_nm': '매각예정자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '5748919485',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '',
      'ord': '9'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'dart_OtherCurrentAssets',
      'account_nm': '기타유동자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '232535002444',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '193921211695',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '223605355896',
      'ord': '10'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_NoncurrentAssets',
      'account_nm': '비유동자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '6987396305870',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '5907561301729',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '5100591690675',
      'ord': '11'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'dart_LongTermDepositsNotClassifiedAsCashEquivalents',
      'account_nm': '장기금융상품',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '10173823192',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '50061893487',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '50688543876',
      'ord': '12'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_NoncurrentFinancialAssetsAtFairValueThroughProfitOrLossDesignatedUponInitialRecognition',
      'account_nm': '당기손익-공정가치 측정 금융자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '303221657755',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '127148333704',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '104846044343',
      'ord': '13'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_NoncurrentFinancialAssetsMeasuredAtFairValueThroughOtherComprehensiveIncome',
      'account_nm': '기타포괄손익-공정가치 측정 금융자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '702575375014',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '419265484312',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '329344718026',
      'ord': '14'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_InvestmentsInAssociates',
      'account_nm': '관계기업 및 공동기업 투자',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '1504418504255',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '978943948141',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '305187605646',
      'ord': '15'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_OtherNoncurrentFinancialAssets',
      'account_nm': '기타비유동금융자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '121290880174',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '92354874203',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '58320579268',
      'ord': '16'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_PropertyPlantAndEquipment',
      'account_nm': '유형자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '430667589746',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '349818319321',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '314811843577',
      'ord': '17'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'dart_OtherIntangibleAssetsGross',
      'account_nm': '무형자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '3351553299735',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '3548415767971',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '3865264987475',
      'ord': '18'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_InvestmentProperty',
      'account_nm': '투자부동산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '0',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '2843015277',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '0',
      'ord': '19'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_RightofuseAssets',
      'account_nm': '사용권자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '345324766938',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '227458396560',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '0',
      'ord': '20'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'dart_OtherNonCurrentAssets',
      'account_nm': '기타비유동자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '166486922465',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '50547988423',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '16411840218',
      'ord': '21'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': 'ifrs-full_DeferredTaxAssets',
      'account_nm': '이연법인세자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '51683486596',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '60703280330',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '55715528246',
      'ord': '22'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': '-표준계정코드 미사용-',
      'account_nm': '금융업자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '503649732037',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '',
      'ord': '23'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': '-표준계정코드 미사용-',
      'account_nm': '현금및현금성자산',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '16802652216',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '',
      'ord': '24'},
     {'rcept_no': '20210318001373',
      'reprt_code': '11011',
      'bsns_year': '2020',
      'corp_code': '00258801',
      'sj_div': 'BS',
      'sj_nm': '재무상태표',
      'account_id': '-표준계정코드 미사용-',
      'account_nm': '예치금',
      'account_detail': '-',
      'thstrm_nm': '제 26 기',
      'thstrm_amount': '19829317800',
      'frmtrm_nm': '제 25 기',
      'frmtrm_amount': '',
      'bfefrmtrm_nm': '제 24 기',
      'bfefrmtrm_amount': '',
      'ord': '25'}]



### 필요한 정보 추출
하나의 **데이터포인트(딕셔너리)** 에서 저희가 필요한 정보는

> (sj_div) **어떤 재무제표**의 : BS(재무상태표) / CIS(포괄손익계산서) / CF(현금흐름표) 

> (account_nm) **어떤 계정과목**인가 

> (thstrm_amount, frmtrm_amount, bfefrmtrm_amount) 올해, 작년, 재작년의 **수치는 얼마인가**

기업을 분석하기 위해서는 이러한 정보가 필요하겠죠. 그래서 필요한 정보만 수집해주도록 하겠습니다.


```python
Year_FS = pd.DataFrame.from_dict(Year_FS)
Year_FS = Year_FS.loc[:,['sj_div', 'account_nm', 'thstrm_amount', 'frmtrm_amount', 'bfefrmtrm_amount']]
Year_FS = Year_FS[Year_FS['sj_div'].isin(['BS', 'CIS' , 'CF'])]
```


```python
Year_FS
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
      <th>sj_div</th>
      <th>account_nm</th>
      <th>thstrm_amount</th>
      <th>frmtrm_amount</th>
      <th>bfefrmtrm_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BS</td>
      <td>유동자산</td>
      <td>4462924201049</td>
      <td>2829694454045</td>
      <td>2858950414592</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BS</td>
      <td>현금및현금성자산</td>
      <td>2877513939692</td>
      <td>1918225198949</td>
      <td>1247013249518</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BS</td>
      <td>단기금융상품</td>
      <td>694068762001</td>
      <td>207766855476</td>
      <td>924215115420</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BS</td>
      <td>유동 당기손익-공정가치 측정 지정 금융자산</td>
      <td>134502747309</td>
      <td>49512474331</td>
      <td>25780947342</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BS</td>
      <td>파생상품자산</td>
      <td>311605977</td>
      <td>1111591977</td>
      <td>1174289977</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>141</th>
      <td>CF</td>
      <td>기타재무활동으로 인한 현금 유출액</td>
      <td>-8850578822</td>
      <td>528734364</td>
      <td>-23996900</td>
    </tr>
    <tr>
      <th>142</th>
      <td>CF</td>
      <td>현금및현금성자산에 대한 환율변동효과</td>
      <td>-39669604010</td>
      <td>10527667973</td>
      <td>8955237814</td>
    </tr>
    <tr>
      <th>143</th>
      <td>CF</td>
      <td>현금및현금성자산의순증가(감소)</td>
      <td>976091392959</td>
      <td>671211949431</td>
      <td>130220177009</td>
    </tr>
    <tr>
      <th>144</th>
      <td>CF</td>
      <td>기초현금및현금성자산</td>
      <td>1918225198949</td>
      <td>1247013249518</td>
      <td>1116793072509</td>
    </tr>
    <tr>
      <th>145</th>
      <td>CF</td>
      <td>기말현금및현금성자산</td>
      <td>2894316591908</td>
      <td>1918225198949</td>
      <td>1247013249518</td>
    </tr>
  </tbody>
</table>
<p>146 rows × 5 columns</p>
</div>



### 재무제표 저장
이렇게 **재무상태표, 포괄손익계산서, 현금흐름표를 수집**하였습니다.

하지만 컬럼제목까지 사업연도로 바꿔주면 조금 더 보기 편한 재무제표가 될 것 같네요.

**컬럼제목을 각 사업연도로 바꿔준 이후**에 재무제표를 csv 파일로 저장해주겠습니다.

저장까지하고 나면 여러분의 wording directory에 '카카오_최근재무제표.csv' **파일이 생성**된 것을 확인할 수 있습니다.


```python
Year_FS.columns = ['fs', 'account', str(int(year_fs)-2), str(int(year_fs)-1), year_fs]
```


```python
Year_FS.to_csv('카카오_최근재무제표.csv', index=False)
```

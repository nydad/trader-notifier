# KOSPI 단타 자동분석 시스템 - 실시간 데이터 소스 종합 리서치

> 조사일: 2026-03-11
> 조사 방법: Claude Opus 4.6 + GPT Codex Spark 5-Agent Panel 병렬 리서치

---

## 1. 실시간 KOSPI 지수 및 가격 데이터

### 1.1 KRX (한국거래소) 공식 데이터 API

#### KRX Open API (openapi.krx.co.kr)
- URL: https://openapi.krx.co.kr
- 데이터 엔드포인트: https://data-dbg.krx.co.kr/svc/apis/
- 주요 API 경로:
  - svc/apis/sto/stk_bydd_trd : KOSPI 일별매매정보
  - svc/apis/sto/ksq_bydd_trd : KOSDAQ 일별매매정보
  - svc/apis/idx/kospi_dd_trd : KOSPI 지수 일별
  - svc/apis/sto/stk_isu_base_info : 종목기본정보
  - svc/apis/drv/fut_bydd_trd : 선물 일별매매정보
- 가격: 회원가입 + API 서비스 신청/승인 필요
- 업데이트 주기: 일별 데이터 중심 (실시간 아님!)
- 포맷: JSON/XML
- 인증: AUTH_KEY 헤더
- 한계: 실시간 시세 미제공, T+1 기준

#### KRX 정보데이터시스템 (data.krx.co.kr) 스크래핑 방식
- URL: https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd
- 방식: OTP 토큰 생성 후 데이터 요청 (2-step)
- 주요 bld 코드:
  - dbms/MDC/STAT/standard/MDCSTAT01501 : 투자자별 매매동향
  - dbms/MDC/STAT/standard/MDCSTAT03901 : 프로그램 매매
  - dbms/MDC/STAT/standard/MDCSTAT04001 : 파생상품 투자자별
- 가격: 무료 (웹 스크래핑)
- 업데이트 주기: 장중 준실시간 (수분 간격)
- 포맷: JSON

### 1.2 Naver Finance (비공식 API)

#### 실시간 Polling API
- URL: https://polling.finance.naver.com/api/realtime
- 파라미터: query=SERVICE_ITEM:{종목코드} 또는 query=SERVICE_INDEX:KOSPI
- 업데이트: ~7초 폴링
- 가격: 무료, 인증 불필요 (User-Agent 필요)
- 포맷: JSON

#### 차트 데이터
- URL: https://fchart.stock.naver.com/sise.nhn
- 파라미터: symbol={코드}&timeframe=day|week|month&count={N}&requestType=0
- 포맷: XML

#### 모바일 API
- 지수: https://m.stock.naver.com/api/index/KOSPI/price?pageSize=20&page=1
- 종목: https://m.stock.naver.com/api/stock/{종목코드}/price

#### KOSPI200 지수 페이지
- https://finance.naver.com/sise/sise_index.naver?code=KPI200
- https://finance.naver.com/sise/sise_index_time.naver?code=KPI200 (시간별)
- https://finance.naver.com/sise/sise_index_day.naver?code=KPI200 (일별)

### 1.3 Daum/Kakao Finance (비공식 API)
- 종목 시세: https://finance.daum.net/api/quotes/A005930
- 일별: https://finance.daum.net/api/quote/A005930/days?symbolCode=A005930&page=1&perPage=10
- 지수 요약: https://finance.daum.net/api/exchanges/summaries
- 포맷: JSON, Referer+User-Agent 필수, 403 빈번

### 1.4 한국투자증권 (KIS) Open API [핵심 추천]

- 포털: https://apiportal.koreainvestment.com
- REST 운영: https://openapi.koreainvestment.com:9443
- REST 모의: https://openapivts.koreainvestment.com:29443
- WebSocket 운영: wss://openapi.koreainvestment.com:9443/websocket
- WebSocket 모의: ws://ops.koreainvestment.com:21000
- 가격: 무료 (계좌 개설 필요, 모의투자 가능)
- 업데이트: 실시간 WebSocket 틱 데이터

#### 인증
- POST /oauth2/tokenP (grant_type=client_credentials, appkey, appsecret)
- 토큰 유효: ~24시간
- WebSocket: POST /oauth2/Approval -> approval_key

#### 주요 REST API
- GET /uapi/domestic-stock/v1/quotations/inquire-price : 현재가
- GET /uapi/domestic-stock/v1/quotations/inquire-daily-price : 기간별 시세
- GET /uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn : 호가
- GET /uapi/domestic-stock/v1/quotations/inquire-investor : 투자자별
- GET /uapi/domestic-stock/v1/quotations/investor-program-trade-today : 프로그램매매
- GET /uapi/domestic-futureoption/v1/quotations/inquire-price : 선물/옵션 현재가
- GET /uapi/domestic-futureoption/v1/quotations/inquire-daily-fuopchartprice : 선물 일별
- GET /uapi/domestic-futureoption/v1/quotations/inquire-time-fuopchartprice : 선물 분별
- GET /uapi/etfetn/v1/quotations/inquire-price : ETF/ETN 시세

#### WebSocket TR_ID
- H0STASP0 : 국내주식 실시간 호가
- H0STCNT0 : 국내주식 실시간 체결
- H0STCNI0 : 실시간 체결통보 (주문용)
- H0IFASP0 : 선물옵션 실시간 호가
- H0IFCNT0 : 선물옵션 실시간 체결

#### Rate Limits
- REST: 초당 20건 (일부 초당 1건)
- WebSocket: 동시 40종목

#### Python: mojito2 (pip), pykis (github)

### 1.5 eBest (LS증권)
- Open API: https://openapi.ls-sec.co.kr (REST+WS)
- xingAPI: COM 기반 (Windows 전용)
- 주요 선물 TR: t8401, o3101, o3105, o3123

### 1.6 키움증권 Open API+
- 레거시: COM (Windows/32bit)
- 신규 REST: https://openapi.kiwoom.com
- TR: opt10001, opt50001, opt50028
- Python: pykiwoom, koapy
- 한계: Windows, 32bit, 1초 5건

### 1.7 pykrx
- pip install pykrx (v1.2.4)
- 일별만, 분봉/틱 불가
- 주요: get_market_ohlcv_by_date, get_market_trading_value_by_investor,
  get_exhaustion_rates_of_foreign_investment, get_index_ohlcv_by_date

### 1.8 기타
- FinanceDataReader: KS11(KOSPI), KS200(KOSPI200)
- Yahoo Finance: 005930.KS, ^KS11, ^KS200 (15분 지연)

---

## 2. 외국인 선물/옵션 포지션 데이터

### 2.1 KRX 파생상품 투자자별
- data.krx.co.kr 메뉴 MDC0302020405
- bld: MDCSTAT04001 (추정)
- 장중 수분 간격, 무료

### 2.2 KIS API
- GET /uapi/domestic-futureoption/v1/quotations/inquire-investor
- 실시간 REST

### 2.3 Naver
- https://finance.naver.com/sise/investorDealTrendDay.nhn
- https://finance.naver.com/sise/investorDealTrendTime.nhn
- HTML 스크래핑

### 2.4 핵심 데이터 포인트

| 데이터 | 최적 소스 | 업데이트 |
|--------|----------|---------|
| 외국인 선물 누적순매수 | KRX 스크래핑 | 장중 수분 |
| 외국인 선물 당일 변화 | KIS REST/WS | 실시간 |
| 외국인 콜/풋 순매수 | KRX 파생 투자자별 | 장중 수분 |
| 외국인 미결제약정 | KRX/KIS | 장중/일별 |

---

## 3. Backwardation vs Contango 탐지

### 3.1 필요 데이터

| 항목 | 소스 | 엔드포인트 |
|------|------|----------|
| KOSPI200 현물 | Naver | sise_index.naver?code=KPI200 |
| KOSPI200 현물 | KIS | /uapi/domestic-stock/v1/quotations/inquire-price |
| KOSPI200 선물 | KIS | /uapi/domestic-futureoption/v1/quotations/inquire-price |
| 무위험이자율 | BOK ECOS | ecos.bok.or.kr/api/ 통계코드 722Y001 |
| 배당수익률 | KRX | 지수 기본정보 |

### 3.2 선물 종목코드 체계
- KOSPI200 선물: 101S{YY}{MM} (예: 101S2603=2026/03)
- 만기: 매월 둘째 주 목요일
- 주요 분기물: 3, 6, 9, 12월

### 3.3 베이시스 계산 공식
- 시장 베이시스 = 선물가격 - 현물지수
- 이론 베이시스 = 현물 x (금리 - 배당) x (잔존일/365)
- 스프레드 = 시장 베이시스 - 이론 베이시스
- 스프레드 < 0 -> 백워데이션 (하락 신호)
- 스프레드 > 0 -> 콘탱고 (정상/상승)

### 3.4 한국은행 ECOS API
- URL: https://ecos.bok.or.kr/api/StatisticSearch/{KEY}/json/kr/1/1/722Y001/M/{시작}/{끝}/0101000
- 무료 발급

---

## 4. 수급 데이터

### 4.1 소스 비교

| 소스 | 외국인 | 기관세분류 | 개인 | 프로그램 | 업데이트 | 비용 |
|------|--------|-----------|------|---------|---------|------|
| KRX 스크래핑 | O | O | O | O | 수분 | 무료 |
| Naver | O | 제한적 | O | O | 수분 | 무료 |
| KIS API | O | O | O | O | 실시간 | 무료(계좌) |
| pykrx | O | O | O | - | 일별 | 무료 |

### 4.2 프로그램 매매
- Naver: https://finance.naver.com/sise/sise_program.nhn?sosok=01
- KIS: GET /uapi/domestic-stock/v1/quotations/investor-program-trade-today
- 차익/비차익 매수/매도 분류

---

## 5. ETF 레버리지 데이터

### 5.1 주요 ETF

| ETF | 코드 | 배수 |
|-----|------|------|
| KODEX 레버리지 | 122630 | 2X |
| KODEX 인버스 | 114800 | -1X |
| KODEX 200선물인버스2X | 252670 | -2X |
| KODEX 200 | 069500 | 1X |
| TIGER 레버리지 | 123320 | 2X |
| TIGER 인버스 | 123310 | -1X |
| TIGER 200선물인버스2X | 252710 | -2X |

### 5.2 iNAV 소스
- KRX ETF: 장중 10초 간격
- Naver: finance.naver.com/item/main.naver?code=122630
- 공공데이터포털: API 15094806
- 자산운용사: samsungfund.com, tigeretf.com
- k-etf.com/etf/{코드}

### 5.3 괴리율 = (시장가 - NAV) / NAV x 100
### 5.4 레버리지/인버스 비율: 거래량 비율 > 2.0 강세, < 0.5 약세

---

## 6. 종합 매트릭스

### 실시간 (단타 필수)

| 데이터 | 1순위 | 2순위 | 업데이트 |
|--------|-------|-------|---------|
| KOSPI 지수 | KIS WebSocket | Naver 7s | 틱/7초 |
| 종목 체결 | KIS H0STCNT0 | Naver | 틱 |
| 선물 체결 | KIS H0IFCNT0 | eBest o3101 | 틱 |
| 외국인 순매수 | KIS REST | Naver 투자자별 | 실시간/수분 |
| 외국인 선물 | KRX 스크래핑 | KIS REST | 수분 |
| 프로그램매매 | KIS REST | Naver | 실시간/수분 |
| ETF 시세 | KIS WebSocket | Naver | 틱/7초 |
| ETF iNAV | KRX | Naver | 10초 |

### 과거 (백테스팅)

| 데이터 | 소스 |
|--------|------|
| OHLCV | pykrx, FinanceDataReader |
| 투자자별 | pykrx |
| 지수 | pykrx |
| 선물 | KRX 스크래핑, KIS |
| ETF | pykrx |

---

## 7. 권장 아키텍처

### 7.1 최소 실행 구성 (모두 무료)
1. 한국투자증권 계좌 + API Key
2. pykrx (pip install pykrx)
3. Naver Finance 스크래핑 (보조)
4. 한국은행 ECOS API Key

### 7.2 Python 패키지
- pykrx, mojito2, websockets, aiohttp
- beautifulsoup4, lxml, pandas, numpy
- redis(선택), finance-datareader(선택)

### 7.3 주의사항
1. KIS: 초당 20건, WS 40종목
2. Naver: 비공식, 차단 위험
3. KRX: OTP 기반 자동화 주의
4. 키움/eBest 레거시: Windows only
5. 데이터 교차 검증 필수
6. NTP 시간 동기화
7. 정규장 09:00~15:30
8. 선물 만기: 매월 둘째 주 목요일

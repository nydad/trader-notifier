---
name: kospi-etf-analysis
description: KOSPI ETF 레버리지 단타를 위한 실시간 시그널 시스템. "장전체크", "시그널", "premarket", "시장 분석", "상관분석", "뉴스 체크", "루프", "트레이딩" 요청 시 사용.
---

# KOSPI ETF 레버리지 단타 시그널 시스템

## 사용자 매매 패턴

**숏: KODEX 200선물인버스2X (252670)** — KOSPI 하락 베팅, 고물량(10만주+), 250~268원 대 스캘핑
**롱: 다양한 레버리지 ETF** — 반도체/방산/KOSPI/코스닥 레버리지로 상승 베팅
**보조: KODEX 코스닥150레버리지 (233740)** — 코스닥 롱, 소량, 빠른 스캘핑

→ SHORT 시그널 = 인버스2X **매수** 기회
→ LONG 시그널 = 레버리지 ETF **매수** 기회 (종목 추천 포함)

## 핵심 실행 (2개만 기억)

### 1. 통합 트레이딩 루프 (메인 — 장중 자동 실행)
```bash
cd E:/workspace/market && PYTHONPATH=src python scripts/trading_loop.py
PYTHONPATH=src python scripts/trading_loop.py --signal-interval 3   # 3분 시그널
PYTHONPATH=src python scripts/trading_loop.py --no-pattern           # 패턴분석 OFF (빠른 시작)
```
**기능:**
- **5분 주기**: Bayesian 시그널 + 역사적 패턴 확률 → Discord 전송
- **30분 주기**: 뉴스 + 외국인/기관 동향 + 종합 리포트 → Discord 전송
- **포지션 관리**: 매수 보고 → 매도 시그널 + 전략 안내
- **Discord 명령**: `!시그널`, `!뉴스`, `!포지션`, `!청산`, `!도움말`
- **매수/매도 파싱**: "인버스 250원 10만주 매수" → 자동 포지션 등록 + 전략 안내

### 2. 장전 체크 (매일 08:30 전, CLI 전용)
```bash
cd E:/workspace/market && PYTHONPATH=src python scripts/premarket.py
PYTHONPATH=src python scripts/premarket.py --discord   # Discord로도 전송
```

## 프로젝트 구조

```
scripts/                            ← 실전 스크립트 (2개)
├── trading_loop.py                 ← 통합 루프 (시그널+리포트+포지션+Discord)
├── premarket.py                    ← 장전 체크 (CLI)
├── analyze.py                      ← 상관분석 (연구용)
└── _archive/                       ← 구버전 (참고용)

src/kospi_corr/                     ← 코어 라이브러리
├── engine/                         ← 시그널 엔진 (실전 핵심)
│   ├── bayesian.py                 ← Bayesian Log-Odds Pooling
│   ├── scorer.py                   ← WeightedScorer
│   └── confidence.py               ← 신뢰도 계산
├── analysis/
│   ├── pattern_matcher.py          ← 역사적 조건부 확률 분석 ★NEW
│   ├── correlation/                ← 상관분석 (연구/백테스트용)
│   └── event_study.py              ← 이벤트 스터디
├── data/providers/
│   ├── naver_scraper.py            ← 외국인/기관 매매동향 스크래핑 ★NEW
│   ├── yfinance_provider.py        ← yfinance 글로벌 데이터
│   ├── krx.py                      ← KRX (pykrx)
│   ├── fdr_provider.py             ← FinanceDataReader
│   └── fred_provider.py            ← FRED
├── collectors/news.py              ← 뉴스 키워드 RSS 모니터
├── signals/regime.py               ← 레짐 게이트
├── backtest/                       ← 백테스트 (연구용)
└── notification/                   ← Discord 알림

config/                             ← 설정
data/watchlist.json                 ← 실전 종목
매매패턴.txt                         ← 사용자 매매 기록
```

## 시그널 로직 (3단계)

### 1단계: Bayesian 시그널 (실시간)
- yfinance 실시간 호가 → z-score → sigmoid → P(LONG)
- Bayesian Pooling: P(LONG) = sigmoid(Σ(weight × log-odds))
- 지표: USD/KRW, S&P선물, NQ선물, WTI, VIX, DXY

### 2단계: 역사적 패턴 (조건부 확률)
- 현재 조건 매칭 → 2년 역사 데이터 검색
- "환율↑ + S&P↓ → KOSPI 하락 73% (47건)" 형태
- 복합 패턴: 리스크오프, 리스크온, 유가 충격 등

### 3단계: 수급 + 뉴스 (30분 리포트)
- 외국인/기관 매매동향 (Naver Finance 스크래핑)
- 뉴스 키워드 (연합/한경/매경 RSS)
- 종합 고려사항 자동 생성

## 매매 해석

| 시그널 | 인버스2X (숏) | 레버리지 (롱) |
|--------|--------------|--------------|
| SHORT  | **매수** 기회 | 매도/미진입 |
| LONG   | 매도/미진입   | **매수** 기회 |
| 방향전환 | 즉시 매도 고려 | 즉시 매도 고려 |

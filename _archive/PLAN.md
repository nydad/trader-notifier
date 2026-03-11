# KOSPI ETF 레버리지 단타 스킬 — 최종 실행 계획

> 15+ Spark Panel 에이전트 조사 결과 종합 (2026-03-11)

## 프로젝트 목표

코스피 ETF 레버리지 롱/숏 단타 투자를 위한 Claude Code 스킬.
모든 데이터를 종합하여 **LONG 확률 / SHORT 확률**을 산출.

## 핵심 발견 (조사에서 밝혀진 사실)

1. **유가는 최고 상관계수 지표가 아님** — 야간선물, 환율, S&P500이 더 강함
2. **환율(USD/KRW)이 유가보다 직접적** — 유가 효과의 상당 부분이 환율을 통해 매개
3. **옵션 PCR, 베이시스는 방향 지표 아님** — 감정/구조 지표로만 사용
4. **1-2개월 백테스트는 통계적 유의성 없음** — 최소 6개월, 권장 1년+
5. **현실적 승률은 50-58%** — 65%는 비현실적
6. **ETF별로 최적 지표가 완전히 다름** — 반도체는 SOX, 방산은 지정학

## 시그널 아키텍처 (최종)

### Layer 1: Direction Driver
| 시그널 | 소스 | 비용 |
|--------|------|------|
| KOSPI200 야간선물 | yfinance / KIS API | 무료 |
| S&P500 / NASDAQ 선물 | yfinance | 무료 |
| USD/KRW 환율 | FinanceDataReader | 무료 |
| WTI 유가 | yfinance (CL=F) | 무료 |

### Layer 2: Sector Driver
| ETF 그룹 | 최적 지표 | 소스 |
|----------|----------|------|
| KOSPI200 (122630, 252670) | 야간선물, S&P500, 환율 | yfinance |
| 반도체 (091170, 396520) | SOX지수, NVIDIA | yfinance |
| 방산 (472170, 472160) | 지정학 뉴스 | RSS/뉴스API |
| 코스닥 (233740, 229200) | KOSDAQ150선물 | pykrx |

### Layer 3: Confirmation
| 시그널 | 소스 | 지연 |
|--------|------|------|
| 외국인 선물 순매수 | KRX/KIS API | 종가 후 |
| 외인+기관 수급 | pykrx / KRX | 15분 지연 |
| 프로그램매매 | KRX | 15분 지연 |

### Layer 4: Regime Gate
| 시그널 | 역할 |
|--------|------|
| VKOSPI | 시그널 신뢰도 판단 |
| 뉴스 키워드 (이란/트럼프/NVIDIA 등) | 이벤트 필터 |
| 만기일/CB/사이드카 | 자동 거래 중단 |

## 결합 엔진

Bayesian Log-Odds Pooling (5개 Panel 합의)
- 각 시그널 → 정규화 → log-odds 변환 → 레짐별 가중치 → sigmoid → 확률

## 기술 스택

- Python 3.11 (이미 설치)
- pykrx + FinanceDataReader: 한국 데이터
- yfinance: 글로벌 데이터 (유가/환율/미국지수)
- pandas + numpy + scipy: 분석
- ta: 기술적 분석
- rich: CLI 출력
- seaborn + matplotlib: 시각화
- SQLite: 캐시/히스토리
- pydantic: 데이터 모델

## 구현 단계

### Phase 1: 상관분석 MVP (1일)
- 6개월치 데이터 수집 (pykrx + FDR)
- 12개 ETF + 유가 + 환율 상관 매트릭스
- rich CLI + seaborn heatmap PNG

### Phase 2: 시그널 시스템 (3-5일)
- 데이터 수집 파이프라인 (collectors/)
- 시그널 프로세서 6개
- Weighted Scoring → 확률 산출
- CLI 출력 포매팅

### Phase 3: 고도화 (1-2주)
- Bayesian log-odds 결합
- 레짐 탐지
- 뉴스 키워드 모니터링
- Rolling correlation
- 백테스트 프레임워크

### Phase 4: 스킬화 (2-3일)
- SKILL.md 작성
- 헬퍼 스크립트 정리
- 설치/설정 가이드
- 테스트

## 파일 구조

```
market/
├── PLAN.md
├── config/
│   ├── symbols.yaml        # watchlist 종목
│   ├── weights.yaml        # 시그널 가중치
│   └── keywords.yaml       # 뉴스 키워드
├── src/
│   ├── cli.py              # 메인 엔트리포인트
│   ├── config.py           # 설정 로더
│   ├── models.py           # 데이터 모델
│   ├── collectors/
│   │   ├── base.py
│   │   ├── krx.py          # pykrx + KRX
│   │   ├── global_market.py # yfinance
│   │   └── news.py         # RSS/뉴스API
│   ├── signals/
│   │   ├── base.py
│   │   ├── direction.py    # 1차 방향 시그널
│   │   ├── sector.py       # 섹터별 드라이버
│   │   ├── flow.py         # 수급 확인
│   │   └── regime.py       # 레짐 필터
│   ├── engine/
│   │   ├── scorer.py       # Weighted Scoring (Phase 2)
│   │   ├── bayesian.py     # Bayesian (Phase 3)
│   │   └── confidence.py   # 신뢰도 계산
│   ├── analysis/
│   │   ├── correlation.py  # 상관 매트릭스
│   │   ├── backtest.py     # 백테스트
│   │   └── event_study.py  # 뉴스 이벤트 분석
│   ├── output/
│   │   ├── cli_rich.py     # 컬러 CLI
│   │   ├── json_out.py     # JSON
│   │   └── heatmap.py      # seaborn PNG
│   └── storage/
│       ├── db.py           # SQLite
│       └── cache.py        # TTL 캐시
├── data/
│   ├── watchlist.json
│   ├── indicators.json
│   └── cache.db
└── requirements.txt
```

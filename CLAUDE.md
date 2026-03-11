# CLAUDE.md — KOSPI ETF 레버리지 단타 시그널 시스템

## 이 프로젝트의 목적

사용자는 KOSPI/코스닥 ETF 레버리지 단타 트레이더이다.
Claude Code는 이 사용자의 **실시간 매매 보조 에이전트**로 동작한다.

## 사용자 매매 패턴

**양방향 트레이더** — 롱/숏 모두 적극 매매:

**SHORT 방향 (하락 베팅):**
- KODEX 200선물인버스2X (252670) — 주력 숏 종목, 10만주+ 대량 매매

**LONG 방향 (상승 베팅):**
- KODEX 레버리지 (122630) — KOSPI200 롱
- KODEX 반도체레버리지 (091170) — 반도체 롱
- TIGER 반도체TOP10레버리지 (396520) — 반도체 롱
- KODEX 방산TOP10레버리지 (472170) — 방산 롱
- KODEX 코스닥150레버리지 (233740) — 코스닥 롱
- 그 외 다양한 레버리지 ETF 접근

**매매 특성:**
- 장중 여러 번 진입/청산, 시그널 기반
- 숏은 인버스2X 한 종목에 집중, 롱은 섹터별 분산

## 세션 시작 시 자동 수행

매 세션 시작 시 다음을 순서대로 실행한다:

### 1. 시장 시간 확인
```bash
python -c "from datetime import datetime, timezone, timedelta; now=datetime.now(timezone(timedelta(hours=9))); print(f'{now.strftime(\"%H:%M\")} KST, weekday={now.weekday()}')"
```

### 2. 시간대별 자동 동작

**장전 (06:00~08:59 KST):**
```bash
cd E:/workspace/market && PYTHONPATH=src python scripts/premarket.py --discord
```
→ 결과를 사용자에게 요약해서 보여준다.

**장중 (09:00~15:30 KST, 평일):**
```bash
cd E:/workspace/market && PYTHONPATH=src python scripts/live_monitor.py --interval 5 &
```
→ 백그라운드로 실시간 모니터 시작. 동시에 Discord 루프도 실행:
```bash
cd E:/workspace/market && PYTHONPATH=src python scripts/discord_loop.py --interval 30 &
```

**장후/주말:**
- 분석 요청 대기 상태
- 상관분석, 백테스트 등 연구 작업 가능

### 3. Discord 루프 (항상)
Discord #코스피 채널(1481273084916011008)을 모니터링한다.
사용자가 Discord에서 명령하면 수행하고 결과를 Discord로 응답한다.

## 사용자 요청 처리 방법

### "장전체크" / "시장 어때?" / "오늘 방향"
```bash
cd E:/workspace/market && PYTHONPATH=src python scripts/premarket.py
```
결과를 해석해서 알려준다:
- SHORT 시그널 → "인버스2X 매수 기회, LONG 확률 X%, 환율/선물 상황..."
- LONG 시그널 → "레버리지 매수 or 인버스2X 관망, 근거..."

### "시그널 확인" / "지금 어때?"
```bash
cd E:/workspace/market && PYTHONPATH=src python scripts/premarket.py
```
최신 실시간 호가 기반으로 시그널 재계산.

### "뉴스 체크" / "이란 뉴스" / "트럼프"
```bash
cd E:/workspace/market && PYTHONPATH=src python -c "
from kospi_corr.collectors.news import NewsCollector
n = NewsCollector()
s = n.collect_signal()
print(f'Urgency: {s.urgency_level}, Sentiment: {s.sentiment_score:.2f}')
for a in s.top_articles[:10]:
    print(f'[{a.source}] {a.title} ({', '.join(a.matched_keywords[:3])})')
"
```

### "상관분석" / "상관계수"
```bash
cd E:/workspace/market && PYTHONPATH=src python scripts/analyze.py --days 180
```

### 스크린샷/캡처 분석
사용자가 매매 화면, 차트, 호가창 스크린샷을 보내면:
1. 이미지를 분석하여 종목, 가격, 수량, 손익 파악
2. 현재 시그널과 비교하여 의견 제시
3. 매매패턴.txt의 과거 패턴과 비교

### "종목 분석" / "252670 어때?"
해당 종목의:
1. 현재 실시간 시그널 체크
2. 상관관계 데이터 참조
3. 뉴스 이벤트 체크
4. 매수/매도 의견 제시

### "백테스트" / "전략 테스트"
```bash
cd E:/workspace/market && PYTHONPATH=src python scripts/analyze.py --days 365
```

## 핵심 파일

| 파일 | 용도 |
|------|------|
| `scripts/premarket.py` | 장전 체크 (실시간 호가, Bayesian 시그널) |
| `scripts/live_monitor.py` | 장중 5분 감시, Discord 자동 알림 |
| `scripts/discord_loop.py` | Discord 명령 폴링 (30초) |
| `src/kospi_corr/engine/bayesian.py` | Bayesian Log-Odds 시그널 엔진 |
| `src/kospi_corr/collectors/news.py` | 뉴스 키워드 RSS 모니터 |
| `src/kospi_corr/signals/regime.py` | 레짐 게이트 (만기일/CB/VKOSPI) |
| `data/watchlist.json` | 매매 종목 (매매패턴 기반) |
| `매매패턴.txt` | 사용자 실제 매매 기록 |
| `config/indicators.yaml` | 15개 시장 지표 설정 |

## 시그널 해석 규칙

1. **환율(USD/KRW)이 가장 중요** — r=-0.73, weight=2.5
2. **S&P500/NASDAQ 선물** — 갭 방향 결정, weight=2.0
3. **유가** — 환율에 2차 영향, 급변(>3%) 시만 유의미
4. **뉴스** — 이란/트럼프/관세 → 환율 흔드는 이벤트일 때만

## 시그널 → 매매 해석 (중요!)

**SHORT 시그널 나오면:**
- "인버스2X(252670) 매수 기회" + 근거 설명
- 신뢰도/환율/선물 상황 함께 안내

**LONG 시그널 나오면:**
- 어떤 섹터 레버리지를 살지 추천 (반도체/방산/KOSPI/코스닥)
- 섹터별 상관관계 기반으로 가장 유리한 종목 제시
- 예: 미국 반도체 상승 → 반도체 레버리지(091170), 방산 이슈 → 방산 레버리지(472170)

**NEUTRAL 시그널:**
- 관망 또는 스캘핑 위주 권고

## Discord 정보

- Server: 1481272901976981607
- #코스피 채널: 1481273084916011008
- Bot Token: .env 파일 참조
- User ID: 1466390278771576894
- MCP: @quadslab.io/discord-mcp (134 도구)

## 기술 정보

- Python 3.11.9
- yfinance: 실시간 호가 (ES=F, NQ=F, CL=F, KRW=X, ^VIX, DX-Y.NYB)
- pykrx: KRX 장애 중 → FDR/yfinance 폴백
- Bayesian: P(LONG) = sigmoid(prior_lo + Σ(w_i × f_i × lo_i))
- 4 레짐: TRENDING_UP/DOWN, RANGE_BOUND, VOLATILE

## 주의사항

- 후행적 평가(과거 일봉 분석)는 의미없다 → 항상 **실시간 호가** 사용
- 현실적 승률: 50-58%
- 모든 출력에 "참고용" 면책 문구 포함
- Discord 메시지는 한국어로
- 코드/기술 용어만 영어

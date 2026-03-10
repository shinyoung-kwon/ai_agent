# LangGraph 바이오마커 발견 파이프라인 설계

## 목표

LangGraph 기반 4단계 순차 파이프라인으로 바이오마커 후보를 발견하고 검증하는 멀티 에이전트 시스템을 구축한다.
이번 구현은 MCP 서버 연결 검증에 집중하며, State와 RAG는 최소 구현으로 시작한다.

## 핵심 결정 사항

| 항목 | 결정 |
|------|------|
| Planner 에이전트 | 없음 — 고정 순차 파이프라인 (나중에 조건부 엣지로 확장) |
| 그래프 구조 | 서브그래프 방식 — 에이전트별 독립 tool_call 루프 |
| LLM 도구 호출 | LLM 판단형 — LLM이 tool_call 결정 |
| MCP 연결 | LangChain Tool로 변환 후 bind_tools |
| LLM 모델 | Claude (전 에이전트 동일, config에서 교체 가능) |
| Agent C (RAG) | 이번엔 mock — 나중에 별도 구현 |
| State | 최소 타입으로 시작, MCP 완성 후 구체화 |
| 프롬프트 | configs/prompts.yaml에서 에이전트별 관리 |
| LangSmith | .env 환경변수로 트레이싱 활성화 |
| 사용자 입력 | main.py에서 CLI input()으로 심플하게 |

## 아키텍처

### 전체 흐름

```
메인 그래프 (builder.py)
START
  → discovery_agent (서브그래프)  ← MCP: discovery-server
  → network_agent (서브그래프)    ← MCP: network-server
  → reasoning_agent (서브그래프)  ← mock (RAG 미구현)
  → validation_agent (서브그래프) ← MCP: validation-server
  → END
```

### 서브그래프 내부 구조

각 에이전트는 자체 tool_call 루프를 가진 서브그래프:

```
llm_call → should_continue? → tool_node → llm_call (루프)
               ↓ (tool_call 없으면)
           결과를 AgentState에 기록
```

- LLM에게 프롬프트 + State + 에이전트별 도구 목록 전달
- LLM이 tool_call 결정 → MCP 도구 실행 → 결과를 다시 LLM에게
- LLM이 최종 응답 생성 → State 업데이트

## 파일 구조

```
apps/agent/app/
├── main.py                    # 사용자 입력 → 그래프 실행 → 결과 출력
├── graph/
│   ├── state.py               # AgentState 정의
│   ├── builder.py             # 메인 그래프 조립 (4개 서브그래프 연결)
│   └── nodes/                 # 에이전트 서브그래프
│       ├── __init__.py
│       ├── discovery.py       # Agent A 서브그래프
│       ├── network.py         # Agent B 서브그래프
│       ├── reasoning.py       # Agent C 서브그래프 (mock)
│       └── validation.py      # Agent D 서브그래프
├── tools/
│   ├── registry.py            # MCP 클라이언트 연결 + LangChain Tool 변환
│   └── profiles/
│       ├── discovery.yaml     # Agent A 허용 도구
│       ├── network.yaml       # Agent B 허용 도구
│       ├── reasoning.yaml     # Agent C (빈 목록)
│       └── validation.yaml    # Agent D 허용 도구
├── services/                  # 비즈니스 로직 (나중에 확장)
├── retrieval/                 # RAG (나중에 확장)
└── utils/
    └── config.py              # dev.yaml, .env 로딩, LLM 팩토리

configs/
├── dev.yaml                   # 모델명, MCP 서버 경로 등
└── prompts.yaml               # 에이전트별 시스템 프롬프트

data/                          # MCP 서버 참조 데이터
├── raw/
├── processed/
├── indices/
└── cache/

mcp-servers/                   # MCP 테스트 서버 (mock)
├── test_discovery_server.py
├── test_network_server.py
└── test_validation_server.py
```

## 데이터 흐름 (AgentState)

```python
class AgentState(TypedDict):
    query: str                                          # 사용자 입력
    messages: Annotated[list[BaseMessage], operator.add] # 누적 대화 이력
    candidates: list[str]                               # Agent A 출력
    network_data: dict                                  # Agent B 출력
    reasoning: str                                      # Agent C 출력
    validation_results: list[dict]                      # Agent D 출력
```

각 에이전트가 자기 필드만 업데이트하고 이전 에이전트의 출력을 참조:

- Agent A: query → DEG 조회 (MCP) → candidates 기록
- Agent B: candidates → 레귤론 조회 (MCP) → network_data 기록
- Agent C: candidates + network_data → 분석 (mock) → reasoning 기록
- Agent D: candidates + reasoning → 시뮬레이션 (MCP) → validation_results 기록

## MCP 연결 방식

1. `configs/dev.yaml`에서 MCP 서버 경로 읽기
2. `registry.py`가 각 서버를 stdio 방식으로 subprocess 실행
3. MCP 도구를 LangChain `BaseTool`로 변환
4. `profiles/*.yaml` 기준으로 에이전트별 도구 세트 반환
5. 각 서브그래프에서 `llm.bind_tools(get_tools("discovery"))` 형태로 바인딩

## 설정

### configs/dev.yaml

```yaml
llm:
  model: claude-sonnet-4-20250514
  temperature: 0

mcp_servers:
  discovery: "python mcp-servers/test_discovery_server.py"
  network: "python mcp-servers/test_network_server.py"
  validation: "python mcp-servers/test_validation_server.py"

langsmith:
  project: ai-agent-biomarker
```

### configs/prompts.yaml

```yaml
discovery_agent:
  system: |
    당신은 유전자 발현 데이터를 분석하는 Discovery Agent입니다.
    주어진 질환/주제에 대해 차등발현유전자(DEG)를 조회하고,
    바이오마커 후보 유전자 목록을 도출하세요.
    도구를 사용하여 발현 데이터를 조회하고 분석하세요.

network_agent:
  system: |
    당신은 유전자 네트워크를 분석하는 Network Agent입니다.
    이전 단계에서 발견된 후보 유전자들의 레귤론 정보와
    TF-Target 네트워크를 조회하여 조절 관계를 파악하세요.

reasoning_agent:
  system: |
    당신은 바이오마커 후보를 평가하는 Reasoning Agent입니다.
    이전 단계의 후보 유전자와 네트워크 데이터를 바탕으로
    생물학적 타당성을 분석하고 후보군의 우선순위를 매기세요.

validation_agent:
  system: |
    당신은 인실리코 검증을 수행하는 Validation Agent입니다.
    추천된 후보 유전자들에 대해 시뮬레이션을 실행하고,
    최종 마스터 레귤레이터를 확정하세요.
```

### .env (환경변수)

```
ANTHROPIC_API_KEY=<your-key>
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-key>
LANGCHAIN_PROJECT=ai-agent-biomarker
```

## 향후 확장

- Agent C에 RAG 파이프라인 연결 (retrieval/ 폴더)
- 조건부 엣지 추가 (예: Agent C 결과 부족 시 Agent A로 되돌아감)
- 에이전트별 다른 LLM 모델 지정
- services/ 폴더에 비즈니스 로직 분리
- 실제 MCP 서버 구현 후 State 타입 구체화

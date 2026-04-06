import os
from langchain_google_genai import ChatGoogleGenerativeAI
# OpenAI: from langchain_openai import ChatOpenAI
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# --- 0. 환경 설정: API 키 로드 ---
load_dotenv()
# OpenAI API 키 설정
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")
# OpenAI: os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
# CLAUDE: Claude API 키를 설정해야 합니다.
# CLAUDE: os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY")
# GEMINI: Google API 키를 설정해야 합니다.
# GEMINI: os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")

# '헌법'의 복잡한 규칙을 잘 따르도록 gpt-4o와 같은 고성능 모델을 사용합니다.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)
# GEMINI: llm = ChatGoogleGenerativeAI(model="gemini-3.0", temperature=0.7)

def get_ai_response(messages: list) -> str:
    """ChatModel에 메시지 리스트를 전달하고 응답을 받습니다."""
    print(f"\n--- AI 조감독에게 장면 생성을 요청합니다 (총 메시지 길이: {len(str(messages))}자) ---")
    try:
        response = llm.invoke(messages)
        #return response.content.strip()
        if isinstance(response.content, list):
            # 리스트 일 경우
            # return "".join([str(block) for block in response.content]).strip()
            # 딕셔너리 구조일 경우 (예상)
            return "".join([
                block.get('text', '') if isinstance(block, dict) else str(block) 
                for block in response.content
            ]).strip()
        else:
            # 문자열인 경우 기존대로 처리
            return response.content.strip()
    except Exception as e:
        return f"API 호출 중 오류 발생: {e}"

# --- 1. '엘라라 연대기'의 핵심 설정을 담은 '헌법' ---
story_constitution = """
# 작품 헌법: 엘라라 연대기

## 제1장: 주요 인물

### 1조 (주인공: 엘라라)
- **핵심 정체성:** 평화주의자 물의 마법사.
- **외모:** 칠흑 같은 긴 생머리, 에메랄드빛 녹색 눈동자. 왼손 손목에 어린 시절 생긴 작은 화상 흉터가 있음.
- **성격:** 내성적이고 신중함 (INFJ). 평소에는 말이 없지만, 불의나 약자가 고통받는 상황을 목격하면 누구보다 용감해짐.
- **핵심 가치관:** 비폭력주의. 대화와 이해를 통해 문제를 해결하는 것을 최우선으로 삼는다.
- **트라우마와 맹세:** 10살 때 겪은 대화재로 부모님을 잃고 극심한 불 공포증(Pyrophobia)을 앓고 있음. 그 이후 스스로에게 '어떤 상황에서도 불의 힘을 탐하거나 사용하지 않겠다'고 맹세함. 이 맹세는 그녀의 마법과 삶의 근간을 이룬다.
- **능력:**
  - 주력 마법: 물(Water)과 바람(Wind)의 원소 마법에 천재적인 재능을 보임. 특히 물을 얼려 방어벽을 만들거나, 안개를 생성하여 적을 교란시키는 데 능함.
  - 약점: 불 마법은 전혀 사용 불가. 불을 보는 것만으로도 공황 상태에 빠질 수 있음. 신체 능력이 매우 약해 근접전은 절대적으로 피해야 함.
- **말투:** 차분하고 부드러운 존댓말을 사용함.

## 제2장: 세계관

### 2조 (마법 시스템)
- **원리:** 세상의 4대 원소(물, 바람, 흙, 불)와 마법사의 정신력이 교감하여 발현됨.
- **규칙:** 마법사는 선천적으로 하나의 주력 원소와 하나의 보조 원소, 총 두 가지만 다룰 수 있다.
- **대가:** 강력한 마법을 사용할수록 정신력 소모가 극심하며, 한계를 넘으면 '마나 번아웃'으로 며칠간 기절할 수 있음.

## 제3장: 톤앤매너
- **장르:** 하이 판타지, 성장 드라마.
- **테마:** 트라우마의 극복, 용서, 자연과의 조화.
- **분위기:** 전반적으로 서정적이고 진지함. 폭력적이거나 잔인한 묘사는 최소화하고, 인물의 내면 심리 묘사에 집중한다.
"""

# --- 2. 프롬프트 생성 함수 ---
def generate_story_prompt_messages(constitution: str, user_request: str) -> list:
    """
    작품 헌법(System)과 작가의 요청(User)을 결합하여
    LangChain의 ChatModel에 전달할 메시지 리스트를 생성합니다.
    """
    
    # 1. 헌법 + AI의 역할을 SystemMessage로 정의
    system_prompt = f"""
당신은 판타지 소설 '엘라라 연대기'를 집필하는 스토리텔링 전문 AI 조감독입니다.
당신의 최우선 임무는 아래에 제시된 '[작품 헌법]'을 완벽하게 이해하고, 모든 내용을 이 헌법에 따라 일관되게 생성하는 것입니다.

**[매우 중요한 규칙]**
1.  답변을 생성하기 전에, 반드시 '[작품 헌법]'의 모든 조항, 특히 인물의 성격과 트라우마, 세계관 규칙을 먼저 확인하십시오.
2.  헌법의 내용과 모순되는 장면, 대사, 설정을 절대로 만들어서는 안 됩니다.
3.  헌법에 명시된 캐릭터의 성격과 가치관에 부합하는 행동과 대사만을 생성해야 합니다.
4.  작품의 전체적인 톤앤매너를 반드시 유지해야 합니다.

---
[작품 헌법]

{constitution}
---
"""
    
    # 2. 실제 작가의 요청을 HumanMessage로 정의
    user_prompt = f"""
[작가의 요청]
{user_request}

[AI 조감독의 장면 생성]
"""
    
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

# --- 3. 시나리오별 실행 및 비교 ---
if __name__ == "__main__":
    
    # 작가의 동일한 요청
    user_request = "고블린 무리에게 포위당한 엘라라가 절체절명의 위기에서 탈출하는 멋진 장면을 묘사해 줘."

    print("="*60)
    print("--- [사례 1] '헌법' (컨텍스트) 없이 단순하게 요청할 경우 ---")
    print("="*60)
    
    # AI에게 '엘라라'에 대한 배경지식(컨텍스트)을 전혀 주지 않음
    simple_messages = [
        HumanMessage(content=f"판타지 소설 작가로서, {user_request}")
    ]
    scene_no_context = get_ai_response(simple_messages)
    print(scene_no_context)
    print("\n[분석: AI가 '엘라라'의 규칙(불 공포증, 비폭력)을 모르기 때문에,")
    print("       화려한 불 마법이나 공격적인 마법을 사용하는 등, 설정을 파괴했을 가능성이 높습니다.]")


    print("\n\n" + "="*60)
    print("--- [사례 2] '헌법' (컨텍스트) 주입 후 동일하게 요청할 경우 ---")
    print("="*60)
    
    # '헌법'을 시스템 프롬프트로 주입하여 최종 메시지 리스트 생성
    messages_with_context = generate_story_prompt_messages(story_constitution, user_request)
    
    scene_with_context = get_ai_response(messages_with_context)
    print(scene_with_context)
    print("\n[분석: AI가 '헌법'을 준수하여, 엘라라의 성격(신중함), 능력(물/바람),")
    print("       약점(불 공포증), 가치관(비폭력)에 맞는 장면을 생성했습니다.]")

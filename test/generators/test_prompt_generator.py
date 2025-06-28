import os
import asyncio
from dotenv import load_dotenv

from src.managers.stores.opensearch_manager import OpenSearchManager
from src.managers.models.openai_manager import OpenAIManager
from src.managers.models.exaone_manager import ExaoneManager
from src.managers.models.anthropic_manager import AnthropicManager
from src.generators.prompt_generator import PromptGenerator

# .env 파일에서 환경 변수 로드
load_dotenv()

async def main():
  """
  비동기 테스트 로직을 실행하기 위한 main 함수.
  """
  print("--- Chatbot Integration Test (Using Managers) ---")
  print("Make sure your .env file is correctly configured.\n")

  os_manager = None
  try:
    # --- 1. `app.py`처럼 필요한 매니저들을 생성합니다 ---

    # OpenSearch 매니저 생성
    os_manager = OpenSearchManager()

    # 환경 변수에 따라 사용할 LLM 매니저를 동적으로 선택
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    llm_manager = None

    if llm_provider == "openai":
      llm_manager = OpenAIManager()
    elif llm_provider == "exaone":
      llm_manager = ExaoneManager()
    elif llm_provider == "anthropic":
      llm_manager = AnthropicManager()

    else:
      raise ValueError(f"지원하지 않는 LLM 제공자입니다: {llm_provider}")

    print(f"Selected LLM Provider: {llm_provider.upper()}")

    # --- 2. PromptGenerator를 생성하고, 필요한 매니저들을 주입(DI)합니다 ---
    prompt_generator = PromptGenerator(
        os_manager=os_manager,
        llm_manager=llm_manager
    )

    # --- 3. 서비스 연결 및 모델 로드 (Startup 시뮬레이션) ---
    print("\nStep 1: Connecting to services and loading models...")
    os_manager.connect()
    prompt_generator.load_embedding_model()  # PromptGenerator 내부의 임베딩 모델 로드
    if isinstance(llm_manager, ExaoneManager):
      llm_manager.load_model()  # 로컬 LLM의 경우 모델 로딩
    print("...Services connected and models loaded successfully.\n")

    # --- 4. 대화 시나리오 시뮬레이션 ---
    session_id = 'moongchi-session'
    questions = [
        "면도용품",
        "쉐이빙폼 사용방법",
        "달달한거 먹고싶다"
    ]

    for i, question in enumerate(questions, 1):
      print(f"--- Turn {i}: User Asks -> '{question}' ---")
      # `generate_response`는 비동기 함수이므로 await로 호출합니다.
      response = await prompt_generator.generate_response(
          message=question,
          session_id=session_id
      )
      # 다음 턴을 위해 세션 ID를 저장합니다.
      session_id = response['session_id']

      print("\n[AI Mungchi's Response]")
      print(response['bot_response'])
      print("-" * 20 + "\n")

  except Exception as e:
    print(f"\nAN ERROR OCCURRED: {e}")
  finally:
    # --- 5. 모든 외부 연결을 안전하게 종료 (Shutdown 시뮬레이션) ---
    print("Step 3: Cleaning up all connections...")
    if os_manager and os_manager.client:
      os_manager.disconnect()
    print("--- Test Finished ---")


# 이 파일이 직접 실행될 때 비동기 main 함수를 호출합니다.
if __name__ == '__main__':
  asyncio.run(main())

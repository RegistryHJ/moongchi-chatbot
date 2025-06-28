import os
from dotenv import load_dotenv
from src.managers.models.anthropic_manager import AnthropicManager

# .env 파일에서 환경 변수 로드
load_dotenv()

if __name__ == '__main__':
  print("--- Anthropic Manager Test ---")

  # 1. API 키 확인
  if not os.getenv('ANTHROPIC_API_KEY'):
    print("Error: ANTHROPIC_API_KEY not found in your .env file.")
  else:
    try:
      # 2. 매니저 생성
      manager = AnthropicManager()

      # 3. 테스트 프롬프트 설정
      prompt = "인공지능의 미래에 대해 한 문장으로 긍정적인 전망을 말해주세요."
      system_message = "You are a futurist AI assistant."

      print(f"Generating text with model: {manager.model_name}")
      print(f"Prompt: {prompt}")

      # 4. 텍스트 생성 실행
      response = manager.generate(prompt, system_message, max_new_tokens=100)

      print("\n--- Response ---")
      print(response)
      print("------------------")

    except Exception as e:
      print(f"\nAn error occurred: {e}")

  print("\n--- Test Finished ---")

import os
from dotenv import load_dotenv
from src.managers.models.exaone_manager import ExaoneManager

# .env 파일에서 환경 변수 로드
load_dotenv()

if __name__ == '__main__':
  print("--- Exaone Manager Test ---")

  try:
    # 1. 모델 매니저 생성
    manager = ExaoneManager()

    # 2. 모델 로드 (시간이 오래 걸릴 수 있습니다)
    print("Loading Exaone model. This may take a while...")
    manager.load_model()

    # 3. 모델이 성공적으로 로드되었는지 확인 후 텍스트 생성
    if manager.model:
      prompt = "인공지능의 미래에 대해 한 문장으로 긍정적인 전망을 말해주세요."
      system_message = "You are a futurist AI assistant."

      print(f"\nGenerating text with model: {manager.model_id}")
      print(f"Prompt: {prompt}")

      response = manager.generate(prompt, system_message, max_new_tokens=100)

      print("\n--- Response ---")
      print(response)
      print("------------------")
    else:
      print("Model could not be loaded, skipping generation.")

  except Exception as e:
    print(f"\nAn error occurred: {e}")

  print("\n--- Test Finished ---")

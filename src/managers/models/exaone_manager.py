import torch
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

class ExaoneManager:
  """
  Hugging Face Hub에서 EXAONE 모델과 토크나이저를 로드하고
  텍스트 생성을 관리하는 클래스입니다.
  """

  def __init__(self):
    """모델 메니저를 초기화합니다."""
    self.model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    self.model = None
    self.tokenizer = None
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {self.device}")

  def load_model(self):
    """
    지정된 model_id를 사용하여 모델과 토크나이저를 로드합니다.
    메모리 부족을 피하기 위해 bfloat16 타입을 사용합니다.
    """
    if self.model is None:
      print(f"Loading model: {self.model_id}...")
      try:
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,  # 메모리 사용량 감소를 위한 데이터 타입
            trust_remote_code=True,    # 모델 저장소의 커스텀 코드 실행 허용
            device_map="auto"          # 사용 가능한 장치(GPU/CPU)에 모델 레이어를 자동으로 분배
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        print("Model and tokenizer loaded successfully.")
      except Exception as e:
        print(f"Failed to load model: {e}")
        self.model = None
        self.tokenizer = None
        raise

  async def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.", max_new_tokens: int = 1024):
    """
    주어진 프롬프트를 기반으로 텍스트를 생성합니다.

    Args:
        prompt (str): 사용자 입력 프롬프트.
        system_prompt (str): 모델의 역할을 정의하는 시스템 메시지.
        max_new_tokens (int): 생성할 최대 토큰 수.

    Returns:
        str: 생성된 텍스트.
    """
    if not self.model or not self.tokenizer:
      print("Model is not loaded. Please call load_model() first.")
      return None

    def _blocking_generate():
      messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt}
      ]

      input_ids = self.tokenizer.apply_chat_template(
          messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
      ).to(self.device)

      output_tokens = self.model.generate(
          input_ids,
          eos_token_id=self.tokenizer.eos_token_id,
          max_new_tokens=max_new_tokens,
          pad_token_id=self.tokenizer.eos_token_id
      )

      response_tokens = output_tokens[0][input_ids.shape[-1]:]
      return self.tokenizer.decode(response_tokens, skip_special_tokens=True)

    loop = asyncio.get_running_loop()
    response_text = await asyncio.to_thread(_blocking_generate)
    return response_text

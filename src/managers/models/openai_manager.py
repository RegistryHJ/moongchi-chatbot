import os
import asyncio
from openai import AsyncOpenAI, OpenAIError

class OpenAIManager:
  """
  OpenAI API를 사용하여 텍스트 생성을 관리하는 클래스입니다.
  """

  def __init__(self, model_name: str = "gpt-4o"):
    """
    OpenAI 클라이언트를 초기화합니다.
    API 키는 환경 변수 'OPENAI_API_KEY'에서 가져옵니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
      raise ValueError("환경 변수에서 'OPENAI_API_KEY'를 찾을 수 없습니다.")

    self.client = AsyncOpenAI(api_key=api_key)
    self.model_name = model_name
    print(f"OpenAIManager initialized with model: {self.model_name}")

  async def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant.", max_new_tokens: int = 1024):
    """
    주어진 프롬프트를 기반으로 텍스트를 생성합니다.
    """
    try:
      completion = await self.client.chat.completions.create(
          model=self.model_name,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": prompt}
          ],
          max_tokens=max_new_tokens
      )
      return completion.choices[0].message.content
    except OpenAIError as e:
      print(f"Error calling OpenAI API: {e}")
      return f"죄송합니다. OpenAI API 호출 중 오류가 발생했습니다: {e}"

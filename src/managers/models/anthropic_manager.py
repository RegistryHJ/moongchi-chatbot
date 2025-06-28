import os
import anthropic

class AnthropicManager:
  """
  Anthropic API를 사용하여 텍스트 생성을 관리하는 클래스입니다.
  """

  def __init__(self, model_name: str = "claude-4-sonnet-20250514"):
    """
    Anthropic 클라이언트를 초기화합니다.
    API 키는 환경 변수 'ANTHROPIC_API_KEY'에서 가져옵니다.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
      raise ValueError("환경 변수에서 'ANTHROPIC_API_KEY'를 찾을 수 없습니다.")

    self.client = anthropic.AsyncAnthropic(api_key=api_key)
    self.model_name = model_name
    print(f"AnthropicManager initialized with model: {self.model_name}")

  async def generate(self, prompt: str, system_message: str, max_new_tokens: int = 1024):
    """
    주어진 프롬프트를 기반으로 텍스트를 생성합니다.
    """
    try:
      message = await self.client.messages.create(
          model=self.model_name,
          max_tokens=max_new_tokens,
          system=system_message,
          messages=[
              {"role": "user", "content": prompt}
          ]
      )
      return message.content[0].text
    except anthropic.APIError as e:
      print(f"Error calling Anthropic API: {e}")
      return f"죄송합니다. Anthropic API 호출 중 오류가 발생했습니다: {e}"

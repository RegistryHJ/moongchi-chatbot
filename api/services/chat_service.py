from typing import Optional, Dict, Any

from src.generators.prompt_generator import PromptGenerator

class ChatService:
  def __init__(self, prompt_generator: PromptGenerator):
    """
    의존성 주입을 통해 PromptGenerator를 받습니다.
    """
    self.prompt_generator = prompt_generator

  async def create_chat_response(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    사용자 메시지에 대한 챗봇 응답을 생성하는 비즈니스 로직을 수행합니다.
    """
    print(f"Service layer received message: '{message}' for session: {session_id}")

    # 주입받은 prompt_generator를 사용하여 응답 생성 로직을 호출합니다.
    response = await self.prompt_generator.generate_response(
        message=message,
        session_id=session_id
    )

    # 여기에 추가적인 비즈니스 로직(예: 로깅, 분석 데이터 저장 등)을 넣을 수 있습니다.
    print(f"Service layer generated response for session: {response['session_id']}")

    return response

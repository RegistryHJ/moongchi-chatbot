from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict

from api.services.chat_service import ChatService
from src.app import get_chat_service

# API 라우터 객체 생성
router = APIRouter()

# --- Pydantic 모델을 사용하여 요청 본문(Request Body)의 형식을 정의합니다 ---
class ChatController(BaseModel):
  message: str
  session_id: Optional[str] = None

# --- FastAPI의 Depends를 사용하여 서비스 객체를 주입받습니다 ---
@router.post("", response_model=Dict[str, Any])
async def handle_chat(
    request: ChatController,
    service: ChatService = Depends(get_chat_service)
):
  """
  POST /api/v1/chatbot 요청을 처리합니다.
  - 요청 본문을 파싱하여 ChatController 객체로 변환합니다.
  - ChatbotService를 주입받아 비즈니스 로직을 호출합니다.
  - 서비스의 결과를 클라이언트에게 JSON 형태로 반환합니다.
  """
  try:
    return await service.create_chat_response(
        message=request.message,
        session_id=request.session_id
    )
  except Exception as e:
    # 서비스 로직에서 예외가 발생할 경우, 500 에러를 반환합니다.
    print(f"Error in controller: {e}")
    raise HTTPException(status_code=500, detail="Internal Server Error")

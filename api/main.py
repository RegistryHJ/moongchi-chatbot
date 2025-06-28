from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.app import app_container
from api.controllers import chat_controller

# FastAPI 2.0 스타일의 lifespan 관리자: 앱의 시작과 종료 시점을 관리합니다.
@asynccontextmanager
async def lifespan(app: FastAPI):
  # 애플리케이션 시작 시
  print("API is starting up...")
  app_container.startup()
  yield  # 이 시점에서 애플리케이션이 실행됩니다.
  # 애플리케이션 종료 시
  print("API is shutting down...")
  app_container.shutdown()

# FastAPI 애플리케이션 객체 생성
app = FastAPI(
    lifespan=lifespan,
    title="MoongChi LLM Search Chatbot API",
    description="공동구매 상품 추천 및 Q&A를 위한 챗봇 API",
    version="1.0.0"
)

# ChatbotController의 라우터를 메인 앱에 포함시킵니다.
# /api/v1/chatbot 경로로 들어오는 모든 요청은 chatbot_controller가 처리합니다.
app.include_router(
    chatbot_controller.router,
    prefix="/api/v2/chat",
    tags=["Chatbot"]
)

# 루트 경로에 대한 간단한 엔드포인트
@app.get("/")
def read_root():
  return {"message": "Welcome to MoongChi LLM Search Chatbot API"}

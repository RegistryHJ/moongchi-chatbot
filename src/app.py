import os
from dotenv import load_dotenv

from src.managers.stores.opensearch_manager import OpenSearchManager
from src.managers.stores.mysql_manager import MySQLManager
from src.managers.models.openai_manager import OpenAIManager
from src.managers.models.anthropic_manager import AnthropicManager
from src.managers.models.exaone_manager import ExaoneManager
from src.generators.prompt_generator import PromptGenerator
from src.generators.embedding_generator import EmbeddingGenerator
from api.services.chat_service import ChatService

class ApplicationContainer:
  """
  애플리케이션의 모든 서비스와 매니저를 생성하고 주입하는 중앙 DI 컨테이너.
  """

  def __init__(self):
    """
    컨테이너 생성 시, 모든 객체를 초기화하고 의존성을 연결합니다.
    """
    load_dotenv()
    print("Initializing Application Container...")

    self.os_manager = OpenSearchManager()
    self.mysql_manager = MySQLManager()

    llm_provider = os.getenv("LLM_PROVIDER", "exaone").lower()
    opensearch_index = os.getenv("OPENSEARCH_INDEX")

    if llm_provider == "openai":
      self.llm_manager = OpenAIManager()
    elif llm_provider == "exaone":
      self.llm_manager = ExaoneManager()
    elif llm_provider == "anthropic":
      self.llm_manager = AnthropicManager()
    else:
      raise ValueError(f"지원하지 않는 LLM 제공자입니다: {llm_provider}")

    print(f"Selected LLM Provider: {llm_provider.upper()}")

    self.prompt_generator = PromptGenerator(
        os_manager=self.os_manager,
        llm_manager=self.llm_manager
    )

    self.embedding_generator = EmbeddingGenerator(
        mysql_manager=self.mysql_manager,
        os_manager=self.os_manager,
        opensearch_index=opensearch_index
    )

    self.chat_service = ChatService(
        prompt_generator=self.prompt_generator
    )

  def startup(self):
    """
    애플리케이션 시작 시 수행해야 할 작업을 관리합니다.
    """
    print("Container is starting up services...")
    self.os_manager.connect()
    self.mysql_manager.connect()

    self.prompt_generator.load_embedding_model()

    if hasattr(self.llm_manager, 'load_model'):
      self.llm_manager.load_model()

    print("Application container started up successfully.")

  def shutdown(self):
    """
    애플리케이션 종료 시 수행해야 할 작업을 관리합니다.
    """
    print("Container is shutting down services...")
    self.os_manager.disconnect()
    self.mysql_manager.disconnect()
    print("Application container shut down.")


app_container = ApplicationContainer()

def get_chat_service() -> ChatService:
  """ChatbotService의 인스턴스를 반환하는 의존성 주입용 함수입니다."""
  return app_container.chat_service

import os
from dotenv import load_dotenv
from src.managers.stores.mysql_manager import MySQLManager
from src.managers.stores.opensearch_manager import OpenSearchManager
from src.generators.embedding_generator import EmbeddingGenerator

# .env 파일에서 환경 변수 로드
load_dotenv()

if __name__ == '__main__':

  print("--- Embedding Generator Integration Test ---")
  print("Make sure your .env file is correctly configured.\n")

  # 리소스 정리를 위해 매니저 변수를 미리 초기화합니다.
  mysql_manager = None
  os_manager = None

  try:
    # 1. 필수 환경 변수 확인
    opensearch_index_name = os.getenv("OPENSEARCH_INDEX")
    if not opensearch_index_name:
      raise ValueError("OPENSEARCH_INDEX must be set in your .env file.")

    # 2. 모든 매니저 인스턴스화
    mysql_manager = MySQLManager()
    os_manager = OpenSearchManager()

    # 3. EmbeddingGenerator에 실제 매니저와 인덱스 이름 주입
    generator = EmbeddingGenerator(
        mysql_manager=mysql_manager,
        os_manager=os_manager,
        opensearch_index=opensearch_index_name
    )

    # 4. 서비스 연결 및 모델 로딩
    print("Step 1: Connecting to services and loading embedding model...")
    mysql_manager.connect()
    os_manager.connect()
    generator.load_embedding_model()
    print("...Services connected and model loaded successfully.\n")

    # 5. 실제 임베딩 생성 및 인덱싱 작업 실행
    print("Step 2: Starting the embedding generation and indexing process...")
    generator.generate_and_index_embeddings()
    print("\n...Embedding process completed.")

  except Exception as e:
    print(f"\nAN ERROR OCCURRED: {e}")

  finally:
    # 6. 모든 외부 연결을 안전하게 종료
    print("\nStep 3: Cleaning up all connections...")
    if mysql_manager and mysql_manager.conn:
      mysql_manager.disconnect()
    if os_manager and os_manager.client:
      os_manager.disconnect()
    print("--- Test Finished ---")

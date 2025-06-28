import os
from dotenv import load_dotenv
from src.managers.stores.opensearch_manager import OpenSearchManager

load_dotenv()

if __name__ == '__main__':
  print("--- OpenSearch Manager Test ---")

  manager = None
  try:
    # 1. OpenSearchManager 인스턴스 생성 및 연결
    manager = OpenSearchManager()
    print("Attempting to connect to OpenSearch cluster...")
    # manager의 connect()가 호출되면서 강화된 환경 변수 체크 실행
    manager.connect()

    # 2. 기본 인덱스에 대한 간단한 검색 실행
    search_query = {
        "size": 1,
        "query": {"match_all": {}}
    }
    print(f"\nExecuting a simple search on index: '{manager.index_name}'")
    result = manager.search(query_body=search_query)

    # 3. 결과 확인 및 출력
    if result is not None:
      print("Search query executed successfully!")
      if result:
        print(f"Found {len(result)} document(s). Sample result:")
        print(result)
      else:
        print("The query ran, but no documents were found in the index.")
    else:
      print("Search query failed.")

  except Exception as e:
    # 강화된 Manager 덕분에 환경 변수 문제가 여기서 명확하게 출력됩니다.
    print(f"\nAn error occurred during the test: {e}")
  finally:
    # 4. 연결 해제
    if manager and manager.client:
      manager.disconnect()
    print("\n--- Test Finished ---")

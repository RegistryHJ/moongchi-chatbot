import os
from dotenv import load_dotenv
from src.managers.stores.mysql_manager import MySQLManager

# .env 파일에서 환경 변수 로드
load_dotenv()

if __name__ == '__main__':
  print("--- MySQL Manager Test ---")

  if not os.getenv('MYSQL_HOST'):
    print("Error: MySQL environment variables not found. Make sure .env file is set up.")
  else:
    manager = MySQLManager()
    try:
      # 쿼리 실행 시 별칭(alias)에 백틱(`)을 사용하여 명확하게 지정
      query = "SELECT NOW() as `current_time`"
      print(f"Executing a simple query: '{query}'")

      result = manager.execute_query(query)

      if result:
        print("Query successful!")
        print("Result:", result)
      else:
        print("Query failed.")

    except Exception as e:
      print(f"An error occurred during the test: {e}")
    finally:
      if manager.conn:
        manager.disconnect()
        print("--- Test Finished ---")

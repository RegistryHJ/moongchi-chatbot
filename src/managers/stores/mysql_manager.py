import os
import pymysql
from pymysql.cursors import DictCursor

class MySQLManager:
  """
  pymysql과 python-dotenv를 사용하여 MySQL 데이터베이스에 연결하고 관리하는 클래스.
  """

  def __init__(self):
    """환경 변수와 연결 상태 변수만 초기화합니다."""
    self.host = os.getenv('MYSQL_HOST')
    self.port = int(os.getenv('MYSQL_PORT', 3306))
    self.user = os.getenv('MYSQL_USERNAME')
    self.password = os.getenv('MYSQL_PASSWORD')
    self.db = os.getenv('MYSQL_DATABASE')
    self.conn = None
    self.cursor = None

  def connect(self):
    """
    환경 변수에서 직접 접속 정보를 읽어와 데이터베이스에 연결합니다.
    이미 연결된 경우, 새로운 연결을 만들지 않습니다.
    """
    if self.conn:
      return

    try:
      # connect 메서드 내에서 환경 변수 로드
      host = self.host
      port = self.port
      user = self.user
      password = self.password
      db = self.db

      # 필수 환경 변수가 설정되었는지 확인
      if not all([host, user, password, db]):
        raise ValueError("Database connection information is missing in environment variables.")

      self.conn = pymysql.connect(
          host=host,
          port=port,
          user=user,
          password=password,
          db=db,
          charset='utf8mb4',  # 이모지 등을 지원하기 위해 utf8mb4 사용 권장
          cursorclass=DictCursor
      )
      self.cursor = self.conn.cursor()
      print("Database connected successfully.")
    except (pymysql.MySQLError, ValueError) as e:
      print(f"Error connecting to MySQL database: {e}")
      # 연결 실패 시 상태를 확실히 None으로 유지
      self.conn = None
      self.cursor = None
      raise  # 예외를 다시 발생시켜 호출자에게 실패를 알림

  def disconnect(self):
    """데이터베이스 연결을 닫습니다."""
    if self.conn:
      self.conn.close()
      self.conn = None
      self.cursor = None
      print("Database disconnected.")

  def _ensure_connected(self):
    """연결이 되어있는지 확인하고, 안 되어있으면 연결합니다."""
    if not self.conn:
      self.connect()

  def execute_query(self, sql, args=None):
    """SELECT 쿼리를 실행하고 결과를 반환합니다."""
    self._ensure_connected()
    try:
      self.cursor.execute(sql, args)
      return self.cursor.fetchall()
    except pymysql.MySQLError as e:
      print(f"Query execution failed: {e}")
      return None

  def execute_update(self, sql, args=None):
    """INSERT, UPDATE, DELETE 쿼리를 실행하고 커밋합니다."""
    self._ensure_connected()
    try:
      self.cursor.execute(sql, args)
      self.conn.commit()
      return self.cursor.lastrowid
    except pymysql.MySQLError as e:
      print(f"Update execution failed: {e}")
      self.conn.rollback()
      return None

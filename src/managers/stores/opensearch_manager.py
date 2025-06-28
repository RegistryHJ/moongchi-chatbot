import os
from opensearchpy import OpenSearch, exceptions

class OpenSearchManager:
  """
  opensearch-py와 python-dotenv를 사용하여 OpenSearch 클러스터에 연결하고 관리하는 클래스.
  기본 인덱스 이름을 환경 변수에서 관리할 수 있습니다.
  """

  def __init__(self):
    """환경 변수와 클라이언트 상태 변수만 초기화합니다."""
    self.host = os.getenv('OPENSEARCH_HOST')
    self.port = int(os.getenv('OPENSEARCH_PORT', 9200))
    self.user = os.getenv('OPENSEARCH_USERNAME')
    self.password = os.getenv('OPENSEARCH_PASSWORD')
    self.index_name = os.getenv('OPENSEARCH_INDEX')  # 기본 인덱스 이름 추가
    self.client = None

  def connect(self):
    """
    환경 변수에서 직접 접속 정보를 읽어와 클러스터에 연결합니다.
    이미 연결된 경우, 새로운 연결을 만들지 않습니다.
    """
    if self.client:
      return

    try:
      if not all([self.host, self.user, self.password]):
        raise ValueError("OpenSearch connection information is missing in environment variables.")

      self.client = OpenSearch(
          hosts=[{'host': self.host, 'port': self.port}],
          http_auth=(self.user, self.password),
          use_ssl=True,
          verify_certs=False,
          ssl_assert_hostname=False,
          ssl_show_warn=False,
      )

      if not self.client.ping():
        raise exceptions.ConnectionError("Ping to OpenSearch cluster failed.")

      print("OpenSearch connected successfully.")
    except (ValueError, exceptions.OpenSearchException) as e:
      print(f"Error connecting to OpenSearch: {e}")
      self.client = None
      raise

  def disconnect(self):
    """OpenSearch 클라이언트 연결을 닫습니다."""
    if self.client:
      self.client.close()
      self.client = None
      print("OpenSearch connection closed.")

  def _ensure_connected(self):
    """연결이 되어있는지 확인하고, 안 되어있으면 연결합니다."""
    if not self.client:
      self.connect()

  def _get_index_name(self, index_name=None):
    """사용할 인덱스 이름을 결정합니다. 인자가 없으면 환경 변수 값을 사용합니다."""
    final_index = index_name or self.index_name
    if not final_index:
      raise ValueError("Index name must be provided as an argument or set via OPENSEARCH_INDEX environment variable.")
    return final_index

  def search(self, query_body, index_name=None):
    """
    지정된 쿼리(Query DSL)를 사용하여 문서를 검색하고 결과를 반환합니다.
    index_name이 없으면 환경 변수의 기본 인덱스를 사용합니다.
    """
    self._ensure_connected()
    target_index = self._get_index_name(index_name)
    try:
      response = self.client.search(
          index=target_index,
          body=query_body
      )
      return response['hits']['hits']
    except exceptions.OpenSearchException as e:
      print(f"Search failed in index '{target_index}': {e}")
      return []

  def index_document(self, document, doc_id=None, index_name=None):
    """
    문서를 인덱싱합니다.
    index_name이 없으면 환경 변수의 기본 인덱스를 사용합니다.
    """
    self._ensure_connected()
    target_index = self._get_index_name(index_name)
    try:
      response = self.client.index(
          index=target_index,
          body=document,
          id=doc_id,
          refresh=True
      )
      return response['_id']
    except exceptions.OpenSearchException as e:
      print(f"Failed to index document in '{target_index}': {e}")
      return None

  def update_document(self, doc_id, partial_document, index_name=None):
    """
    기존 문서의 특정 필드만 부분적으로 업데이트합니다.
    index_name이 없으면 환경 변수의 기본 인덱스를 사용합니다.
    """
    self._ensure_connected()
    target_index = self._get_index_name(index_name)
    try:
      response = self.client.update(
          index=target_index,
          id=doc_id,
          body={'doc': partial_document},
          refresh=True
      )
      print(f"Document '{doc_id}' in index '{target_index}' updated successfully.")
      return response['result']
    except exceptions.NotFoundError:
      print(f"Document with ID '{doc_id}' not found in index '{target_index}' for update.")
      return None
    except exceptions.OpenSearchException as e:
      print(f"Failed to update document '{doc_id}' in index '{target_index}': {e}")
      return None

  def delete_document(self, doc_id, index_name=None):
    """
    ID로 특정 문서를 삭제합니다.
    index_name이 없으면 환경 변수의 기본 인덱스를 사용합니다.
    """
    self._ensure_connected()
    target_index = self._get_index_name(index_name)
    try:
      response = self.client.delete(index=target_index, id=doc_id, refresh=True)
      return response['result'] == 'deleted'
    except exceptions.NotFoundError:
      print(f"Document with ID '{doc_id}' not found in index '{target_index}' for deletion.")
      return False
    except exceptions.OpenSearchException as e:
      print(f"Failed to delete document '{doc_id}' in index '{target_index}': {e}")
      return False

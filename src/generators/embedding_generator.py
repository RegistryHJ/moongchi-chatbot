import torch
from opensearchpy.helpers import bulk
from opensearchpy import exceptions
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
  """
  MySQL에서 데이터를 가져와 임베딩을 생성하고 OpenSearch에 저장하는 클래스.
  """

  _PRODUCT_INDEX_SCHEMA = {
      "mappings": {
          "properties": {
              "product_id": {"type": "integer"},
              "product_name": {"type": "text", "analyzer": "nori"},
              "price": {"type": "integer"},
              "product_url": {"type": "keyword"},
              "image_url": {"type": "keyword"},
              "category_path": {"type": "text", "analyzer": "nori"},
              "embedding": {
                  "type": "dense_vector",
                  "dims": 768,
                  "index": True,
                  "similarity": "cosine"
              }
          }
      }
  }

  def __init__(self, mysql_manager, os_manager, opensearch_index):
    """
    MySQL 및 OpenSearch 매니저만 주입받고, 나머지 설정은 모두 내부에서 정의합니다.
    """
    print("Initializing EmbeddingGenerator with fixed, hard-coded settings...")
    self.mysql_manager = mysql_manager
    self.os_manager = os_manager

    # 환경 변수 대신 고정된 값으로 직접 정의
    self.model_name = "jhgan/ko-sbert-nli"
    self.opensearch_index = opensearch_index

    # 모델 관련 속성 초기화
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = None  # 모델은 아직 로드하지 않음

  def load_embedding_model(self):
    """
    무거운 임베딩 모델을 메모리에 로드합니다.
    애플리케이션 시작 시 한 번만 호출되는 것을 권장합니다.
    """
    if self.model is None:
      print(f"Loading model '{self.model_name}' to {self.device}...")
      self.model = SentenceTransformer(self.model_name, device=self.device)
      print("Embedding model loaded successfully.")

  def ensure_index_exists(self):
    """자신의 작업에 필요한 인덱스가 존재하는지 확인하고, 없으면 생성합니다."""
    client = self.os_manager.client
    if not client.indices.exists(index=self.opensearch_index):
      print(f"Index '{self.opensearch_index}' not found. Creating...")
      try:
        client.indices.create(index=self.opensearch_index, body=self._PRODUCT_INDEX_SCHEMA)
        print(f"Index '{self.opensearch_index}' created successfully.")
      except exceptions.RequestError as e:
        # 이미 생성된 경우의 경쟁 상태(race condition)를 처리
        if 'resource_already_exists_exception' not in str(e):
          raise e
    else:
      print(f"Index '{self.opensearch_index}' already exists.")

  def _fetch_product_data(self):
    """주입받은 MySQL 매니저를 사용해 상품 데이터를 가져옵니다."""
    print("Fetching product data from MySQL...")
    query = """
        SELECT 
          p.product_id, p.name AS product_name, p.price, p.product_url, p.img_url,
          c.large_category, c.medium_category, c.small_category
        FROM products p JOIN categories c ON p.category_id = c.category_id
    """
    return self.mysql_manager.execute_query(query)

  def _create_embedding(self, text: str):
    """내부적으로 로드된 모델을 사용하여 임베딩을 생성합니다."""
    if self.model is None:
      raise RuntimeError("Embedding model is not loaded. Call load_embedding_model() first.")

    embedding_vector = self.model.encode(text)
    return embedding_vector.tolist()

  def generate_and_index_embeddings(self):
    """임베딩 생성 및 인덱싱의 전체 프로세스를 실행합니다."""
    self.ensure_index_exists()

    products = self._fetch_product_data()
    if not products:
      print("No products to process. Exiting.")
      return

    actions = []
    print(f"Generating embeddings with fixed model '{self.model_name}'...")

    for product in tqdm(products, desc="Processing products"):
      category_parts = [product['large_category'], product['medium_category'], product['small_category']]
      category_path = " > ".join(filter(None, category_parts))
      text_to_embed = f"상품명: {product['product_name']}, 카테고리: {category_path}"

      try:
        embedding_vector = self._create_embedding(text_to_embed)
        document = {
            "product_id": product['product_id'], "product_name": product['product_name'],
            "price": product['price'], "product_url": product['product_url'],
            "image_url": product['image_url'], "category_path": category_path,
            "embedding": embedding_vector
        }
        action = {"_index": self.opensearch_index, "_id": product['product_id'], "_source": document}
        actions.append(action)
      except Exception as e:
        print(f"Error processing product_id {product['product_id']}: {e}")

    if actions:
      print(f"\nBulk indexing {len(actions)} documents to '{self.opensearch_index}'...")
      success, failed = bulk(self.os_manager.client, actions, raise_on_error=False)
      print(f"Indexing complete. Success: {success}, Failed: {len(failed)}")
      if failed:
        print("Failed documents (sample):", failed[:5])

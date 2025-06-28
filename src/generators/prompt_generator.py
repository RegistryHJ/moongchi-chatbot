import uuid
import torch
import asyncio
import textwrap
from datetime import datetime
from typing import Optional, Dict, Any, List
from sentence_transformers import SentenceTransformer

class PromptGenerator:
  """
  OpenSearch 검색 및 LLM 생성을 통해 RAG 답변을 생성하는 클래스.
  """

  def __init__(self, os_manager, llm_manager):
    """
    OpenSearch 매니저와 범용 LLM 매니저를 주입받습니다.
    임베딩 모델 관련 설정은 내부적으로 처리합니다.
    """
    print("Initializing PromptGenerator...")
    self.os_manager = os_manager
    self.llm_manager = llm_manager  # 모든 텍스트 생성을 책임질 범용 LLM 매니저

    # 임베딩 모델 관련 설정 (내부 책임)
    self.embedding_model_name = "jhgan/ko-sbert-nli"
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.embedding_model = None

    self.sessions: Dict[str, List[Dict[str, str]]] = {}

  def load_embedding_model(self):
    """내부에서 사용할 임베딩 모델을 메모리에 로드합니다."""
    if self.embedding_model is None:
      print(f"Loading embedding model '{self.embedding_model_name}' to {self.device}...")
      self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
      print("Chatbot embedding model loaded successfully.")

  def _get_embedding(self, text: str) -> List[float]:
    """내부적으로 로드된 모델을 사용하여 임베딩 벡터를 생성합니다."""
    if self.embedding_model is None:
      raise RuntimeError("Embedding model is not loaded. Call load_embedding_model() first.")
    return self.embedding_model.encode(text).tolist()

  async def _generate_product_description(self, product_name: str) -> str:
    """주입된 LLM 매니저를 사용하여 'Ai 뭉치'의 설명을 생성합니다."""
    system_prompt = textwrap.dedent("""
        당신은 'Ai 뭉치'라는 이름의 공동구매 전문 쇼핑 큐레이터입니다.
        당신의 임무는 주어진 상품명에 대해, 상품 설명을 매력적이고 친근한 설명을 1문장으로 생성하는 것입니다.
        항상 밝고 긍정적인 톤을 유지하고, 이 상품을 왜 사야 하는지 핵심 장점을 부각시켜주세요. 인삿말은 안해도 됩니다.
    """).strip()

    try:
      return await self.llm_manager.generate(
          prompt=f"상품명: '{product_name}'", system_prompt=system_prompt
      )
    except Exception as e:
      print(f"Error generating description for '{product_name}': {e}")
      return f"{product_name}의 상품 설명을 생성하지 못하였습니다."

  async def _search_similar_products(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    query_vector = self._get_embedding(query)
    query_body = {"size": top_k, "query": {"knn": {"embedding": {"vector": query_vector, "k": top_k}}}}
    try:
      response = self.os_manager.client.search(index=self.os_manager.index_name, body=query_body)
      hits = response.get('hits', {}).get('hits', [])
      products_from_search = [
          {
              "product_id": hit['_source'].get('product_id'), "name": hit['_source'].get('product_name'),
              "price": hit['_source'].get('price'), "category_name": hit['_source'].get('category_path'),
              "img_url": hit['_source'].get('image_url'), "product_url": hit['_source'].get('product_url'),
              "similarity": round(hit.get('_score', 0.0) * 100, 2)
          } for hit in hits
      ]
      description_tasks = [self._generate_product_description(p['name']) for p in products_from_search]
      generated_descriptions = await asyncio.gather(*description_tasks)
      for product, description in zip(products_from_search, generated_descriptions):
        product['description'] = description
      return products_from_search
    except Exception as e:
      print(f"Error in search or description generation: {e}")
      return []

  def _format_recommendation_response(self, query: str, products: List[Dict[str, Any]]) -> str:
    if not products:
      return "죄송해요, 리더님. 지금은 딱 맞는 상품을 찾지 못했어요. 다른 키워드로 질문해주시겠어요?"
    conversational_keywords = [
        # 욕구/희망 표현
        '싶어', '싶다', '원해', '땡겨', '땡긴다',
        # 요청/질문 표현
        '추천', '알려줘', '찾아줘', '어때', '궁금', '뭐가', '어떤', '어떻게',
        # 일반적인 동사/서술어
        '먹고', '하고', '쓰는', '쓸만한', '입을', '바를', '좋아', '필요해',
        # 구어체 표현
        '좀', '같은', '같은거', '있을까', '있나요'
    ]
    is_conversational = len(query) > 10 or any(kw in query for kw in conversational_keywords)

    if is_conversational:
      response_lines = [f"리더님의 요청에 딱 맞는 상품을 찾아봤어요! Ai 뭉치가 몇 가지 추천해 드릴게요!", "---", "### 추천 상품"]
    else:
      response_lines = [f"'{query}'에 대한 상품을 찾으시는군요! Ai 뭉치가 몇 가지 추천해 드릴게요!", "---", "### 추천 상품"]
    wrapper = textwrap.TextWrapper(width=60, subsequent_indent="  ")
    for i, p in enumerate(products, 1):
      response_lines.extend([f"{i}. [{p['name']}]({p['product_url']})", f"- **가격**: {p['price']:,}원", f"- **유사도**: {p['similarity']:.2f}%"])
      wrapped_description = wrapper.fill(text=p['description'])
      response_lines.append(f"- **Ai 뭉치 설명**: {wrapped_description}")
    return "\n".join(response_lines)

  def _is_new_topic_question(self, message: str, history: List[Dict[str, str]]) -> bool:
    if not history:
      return True
    related_keywords = ["사용법", "사용방법", "어떻게써", "어떻게사용", "방법알려줘", "매뉴얼", "설명서", "가이드", "상세정보", "자세히", "스펙", "사양", "성분", "재질", "구성품", "크기", "무게", "용량", "색상", "종류", "차이점", "비교", "다른점", "뭐가달라", "장단점", "호환", "같이쓸수", "전용", "필요한거", "조건", "어디서사", "구매처", "궁금", "질문", "문의", "알려줘", "알려주세요", "추천해준것", "아까말한", "방금그상품"]
    processed_message = message.lower().replace(" ", "")
    for kw in related_keywords:
      if kw in processed_message:
        return False
    return True

  async def _generate_follow_up_response(self, query: str, history: List[Dict[str, str]]) -> str:
    """후속 질문에 대한 답변도 주입된 LLM 매니저로 생성합니다."""
    system_prompt = "당신은 'Ai 뭉치'라는 이름의 친절한 쇼핑 도우미입니다. 이전 대화 내용을 참고하여 사용자의 질문에 상세히 답변해주세요."
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    full_prompt = f"이전 대화:\n{history_str}\n\n사용자 질문: {query}"
    return await self.llm_manager.generate(prompt=full_prompt, system_prompt=system_prompt)

  async def generate_response(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    if not session_id or session_id not in self.sessions:
      session_id = str(uuid.uuid4())
      self.sessions[session_id] = []
    history = self.sessions[session_id]
    if self._is_new_topic_question(message, history):
      products = await self._search_similar_products(message)
      bot_response = self._format_recommendation_response(message, products)
      recommended_products = [{k: v for k, v in p.items() if k not in ['similarity', 'description']} for p in products]
    else:
      bot_response = await self._generate_follow_up_response(message, history)
      recommended_products = []
    history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": bot_response}])
    return {"success": True, "bot_response": bot_response, "recommended_products": recommended_products, "session_id": session_id, "message_count": len(history), "timestamp": datetime.now().isoformat()}

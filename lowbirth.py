import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # 출력 깔끔하게 하기 위해 추가

# 1. 환경 변수 로드
load_dotenv(override=True)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# 2. LLM 및 프롬프트 설정
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 조수입니다. 제공된 context의 내용만을 바탕으로 답변해주세요."),
    ("human", "질문: {question}\n\n참고 내용(context): {context}"),
])

# 3. 검색 함수 정의
def search_top_k(question_text):
    # Pinecone 연결
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("lowbirth") # index 객체 정의 필수!

    # 엠베딩 설정 (질문 하나이므로 embed_query 사용이 효율적)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    embedded_question = embedding.embed_query(question_text) # 리스트가 아닌 문자열 전달

    # 유사도 검색
    query_result = index.query(
        namespace="lowbirth_1",
        vector=embedded_question,
        top_k=3,
        include_metadata=True
    )

    # 검색된 결과 결합
    context_list = []
    for match in query_result.matches:
        if "chunk_text" in match.metadata:
            context_list.append(match.metadata["chunk_text"])
    
    return "\n\n".join(context_list)

# 4. 메인 실행부 (RAG 파이프라인)
if __name__ == "__main__":
    # 터미널 입력 받기
    user_question = input("질문을 입력해주세요: ")

    # 1단계: 관련 문맥 검색 (Retrieval)
    print("관련 내용을 검색 중입니다...")
    top_k_context = search_top_k(user_question)

    # 2단계: 답변 생성 (Generation)
    # StrOutputParser를 추가하면 response.content 대신 바로 문자열을 얻습니다.
    chain = chat_template | llm | StrOutputParser()
    
    print("\n--- 답변 중 ---")
    response = chain.invoke({
        "question": user_question,
        "context": top_k_context
    })

    print(response)
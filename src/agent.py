import streamlit as st
import os
from torch import argsort
from langchain.load import loads, dumps
from sentence_transformers import SentenceTransformer
from seed_data import get_retriever
from langchain.schema import Document as LC_Document
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI
# from langchain.tools.retriever import create_retriever_tool
# from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.prompts import  PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCekUE-sNiAc_Jw-TFaLO11Xn18lLc-Lkw"
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

@st.cache_resource
def get_gemini_llm() -> GoogleGenerativeAI:
    return GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY, temperature=0)

@st.cache_resource
def get_encoder_model(model_name: str ="hiieu/halong_embedding") -> SentenceTransformer:
    return SentenceTransformer(model_name)

def fusion_ranker(query: str, vectorstore: PineconeVectorStore) -> list:
    # Gọi llm
    llm = get_gemini_llm()
    #Gọi retriever
    retriever = get_retriever(index_name="new-documents-hybrid")
    retriever.top_k = 15
    #Prompting để tạo ra 3 câu query liên quan từ câu query đầu vào (Multi-query)
    system_template = """
        Bạn là một chuyên gia tạo ra nhiều câu hỏi liên quan từ câu query đầu vào của người dùng. 
        Trong mỗi câu output đừng giải thích gì thêm cả, chỉ cần tạo ra câu hỏi liên quan từ câu query đầu vào.
        Hãy tạo ra 3 câu query liên quan tới query sau: "{query}"
        Output trả về sẽ là 3 câu query liên quan và câu query đầu vào.
        Output:
        ...
        ...
        ...
        ...
    """
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=system_template,
    )
    prompt = prompt_template.format(query=query)
    get_response = llm(prompt)
    
    #Xử lý list các câu query trả về từ llm
    list_query = [line.strip() for line in get_response.split("\n") if line.strip()]
    retrieved_list = [retriever.invoke(query) for query in list_query]
    
    # Dùng RMM để rerank các docs được trả về từ retriever theo score
    lst=[]
    for ddxs in retrieved_list:
        for ddx in ddxs:
            if ddx.page_content not in lst:
                lst.append(ddx.page_content)
                
    fused_scores = {}
    k=60
    for docs in retrieved_list:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ][:5]
    return reranked_results

def original_ranker(query: str, ranker: SentenceTransformer, vectorstore: PineconeVectorStore) -> list:
    # Goi retriever
    retriever = get_retriever(index_name="new-documents-hybrid")
    retriever.top_k = 15
    retrieved_docs = retriever.invoke(query) # Trả về list các LC_Document

    # Tính similarity giữa câu query và các docs được trả về
    docs_list = [doc.page_content for doc in retrieved_docs]
    embedded_query = ranker.encode([query])
    embedded_sentences = ranker.encode(docs_list)
    similarities = ranker.similarity(embedded_query, embedded_sentences).flatten()
    
    # Sắp xếp các docs theo thứ tự similarity và chỉ lấy top 5
    sorted_indices = argsort(similarities, descending=True)
    ranked_result = [retrieved_docs[i] for i in sorted_indices][:5]
    return ranked_result

def get_router(query: str, llm) -> str: # Vì chưa sử dụng được Agent nên sẽ tạm định nghĩa router ở đây
    system_template = """
        Bạn là một chuyên gia về phân loại câu hỏi.
        Công việc của bạn là phân loại câu hỏi xem câu hỏi đưa và có phải là một câu hỏi về luật giao thông hay không.
        Đối với những câu hỏi như "bạn là chatbot về luật gì?", hay "bạn hỗ trợ người dùng những vấn đề nào?" thì hãy trả lời là chatbot về luật giao thông.
        Nếu câu hỏi là một câu hỏi về luật giao thông thì hãy phản hồi "yes"
        Nếu là những câu hỏi về chào hỏi thông thường thì phản hồi "no"
        Với những câu hỏi dùng để hỏi lại lịch sử trò chuyện trước đó mà liên quan đến yếu tố luật giao thông thì phản hồi "yes"
        Với những câu hỏi dùng để hỏi lại lịch sử trò chuyện trước đó mà không liên quan đến luật giao thông thì cũng quy vào normal chatting và phản hổi "no"
        Nếu là những câu hỏi về các vấn đề khác không phải luật giao thông và những luật không phải là luật giao thông thì phản hồi "fail"
        Dưới đây là query của người dùng:
        <query>
        {query}
        </query>
    """
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=system_template,
    )
    prompt = prompt_template.format(query=query)
    response = llm(prompt).strip()
    return response

def legal_response(query: str, llm: any, context: list[LC_Document], history: any) -> str:
    examples = [
        {
            "input": "Người đi bộ có được phép băng qua đường tại nơi không có vạch kẻ đường không?",
            "output": "Người đi bộ chỉ được phép băng qua đường tại các vị trí có vạch kẻ đường hoặc nơi có tín hiệu giao thông cho phép. Nếu không có vạch kẻ đường, người đi bộ phải đảm bảo an toàn và không gây cản trở giao thông.",
        },
        {
            "input": "Tốc độ tối đa được phép chạy trong khu dân cư là bao nhiêu?",
            "output": "Không thể trả lời câu hỏi này.",
        },
    ]
    example_template = PromptTemplate(
        input_variables=["input", "output"],
        template="Human: {input}\nAI: {output}\n",
    )
    
    prefix = """
        # Bạn là một chuyên gia về luật giao thông

        Nhiệm vụ của bạn là cung cấp câu trả lời câu hỏi của người dùng thông qua context được truyền vào.
        Nếu trong context không bao gồm nội dung nào liên quan để có thể trả lời được câu hỏi thì hãy trả lời "Không thể trả lời câu hỏi này." và không nói gì thêm.
        Tuyệt đối chỉ được dùng thông tin trong context để trả lời câu hỏi.
        
        Dưới đây context được cung cấp để bạn trả lời câu hỏi:   
        <context>
        {context}
        </context>
    """
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix=prefix,
        suffix="Human: {question}\nAI:",
        input_variables=["question", "context"],
    )
    context = [ctx.page_content for ctx in context]
    final_prompt = few_shot_prompt.format(question = query, context= context)
    
    response = llm(final_prompt)
    return response

def normal_response(query: str, llm: GoogleGenerativeAI, history) -> str:
    # Định nghĩa template
    template = ChatPromptTemplate([
        ("system", "Bạn là một chatbot trả lời những câu hỏi về normal chatting"),
        ("human", "{query}"),
    ])
    
    # Tạo prompt từ template và thay thế placeholder
    prompt_value = template.invoke({"query": query})  # Trả về ChatPromptValue
    prompt = prompt_value.to_string()  # Chuyển đổi thành chuỗi
    
    # Gửi prompt đến LLM và nhận phản hồi
    response = llm(prompt)
    return response

# prompt = "Người đi bộ có được phép băng qua đường tại nơi không có vạch kẻ đường không?"
# llm = get_gemini_llm()
# retriever = get_retriever("new-documents-hybrid")
# encode_model = get_encoder_model()
# result = original_ranker(prompt, encode_model, retriever)
# print(result)
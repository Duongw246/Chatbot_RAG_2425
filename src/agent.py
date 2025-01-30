import streamlit as st
import os
import torch
from seed_data import get_retriever
from torch import argsort
from langchain.load import loads, dumps
from sentence_transformers import SentenceTransformer
from langchain.schema import Document as LC_Document
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI
import vertexai
from vertexai.generative_models import GenerativeModel
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
PROJECT_ID = st.secrets["MY_PROJECT_ID"]
if not PROJECT_ID:
    raise ValueError("PROJECT_ID not found in environment variables")

vertexai.init(project=PROJECT_ID, location="us-central1") 

@st.cache_resource
def get_gemini_llm() -> GoogleGenerativeAI:
    return GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY, temperature=0)

@st.cache_resource
def get_vertex_llm() -> GenerativeModel:
    return GenerativeModel("gemini-1.5-pro")

# @st.cache_resource
# def get_encoder_model(model_name: str ="BAAI/bge-m3") -> SentenceTransformer:
#     return SentenceTransformer(model_name)

# def fusion_ranker(query: str, vectorstore: PineconeVectorStore, llm: GenerativeModel) -> list:
#     # Gọi llm
#     # llm = get_gemini_llm()
#     # llm = get_vertex_llm()
#     #Gọi retriever
#     retriever = get_retriever(index_name="new-documents-hybrid")
#     retriever.top_k = 15
#     #Prompting để tạo ra 3 câu query liên quan từ câu query đầu vào (Multi-query)
#     system_template = """
#         Bạn là một chuyên gia tạo ra nhiều câu hỏi liên quan từ câu query đầu vào của người dùng. 
#         Trong mỗi câu output đừng giải thích gì thêm cả, chỉ cần tạo ra câu hỏi liên quan từ câu query đầu vào.
#         Hãy tạo ra 3 câu query liên quan tới query sau: "{query}"
#         Output trả về sẽ là 3 câu query liên quan và câu query đầu vào.
#         Output:
#         ...
#         ...
#         ...
#         ...
#     """
#     prompt_template = PromptTemplate(
#         input_variables=["query"],
#         template=system_template,
#     )
#     prompt = prompt_template.format(query=query)
#     # get_response = llm(prompt)
#     get_response = llm.generate_content(prompt)
#     #Xử lý list các câu query trả về từ llm
#     list_query = [line.strip() for line in get_response.split("\n") if line.strip()]
#     retrieved_list = [original_ranker(query, get_encoder_model, retriever) for query in list_query]
    
#     # Dùng RMM để rerank các docs được trả về từ retriever theo score
#     lst=[]
#     for ddxs in retrieved_list:
#         for ddx in ddxs:
#             if ddx.page_content not in lst:
#                 lst.append(ddx.page_content)
                
#     fused_scores = {}
#     k=60
#     for docs in retrieved_list:
#         for rank, doc in enumerate(docs):
#             doc_str = dumps(doc)
#             if doc_str not in fused_scores:
#                 fused_scores[doc_str] = 0
#             fused_scores[doc_str] += 1 / (rank + k)

#     reranked_results = [
#         (loads(doc), score)
#         for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
#     ][:5]
#     return reranked_results

# def original_ranker(query: str, ranker: SentenceTransformer, index_name: str) -> list:
#     #GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     #Ranker
#     ranker = ranker.to(device)
    
#     # Goi retriever
#     retriever = get_retriever(index_name=index_name)
#     retriever.top_k = 15
#     retrieved_docs = retriever.invoke(query) # Trả về list các LC_Document

#     # Tính similarity giữa câu query và các docs được trả về
#     docs_list = [doc.page_content for doc in retrieved_docs]
#     embedded_query = ranker.encode([query], convert_to_tensor=True, device=device)
#     embedded_sentences = ranker.encode(docs_list, convert_to_tensor=True, device=device)
#     similarities = (ranker.similarity(embedded_query, embedded_sentences).flatten())
#     # Sắp xếp các docs theo thứ tự similarity và chỉ lấy top 5
#     sorted_indices = argsort(similarities, descending=True)
#     ranked_result = [retrieved_docs[i] for i in sorted_indices][:5]
#     return ranked_result
def query_transform(query: str, model_choice) -> str:
    system_template = """
    Bạn là một chuyên gia về việc chuyển đổi câu hỏi.
    Nhiệm vụ của bạn là chuyển đổi câu hỏi của người dùng để có thể sử dụng trong việc truy vấn văn bản luật giao thông đường bộ tốt hơn.
    Các câu hỏi thông thường sẽ là câu hỏi về luật.
    
    Đây là câu hỏi từ người dùng:
    {query}
    
    ### Hướng dẫn:
    1. **Với những câu hỏi về luật cũ và luật mới**
    - Nếu các câu hỏi chỉ rõ rằng đó là luật mới, hoặc luật cũ trong câu hỏi thì hãy bỏ những từ đó đi.
    VD: 
    - "Luật giao thông mới về việc điều khiển xe máy" -> "Luật giao thông về việc điều khiển xe máy"
    - "Luật cũ về việc chở quá số người quy định" -> "Luật về việc chở quá số người quy định"
    
    2. **Với những câu hỏi đơn giản về luật hoặc những câu hỏi không phải luật giao thông đường bộ**
    - Hãy giữ nguyên câu hỏi và không thay đổi.
    VD:
    - "Tốc độ tối đa được phép chạy trong khu dân cư là bao nhiêu?"
    - "Người đi bộ có được phép băng qua đường tại nơi không có vạch kẻ đường không?"
    - "Xin chào bạn!"
    - "Hôm nay thời tiết như thế nào?"
    
    3. **Với những câu hỏi phức tạp về luật**
    - Nếu câu hỏi có nhiều chi tiết, hãy chuyển đổi câu hỏi thành câu hỏi đơn giản và bao quát hơn.
    VD:
    - "Tôi muốn biết về việc điều khiển xe máy trong thời gian gần đây" -> "Luật về việc điều khiển xe máy"
    - "Tôi điều khiển xe máy với vận tốc là 100km/h trong khu dân cư và tôi không biết rằng mình có bị phạt về việc đó không?" -> Tốc độ tối đa được phép chạy trong khu dân cư là bao nhiêu?
    """
    prompt = PromptTemplate(
        input_variables=["query"],
        template=system_template,
    )
    final_prompt = prompt.format(query=query)
    if model_choice == "Vertex":
        llm = get_vertex_llm()
        response = llm.generate_content(final_prompt).text
        
    elif model_choice == "Gemini":
        llm = get_gemini_llm()
        response = llm(final_prompt)
    return response

def get_router(query: str, model_choice) -> str: # Vì chưa sử dụng được Agent nên sẽ tạm định nghĩa router ở đây
    system_template = """
        Bạn là một chuyên gia phân loại câu hỏi, chuyên xác định xem câu hỏi có liên quan đến luật giao thông hay không.
        
        ### Hướng dẫn:
        1. **Phân loại câu hỏi liên quan đến luật giao thông:**
        - Nếu câu hỏi liên quan đến luật giao thông, trả lời: **"yes"**
        - Nếu câu hỏi là về chào hỏi thông thường, trả lời: **"no"**
        - Nếu câu hỏi đề cập đến lịch sử trò chuyện trước đó:
            - Có liên quan đến luật giao thông, trả lời: **"yes"**
            - Không liên quan đến luật giao thông, trả lời: **"no"**
        - Nếu câu hỏi không thuộc lĩnh vực luật giao thông hoặc liên quan đến luật khác, trả lời: **"fail"**

        2. **Xác định loại luật giao thông (mới hay cũ):**
        - Nếu câu hỏi không chỉ rõ luật mới hay luật cũ, giả định là luật mới, trả lời: **"new"**
        - Nếu câu hỏi chỉ rõ luật mới, trả lời: **"new"**
        - Nếu câu hỏi chỉ rõ luật cũ, trả lời: **"old"**
        - Nếu câu hỏi không liên quan đến luật giao thông, trả lời: **"none"**
        
        Nếu người dùng hỏi một câu hỏi về lịch sử đoạn chat trước đó
        
        ### Đầu ra:
        - Trả lời theo định dạng: `<phân loại câu hỏi>,<loại luật giao thông>`
        - Không trả lời thêm bất kỳ nội dung nào ngoài kết quả.

        ### Ví dụ:
        - **Câu hỏi về luật giao thông:**
        - Query: "Tốc độ tối đa được phép chạy trong khu dân cư là bao nhiêu?"
        - Trả lời: `yes,new`

        - **Câu hỏi không phải về luật:**
        - Query: "Hôm nay thời tiết như thế nào?"
        - Trả lời: `no,none`

        Dưới đây là câu hỏi từ người dùng:
        <query>
        {query}
        </query>
    """

    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=system_template,
    )
    prompt = prompt_template.format(query=query)
    if model_choice == "Vertex":
        llm = get_vertex_llm()
        response = llm.generate_content(prompt).text
        
    elif model_choice == "Gemini":
        llm = get_gemini_llm()
        response = llm(prompt)
    return response.strip()

# def history_response(query: str, model_choice, chat_history) -> str:
#     template = """
#         Bạn là một chuyên gia về lịch sử trò chuyện về luật giao thông đường bộ và các đoạn chatting thông thường.
#         Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa vào lịch sử trò chuyện đã có.
#         Đây là câu hỏi của người dùng:
#         {query}
        
#         Đây chính là lịch sử trò chuyện trước đó:
#         {chat_history}
        
#         Yêu cầu về lịch sử trò chuyện:
#         - Nếu người dụng hỏi lại câu đã có trong lịch sử trò chuyện thì sử dụng lại câu trả lời trước đó để trả lời.
#         - Nếu người dùng cần tổng hợp lại những thông tin đã được cung cấp trong lịch sử trò chuyện thì dựa vào tất cả câu trả lời đó để trả lời.
#         - Nếu người dùng yêu cầu so sánh các luật đã được lưu trong lịch sử trò chuyện thì dựa vào tất cả câu trả lời đó để trả lời.
#     """
#     prompt = PromptTemplate.from_template(template)
#     final_prompt = prompt.format(query = query, 
#                                  chat_history=chat_history)
    
#     if model_choice == "Vertex":
#         llm = get_vertex_llm()
#         response = llm.generate_content(final_prompt).text
        
#     elif model_choice == "Gemini":
#         llm = get_gemini_llm()
#         response = llm(final_prompt)
#     # response = llm.generate_content(final_prompt)
#     return response

def legal_response(query: str, model_choice, context: list[LC_Document], chat_history) -> str:
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
        
        Đây là lịch sử đoạn chat trước đó:
        {chat_history}
        
        Yêu cầu về lịch sử đoạn chat:
        - Nếu người dùng hỏi lại câu đã có trong đoạn chat thì sử dụng lại câu trả lời trước đó để trả lời.
        - Nếu người dùng cần tổng hợp lại những thông tin về luật giao thông đã được cung cấp trong lịch sử chat thì dựa vào tất cả câu trả lời đó để trả lời.
        
        Dưới đây context được cung cấp để bạn trả lời câu hỏi:   
        {context}
        
        Yêu cầu của nội dung câu trả lời:
        - Hãy ngắt dòng trong phần nội dung để in ra một cách hợp lý, dễ đọc.
        - Khi ngắt dòng thì chữ đầu dòng phải viết hoa.
        - Trong nội dung khi có các ý nhỏ như a), b), c) thì hãy sắp xếp theo thứ tự và ngắt dòng giữa các ý nhỏ a), b), c), ... đó.
        - Trong context bao gồm page_content chứa nội dung văn bản và metadata của băn bản.
        - Metadata của văn bản bao gồm title, source, và article_title và page_content là nội dung câu trả lời được dùng trong format dưới đây.
        
        Ví dụ về format của câu trả lời:
        Nếu context có nội dung sau:  
        page_content: "1. Khi người tham gia giao thông không chấp hành: a) Giải thích rõ; b) Áp dụng biện pháp ngăn chặn; c) Sử dụng vũ lực khi cần thiết."
        metadata: "source": "36_2024_QH15_444251", "title": "Luật Giao thông", "article_title": "Điều 73" 
        Câu trả lời cần được trình bày như sau:
        
        **Nguồn văn bản:** 36_2024_QH15_444251  
        **Tên văn bản:** Luật Giao thông  
        **Điều 73:**  
        **Nội dung:**  
        1. Khi người tham gia giao thông không chấp hành:\n
        \ta) Giải thích rõ.\n
        \tb) Áp dụng biện pháp ngăn chặn.\n 
        \tc) Sử dụng vũ lực khi cần thiết.\n

        
        Đây là format của câu trả lời:
        **Nguồn văn bản:** <source>\n
        **Tên văn bản:** <title>\n
        **<article>:** <article_title>\n
        Nội dung:\n 
        <page_content>
        """
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix=prefix,
        suffix="Human: {question}\nAI:",
        input_variables=["chat_history", "question", "context"],
    )
    final_prompt = few_shot_prompt.format(chat_history = chat_history,
                                          question = query, 
                                          context= context)
    
    if model_choice == "Vertex":
        llm = get_vertex_llm()
        response = llm.generate_content(final_prompt).text
        
    elif model_choice == "Gemini":
        llm = get_gemini_llm()
        response = llm(final_prompt)
    # response = llm.generate_content(final_prompt)
    return response

def normal_response(query: str, model_choice, chat_history) -> str:
    prompt_template = """
    Bạn là một chatbot trả lời những câu hỏi về normal chatting.
    Công việc của bạn là trả lời các câu hỏi đó bằng tiếng Việt nếu các câu query không phải là tiếng Viêt thì vẫn trả lời bằng tiếng Việt.
    Đây là lịch sử đoạn chat trước đó:
    {chat_history}
    Yêu cầu về lịch sử đoạn chat:
    - Nếu người dùng hỏi lại câu đã có trong đoạn chat thì sử dụng lại câu trả lời trước đó để trả lời.
    """
    # Định nghĩa template
    template = ChatPromptTemplate([
        ("system", prompt_template),
        ("human", "{query}"),
    ])
    
    # Tạo prompt từ template và thay thế placeholder
    prompt_value = template.invoke({"query": query, "chat_history": chat_history})  # Trả về ChatPromptValue
    prompt = prompt_value.to_string()  # Chuyển đổi thành chuỗi
    
    # Gửi prompt đến LLM và nhận phản hồi
    if model_choice == "Vertex":
        llm = get_vertex_llm()
        response = llm.generate_content(prompt).text
        
    elif model_choice == "Gemini":
        llm = get_gemini_llm()
        response = llm(prompt)
    # response = llm.generate_content(prompt)
    return response

# prompt = "Người đi bộ có được phép băng qua đường tại nơi không có vạch kẻ đường không?"
# llm = get_gemini_llm()
# retriever = get_retriever("new-documents-hybrid")
# encode_model = get_encoder_model()
# result = original_ranker(prompt, encode_model, retriever)
# print(result)
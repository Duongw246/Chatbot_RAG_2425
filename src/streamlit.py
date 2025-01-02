import os, time
import streamlit as st
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from langchain.load import loads, dumps
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.retrievers import PineconeHybridSearchRetriever

# Load API keys
# os.environ["GOOGLE_API_KEY"] = "AIzaSyD6fv2qZAcRc30uDjn96CbsM6pUJwLkdFE"
google_api_key = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = "AIzaSyCekUE-sNiAc_Jw-TFaLO11Xn18lLc-Lkw"
# os.environ["PINECONE_API_KEY"] = "pcsk_3EcDrL_3mUVa7rhMVLMFBZJFPvgGEaunymPs7T5XcZXjBp9dF55S73miNRzeW2FMsFWcEb"
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

@st.cache_resource
def retrieval_definition():
    model_name = "hiieu/halong_embedding"
    hf = HuggingFaceEmbeddings(model_name=model_name)

    # Pinecone storage and BM25 retriever
    test_index = pc.Index(name="hybrid-rag")
    bm25_encoder = BM25Encoder().default()
    retriever = PineconeHybridSearchRetriever(embeddings=hf, sparse_encoder=bm25_encoder, index=test_index)
    return retriever

retriever = retrieval_definition()

@st.cache_resource
def model_definition():
    llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0)
    return llm
llm = model_definition()

def fusion_query(query, llm, retriever):
    system_template = """
        Bạn là một chuyên gia tạo ra nhiều câu hỏi liên quan từ câu query đầu vào của người dùng. 
        Trong mỗi câu output đừng giải thích gì thêm cả, chỉ cần tạo ra câu hỏi liên quan từ câu query đầu vào.
        Hãy tạo ra 4 câu query liên quan tới query sau: "{query}"
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
    list_query = [line.strip() for line in get_response.split("\n") if line.strip()]
    retriever.top_k = 10
    retrieved_list = [retriever.invoke(query) for query in list_query]
    
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
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

def routing(query, llm):
    system_template = """
        Bạn là một chuyên gia về phân loại câu hỏi.
        Công việc của bạn là phân loại câu hỏi xem câu hỏi đưa và có phải là một câu hỏi về luật giao thông hay không.
        Đối với những câu hỏi như "bạn là chatbot về luật gì?", hay "bạn hỗ trợ người dùng những vấn đề nào?" thì hãy trả lời là chatbot về luật giao thông.
        Nếu câu hỏi là một câu hỏi về luật giao thông thì hãy phản hồi "yes"
        Ngược lại hãy phản hồi "no"
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

def few_shot_fusion(query, llm, context):
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
    context = [ctx[0].page_content for ctx in context[:11]]
    final_prompt = few_shot_prompt.format(question = query, context= context)
    
    response = llm(final_prompt)
    return response

st.title("Legal Assistant")

with st.chat_message("assistant"):
    st.markdown("Xin chào! Tôi là trợ lý pháp lý của bạn. Hãy nhập câu hỏi của bạn vào ô bên dưới để bắt đầu trò chuyện với tôi.")

if "messages" not in st.session_state:
    st.session_state.messages = []

def build_prompt(messages):
    """Tạo prompt từ lịch sử hội thoại."""
    prompt = ""
    for message in messages:
        role = "user" if message["role"] == "user" else "assistant"
        prompt += f"{role}: {message['content']}\n"
    return prompt
   
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message['content'])
        
prompt = st.chat_input("Type something...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt) 
    start_time = time.time()   
    history = build_prompt(st.session_state.messages)
    full_prompt = f"{history}assistant:"
    
    router = routing(prompt, llm)
    if router == "yes":
        context = fusion_query(prompt, llm, retriever)
        response = few_shot_fusion(full_prompt, llm, context)
    else:
        response = llm(full_prompt)
    end_time = time.time()      
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st.caption(f"Thời gian xử lý: {end_time - start_time:.2f} giây")
            st.markdown(response)  
    st.session_state.messages.append({"role": "assistant", "content": response})

import os
import streamlit as st
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.retrievers import PineconeHybridSearchRetriever

# Load API keys
# os.environ["GOOGLE_API_KEY"] = "AIzaSyD6fv2qZAcRc30uDjn96CbsM6pUJwLkdFE"
google_api_key = st.secrets["GOOGLE_API_KEY"]
# os.environ["PINECONE_API_KEY"] = "pcsk_3EcDrL_3mUVa7rhMVLMFBZJFPvgGEaunymPs7T5XcZXjBp9dF55S73miNRzeW2FMsFWcEb"
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

@st.cache_resource
def retrieval_definition():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    hf = HuggingFaceEmbeddings(model_name=model_name)

    # Pinecone storage and BM25 retriever
    test_index = pc.Index(name="test-index")
    bm25_encoder = BM25Encoder().default()
    retriever = PineconeHybridSearchRetriever(embeddings=hf, sparse_encoder=bm25_encoder, index=test_index)
    return retriever

retriever = retrieval_definition()

@st.cache_resource
def model_definition():
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
    return llm
llm = model_definition()

def few_shot_prompting(query, llm, retriever):
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
        Bạn là một chuyên gia về luật.
        Công việc của bạn là trả lời các câu hỏi của người dùng về luật.
        Format của câu trả lời bao gồm:
        - Nội dung của câu trả lời được trích xuất từ văn bản nào?
        - Nội dung của câu trả lời được trích xuất từ chương nào? .
        - Nội dung của câu trả lời được trích xuất từ điều nào? Hãy ghi rõ nội dung liên quan tới câu hỏi của người dùng.
        Format của câu trả lời: 
        - Tên văn bản - Tên chương - Tên điều: \n
                Nội dung câu trả lời
                
        Nêu văn bản không chia theo chương hoặc điều, hãy bỏ qua phần tương ứng.
        Câu trả lời phải chính xác và đầy đủ thông tin dựa theo câu hỏi của người dùng và nội dung của văn bản.
        Nếu không có thông tin của câu trả lời dựa vào những văn bản được truyền vào, hãy trả lời "Không thể trả lời câu hỏi này" và không nói gì thêm. Không trả lời lan man.
    """
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix=prefix,
        suffix="Human: {question}\nAI:",
        input_variables=["question"],
    )
    contexts = retriever.invoke(query)
    context = [context.page_content for context in contexts]
    
    final_prompt = few_shot_prompt.format(question=f"{query}\nContext: {context}")
    print(final_prompt)
    
    response = llm(final_prompt)
    print(type(response))
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
        
    history = build_prompt(st.session_state.messages)
    full_prompt = f"{history}assistant:"
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = few_shot_prompting(full_prompt, llm, retriever)
            st.markdown(response)  
    st.session_state.messages.append({"role": "assistant", "content": response})

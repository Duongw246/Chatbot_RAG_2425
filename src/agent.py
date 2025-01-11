import streamlit as st
from langchain.load import loads, dumps
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

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

@st.cache_resource
def get_gemini_llm() -> GoogleGenerativeAI:
    return GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY, temperature=0)

def fusion_retriever(query: str, llm, vectorstore: PineconeVectorStore) -> list:
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
    retriever = get_retriever(index_name="hybrid-rag")
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=system_template,
    )
    prompt = prompt_template.format(query=query)
    get_response = llm(prompt)
    list_query = [line.strip() for line in get_response.split("\n") if line.strip()]
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
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def get_router(query: str, llm) -> str:
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

def get_legal_response(query: str, llm: any, context: list[LC_Document], history: any) -> str:
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
    context = [ctx[0].page_content for ctx in context[:5]]
    final_prompt = few_shot_prompt.format(question = query, context= context)
    
    response = llm(final_prompt)
    return response

def get_normal_response(query: str, llm: any, history) -> str:
    prompt_parts = [
        ("system", "Bạn là một chatbot trả lời những câu hỏi về normal chatting")
    ]

    # Process the StreamlitChatMessageHistory to extract past messages
    if history:
        for message in history.messages:
            role = "human" if message["role"] == "user" else "assistant"
            prompt_parts.append((role, message["content"]))
            
    prompt_parts.append(("human", query))

    # Create the prompt
    prompt = ChatPromptTemplate(prompt_parts)
    response = llm(prompt)
    return response
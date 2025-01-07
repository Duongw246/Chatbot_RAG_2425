import streamlit as st
from seed_data import get_retriever
from langchain_google_genai import GoogleGenerativeAI
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY_1"]
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")


@st.cache_resource
def call_retriever()-> PineconeHybridSearchRetriever:
    # Pinecone storage and BM25 retriever
    retriever = get_retriever(index_name="hybrid-rag")
    retriever.top_k = 4
    
    return retriever

retriever_tool = create_retriever_tool(
    call_retriever(),
    "Tìm kiếm thông tin",
    "Tìm kiếm thông tin mới nhất về luật giao thông đường bộ",
)

@st.cache_resource
def get_llm_and_agent() -> AgentExecutor:
    llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY, temperature=0)
    tool = [retriever_tool]
    
    system = """Bạn là một chuyên gia về luật giao thông. Tên bạn là Legal AI"""
    prompt = ChatPromptTemplate([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),   
    ])
        
    agent = create_openai_functions_agent(llm=llm, tools=tool, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tool, verbose=True)

retriever = call_retriever()
agent_executor = get_llm_and_agent()
    
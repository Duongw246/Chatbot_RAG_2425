import time
import streamlit as st
from seed_data import get_retriever
from agent import (legal_response, 
                   normal_response, 
                   fusion_ranker,
                   original_ranker, 
                   get_router,
                   get_gemini_llm,
                   get_encoder_model)
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="Legal Assistant", 
        page_icon="💬",
        layout="wide" 
    )
@st.cache_data
def setup_chat_interface():
    st.title("Legal Assistant")
    st.caption("Trợ lý AI được hỗ trợ bởi LangChain và Google")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi là trợ lý pháp lý của bạn. Hãy nhập câu hỏi của bạn vào ô bên dưới để bắt đầu trò chuyện với tôi."}
        ]
        msgs.add_ai_message(st.session_state.messages[0]["content"])
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    return msgs

def user_input(msgs, llm, encode_model, new_retriever, old_retriever):
    if prompt:= st.chat_input("Hãy hỏi tôi bất cứ điều gì về luật giao thông đường bộ!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)   
        msgs.add_user_message(prompt)
        
        router = get_router(prompt, llm)
     
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st_callback = StreamlitCallbackHandler(st.container())
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                ]
                if router == "yes":
                    context = original_ranker(prompt, encode_model, new_retriever)
                #     # context = fusion_ranker(prompt, llm, new_retriever)
                    response = legal_response(msgs, llm, context, chat_history)
                    # else:
                    #     context = fusion_ranker(prompt, llm, old_retriever)
                    #     response = legal_response(prompt, llm, context, msgs)
                        
                elif router == "no":
                    response = normal_response(prompt, llm, chat_history)
                elif router == "fail" or router == "yes":
                    # response = "Không tìm thấy thông tin cho nội dung bạn tìm kiếm!"
                    response = prompt
                st.session_state.messages.append({"role": "assistant", "content": response})
                msgs.add_ai_message(response)
                st.markdown(response)
                
def main():
    setup_page()
    llm = get_gemini_llm()
    encode_model = get_encoder_model()
    new_retriever = get_retriever("new-documents-hybrid")
    old_retriever = get_retriever("old-documents-hybrid")
    msgs = setup_chat_interface()
    user_input(msgs, llm, encode_model, new_retriever, old_retriever)
    
if __name__ == "__main__":
    main()
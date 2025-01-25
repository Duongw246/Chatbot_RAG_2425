import time
import random
import streamlit as st
from seed_data import get_retriever
from agent import (legal_response, 
                   normal_response, 
                   fusion_ranker,
                   original_ranker, 
                   get_router,
                   get_gemini_llm,
                   get_encoder_model,
                   get_vertex_llm
                   )
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

def reset_conversation():
    """
    Đặt lại toàn bộ trạng thái của cuộc trò chuyện.
    """
    st.session_state['messages'] = [] 

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
    
def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="Legal Assistant", 
        page_icon="💬",
        layout="wide" 
    )

def setup_sidebar():
    with st.sidebar:
        st.title("⚙️ **CẤU HÌNH**")
        st.header(":desktop_computer: Response Model")
        model_choice = st.radio(
            "Chọn Model để trả lời:",
            ["Vertex", "Gemini"]
        )
        
        st.button("Clear chat", on_click=reset_conversation)
        return model_choice
        
def setup_chat_interface(model_choice):
    if model_choice == "Gemini":
        st.title("Legal Assistant")
        st.caption("Trợ lý AI được hỗ trợ bởi LangChain và Google")
    if model_choice == "Vertex":
        st.title("Legal Assistant")
        st.caption("Trợ lý AI được hỗ trợ bởi LangChain và Vertex AI")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if "messages" not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": random.choice(
                [
                    "Xin chào! Tôi là trợ lý pháp lý của bạn. Hãy nhập câu hỏi của bạn vào ô bên dưới để bắt đầu trò chuyện với tôi.",
                    "Chào bạn! Bạn cần tôi giúp gì không? Tôi có thể trả lời mọi câu hỏi của bạn về luật giao thông đường bộ của bạn.",
                    "Bạn muốn biết gì về luật giao thông đường bộ? Hãy nhập câu hỏi của bạn vào ô bên dưới để bắt đầu trò chuyện với tôi.",
            ])}
        ]
        msgs.add_ai_message(st.session_state.messages[0]["content"])
    
    for msg in st.session_state['messages']:
        with st.chat_message(msg["role"]):
            # response = st.write_stream(response_generator(msg["content"]))
            st.markdown(msg["content"])
    return msgs

def user_input(msgs, model_choice, encode_model, new_retriever, old_retriever):
    if prompt:= st.chat_input("Hãy hỏi tôi bất cứ điều gì về luật giao thông đường bộ!"):
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)   
        msgs.add_user_message(prompt)
        router = get_router(prompt, model_choice)
        router, status = router.split(",")
        router = router.strip(" ")
        status = status.strip(" ")
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm thông tin..."):
            # st_callback = StreamlitCallbackHandler(st.container())
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                ]
                if router == "yes":
                    if status == "new":
                        context = original_ranker(prompt, encode_model, "new-documents-hybrid")
                        
                    elif status == "old":
                        context = original_ranker(prompt, encode_model, "old-documents-hybrid")
                    # context = original_ranker(prompt, encode_model, new_retriever)
                    
                    # context = fusion_ranker(prompt, new_retriever)
                    response = legal_response(msgs, model_choice, context, chat_history)
                    # else:
                    #     context = fusion_ranker(prompt, llm, old_retriever)
                    #     response = legal_response(prompt, llm, context, msgs)
                        
                elif router == "no":
                    response = normal_response(prompt, model_choice, chat_history)
                else:
                    response = "Không tìm thấy thông tin cho nội dung bạn tìm kiếm!"
                    # response = prompt
                response = st.write_stream(response_generator(response))
                st.session_state['messages'].append({"role": "assistant", "content": response})
                msgs.add_ai_message(response)
                # st.markdown(response)
def main():
    setup_page()
    model_choice = setup_sidebar()
    # llm = get_vertex_llm()
    encode_model = get_encoder_model()
    new_retriever = get_retriever("new-documents-hybrid")
    old_retriever = get_retriever("old-documents-hybrid")
    msgs = setup_chat_interface(model_choice)
    user_input(msgs, model_choice, encode_model, new_retriever, old_retriever)
    
if __name__ == "__main__":
    main()
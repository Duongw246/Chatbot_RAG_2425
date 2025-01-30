import time
import random
import streamlit as st
from seed_data import postgres_retriever
from agent import (legal_response, 
                   normal_response, 
                   query_transform,
                #    history_response,
                   get_router
                   )
# from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

def reset_conversation():
    """
    Đặt lại toàn bộ trạng thái của cuộc trò chuyện.
    """
    st.session_state['messages'] = [
        {"role": "assistant", "content": random.choice(
            [
                "Xin chào! Tôi là trợ lý pháp lý của bạn. Hãy nhập câu hỏi của bạn vào ô bên dưới để bắt đầu trò chuyện với tôi.",
                "Chào bạn! Bạn cần tôi giúp gì không? Tôi có thể trả lời mọi câu hỏi của bạn về luật giao thông đường bộ của bạn.",
                "Bạn muốn biết gì về luật giao thông đường bộ? Hãy nhập câu hỏi của bạn vào ô bên dưới để bắt đầu trò chuyện với tôi.",
        ])}
    ] 

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
            # response = st.markdown_stream(response_generator(msg["content"]))
            st.markdown(msg["content"])
    return msgs

def user_input(msgs, model_choice, new_retriever, old_retriever):
    if prompt:= st.chat_input("Hãy hỏi tôi bất cứ điều gì về luật giao thông đường bộ!"):
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)   
        msgs.add_user_message(prompt)
        router = get_router(prompt, model_choice)
        router, status = router.split(",")
        router = router.strip(" ")
        status = status.strip(" ")
        # st.markdown(router)
        # st.markdown(status)
        transform_prompt = query_transform(prompt, model_choice)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            timer_placeholder = st.empty()
            start_time = time.time()  # Bắt đầu đo thời gian
            with st.spinner("Đang tìm kiếm thông tin..."):
                def update_timer():
                    while True:
                        elapsed_time = time.time() - start_time
                        timer_placeholder.caption(f"⏱️ Đang xử lý: {elapsed_time:.2f} giây")
                        time.sleep(0.1)

                import threading
                timer_thread = threading.Thread(target=update_timer, daemon=True)
                timer_thread.start()
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                ]
                if router == "yes":
                    if status == "new":
                        context = new_retriever.invoke(transform_prompt)
                        
                    elif status == "old":
                        context = old_retriever.invoke(transform_prompt)

                    response = legal_response(msgs, model_choice, context, chat_history)
                        
                elif router == "no":
                    response = normal_response(transform_prompt, model_choice, chat_history)
                    
                # elif router == "history":
                #     response = history_response(prompt, model_choice, chat_history)
                    
                else:
                    response = random.choice([
                    "Câu hỏi của bạn nằm không nằm trong phạm vi về luật giao thông đường bộ. Hãy nhập câu hỏi khác để tôi giúp bạn nhé!",
                    "Xin lỗi, câu hỏi này không nằm trong phạm vi của tôi. Hãy nhập câu hỏi khác để tôi giúp bạn nhé!",
                    "Kiến thức của tôi chỉ giới hạn trong lĩnh vực luật giao thông đường bộ. Hãy nhập câu hỏi khác để tôi giúp bạn nhé!",
                ])  
                response_content = ""
                for chunk in response:
                    for char in chunk:
                        response_content += char
                        response_placeholder.markdown(response_content)
                end_time = time.time()  # Kết thúc đo thời gian
                elapsed_time = end_time - start_time    
                    # response = prompt
                # response = st.write_stream(response_generator(response))
                st.session_state['messages'].append({"role": "assistant", "content": response})
                msgs.add_ai_message(response)
                # st.markdown(response)
                
                elapsed_time = time.time() - start_time
                timer_placeholder.caption(f"⏱️ Thời gian xử lý: {elapsed_time:.2f} giây")
def main():
    setup_page()
    model_choice = setup_sidebar()
    new_retriever = postgres_retriever(collection_name="vectordb", database_name="postgres")
    old_retriever = postgres_retriever(collection_name="old_vectordb", database_name="old_law")
    msgs = setup_chat_interface(model_choice)
    user_input(msgs, model_choice, new_retriever, old_retriever)
    
if __name__ == "__main__":
    main()
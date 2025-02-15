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
        st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>⚙️ Cấu Hình</h1>
        """, unsafe_allow_html=True)
        
        with st.expander("🚀 Chọn Model AI", expanded=True):
            model_choice = st.selectbox(
                "Chọn Model để trả lời:",
                ["vertex-gemini-1.5-pro", "vertex-gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                index=0
            )
            st.caption("🔹 Model AI sẽ ảnh hưởng đến tốc độ và độ chính xác của câu trả lời.")
        
        with st.expander("📂 Chọn nguồn dữ liệu", expanded=True):
            data_source = st.radio(
                "Ưu tiên nguồn dữ liệu:",
                ["Ưu tiên luật mới", "Chỉ dùng dữ liệu cũ", "Kết hợp cả hai"],
                index=2
            )
            st.caption("🔹 Chọn cách hệ thống lấy dữ liệu từ RAG.")
        
        with st.expander("📜 Chọn bộ luật hoặc chủ đề"):
            law_topic = st.selectbox(
                "Chọn chủ đề luật:",
                ["Giao thông đường bộ"],
                index=0
            )
        
        with st.expander("🔍 Điều chỉnh truy vấn"):
            retrieval_depth = st.slider("Chọn độ sâu truy vấn:", 1, 10, 10)
            st.caption("🔹 Độ sâu càng cao, truy xuất càng nhiều tài liệu.")
        
        with st.expander("🛠 Tuỳ Chọn Khác"):
            if st.button("🗑 Xóa cuộc trò chuyện", use_container_width=True):
                reset_conversation()
            st.markdown("""
            <small style='color: grey;'>Xóa lịch sử chat để bắt đầu cuộc trò chuyện mới.</small>
            """, unsafe_allow_html=True)
        
    return model_choice, data_source, law_topic, retrieval_depth


        
def setup_chat_interface(model_choice):
    st.title("Legal Assistant")
    if model_choice == "vertex-gemini-1.5-pro" or model_choice == "vertex-gemini-1.5-flash":
        st.caption("Trợ lý AI được hỗ trợ bởi LangChain và Vertex AI")
    if model_choice == "gemini-1.5-pro" or model_choice == "gemini-1.5-flash":
        st.caption("Trợ lý AI được hỗ trợ bởi LangChain và Google")

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
            start_time = time.time()
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
                st.session_state['messages'].append({"role": "assistant", "content": response})
                msgs.add_ai_message(response)
                elapsed_time = time.time() - start_time
                timer_placeholder.caption(f"⏱️ Thời gian phản hồi: {elapsed_time:.2f} giây")
def main():
    setup_page()
    model_choice = setup_sidebar()
    new_retriever = postgres_retriever(collection_name="vectordb", database_name="postgres")
    old_retriever = postgres_retriever(collection_name="old_vectordb", database_name="old_law")
    msgs = setup_chat_interface(model_choice)
    user_input(msgs, model_choice, new_retriever, old_retriever)
    
if __name__ == "__main__":
    main()
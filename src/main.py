import time
import streamlit as st
from agent import get_response, fusion_retriever, get_router, get_gemini_llm, call_retriever
from seed_data import get_vectorstore
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

def setup_chat_interface():
    st.title("Legal Assistant")
    st.caption("Trợ lý AI được hỗ trợ bởi LangChain và Google")

    with st.chat_message("assistant"):
        st.markdown("Xin chào! Tôi là trợ lý pháp lý của bạn. Hãy nhập câu hỏi của bạn vào ô bên dưới để bắt đầu trò chuyện với tôi.")

    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
        ]
        msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg['content'])
    return msgs

def user_input(msgs, llm, retriever):
    if prompt:= st.chat_input("Hãy hỏi tôi bất cứ điều gì về luật giao thông đường bộ!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt) 
        start_time = time.time()   
        msgs.add_user_message(prompt)
        
        # router = get_router(prompt, llm)
        # if router == "yes":
        #     context = fusion_retriever(prompt, llm, retriever)
        #     response = get_response(msgs, llm, context)
        # elif router == "no":
        #     response = llm(msgs)
        # else:
        #     response = "Không tìm thấy thông tin cho nội dung bạn tìm kiếm!"
        end_time = time.time()      
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st_callback = StreamlitCallbackHandler(st.container())
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                ]
                context = fusion_retriever(prompt, llm, retriever)
                response = get_response(prompt,llm, context, chat_history)
                st.session_state.messages.append({"role": "assistant", "content": response})
                msgs.add_ai_message(response)
                st.markdown(response)
                
def main():
    setup_page()
    llm = get_gemini_llm()
    retriever = call_retriever()
    msgs = setup_chat_interface()
    user_input(msgs, llm, retriever)
    
if __name__ == "__main__":
    main()
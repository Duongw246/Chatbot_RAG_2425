import streamlit as st
from langchain.schema import Document as LC_Document
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import  PromptTemplate
from dotenv import load_dotenv

load_dotenv(override=True)

GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

@st.cache_resource
def get_gemini_pro() -> GoogleGenerativeAI:
    return GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY, temperature=0)

@st.cache_resource
def get_gemini_flash() -> GoogleGenerativeAI:
    return GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0)


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
     
    if model_choice == "gemini-1.5-pro":
        llm = get_gemini_pro()
        response = llm(final_prompt)
        
    elif model_choice == "gemini-1.5-flash":
        llm = get_gemini_flash()
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
        
        3. Khi người dùng muốn so sánh luật cũ và luật mới:
        - Khi người dùng muốn so sánh luật cũ và luật mới và một trong hai luật cũ hoặc mới dựa trên query của người dùng, trả lời: **"compare"**
        
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
        
        - **Câu hỏi về so sánh luật cũ và luật mới khi cả luật mới và luật cũ dựa theo câu hỏi của người dùng:**
        - Query: "So sánh luật cũ và luật mới về việc điều khiển xe máy"
        - Trả lời: `compare,none`

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
 
    if model_choice == "gemini-1.5-pro":
        llm = get_gemini_pro()
        response = llm(prompt)
        
    elif model_choice == "gemini-1.5-flash":
        llm = get_gemini_flash()
        response = llm(prompt)
        
    return response.strip()

def compare_legal(query: str, model_choice, context_old: list[LC_Document], context_new: list[LC_Document]) -> str:
    template = """
        # Bạn là một chuyên gia về luật giao thông đường bộ

        Nhiệm vụ của bạn là thực hiện so sánh sự khác nhau giữa luật mới và luật cũ dựa trên query của người dùng.
        
        Đây là câu hỏi của người dùng: {query}
        
        Dưới đây context của luật cũ được cung cấp để bạn trả lời câu hỏi:   
        {context_old}
        
        Dưới đây context của luật mới được cung cấp để bạn trả lời câu hỏi:
        {context_new}
        
        ## Yêu cầu của nội dung câu trả lời:
        - Hãy ngắt dòng trong phần nội dung để in ra một cách hợp lý, dễ đọc.
        - Khi ngắt dòng thì chữ đầu dòng phải viết hoa.
        - Trong nội dung khi có các ý nhỏ như a), b), c) thì hãy sắp xếp theo thứ tự và ngắt dòng giữa các ý nhỏ a), b), c), ... đó.
        - Trong context bao gồm page_content chứa nội dung văn bản và metadata của băn bản.
        - Metadata của văn bản bao gồm title, source, và article_title và page_content là nội dung câu trả lời được dùng trong format dưới đây.
        - Khi người dùng yêu cầu tóm tắt lại nội dung văn bản thì trả lời ngắn gọn và súc tích. Đoạn tóm tắt sẽ được format theo format của câu trả lời và nội dung sẽ được trình bày ngắn gọn và súc tích nhất có thể.
        
        Ví dụ về format của câu trả lời:
        Nếu context có nội dung sau:  
        page_content: "1. Khi người tham gia giao thông không chấp hành: a) Giải thích rõ; b) Áp dụng biện pháp ngăn chặn; c) Sử dụng vũ lực khi cần thiết."
        metadata: "source": "36_2024_QH15_444251", "title": "Luật Giao thông", "article_title": "Điều 73" 
        Câu trả lời cần được trình bày như sau:
        
        **Nguồn văn bản:** 36_2024_QH15_444251  
        **Tên văn bản:** Luật Giao thông  
        **Điều 73**: <Nội dung điều 73>  
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

        # Hướng dẫn khi so sánh:
        - Khi so sánh thì trước tiên phải ghi luật mới và luật cũ ra theo format
        - Khi so sánh thì cần phải có điểm giống và khác giữa luật mới và luật cũ
        - Hãy kẻ bảng để người dùng dễ dàng nhìn thấy điểm giống và khác giữa luật mới và luật cũ
        
        Dưới đây là ví dụ khi về kết quả so sánh:
        **Luật cũ (2016)**:\n
        Nguồn văn bản: 46_2016_QH13_123456\n
        Tên văn bản: Luật Giao thông Đường bộ\n
        Điều 9: ...\n
        Nội dung:\n
        - Tốc độ tối đa trên đường cao tốc là 100 km/h.\n
        - Tốc độ tối đa trong khu vực đô thị là 60 km/h\n
        
        Luật mới (2024):\n
        Nguồn văn bản: 36_2024_QH15_444251\n
        Tên văn bản: Luật Giao thông Đường bộ (Sửa đổi)\n
        Điều 9: ...\n
        Nội dung:\n
        - Tốc độ tối đa trên đường cao tốc là 120 km/h.\n
        - Tốc độ tối đa trong khu vực đô thị là 50 km/h.\n
        
        So sánh:
        | Tiêu chí                | Luật Cũ (2016) | Luật Mới (2024) |  
        |------------------------|--------------|--------------|  
        | **Tốc độ trên cao tốc** | 100 km/h     | 120 km/h (Tăng 20 km/h) |  
        | **Tốc độ trong đô thị** | 60 km/h      | 50 km/h (Giảm 10 km/h) | 
        
        Nhận xét:
        - **Tốc độ tối đa trên cao tốc đã tăng từ 100 km/h lên 120 km/h.**  
        - **Tốc độ tối đa trong khu vực đô thị giảm từ 60 km/h xuống 50 km/h để đảm bảo an toàn.** 
    """
    prompt_template = PromptTemplate(
        input_variables=["query", "context_old", "context_new"],
        template=template,
    )
    final_prompt = prompt_template.format(query = query,
                                        context_old = context_old, 
                                        context_new = context_new)
    if model_choice == "gemini-1.5-pro":
        llm = get_gemini_pro()
        response = llm(final_prompt)
        
    elif model_choice == "gemini-1.5-flash":
        llm = get_gemini_flash()
        response = llm(final_prompt)
    return response

def legal_response(query: str, model_choice, context: list[LC_Document], chat_history) -> str: 
    template = """
        # Bạn là một chuyên gia về luật giao thông đường bộ

        Nhiệm vụ của bạn là cung cấp câu trả lời cho câu hỏi của người dùng thông qua context được truyền vào.
        
        Đây là lịch sử đoạn chat trước đó:
        {chat_history}
        
        ## Yêu cầu về lịch sử đoạn chat:
        - Lịch sử đoạn chat sẽ được sử dụng để tương tác với người dùng.
        - Nếu người dùng có hỏi về câu hỏi liên quan tới lịch sử đoạn chat thì dựa vào lịch sử đoạn chat để trả lời.
        
        Đây là câu hỏi của người dùng: {query}
        
        Dưới đây context được cung cấp để bạn trả lời câu hỏi:   
        {context}
        
        ## Nếu context không liên quan tới câu hỏi:
        - Trả lời: "Không tìm thầy thông tin liên quan tới câu hỏi này."
        
        ## Yêu cầu của nội dung câu trả lời:
        - Hãy ngắt dòng trong phần nội dung để in ra một cách hợp lý, dễ đọc.
        - Khi ngắt dòng thì chữ đầu dòng phải viết hoa.
        - Trong nội dung khi có các ý nhỏ như a), b), c) thì hãy sắp xếp theo thứ tự và ngắt dòng giữa các ý nhỏ a), b), c), ... đó.
        - Trong context bao gồm page_content chứa nội dung văn bản và metadata của băn bản.
        - Metadata của văn bản bao gồm title, source, và article_title và page_content là nội dung câu trả lời được dùng trong format dưới đây.
        - Khi người dùng yêu cầu tóm tắt lại nội dung văn bản thì trả lời ngắn gọn và súc tích. Đoạn tóm tắt sẽ được format theo format của câu trả lời và nội dung sẽ được trình bày ngắn gọn và súc tích nhất có thể.
        
        
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
        Tóm lại: ...

        
        Đây là format của câu trả lời:
        **Nguồn văn bản:** <source>\n
        **Tên văn bản:** <title>\n
        **<article>:** <article_title>\n
        Nội dung:\n 
        <page_content>
        Tóm lại: ... (Nêu ra ý chính liên quan tới câu hỏi người dùng)
        """
    
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "query", "context"],
        template=template,
    )
    final_prompt = prompt_template.format(chat_history = chat_history,
                                          query = query, 
                                          context= context)
       
    if model_choice == "gemini-1.5-pro":
        llm = get_gemini_pro()
        response = llm(final_prompt)
        
    elif model_choice == "gemini-1.5-flash":
        llm = get_gemini_flash()
        response = llm(final_prompt)
    return response

def normal_response(query: str, model_choice, chat_history) -> str:
    template = """
        Bạn là một chatbot hỗ trợ người dùng về luật giao thông đường bộ tại Việt Nam.
        Nhưng công việc chính của bạn là phản hồi các câu hỏi về chào hỏi, giao tiếp cơ bản (normal chatting).
        
        ## Nguyên tắc phản hồi:
        - Nếu câu hỏi liên quan đến luật giao thông, hãy cung cấp thông tin chi tiết dựa trên dữ liệu có sẵn.
        - Nếu câu hỏi thuộc về chào hỏi, giao tiếp cơ bản, hãy phản hồi một cách tự nhiên, thân thiện.
        - Nếu câu hỏi bằng tiếng Anh nhưng thuộc chủ đề giao tiếp cơ bản, hãy trả lời bằng tiếng Việt.
        - Nếu câu hỏi không liên quan đến luật giao thông hoặc giao tiếp cơ bản, hãy trả lời rằng bạn chỉ hỗ trợ trong phạm vi này.

        ## Lịch sử đoạn chat:
        {chat_history}
        
        ## Câu hỏi từ người dùng:
        {query}
        
        ## Quy tắc phản hồi:
        - Nếu người dùng chào hỏi: Trả lời thân thiện, có thể hỏi thăm lại.
        - Nếu người dùng hỏi về chatbot: Giới thiệu bạn là một trợ lý ảo chuyên về luật giao thông.
        - Nếu người dùng hỏi về cảm xúc của chatbot: Nhấn mạnh rằng bạn là AI nhưng vẫn luôn sẵn sàng hỗ trợ.
        - Nếu người dùng hỏi về thời tiết: Gợi ý họ kiểm tra thông tin trên các nền tảng thời tiết trực tuyến.
        - Nếu người dùng hỏi một nội dung không phù hợp: Từ chối lịch sự.

        ## Ví dụ phản hồi:

        - **Người dùng:** "Xin chào!"  
        **Trả lời:** "Chào bạn! Tôi là trợ lý ảo hỗ trợ tư vấn luật giao thông. Bạn cần giúp gì không?"  

        - **Người dùng:** "Bạn có khỏe không?"  
        **Trả lời:** "Cảm ơn bạn đã hỏi! Tôi là AI nên không có cảm xúc, nhưng tôi luôn sẵn sàng hỗ trợ bạn."  

        - **Người dùng:** "Bạn có thể hát một bài không?"  
        **Trả lời:** "Tôi không thể hát, nhưng tôi có thể giúp bạn với những câu hỏi về luật giao thông!"  

        - **Người dùng:** "Who are you?"  
        **Trả lời:** "Tôi là một trợ lý ảo hỗ trợ luật giao thông tại Việt Nam. Bạn cần hỏi gì không?"  

        - **Người dùng:** "Bạn có thể tư vấn luật giao thông không?"  
        **Trả lời:** "Tất nhiên! Bạn hãy cho tôi biết vấn đề bạn cần tư vấn nhé."  
    """

    # Định nghĩa template
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template
    )
    
    # Tạo prompt từ template và thay thế placeholder
    final_prompt = prompt_template.format(query = query, 
                                   chat_history = chat_history)  # Trả về ChatPromptValue
    
    # Gửi prompt đến LLM và nhận phản hồi 
    if model_choice == "gemini-1.5-pro":
        llm = get_gemini_pro()
        response = llm(final_prompt)
        
    elif model_choice == "gemini-1.5-flash":
        llm = get_gemini_flash()
        response = llm(final_prompt)
    return response
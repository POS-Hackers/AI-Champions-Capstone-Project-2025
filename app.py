import streamlit as st
import os
# import dotenv (commented out as Streamlit community cloud dont support .env)
import uuid
import random  
import hmac 
import requests

# # --- Function to Check Password --
def check_password():  
    """Returns `True` if the user had the correct password."""  
    def password_entered():  
        """Checks whether a password entered by the user is correct."""  
        if hmac.compare_digest(st.session_state["password"], st.secrets["APP_PASSWORD"]):  
            st.session_state["password_correct"] = True  
            del st.session_state["password"]  # Don't store the password.  
        else:  
            st.session_state["password_correct"] = False  
    # Return True if the passward is validated.  
    if st.session_state.get("password_correct", False):  
        return True  
    # Show input for password.  
    st.text_input(  
        "Password", type="password", on_change=password_entered, key="password"  
    )  
    if "password_correct" in st.session_state:  
        st.error("😕 Password incorrect")  
    return False

# # --- App Setup ---
# Check if it's linux so it works on Streamlit Cloud
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Assign USER_AGENT
os.environ["USER_AGENT"] = st.secrets["USER_AGENT"]

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek

from rag_methods import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

# Load LLM API Tokens and relevant docs URL from Github
# Commented as Streamlit coomunity cloud dont support .env
# openai_api_key = os.getenv("OPENAI_API_KEY")
# deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
# about_us_url = os.getenv("ABOUT_US_DOC_URL")
# methodology_url = os.getenv("METHODOLOGY_DOC_URL")

openai_api_key = st.secrets["OPENAI_API_KEY"]
deepseek_api_key = st.secrets["DEEPSEEK_API_KEY"]
about_us_url = st.secrets["ABOUT_US_DOC_URL"]
methodology_url = st.secrets["METHODOLOGY_DOC_URL"]

about_us_response = requests.get(about_us_url)
methodology_response = requests.get(methodology_url)

# List available LLM models
if "AZ_OPENAI_API_KEY" not in os.environ:
    MODELS = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "deepseek/deepseek-chat",
    ]
else:
    MODELS = ["azure-openai/gpt-4o"]

# Set Page Config
st.set_page_config(
    page_title="SysBuddy", 
    page_icon="🤖", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Set Header
st.html("""<h2 style="text-align: center;"><i> SysBuddy - AI Chatbot for System Management </i> </h2>""")

# Session State Initialisation
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I assist you?"}
]

# Initial password check
if not check_password():  
    st.stop()

# Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
missing_openai = openai_api_key == "" or openai_api_key is None
missing_deepseek = deepseek_api_key == "" or deepseek_api_key is None

if missing_openai and missing_deepseek:
    st.write("#")
    st.warning("⬅️ Please set API key in system environment and reboot app...")

# # --- Disclaimer ---
with st.expander("📢 **Disclaimer**"):
    st.markdown("""
    **IMPORTANT NOTICE:** This web application is developed as a proof-of-concept prototype.  
    The information provided here is **NOT** intended for actual usage and should not be relied upon for making any decisions, especially those related to **financial**, **legal**, or **healthcare** matters.

    Furthermore, please be aware that the **LLM may generate inaccurate or incorrect information**.  
    You assume full responsibility for how you use any generated output.

    Always consult with **qualified professionals** for accurate and personalized advice.
    """)

# # --- Sidebar ---
with st.sidebar:

    # -- Sidebar Section #1 --
    st.markdown(
        "<hr style='margin: 5px 0; padding: 0;'>",
        unsafe_allow_html=True
    )
    st.header("LLM Model:")

    models = []
    for model in MODELS:
        if "openai" in model and not missing_openai:
            models.append(model)
        elif "deepseek" in model and not missing_deepseek:
            models.append(model)

    st.selectbox(
        "🤖 Select a Model", 
        options=models,
        key="model",
    )

    cols0 = st.columns(2)
    with cols0[0]:
        is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
        st.toggle(
            "Use RAG", 
            value=is_vector_db_loaded, 
            key="use_rag", 
            disabled=not is_vector_db_loaded,
        )

    with cols0[1]:
        st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

    # -- Sidebar Section #2 --
    st.markdown(
        "<hr style='margin: 3px 0; padding: 0;'>",
        unsafe_allow_html=True
    )
    st.header("RAG Sources:")
        
    # File upload input for RAG with documents
    st.file_uploader(
        "📄 Upload a document", 
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
    )

    # URL input for RAG with websites
    st.text_input(
        "🌐 Introduce a URL", 
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
    )

    # -- Sidebar Section #3 --
    st.markdown(
        "<hr style='margin: 3px 0; padding: 0;'>",
        unsafe_allow_html=True
    )
    st.header("Knowledge Base:")
    with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
        st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

    # -- Sidebar Section #4 --
    st.markdown(
        "<hr style='margin: 3px 0; padding: 0;'>",
        unsafe_allow_html=True
    )
    st.header("Documentation:")
    if about_us_response.status_code == 200:
        st.download_button(
            label="📄 Download About Us",
            data=about_us_response.content,
            file_name="About_Us.pdf",
            mime="application/pdf",
            key="about_us_download"
        )
    else:
        st.error("❌ Failed to load About Us document.")
    
    # Download button for Methodology
    if methodology_response.status_code == 200:
        st.download_button(
            label="📄 Download Methodology",
            data=methodology_response.content,
            file_name="Methodology.pdf",
            mime="application/pdf",
            key="methodology_download"
        )
    else:
        st.error("❌ Failed to load Methodology document.")

# --- Main Chat App ---
model_provider = st.session_state.model.split("/")[0]
if model_provider == "openai":
    llm_stream = ChatOpenAI(
        api_key=openai_api_key,
        model_name=st.session_state.model.split("/")[-1],
        temperature=0.3,
        streaming=True,
    )
elif model_provider == "deepseek":
    llm_stream = ChatDeepSeek(
        api_key=deepseek_api_key,
        model=st.session_state.model.split("/")[-1],
        temperature=0.3,
        streaming=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

        if not st.session_state.use_rag:
            st.write_stream(stream_llm_response(llm_stream, messages))
        else:
            st.write_stream(stream_llm_rag_response(llm_stream, messages))
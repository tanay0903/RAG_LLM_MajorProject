# import basics
import os
from dotenv import load_dotenv
import time

# import google generative ai
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# import streamlit
import streamlit as st

# import langchain
from langchain.agents import AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.tools import tool
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# import supabase db
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# initiating supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Streamlit page configuration
st.set_page_config(page_title="Document Genie", layout="wide")

# Sidebar Instructions
st.sidebar.markdown("""
## Document Genie: Instant Insights from Your Documents

This chatbot uses Google's Generative AI (Gemini-PRO) to process and analyze uploaded PDFs for quick insights.

### How to Use:
1. Upload your PDF,DOCX, or TXT files documents.
2. Submit and ask questions for precise answers based on the content.
""")

# initiating embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# initiating vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="files",
    query_name="match_documents",
)
# initiating llm
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest",
    temperature=0,
    convert_system_message_to_human=True,  # Important for chat prompts
    #stream=False
)

# pulling prompt from hub
prompt = hub.pull("hwchase17/openai-functions-agent")


# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# initiating streamlit app
st.title("🧞 Docs_Genie")

import os
import time
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sidebar for Upload
st.sidebar.title("📄 Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF, DOCX, or TXT files", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True,
    key="file_uploader",   # Important to assign a key
)

# Initialize temp variables
if "uploaded_file_paths" not in st.session_state:
    st.session_state.uploaded_file_paths = []
if "files_processed" not in st.session_state:
    st.session_state.files_processed = False

upload_dir = os.path.join("uploads", "temp")
os.makedirs(upload_dir, exist_ok=True)

# Handle uploaded files ONLY ONCE
if uploaded_files and not st.session_state.files_processed:
    all_docs = []

    for uploaded_file in uploaded_files:
        temp_file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.uploaded_file_paths.append(temp_file_path)

        # Decide loader based on file type
        if uploaded_file.name.endswith(".pdf"):
            loader = PyMuPDFLoader(temp_file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(temp_file_path, encoding="utf-8")
        else:
            st.warning(f"Unsupported file format: {uploaded_file.name}")
            continue

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        all_docs.extend(docs)

    # Add to vector store
    if all_docs:
        vector_store.add_documents(all_docs)
        
        success_placeholder = st.empty()
        success_placeholder.success(f"✅ Uploaded and processed {len(uploaded_files)} file(s) successfully!")
        time.sleep(3)
        success_placeholder.empty()

    st.session_state.files_processed = True

# Deleting Files when no uploaded files
if not uploaded_files and st.session_state.files_processed:
    # Files were uploaded before but now removed
    for temp_file_path in st.session_state.uploaded_file_paths:
        try:
            os.remove(temp_file_path)
            info_placeholder = st.empty()
            info_placeholder.info(f"🧹 Cleaned up temporary file: {os.path.basename(temp_file_path)}")
            time.sleep(2)
            info_placeholder.empty()
        except Exception as e:
            st.warning(f"⚠️ Could not delete temporary file: {e}")

    # Reset the session state after all files are removed
    st.session_state.uploaded_file_paths = []
    st.session_state.files_processed = False

# Keep track of which files are removed
if uploaded_files and len(uploaded_files) < len(st.session_state.uploaded_file_paths):
    # If files are removed but new ones are still uploaded
    removed_files = set(st.session_state.uploaded_file_paths) - set([os.path.join(upload_dir, f.name) for f in uploaded_files])
    for file in removed_files:
        info_placeholder = st.empty()
        info_placeholder.info(f"🧹 Cleaned up temporary file: {os.path.basename(file)}")
        time.sleep(2)
        info_placeholder.empty()

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# create the bar where we can type messages
user_question = st.chat_input("How are you?")

# did the user submit a prompt?
if user_question:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))


    # invoking the agent
    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})

    ai_message = result["output"]

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
# from langchain.embeddings import openai
from htmlTemplate import css, bot_template, user_template



def get_pdf_text(pdf_docs):
    text = " "
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
                text += page.extract_text()
    return text        

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore 

def get_conversation_chain(vectorstore):
    #llm = HuggingFaceHub(repo_id="mistralai/mathstral-7B-v0.1", model_kwargs={"temperature":0 , "max_length":512})
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    #llm = HuggingFaceHub(repo_id="meta-llama/Llama-2-7b", model_kwargs={"temperature":0 , "max_length":512})
    memory = ConversationBufferMemory(memory_key='Chat_History', return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(),
        memory=memory
    )
    return conversational_chain
    
def handle_userinput(user_question):
    response = st.session_state.conversation({"question":user_question})
    # st.write(response)
    st.session_state.Chat_History = response['Chat_History']

    for i, message in enumerate(st.session_state.Chat_History):
        if i & 2 == 0:
            st.write(user_template.replace("{{MSG}}" , message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}" , message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDF",page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "Chat_History" not in st.session_state:
        st.session_state.Chat_History = None

    st.header("Chat With multi PDF :books:")
    user_question = st.text_input("Ask a question about Your Documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Document")
        pdf_docs = st.file_uploader("UPLOAD UR PDF HERE & CLICK ON 'PROCESS'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("processing"):
                # get pdf
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain 
                st.session_state.conversation = get_conversation_chain(vectorstore)
                

if __name__ == '__main__':
    main()

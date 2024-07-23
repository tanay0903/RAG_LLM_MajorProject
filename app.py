import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
#from langchain.embeddings import openai
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval



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
    llm = ""
    memory = ConversationBufferMemory(memory_key='Chat_History', return_messages=True)
    conversation_chain = conversational_retrieval.from_llm(
        llm=llm
        )




def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDF",page_icon=":books:")

    st.header("Chat With multi PDF :books:")

    st.text_input("Ask a question about Your Documents:")

    with st.sidebar:
        st.subheader("Your Document")
        pdf_docs = st.file_uploader("UPLOAD UR PDF HERE & CLICK ON 'UPLOAD'", accept_multiple_files=True)

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
                conversation = get_conversation_chain(vectorstore)
                
    

if __name__ == '__main__':
    main()

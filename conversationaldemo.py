import os
from json import load
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr
import streamlit as st
from torch import mode
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain



def get_llm():
    groq_api_key = st.secrets["GROQ_API_KEY"]
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=SecretStr(grop_api_key))


def get_embedding():
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.set_page_config(page_title="Demo",layout="wide")
st.title("Converstional Chat")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


#side bar
with st.sidebar:
    st.header("Data Source")
    embedding=get_embedding()
    soure_type =st.radio("Choose data source",["Web URL","PDF"])
    if soure_type =="Web URL":
        web_url = st.text_input("Enter URL:", placeholder="https://example.com/article")
        if st.button("Load Web Data") and web_url:
            try:
                with st.spinner("loading data"):
                    loader = WebBaseLoader(web_path=web_url)
                    docs = loader.load()
                    if not docs or not docs[0].page_content.strip():
                        st.error("No content found at this URL. Check the link and try again.")
                    else:
                        spilt  = splitter.split_documents(docs)
                        st.session_state["vector_db"] = Chroma.from_documents(spilt,embedding=embedding)
                        st.session_state["data_source"] = web_url
                        st.success("Web data loaded!")
            except ConnectionError:
                st.error("No internet connection. Please check your network and try again.")
            except Exception as e:
                st.error(f"Failed to load web data: {e}")
    else:
         uploaded_file = st.file_uploader("upload the pdf",type ="pdf")
         st.info("loaded from pdf")
         if st.button("Upload the pdf") and uploaded_file:
             tmp_path = f"tmp_{uploaded_file.name}"
             try:
                 with st.spinner("uploading file"):
                     with open(tmp_path ,"wb") as f:
                         f.write(uploaded_file.getbuffer())
                     docs = PyPDFLoader(tmp_path).load()
                     if not docs:
                         st.error("PDF appears to be empty or unreadable.")
                     else:
                         split = splitter.split_documents(docs)
                         st.session_state["vector_db"] = Chroma.from_documents(split, embedding=embedding)
                         st.session_state["data_source"] = uploaded_file.name
                         st.success("loaded from pdf")
             except Exception as e:
                 st.error(f"Failed to load PDF: {e}")
             finally:
                 if os.path.exists(tmp_path):
                     os.remove(tmp_path)
                 
# hardcode session 
st.session_state.get("current_session", "sesssion")
 
         
if "data_source" in st.session_state:
    st.info(f"Selected Data Source {st.session_state['data_source']}", )
    
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Re-render all previous messages on every rerun
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your data...")
if prompt:
    if "vector_db" not in st.session_state:
        st.error("Please load a data source first (Web URL or PDF) from the sidebar.")
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        chat_history = []
        for msg in st.session_state["messages"][:-1]:
            if msg["role"]=="user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))
        llm = get_llm()
        
        #retriver
        retriver=st.session_state["vector_db"].as_retriever()
        
       
        context_aware_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ])
        
        rag_chain = create_retrieval_chain(
            create_history_aware_retriever(llm,retriver,context_aware_prompt),
            create_stuff_documents_chain(llm,qa_prompt)
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = rag_chain.invoke({"input": prompt, "chat_history": chat_history})["answer"]
                st.markdown(answer)

        # Save assistant response so it persists across reruns
        st.session_state["messages"].append({"role": "assistant", "content": answer})

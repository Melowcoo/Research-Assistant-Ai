# Gerekli kütüphaneleri yükleyin
import os
import streamlit as st
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# OpenAI API anahtarını tanımlayın
os.environ['OPENAI_API_KEY'] = "sk-proj-rHgUN0wOjktmwnP4p2A9T3BlbkFJuqjlN84vkXRp3S15Ajkw"
load_dotenv()

# URL'den vektör deposuna veri alın
def get_vectorstore_from_url(url):
    """
    Belirtilen URL'den belge yükleyerek ve parçalayarak vektör deposunu oluşturur.
    """
    loader = WebBaseLoader(url)
    document = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

# Bağlam alıcı zincirini alın
def get_context_retriever_chain(vector_store):
    """
    Kullanıcının girdiği soruya yanıt bulabileceği vektör deposunu dökümanlarını araştırır. Ranking
    """
    llm = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Yukarıdaki konuşmaya göre, konuşmayla ilgili bilgi almak için bir arama sorgusu oluştur")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt) #geçmişi de bir döküman olarak alır.
    
    return retriever_chain
    
# Konuşmalı RAG zincirini alın
def get_conversational_rag_chain(retriever_chain): 
    """
    Alıcı zincirini kullanarak bağlam duyarlı yanıtlar sağlamak için bir alıcı-iyileştirilmiş üretim (RAG) zinciri oluşturur.
    """
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Aşağıdaki bağlama dayanarak kullanıcının sorularını yanıtla:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Yanıt alın
def get_response(user_input):
    """
    Kullanıcının girişine göre bir yanıt döndürür.
    """
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# Uygulama yapılandırması
st.set_page_config(page_title="ChatWeb", page_icon="🤖")
st.title("Web Siteleri ile Sohbet")

# Yan panel
with st.sidebar:
    st.header("Ayarlar")
    website_url = st.text_input("Web Sitesi URL'si")

if website_url is None or website_url == "":
    st.info("Lütfen bir web sitesi URL'si girin")

else:
    # Oturum durumu
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Merhaba, Ben bir botum. Size nasıl yardımcı olabilirim?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)    

    # Kullanıcı girdisi
    user_query = st.chat_input("Mesajınızı buraya yazın...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
       

    # Sohbet
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("Yapay Zeka"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("İnsan"):
                st.write(message.content)

import streamlit as st
import openai
import os
import json
import pandas as pd
import faiss
import numpy as np
import requests
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Load OpenAI API Key from Streamlit Secrets
api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI()

# Function to load CSV from GitHub
@st.cache_data
def load_csv_from_github():
    github_url = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO/main/restaurant_data_3.csv"
    df = pd.read_csv(github_url)

    df = df.fillna({
        "restaurant_name": "Unknown",
        "menu_item": "Unknown Item",
        "menu_category": "Uncategorized item",
        "menu_description": "No menu description available",
        "ingredients": "Unknown ingredients",
        "price_description": "Price not listed",
        "number_of_reviews": "reviews unknown",
        "rating_description": "No rating available",
        "category_description": "No category description",
        "location": "Unknown location",
        "rating": 0  
    })

    return df

csv_documents = load_csv_from_github()

# Function to load Wikipedia JSON from GitHub
@st.cache_data
def load_wikipedia_data():
    github_url = "https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPO/main/wikipedia_restaurant_knowledge.json"
    
    try:
        response = requests.get(github_url)
        response.raise_for_status()
        data = response.json()
        st.sidebar.success(f"✅ Successfully loaded {len(data)} Wikipedia articles.")
        return data
    except Exception as e:
        st.sidebar.error(f"❌ Error loading Wikipedia data: {e}")
        return []

wikipedia_data = load_wikipedia_data()

# Function to create FAISS index
@st.cache_resource
def create_faiss_index(_documents):
    if not _documents:
        st.sidebar.warning("⚠️ No documents available to store in FAISS.")
        return None
    embedding_model = HuggingFaceEmbeddings()
    return FAISS.from_documents(_documents, embedding_model)

# Create FAISS vector store
all_documents = csv_documents.to_dict(orient="records") + wikipedia_data if csv_documents is not None and wikipedia_data is not None else []
vector_store = create_faiss_index(all_documents)

st.title('🍽️ MenuData Chatbot')
st.markdown('This chatbot helps answer questions about restaurants, their menus, and ingredients.')

query = st.text_input('Ask a question about a restaurant or menu:')

if st.button('Submit'):
    if query.strip():
        retrieved_chunks = vector_store.similarity_search(query, k=5) if vector_store else []
        answer = "I do not have enough information to answer this query." if not retrieved_chunks else "

".join([doc.page_content for doc in retrieved_chunks])
        
        if answer:
            st.success("✅ Chatbot response generated successfully!")
            st.subheader('🤖 AI Response:')
            st.text_area('', answer, height=150)
        else:
            st.warning("⚠️ No response generated. Please try a different query.")
    else:
        st.warning('⚠️ Please enter a valid question.')

import os
import json
import pandas as pd
import faiss
import numpy as np
import streamlit as st
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download
import openai

# Load OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load Restaurant Data
@st.cache_data
def load_restaurant_data():
    file_path = "./Data/restaurant_data_3.csv"
    return pd.read_csv(file_path)

df = load_restaurant_data()

# Load Wikipedia Data
SAVE_DIR = os.path.join("Data", "Wikipedia_data2")
FILE_PATH = os.path.join(SAVE_DIR, "wikipedia_restaurant_knowledge.json")

@st.cache_data
def load_wikipedia_data():
    try:
        with open(FILE_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"❌ Error loading Wikipedia data: {e}")
        return []

wikipedia_data = load_wikipedia_data()

# Chunk Wikipedia articles
def chunk_wikipedia_articles(data, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for article in data:
        title = article["title"]
        url = article["url"]
        text = article["summary"]
        if text:
            split_texts = text_splitter.split_text(text)
            for chunk in split_texts:
                chunks.append({"title": title, "url": url, "chunk": chunk})
    return chunks

chunked_wikipedia_data = chunk_wikipedia_articles(wikipedia_data)

# Load Sentence Transformer model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Generate embeddings and store in FAISS
chunks_texts = [chunk["chunk"] for chunk in chunked_wikipedia_data]
vectors = embedding_model.embed_documents(chunks_texts)

dimension = len(vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedding_model.embed_query(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = [chunked_wikipedia_data[i] for i in indices[0]]
    return results

# Function to generate response using GPT-4o
def generate_response(query, retrieved_chunks):
    if not retrieved_chunks:
        return "I do not have enough information to answer this query."

    context_text = "\n\n".join([
        f"({i+1}) [{chunk['title']}] {chunk['chunk']}" 
        for i, chunk in enumerate(retrieved_chunks)
    ])

    system_message = f"""
    ### 🍽️ AI Assistant for Restaurant & Culinary Queries

    **User Query:** {query}

    **Relevant Context:**
    {context_text}

    **AI Response:**
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response["choices"][0]["message"]["content"]

# Streamlit UI
st.set_page_config(page_title="🍽️ Restaurant & Culinary AI", page_icon="🍽️")

st.title("🍽️ Restaurant & Culinary AI Chatbot")
st.write("Ask a question about restaurants, menus, or ingredients!")

query = st.text_input("🔍 Enter your query:")

if st.button("Submit"):
    if query:
        with st.spinner("Fetching relevant information..."):
            retrieved_chunks = retrieve_relevant_chunks(query)
            answer = generate_response(query, retrieved_chunks)
            st.write(f"**Answer:** {answer}")

            # Display relevant sources
            st.write("### 🔎 Sources Used:")
            for chunk in retrieved_chunks:
                st.write(f"📌 **{chunk['title']}**: {chunk['chunk'][:200]}... ([Read more]({chunk['url']}))")
    else:
        st.warning("⚠️ Please enter a query.")

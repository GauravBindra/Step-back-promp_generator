import streamlit as st



import openai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("✅ API Key loaded successfully.")
else:
    print("❌ API Key not found. Check your .env file.")



client = openai.OpenAI()

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
from llama_cpp import Llama

import pandas as pd
import re
from langchain.schema import Document

def categorize_menu_item(category):
    """Assigns each row to a general menu category based on known mappings."""
    if pd.isna(category) or category.strip() == "":
        return "uncategorized"  # Handle missing values

    category = category.lower().strip()
    category = re.sub(r"[-_/]", " ", category)  # Normalize by replacing special characters

    # Mapping specific terms to generalized categories
    category_mapping = {
        "appetizers": ["small plates", "starters", "snacks", "antojitos", "dumplings", "wings", "charcuterie boards", "catering appetizers"],
        "main dishes": ["mains", "entrée", "specials", "meat & seafood", "dinner plates", "platos", "house specials", "chef’s recommendations", "hot pressed sandwiches", "bento box"],
        "breakfast": ["breakfast", "brunch", "breakfast omelettes", "breakfast pancakes", "breakfast burritos", "breakfast sandwiches"],
        "burgers & sandwiches": ["burgers", "sandwiches", "biscuit sandwiches", "hot pressed sandwiches"],
        "pasta & pizza": ["pasta", "pizza", "plates & pasta", "mac & cheese"],
        "rice & noodles": ["rice bowls", "rice platters", "fried rice", "pho", "ramen", "vermicelli bowls", "noodle soups", "pad thai", "japanese noodle", "chow mein", "rice & biryani", "tandoori specials"],
        "salads": ["salads", "salads & sides", "greens", "salads & vegetables"],
        "soups": ["soup", "soup & salads", "noodle soup", "vermicelli & noodle soup", "wonton soup"],
        "seafood": ["seafood", "pescado", "fish & chips", "shrimp", "ceviche", "sashimi", "sushi rolls"],
        "mexican cuisine": ["tacos", "burritos", "quesadillas", "enchiladas", "tamales", "taqueria", "mexican tortas"],
        "desserts": ["dessert", "sweets", "desserts & pastries", "pastries & sweets", "tarts", "cakes", "ice cream"],
        "drinks & beverages": ["beverages", "drinks", "coffee", "smoothies", "fruit teas", "milk tea", "non-alcoholic beverages"],
        "vegan & vegetarian": ["vegan menu", "vegetarian", "vegetarian rolls", "plant-based", "vegan sushi"],
        "bar & alcoholic beverages": ["beer", "wine", "cocktails", "sake", "cava", "red wine", "beer & cider", "boozy teapots"]
    }

    # Assign to general category
    for general_category, variations in category_mapping.items():
        if category in variations:
            return general_category

    return "other"  # If not found, assign "other"


@st.cache_data
def load_csv_locally():
    """
    Loads the restaurant CSV from GitHub and converts it into structured text documents.
    """
    local_csv_path = "restaurant_data_3.csv"
    
    df = pd.read_csv(local_csv_path)

    # Ensure missing values are replaced
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
        "rating": 0  # Default rating as numeric 0 instead of NaN
    })

    # Handle missing or incorrectly formatted category_list
    if "category_list" in df.columns:
        df["category_list"] = df["category_list"].apply(lambda x: x if isinstance(x, list) else [])
    else:
        df["category_list"] = [[]] * len(df)

    df["menu_category"] = df["menu_category"].apply(categorize_menu_item)

    documents = []
    
    for _, row in df.iterrows():
        ingredients_list = [ing.strip().lower() for ing in str(row['ingredients']).split(",") if ing.strip()]
        
        text_representation = f"""
        Restaurant: {row['restaurant_name']}
        Menu Item: {row['menu_item']}
        Category: {row['menu_category']}
        Description: {row['menu_description']}
        Ingredients: {', '.join(ingredients_list)}
        Price: {row['price_description']}
        Number of reviews: {row['number_of_reviews']}
        Rating Summary: {row['rating_description']}
        Category Description: {row['category_description']}
        """

        metadata = {
            "source": "csv",
            "restaurant_name": row["restaurant_name"],
            "location": row["location"],
            "rating": float(row["rating"]),
            "categories": row["category_list"],
            "ingredients": ingredients_list,
            "menu_category": row["menu_category"]
        }

        document = Document(page_content=text_representation.strip(), metadata=metadata)
        documents.append(document)

    return documents

csv_documents = load_csv_locally()

import json
import requests
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import json

@st.cache_data
def load_wikipedia_data():
    """
    Loads Wikipedia restaurant knowledge data from a local JSON file.
    """
    local_json_path = "wikipedia_restaurant_knowledge.json"  # Local file path

    try:
        with open(local_json_path, "r", encoding="utf-8") as file:
            data = json.load(file)  # Load JSON data
        st.sidebar.success(f"✅ Successfully loaded {len(data)} Wikipedia articles.")
        return data
    except Exception as e:
        st.sidebar.error(f"❌ Error loading Wikipedia data: {e}")
        return []


# Load Wikipedia data from GitHub
wikipedia_data = load_wikipedia_data()

# ------------------------- Chunk Wikipedia Articles -------------------------
def chunk_wikipedia_articles(data, chunk_size=500, chunk_overlap=50):
    """Chunk Wikipedia articles into smaller LangChain Document objects with metadata."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    documents = []
    
    for article in data:
        title = article["title"]
        url = article["url"]
        text = article["summary"]
        
        if text:
            split_texts = text_splitter.split_text(text)
            for chunk in split_texts:
                # Convert each chunk into a LangChain Document with metadata
                doc = Document(
                    page_content=chunk,
                    metadata={"title": title, "url": url, "source": "wikipedia"}  # Metadata for filtering
                )
                documents.append(doc)
    
    return documents

# Convert Wikipedia data to LangChain Documents
wikipedia_documents = chunk_wikipedia_articles(wikipedia_data)

# Print success message in Streamlit UI
st.sidebar.success(f"✅ Successfully loaded {len(wikipedia_documents)} Wikipedia document chunks.")


# Combine CSV and Wikipedia documents
all_documents = csv_documents + wikipedia_documents if csv_documents and wikipedia_documents else []

# Display success message in Streamlit sidebar
if all_documents:
    st.sidebar.success(f"✅ Total documents loaded: {len(all_documents)} (CSV + Wikipedia)")
else:
    st.sidebar.warning("⚠️ No documents loaded. Please check your GitHub data sources.")


import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Ensure OpenAI API key is loaded from Streamlit secrets
# openai_api_key = st.secrets["openai_api_key"]

# Use OpenAI's embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)

@st.cache_resource
def create_faiss_index(_documents):
    """Create and store FAISS index in memory for retrieval."""
    if not _documents:
        st.sidebar.warning("⚠️ No documents available to store in FAISS.")
        return None
    return FAISS.from_documents(_documents, embedding_model)


# Store Wikipedia + CSV text in FAISS
vector_store = create_faiss_index(all_documents)

if vector_store:
    st.sidebar.success(f"✅ Stored {len(all_documents)} document chunks in FAISS using OpenAI embeddings.")
else:
    st.sidebar.error("❌ FAISS index creation failed. No documents available.")


import re

def get_dynamic_k(query):
    """
    Determines the value of k dynamically based on query intent, complexity, and keyword patterns.
    """
    query = query.lower().strip()
    query_length = len(query.split())  # Count words in query

    # Trend & Comparison Queries (Require more data points)
    if re.search(r"\b(compare|trend|change over time|growth|evolution|pattern)\b", query):
        return 25 if query_length > 8 else 20  # Boost k for longer, complex queries

    # Listing & Discovery Queries (Need a reasonable number of results)
    if re.search(r"\b(find|list|recommend|show me|best|top|near me)\b", query):
        return 15 if query_length > 6 else 10  # Adjust based on specificity

    # General Lookup Queries (Need precise, fewer results)
    if re.search(r"\b(what is|who is|define|meaning of|describe|explain)\b", query):
        return 8 if query_length > 10 else 5  # Smaller k for focused questions

    # Default Fallback (Safe choice)
    return 5  


def retrieve_relevant_chunks(query: str, top_k=5):
    """
    Retrieves relevant chunks from FAISS based on similarity search.
    """
    # filters_applied = extract_filters(query, csv_file_path)
    k = get_dynamic_k(query)

    # Retrieve raw FAISS results
    results = vector_store.similarity_search(query, k=k)

    return results  # Directly return FAISS results without boosting or filtering



def generate_response(query: str, retrieved_chunks: list) -> str:
    """
    Generates a response using OpenAI's GPT-4o with structured and unstructured context.
    """
    if not retrieved_chunks:
        return "I do not have enough information to answer this query."

    # Extract relevant context from retrieved documents
    context_text = "\n\n".join([
        f"({i+1}) [{doc.metadata.get('source', 'unknown')}] {doc.page_content}" 
        for i, doc in enumerate(retrieved_chunks)
    ])
    # print(context_text)
    # Updated system prompt with chatbot purpose and answering guidelines
    system_message = f"""
    ### 🤖 Chatbot Purpose:
    
    This AI chatbot helps users explore culinary-related insights.  
    It uses 'Retrieved Context' from menu-data from various Restaurants as well as related Wikipedia articles to generate helpful and fact-based responses.

    ### 🎯 Key Functions:
    - **Identify and Recommend Trending Flavors & Ingredients**  
      - Analyze menu data to highlight **popular and emerging food trends**.  

    - **Provide Menu Inspiration for New Recipes**  
      - Help users explore **new ingredients and innovative culinary pairings**.

    - **Help Users with Restaurant & Dish Recommendations**  
      - Recommend restaurants and dishes based on user preferences (**restaurant category, dish, location, dietary needs, price, rating**).  

    ### 📖 Answering Guidelines:
    - **Step 1: Identify the Query Type**  
      - Determine if the query is about a **specific restaurant, menu item, pricing, or general food knowledge**.

    - **Step 2: Select the Most Reliable Source**  
      - If the query is about **a restaurant, dish, or pricing**, use structured restaurant data (**CSV**).  
      - If the query is about **food history, cultural significance, or trends**, use **Wikipedia**.  

    - **Step 3: Handling Incomplete, General, or Unavailable Information**  
      - If the retrieved data **partially answers the query**, provide what is available and clarify missing details.  
      - If only **general food knowledge is found**, state that **restaurant-specific data is unavailable**.  
      - If **no relevant data exists**, clearly say: **"I do not have enough information to answer this query."**  
      - **The model must rely on the provided data and avoid making assumptions or speculative responses.**  

    - **Step 4: Maintain Clarity & Present Information Effectively**  
      - Ensure responses are **concise, structured, and easy to understand**.  
      - Prioritize **factual accuracy** and **avoid unnecessary details**.  
      - If the query expects a **list, comparison, or summary**, format the response accordingly for readability.  

    ### ❓ User Query:
    {query}

    ### 📖 Retrieved Context:
    {context_text}

    ### ✅ AI Response:
    """

    # OpenAI API Call (GPT-4o)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content

st.title('🍽️ Restaurant Chatbot')
st.markdown('This chatbot helps answer questions about restaurants, their menus, and ingredients.')

# User input
query = st.text_input('Ask a question about a restaurant or menu:')

# Process query on button click
if st.button('Submit'):
    if query.strip():  # Ensuring the query is not empty
        retrieved_chunks = retrieve_relevant_chunks(query)
        answer = generate_response(query, retrieved_chunks)

        # Display chatbot response
        if answer:
            st.success("✅ Chatbot response generated successfully!")
            st.subheader('🤖 AI Response:')
            st.text_area('', answer, height=150)
        else:
            st.warning("⚠️ No response generated. Please try a different query.")
    else:
        st.warning('⚠️ Please enter a valid question.')


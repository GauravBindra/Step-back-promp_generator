import openai
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env (for local development)
load_dotenv()

# Load API key from environment variable or Streamlit secrets
api_key = os.getenv("OPENAI_API_KEY")  # For local deployment

# Streamlit Secrets fallback (for hosted deployment)
if not api_key:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

if not api_key:
    st.error("🚨 API key not found! Set it as an environment variable or in Streamlit Secrets.")
    st.stop()

# Streamlit UI Setup
st.title("🔄 Step-Back Prompting Chatbot")
st.markdown("### Convert your query into a more general, insightful step-back question!")

# User Input Query
query = st.text_area("📝 Enter your query:")

# Function to generate a step-back query
def generate_step_back_query(original_query):
    prompt = f"""
You are an expert in reformulating user queries into more general questions that better capture the user's intent. 

Your task is to generate a **step-back query**, which is a more abstract and generalized version of the original query. The step-back query should:
- **Preserve the core meaning** of the original query.
- **Uncover implicit sub-questions** that the user might be asking.
- **Rephrase specific details into broader concepts** where appropriate.
- **Make retrieval easier by bridging the gap between the user's query and the information they seek.**
- **Support multi-sentence reformulations** if necessary.

Always output the reformulated question with the prefix **"Step-Back Query:"** followed by the generated query.

### **Examples:**

**Example 1:**  
Original Query: *What are the chemical properties of the element discovered by Marie Curie?*  
Step-Back Query: *What elements were discovered by Marie Curie, and what are their chemical properties?*

**Example 2:**  
Original Query: *Why does my LangGraph agent `astream_events` return a long trace instead of the expected output?*  
Step-Back Query: *How does the `astream_events` function work in LangGraph agents?*

**Example 3:**  
Original Query: *Which school did Estella Leopold attend from August to November 1954?*  
Step-Back Query: *What is the educational history of Estella Leopold?*

**Example 4:**  
Original Query: *Why are Korean BBQ tacos such a big thing in Los Angeles right now?*  
Step-Back Query: *What sparked the popularity of Korean BBQ tacos, and how does their success in Los Angeles reflect the rise of fusion cuisine trends across the country?*

### **Now, generate a step-back query for the following user input:**
Original Query: "{original_query}"
Step-Back Query:
"""

    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)

        # OpenAI API Call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )

        # Extract the response
        step_back_query = response.choices[0].message.content.strip()
        return step_back_query

    except Exception as e:
        return f"⚠️ Error: {e}"

# Button to generate the step-back query
if st.button("Generate Step-Back Query"):
    if query:
        with st.spinner("🔄 Generating Step-Back Query..."):
            step_back_query = generate_step_back_query(query)
        
        # Display the result
        st.markdown("## 🔍 **Query vs Step-Back Query Comparison**")
        st.markdown(f"### **Original Query:**\n> {query}")
        st.markdown(f"### **Generated Step-Back Query:**\n> {step_back_query}")

    else:
        st.warning("🚨 Please enter a query.")

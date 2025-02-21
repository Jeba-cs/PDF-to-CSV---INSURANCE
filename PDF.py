import streamlit as st
import json
import requests
from PyPDF2 import PdfReader
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import datetime
import pandas as pd
import time  # Import for rate limiting
from db_utils import log_data_to_arctic, calculate_token_count  # Assuming this contains necessary functions

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="insurance_db")
collection = chroma_client.get_or_create_collection(name="insurance_embeddings")

# Function: Extract text from PDF
def pdf_process(file):
    pdf_reader = PdfReader(file)
    file_content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return file_content

# Function: Chunk Text
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    return text_splitter.split_text(text)

# Function: Generate Embeddings
def generate_embeddings(text):
    return embedding_model.encode(text).tolist()

# Function: Store Vectorized Chunks in ChromaDB
def store_chunks(chunks):
    for i, chunk in enumerate(chunks):
        embedding = generate_embeddings(chunk)
        chunk_id = str(len(collection.get().get("ids", [])))
        collection.add(ids=[chunk_id], embeddings=[embedding], metadatas=[{"text": chunk}])
        log_data_to_arctic("Vectorization", chunk, json.dumps({"embedding": embedding}), 0, calculate_token_count(chunk))

# Function: Query Stored Embeddings
def query_embeddings(query_text, top_k=3):
    query_embedding = generate_embeddings(query_text)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results

# Function: Chatbot API with Chain of Thought Reasoning (with rate limiting and error handling)
def chatbot_with_deepseek(context, user_query, api_key, max_retries=3):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": f"""Use the following retrieved context to analyze the user's question step-by-step and 
            provide an accurate response.\n\nContext: {context}\n\nQuestion: {user_query}\n\nStep 1: Identify key 
            insurance-related entities in the question.\nStep 2: Search the provided context for matching 
            details.\nStep 3: Formulate a structured response based on the retrieved data."""}]}

    tokens_used = calculate_token_count(payload["messages"][0]["content"])  # Calculate token count for efficiency
    start_time = datetime.datetime.utcnow()

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            response_json = response.json()
            response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "❌ No response")
            response_time = (datetime.datetime.utcnow() - start_time).total_seconds() * 1000

            log_data_to_arctic("DeepSeek Agent", payload["messages"][0]["content"], response_text, response_time, tokens_used)
            return response_text
        except requests.exceptions.RequestException as e:
            wait_time = 2 ** attempt  # Exponential backoff
            st.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    return "❌ Failed to get response from DeepSeek API after multiple retries."

# Define DeepSeek Agent Tool - This now directly calls chatbot_with_deepseek
def deepseek_agent_tool(query, api_key):
    results = query_embeddings(query, top_k=3)
    retrieved_chunks = [meta.get("text", "N/A") for meta in results["metadatas"][0] if isinstance(meta, dict)]

    if not any(retrieved_chunks):
        st.warning("⚠️ No relevant chunks found in vector database.")
        return "No relevant data found in the document."

    context = "\n".join(retrieved_chunks)

    st.markdown("### 📄 Retrieved Context from PDF")
    for i, chunk in enumerate(retrieved_chunks):
        st.markdown(f"**Chunk {i+1}:** {chunk[:300]}...")

    response = chatbot_with_deepseek(context, query, api_key)
    return response

def extract_insurance_data(pdf_text, api_key, max_retries=3):
    fields_to_extract = {
        "Insured Name": "What is the name of the insured?",
        "DOB": "What is the date of birth of the insured?",
        "Insurance Type": "What type of insurance is this policy?",
        "Payments": "What are the payment details for this policy?",
        "Policy Expiry Date": "What is the expiry date of this policy?"
    }

    extracted_data = {}

    # Manually split the PDF text into individual documents based on known markers in the data
    documents = pdf_text.split("Sample Document")

    for doc_num, doc in enumerate(documents[1:], start = 1):  # Skip the first split, cause it is blank
        if not doc.strip():  # Skip blank documents
            continue

        doc = "Sample Document" + doc #add sample doc at the beginning

        temp_extracted_data = {}

        for field, query in fields_to_extract.items():
            # 1. Construct a focused prompt with document context and precise extraction instructions
            prompt = f"""
            You are an expert insurance data extractor. Given the following document, extract the {field}:

            Document: {doc}

            Provide ONLY the value for the {field}. If the information is not present, respond with '❌ Not found in document.'.
            """

            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}

            tokens_used = calculate_token_count(prompt)

            answer = None
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                response.raise_for_status()
                response_json = response.json()
                response_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "❌ Not found in document.")
                answer = response_text

                # 4. Basic post-processing to ensure consistency
                if "not found" in answer.lower():
                    answer = "❌ Not found in document."

            except requests.exceptions.RequestException as e:
                answer = f"❌ API request failed: {e}"

            temp_extracted_data[field] = answer

        # 5. Store Extracted data
        extracted_data[f"Sample Document {doc_num}"] = temp_extracted_data

    return extracted_data

# Streamlit UI Setup
st.set_page_config(page_title="Insurance PDF Analyzer", layout="wide")
st.title("📄 Insurance Document Analyzer with AI")

# File Upload
file = st.file_uploader("Upload an Insurance PDF", type=["pdf"])

# Initialize pdf_text
pdf_text = None

if file:
    st.success("✅ File uploaded successfully!")
    pdf_text = pdf_process(file)

    # Chunking and storing embeddings
    chunks = chunk_text(pdf_text)

    st.markdown(f"📊 **Total Chunks:** {len(chunks)}")

    if st.button("🔄 Convert to Embeddings"):
        store_chunks(chunks)
        st.success("✅ Embeddings created and stored in ChromaDB!")

# User Inputs for Query and API Key
user_query = st.text_input("💬 Ask about the insurance policy:")
deepseek_api_key = st.text_input("🔑 Enter DeepSeek API Key:", type="password")

# Choose between Chatbot or Downloading Excel File
option = st.selectbox("Choose an option:", ["Chatbot", "View/Download Excel"])

if option == "Chatbot":
    if deepseek_api_key and user_query and pdf_text:
        query_tokens = calculate_token_count(user_query)

        results = query_embeddings(query, top_k=3)

        retrieved_chunks = [meta.get("text", "N/A") for meta in results["metadatas"][0] if isinstance(meta, dict)]

        context = "\n".join(retrieved_chunks)

        context_tokens = calculate_token_count(context)

        total_tokens = query_tokens + context_tokens

        st.markdown(f"📝 **Estimated Token Usage:** {total_tokens} (Query: {query_tokens}, Context: {context_tokens})")

        agent_response = deepseek_agent_tool(user_query, deepseek_api_key)

        st.text_area("AI Response:", value=agent_response, height=200)

elif option == "View/Download Excel":
    if pdf_text:

        extracted_data = extract_insurance_data(pdf_text, deepseek_api_key)

        # extracted_data is now a dictionary of dictionaries, each inner dict representing a policy
        df = pd.DataFrame.from_dict(extracted_data, orient='index')  # So it is easier to read

        st.write(df)

        excel_file_name = "insurance_data.xlsx"

        df.to_excel(excel_file_name, index=True)  # Add index to the excel for easier reading

        with open(excel_file_name, "rb") as f:
            st.download_button(
                label="Download Excel File",
                data=f,
                file_name=excel_file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:

         st.warning("⚠️ Please upload a PDF file first.")

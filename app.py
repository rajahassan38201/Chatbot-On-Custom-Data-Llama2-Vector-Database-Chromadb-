import streamlit as st
import ollama
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

st.title("💬 Customer Support Chatbot")

# Load the JSON data
def load_qa_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

qa_data = load_qa_data('Data.json')  # Replace with your JSON filename

# Create sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Create embeddings and add to ChromaDB
embeddings = []
texts = []
ids = []

for i, qa in enumerate(qa_data):
    texts.append(qa['question'] + " " + qa["answer"])
    ids.append(str(i))

embeddings = model.encode(texts).tolist()

# Create ChromaDB client and collection
chroma_client = chromadb.Client()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

try:
    collection = chroma_client.get_or_create_collection(name="pakproperties_collection", embedding_function=sentence_transformer_ef)
    collection.add(
        embeddings=embeddings,
        ids=ids,
        documents=texts
    )
except Exception as e:
    st.error(f"Error initializing ChromaDB: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with PakProperties?"}]

# Write Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

# Function to retrieve from VectorDB
def retrieve_context(question, n_results=3):
    query_embedding = model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    context = ""
    for document in results['documents'][0]:
        context += document + "\n"
    return context

# Generator for Streaming Tokens
def generate_response(question):
    context = retrieve_context(question)
    prompt = f"Context:\n{context}\n\nUser Question: {question}\nAnswer:"

    try:
        response = ollama.chat(model='llama2:7b-chat-q4_0', stream=True, messages=[{"role": "user", "content": prompt}])
        for partial_resp in response:
            token = partial_resp["message"]["content"]
            st.session_state["full_message"] += token
            yield token
    except Exception as e:
        st.error(f"Error generating response from Ollama: {e}")
        yield "Sorry, I couldn't find an answer to your question."

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑").write(prompt)
    st.session_state["full_message"] = ""
    st.chat_message("assistant", avatar="🤖").write_stream(generate_response(prompt))
    st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})
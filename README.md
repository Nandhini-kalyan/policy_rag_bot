# Policy & HR FAQ Assistant (RAG + Streamlit)

An internal assistant that answers staff questions about school and HR policies using Retrieval‑Augmented Generation (RAG) with OpenAI and Streamlit.

## Overview

This app lets teachers and staff ask natural language questions about policies (admissions, fees, attendance, HR leave, conduct, transport) and returns answers grounded in the school’s own policy documents, not just the model’s training data. [web:71][web:74]

High-level flow:
- Ingest a policy text file and split it into smaller chunks.
- Create vector embeddings for each chunk using OpenAI embeddings.
- For each user question, retrieve the most relevant chunks via cosine similarity.
- Call an LLM with the retrieved context + question to generate a grounded answer. [web:71][web:74]

## Features

- RAG-style QA over internal documents (policies.txt).
- Handles queries about:
  - Admissions requirements and intakes
  - Tuition fees and payment rules
  - Student attendance and leave
  - Staff leave and HR policies
  - Code of conduct, communication, and transport
- Uses:
  - `text-embedding-3-small` for document and query embeddings.
  - `gpt-4o-mini` for answer generation.
- Simple chat-style interface with `st.chat_input` and message history. [web:80][web:81]

## File Structure

policy_rag_bot/
├── policy_bot_app.py # Streamlit app (chat UI + RAG answer logic)
├── rag_utils.py # Embedding, indexing, and retrieval utilities
├── data/
│ └── policies.txt # Sample policy data (school + HR policies)
└── requirements.txt # Dependencies


## Setup and Running Locally

1. Clone the repository and enter the folder:

- git clone https://github.com/Nandhini-kalyan/policy_rag_bot.git
- cd policy_rag_bot
  

2. Create and activate a virtual environment (optional but recommended).

3. Install dependencies:

pip install -r requirements.txt

4. Set your OpenAI API key in a `.env` file (do NOT commit this file):

- echo OPENAI_API_KEY=sk-... > .env

5. Ensure you have a `data/policies.txt` file with your policy content (sample included in this repo).

6. Run the Streamlit app:


7. Open the local URL (usually http://localhost:8501), then:
- Type a question like:
  - “What documents are required for new student admission?”
  - “How are tuition fees paid and when are they due?”
  - “What is the attendance requirement for students?”
  - “How should a teacher report sick leave to HR?”
- The app will retrieve relevant policy sections and generate a concise answer.

## How It Works (RAG Logic)

- `rag_utils.py`:
  - `load_policy_chunks()` loads `policies.txt` and splits it into manageable chunks.
  - `get_embeddings()` uses OpenAI embeddings to convert text into vectors.
  - `build_index()` stores normalized embeddings for all chunks in memory.
  - `retrieve_similar()` computes cosine similarity between the query embedding and document embeddings, and returns the top‑k chunks.

- `policy_bot_app.py`:
  - On startup, builds the index once using `@st.cache_resource`.
  - For each user question:
    - Retrieves top chunks with `retrieve_similar()`.
    - Creates a combined prompt with policy context + user question.
    - Calls `openai.chat.completions.create()` with a system prompt that instructs the model to answer only from the given context and say “I don’t know” if not covered. [web:71]


---

Author: Nandhini Kalyanasundaram  





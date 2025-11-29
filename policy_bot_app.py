import os
import streamlit as st
import openai
from dotenv import load_dotenv

from rag_utils import load_policy_chunks, build_index, retrieve_similar

load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"


@st.cache_resource
def get_index():
    chunks = load_policy_chunks("Data/policies.txt")
    return build_index(chunks)


def answer_question(query: str, index) -> str:
    """Use retrieved context + LLM to answer a policy question."""
    top_chunks = retrieve_similar(index, query, k=3)
    context_text = "\n\n---\n\n".join([c["text"] for c in top_chunks])

    system_prompt = """
You are an internal assistant for a group of schools in the UAE.
Answer questions about school and HR policies using ONLY the provided context.
If the answer is not in the context, say you don't know and ask the user to check with HR or Administration.
Be concise and clear.
""".strip()

    user_prompt = f"""
Context from policy documents:
{context_text}

Question:
{query}
""".strip()

    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return resp.choices[0].message.content.strip()


def main():
    st.set_page_config(page_title="Policy & HR FAQ Assistant", layout="wide")
    st.title("📚 Policy & HR FAQ Assistant")
    st.markdown(
        "Ask questions about admissions, fees, attendance, and HR policies. "
        "Answers are grounded in the internal policy documents."
    )

    index = get_index()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    query = st.chat_input("Ask about policies, leave, fees, admissions...")
    if query:
        # User message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(query, index)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------- LOAD MODELS --------
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    return tokenizer, model, embed

tokenizer, model, embed_model = load_models()

# -------- DATA --------
data = [
{
"text": "track order",
"type": "navigation",
"steps": ["Open app", "Go to My Orders", "Click Track Order"],
"images": ["orders_page.png", "track_button.png"]
},

{
"text": "cancel order",
"type": "navigation",
"steps": ["Open app", "Go to My Orders", "Click Cancel Order"],
"images": ["cancel_page.png"]
},

{
"text": "payment failed",
"type": "general",
"answer": "Refund will be processed in 3-5 days"
}
]

# -------- VECTOR DB --------
texts = [d["text"] for d in data]
vectors = embed_model.encode(texts)

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# -------- CHAT MEMORY --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------- AI FUNCTION --------
def generate_ai(question, history):

    context = ""
    for q, a in history[-3:]:
        context += f"User: {q}\nAI: {a}\n"

    prompt = f"""
You are a helpful customer support assistant.

Conversation:
{context}

User: {question}
AI:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=80)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------- UI --------
st.title("💬 AI Customer Support Chatbot")

# show chat history
for q, a in st.session_state.chat_history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

# input
user_input = st.chat_input("Ask your question...")

if user_input:

    st.chat_message("user").write(user_input)

    # -------- SMALL TALK FIX --------
    if user_input.lower() in ["hi", "hello"]:
        answer = "Hello! How can I help you today?"

    else:
        # -------- SIMILARITY CHECK --------
        q_vec = embed_model.encode([user_input])
        D, I = index.search(np.array(q_vec), 1)

        score = D[0][0]

        # -------- FIXED FAST ANSWER --------
        if score < 0.8:

            result = data[I[0][0]]

            if result["type"] == "navigation":
                answer = "Follow these steps:\n" + "\n".join(result["steps"])

                st.chat_message("assistant").write(answer)

                for img in result["images"]:
                    st.image(f"images/{img}")

            else:
                answer = result["answer"]
                st.chat_message("assistant").write(answer)

        # -------- AI ANSWER --------
        else:
            with st.spinner("Thinking..."):
                answer = generate_ai(user_input, st.session_state.chat_history)

            st.chat_message("assistant").write(answer)

    # -------- SAVE MEMORY --------
    st.session_state.chat_history.append((user_input, answer))

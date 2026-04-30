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

# TRACK ORDER
{
"text": "track order where is my order order status",
"keywords": ["track order", "order status", "where is my order"],
"answer": "You can track your order in the 'My Orders' section."
},

# BUY PRODUCT
{
"text": "how to order buy product purchase item",
"keywords": ["buy", "purchase", "how to order", "order product"],
"answer": "To place an order, search for the product, add it to your cart, and proceed to checkout."
},

# CANCEL
{
"text": "cancel order",
"keywords": ["cancel order"],
"answer": "Go to 'My Orders', select your order, and choose cancel."
},

# PAYMENT
{
"text": "payment failed refund",
"keywords": ["payment failed", "refund"],
"answer": "If money was deducted, the refund will be processed within 3–5 working days."
},

# PASSWORD
{
"text": "forgot password reset password",
"keywords": ["forgot password", "reset password"],
"answer": "Click 'Forgot Password' on the login page and follow the steps."
},
]

# -------- VECTOR DB --------
texts = [d["text"] for d in data]
vectors = embed_model.encode(texts)

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# -------- MEMORY --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------- UI --------
st.title("💬 HACSS - Customer Support")

# INTRO
if "started" not in st.session_state:
    st.session_state.started = True
    intro = "Hello! I’m HACSS. How can I help you today?"
    st.chat_message("assistant").write(intro)
    st.session_state.chat_history.append(("system", intro))

# SHOW CHAT
for q, a in st.session_state.chat_history:
    if q != "system":
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

# -------- AI FUNCTION --------
def generate_ai(question):
    prompt = f"""
You are a friendly customer support assistant.
Respond clearly and politely.
Ignore slang words like 'bro'.

User: {question}
Assistant:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=60)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------- INPUT --------
user_input = st.chat_input("Ask your question...")

if user_input:
    st.chat_message("user").write(user_input)
    q = user_input.lower()

    # -------- GREETING (FIXED) --------
    if any(greet in q for greet in ["hi", "hello", "hey"]):
        answer = "Hello! How can I help you today?"
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- THANKS --------
    if "thank" in q:
        answer = "You're welcome! Let me know if you need anything else."
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- KEYWORD MATCH --------
    answer = None
    for item in data:
        if any(kw in q for kw in item["keywords"]):
            answer = item["answer"]
            break

    if answer:
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- VECTOR MATCH --------
    q_vec = embed_model.encode([user_input])
    D, I = index.search(np.array(q_vec), 1)

    if D[0][0] < 1.2:   # stricter threshold (IMPORTANT)
        answer = data[I[0][0]]["answer"]
        if answer and len(answer.strip()) > 5:
            st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append((user_input, answer))
            st.stop()

    # -------- AI FALLBACK --------
    answer = generate_ai(user_input)

    if not answer or len(answer.strip()) < 5:
        answer = "I'm here to help. Could you please clarify your question?"

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append((user_input, answer))

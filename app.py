import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------- LOAD MODEL --------
@st.cache_resource
def load_embed():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed()

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
"answer": "To place an order, search the product, add it to your cart, and proceed to checkout."
},

# PAYMENT FAILED
{
"text": "payment failed refund",
"keywords": ["payment failed", "refund"],
"answer": "If the payment failed but money was deducted, the refund will be processed within 3–5 working days."
},

# REFUND WHERE
{
"text": "refund where will money come bank account wallet",
"keywords": ["refund where", "refund account", "money come account"],
"answer": "The refund will be sent to your original payment method. UPI/card → bank account. Wallet → wallet balance."
},

# REFUND STATUS
{
"text": "check refund status refund received or not",
"keywords": ["refund status", "refund received", "check refund"],
"answer": "You can check your refund status in 'My Orders' or your bank/app transaction history."
},

# CANCEL
{
"text": "cancel order",
"keywords": ["cancel order"],
"answer": "Go to 'My Orders', select your order, and cancel it."
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

# -------- INPUT --------
user_input = st.chat_input("Ask your question...")

if user_input:
    st.chat_message("user").write(user_input)
    q = user_input.lower()

    # -------- GREETING --------
    if any(g in q for g in ["hi", "hello", "hey"]):
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
    for item in data:
        if any(kw in q for kw in item["keywords"]):
            answer = item["answer"]
            st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append((user_input, answer))
            st.stop()

    # -------- VECTOR MATCH --------
    q_vec = embed_model.encode([user_input])
    D, I = index.search(np.array(q_vec), 1)

    if D[0][0] < 1.2:
        answer = data[I[0][0]]["answer"]
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- FALLBACK --------
    answer = "I'm here to help. Could you please explain your issue in more detail?"
    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append((user_input, answer))

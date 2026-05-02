import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# -------- DATA --------
sections = {
    "Order / Delivery": [
        {
            "text": "track order where is my order not delivered late",
            "answer": """1. Open My Orders
2. Check order status
3. Track delivery
4. Contact support if delayed"""
        }
    ],

    "Buy / Product": [
        {
            "text": "how to order buy product purchase food",
            "answer": """1. Search the product
2. Add to cart
3. Go to checkout
4. Enter details
5. Place your order"""
        },
        {
            "text": "after adding cart what next",
            "answer": """1. Open cart
2. Click checkout
3. Enter delivery details
4. Select payment
5. Confirm order"""
        }
    ],

    "Payment / Refund": [
        {
            "text": "payment failed refund",
            "answer": """1. Wait 3–5 working days
2. Check bank/wallet
3. Contact support if needed"""
        }
    ],

    "Account": [
        {
            "text": "forgot password reset login issue",
            "answer": """1. Go to login page
2. Click Forgot Password
3. Enter details
4. Set new password"""
        }
    ]
}

# -------- BUILD INDEX --------
def build_index(data):
    texts = [d["text"] for d in data]
    vectors = embed_model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index

# -------- UI --------
st.title("💬 HACSS - Customer Support")

# SECTION SELECT
selected_section = st.selectbox(
    "Select your issue type:",
    list(sections.keys())
)

# INIT CHAT
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# SHOW CHAT
for q, a in st.session_state.chat_history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

# INPUT
user_input = st.chat_input("Ask your question...")

if user_input:
    st.chat_message("user").write(user_input)

    data = sections[selected_section]
    index = build_index(data)

    q_vec = embed_model.encode([user_input])
    D, I = index.search(np.array(q_vec), 1)

    if D[0][0] < 1.2:
        answer = data[I[0][0]]["answer"]
    else:
        answer = """I didn't understand clearly.

Try asking about:
• Order
• Payment
• Account"""

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append((user_input, answer))

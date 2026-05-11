import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------- DATA --------
sections = {

    "Order": [
        {
            "text": "track order where is my order not delivered late",
            "answer": """1. Open My Orders
2. Check order status
3. Track delivery
4. Contact support if delayed"""
        }
    ],

    "Buy": [
        {
            "text": "how to order buy product food ice cream",
            "answer": """1. Search product
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

    "Payment": [
        {
            "text": "payment failed refund",
            "answer": """1. Wait 3–5 days
2. Check bank/wallet
3. Contact support if needed"""
        }
    ],

    "Account": [
        {
            "text": "forgot password reset login issue",
            "answer": """1. Go to login
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

    index.add(np.array(vectors).astype("float32"))

    return index


# -------- SEARCH FUNCTION --------
def search(data, question):

    index = build_index(data)

    q_vec = embed_model.encode([question])

    q_vec = np.array(q_vec).astype("float32")

    D, I = index.search(q_vec, 1)

    if D[0][0] < 1.2:
        return data[I[0][0]]["answer"]

    return None


# -------- SESSION --------
if "section" not in st.session_state:
    st.session_state.section = None

if "chat" not in st.session_state:
    st.session_state.chat = []


# -------- UI --------
st.title("💬 HACSS - Customer Support")


# -------- STEP 1: SELECT SECTION --------
if st.session_state.section is None:

    st.write("### Choose your issue:")

    col1, col2 = st.columns(2)

    if col1.button("📦 Order"):
        st.session_state.section = "Order"

    if col2.button("🛒 Buy"):
        st.session_state.section = "Buy"

    if col1.button("💳 Payment"):
        st.session_state.section = "Payment"

    if col2.button("👤 Account"):
        st.session_state.section = "Account"


# -------- STEP 2: CHAT --------
else:

    st.write(f"### Selected: {st.session_state.section}")

    # Change section button
    if st.button("🔄 Change Issue"):
        st.session_state.section = None
        st.session_state.chat = []
        st.rerun()

    # Show old chat
    for q, a in st.session_state.chat:

        st.chat_message("user").write(q)

        st.chat_message("assistant").write(a)

    # User input
    user_input = st.chat_input("Ask your question...")

    if user_input:

        st.chat_message("user").write(user_input)

        data = sections[st.session_state.section]

        answer = search(data, user_input)

        if not answer:
            answer = """I didn't understand clearly.

Try asking related to this section."""

        st.chat_message("assistant").write(answer)

        st.session_state.chat.append((user_input, answer))

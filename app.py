import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# -------- DATA GROUPS --------

order_data = [
    {"text": "where is my order track order status",
     "answer": "You can track your order in the 'My Orders' section."},

    {"text": "order not delivered late delivery not came",
     "answer": "Your order may be delayed. Please check 'My Orders' or contact support."}
]

buy_data = [
    {"text": "how to order buy product purchase item",
     "answer": "Search the product, add it to your cart, and proceed to checkout."},

    {"text": "after adding to cart what next checkout process",
     "answer": "Go to cart, click checkout, enter details, and place your order."}
]

payment_data = [
    {"text": "payment failed refund money deducted",
     "answer": "If payment failed, refund will be processed in 3–5 working days."},

    {"text": "refund where money come bank wallet",
     "answer": "Refund goes to your original payment method (bank or wallet)."}
]

account_data = [
    {"text": "forgot password reset account login issue",
     "answer": "Use 'Forgot Password' on login page to reset your account."}
]

# -------- MERGE ALL --------
all_data = order_data + buy_data + payment_data + account_data

# -------- FUNCTION TO BUILD FAISS --------
def build_index(data):
    texts = [d["text"] for d in data]
    vectors = embed_model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index

# -------- UI --------
st.title("💬 HACSS - Customer Support")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    # GREETING
    if any(g in q for g in ["hi", "hello", "hey"]):
        answer = "Hello! How can I help you today?"
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # THANKS
    if "thank" in q:
        answer = "You're welcome! Let me know if you need anything else."
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- SECTION DETECTION --------
    if any(word in q for word in ["refund", "payment"]):
        selected_data = payment_data

    elif any(word in q for word in ["buy", "cart", "checkout", "order food"]):
        selected_data = buy_data

    elif any(word in q for word in ["order", "delivery"]):
        selected_data = order_data

    elif any(word in q for word in ["account", "password", "login"]):
        selected_data = account_data

    else:
        selected_data = all_data

    # -------- FAISS SEARCH ON SELECTED GROUP --------
    index = build_index(selected_data)

    texts = [d["text"] for d in selected_data]
    q_vec = embed_model.encode([user_input])

    D, I = index.search(np.array(q_vec), 1)

    # -------- THRESHOLD CHECK --------
    if D[0][0] < 1.2:
        answer = selected_data[I[0][0]]["answer"]
    else:
        answer = "Sorry, I couldn't understand clearly. Please ask about orders, payments, or account issues."

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append((user_input, answer))

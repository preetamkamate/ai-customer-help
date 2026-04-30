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
"answer": "Go to My Orders to track your order status."
},

# BUY / ORDER PRODUCT
{
"text": "how to order buy product purchase item chocolates",
"keywords": ["buy", "purchase", "how to order", "order product"],
"answer": "To order a product, search the item, add it to cart, and proceed to checkout."
},

# CANCEL
{
"text": "cancel order how to cancel order",
"keywords": ["cancel order"],
"answer": "Open My Orders, select order, and click Cancel."
},

# PAYMENT
{
"text": "payment failed refund money deducted",
"keywords": ["payment failed", "refund"],
"answer": "Refund will be processed in 3-5 working days."
},

# ACCOUNT
{
"text": "reset password forgot password",
"keywords": ["reset password", "forgot password"],
"answer": "Click on Forgot Password to reset your password."
},

]

# -------- EXPAND DATA --------
extra_data = []
topics = ["order", "payment", "delivery", "account", "refund"]

for i in range(200):
    extra_data.append({
        "text": f"customer issue {topics[i % 5]} {i}",
        "keywords": [topics[i % 5]],
        "answer": f"This is system response for {topics[i % 5]} issue {i}."
    })

data += extra_data

# -------- VECTOR DB --------
texts = [d["text"] for d in data]
vectors = embed_model.encode(texts)

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# -------- MEMORY --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------- UI --------
st.title("💬 HACSS - Hybrid AI Customer Support System")

# -------- INTRO --------
if "started" not in st.session_state:
    st.session_state.started = True
    intro = "Hello! I am HACSS, your AI Customer Support Assistant. How can I help you today?"
    st.chat_message("assistant").write(intro)
    st.session_state.chat_history.append(("system", intro))

# -------- SHOW CHAT --------
for q, a in st.session_state.chat_history:
    if q != "system":
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

# -------- AI --------
def generate_ai(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=80)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------- INPUT --------
user_input = st.chat_input("Ask your question...")

if user_input:
    st.chat_message("user").write(user_input)
    q = user_input.lower()

    # -------- GREETING --------
    if q in ["hi", "hello", "hey"]:
        answer = "Hello! I am HACSS, your AI Customer Support Assistant. How can I help you today?"
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- KEYWORD MATCH (PHRASE BASED) --------
    for item in data:
        if any(kw in q for kw in item["keywords"]):
            answer = item["answer"]
            st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append((user_input, answer))
            st.stop()

    # -------- VECTOR MATCH --------
    q_vec = embed_model.encode([user_input])
    D, I = index.search(np.array(q_vec), 1)

    if D[0][0] < 1.5:
        answer = data[I[0][0]]["answer"]
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- AI FALLBACK --------
    answer = generate_ai(user_input)
    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append((user_input, answer))

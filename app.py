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
"answer": "You can track your order by going to the 'My Orders' section and selecting your order."
},

# BUY PRODUCT
{
"text": "how to order buy product purchase item",
"keywords": ["buy", "purchase", "how to order", "order product"],
"answer": "To place an order, search for the product, add it to your cart, and proceed to checkout."
},

# CANCEL ORDER
{
"text": "cancel order",
"keywords": ["cancel order"],
"answer": "To cancel your order, go to 'My Orders', select the order, and choose the cancel option."
},

# PAYMENT
{
"text": "payment failed refund",
"keywords": ["payment failed", "refund"],
"answer": "If your payment failed but the amount was deducted, the refund will be processed within 3-5 working days."
},

# ACCOUNT
{
"text": "reset password forgot password",
"keywords": ["reset password", "forgot password"],
"answer": "You can reset your password by clicking on 'Forgot Password' on the login page."
},

]

# -------- EXPAND DATA --------
extra_data = []
topics = ["order", "payment", "delivery", "account", "refund"]

for i in range(200):
    extra_data.append({
        "text": f"customer issue {topics[i % 5]} {i}",
        "keywords": [topics[i % 5]],
        "answer": f"This is a system response for {topics[i % 5]} related query."
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
    intro = "Hello! I am HACSS, your AI Customer Support Assistant. How can I assist you today?"
    st.chat_message("assistant").write(intro)
    st.session_state.chat_history.append(("system", intro))

# -------- SHOW CHAT --------
for q, a in st.session_state.chat_history:
    if q != "system":
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

# -------- AI FUNCTION --------
def generate_ai(question):
    prompt = f"You are a polite and professional customer support assistant. Answer clearly and respectfully.\n\nUser: {question}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=80)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------- INPUT --------
user_input = st.chat_input("Ask your question...")

if user_input:
    st.chat_message("user").write(user_input)
    q = user_input.lower()

    # -------- GREETING --------
    if q in ["hi", "hello", "hey"]:
        answer = "Hello! I am HACSS, your AI Customer Support Assistant. How can I assist you today?"
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- SMALL TALK --------
    if "thank" in q:
        answer = "You're welcome! I'm happy to assist you. Please let me know if you need anything else."
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    if "love" in q:
        answer = "Thank you for your kind words. I am here to assist you with your queries."
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    if "bro" in q:
        answer = "I will communicate in a professional manner. How can I assist you?"
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

    if D[0][0] < 1.5:
        answer = data[I[0][0]]["answer"]
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- AI FALLBACK --------
    answer = generate_ai(user_input)
    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append((user_input, answer))

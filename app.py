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

# ORDER
{"text": "track order where is my order order status", "keywords": ["order","track"], "answer": "Go to My Orders to track your order status."},
{"text": "cancel order how to cancel my order", "keywords": ["cancel"], "answer": "Open My Orders, select order, click Cancel."},
{"text": "order delayed delivery late", "keywords": ["delivery","delay"], "answer": "Your order may be delayed due to logistics issues."},
{"text": "change delivery address update address", "keywords": ["address"], "answer": "Go to Order Settings to update your address."},

# PAYMENT
{"text": "payment failed refund money deducted", "keywords": ["payment","refund"], "answer": "Refund will be processed in 3-5 working days."},
{"text": "payment methods available", "keywords": ["payment"], "answer": "We support UPI, debit/credit cards, and net banking."},

# ACCOUNT
{"text": "forgot password reset password", "keywords": ["password"], "answer": "Click on Forgot Password to reset your password."},
{"text": "update profile change name", "keywords": ["profile"], "answer": "Go to Account Settings to update your profile."},

# DELIVERY
{"text": "delivery time how long shipping", "keywords": ["delivery"], "answer": "Delivery usually takes 3-7 days."},
{"text": "delivery charges shipping cost", "keywords": ["delivery"], "answer": "Shipping charges depend on your location."},

# RETURNS
{"text": "return product how to return item", "keywords": ["return"], "answer": "Go to My Orders and select Return option."},
{"text": "refund status check refund", "keywords": ["refund"], "answer": "Refund status is available in My Orders."},

]

# -------- AUTO EXPAND DATA --------
extra_data = []
topics = ["order", "payment", "delivery", "account", "refund"]

for i in range(300):
    extra_data.append({
        "text": f"customer issue related to {topics[i % 5]} number {i}",
        "keywords": [topics[i % 5]],
        "answer": f"This is system response for {topics[i % 5]} issue number {i}."
    })

data = data + extra_data

# -------- VECTOR DB --------
texts = [d["text"] for d in data]
vectors = embed_model.encode(texts)

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# -------- CHAT MEMORY --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------- UI --------
st.title("💬 HACSS - Hybrid AI Customer Support System")

# -------- INTRO MESSAGE --------
if "started" not in st.session_state:
    st.session_state.started = True
    intro = "Hello! I am HACSS, your AI Customer Support Assistant. How can I help you today?"
    st.chat_message("assistant").write(intro)
    st.session_state.chat_history.append(("system", intro))

# -------- SHOW HISTORY --------
for q, a in st.session_state.chat_history:
    if q != "system":
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

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

# -------- INPUT --------
user_input = st.chat_input("Ask your question...")

if user_input:

    st.chat_message("user").write(user_input)
    q = user_input.lower()
    answer = None

    # -------- GREETING --------
    if q in ["hi", "hello", "hey"]:
        answer = "Hello! I am HACSS, your AI Customer Support Assistant. How can I help you today?"
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- KEYWORD MATCH --------
    for item in data:
        if any(word in q for word in item["keywords"]):
            answer = item["answer"]
            st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append((user_input, answer))
            st.stop()

    # -------- VECTOR MATCH --------
    q_vec = embed_model.encode([user_input])
    D, I = index.search(np.array(q_vec), 1)
    score = D[0][0]

    if score < 1.5:
        result = data[I[0][0]]
        answer = result["answer"]
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- AI FALLBACK --------
    with st.spinner("Thinking..."):
        answer = generate_ai(user_input, st.session_state.chat_history)

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append((user_input, answer))

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------- PAGE --------
st.set_page_config(page_title="HACSS", page_icon="💬")

# -------- LOAD MODELS --------
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    return tokenizer, model, embed

tokenizer, model, embed_model = load_models()

# -------- BASE DATA --------
data = [

# ORDER
{
    "text": "track order where is my order order status",
    "type": "Orders",
    "keywords": ["order", "track"],
    "answer": """1. Open App
2. Go to My Orders
3. Select your order
4. Click Track Order"""
},

{
    "text": "cancel order how to cancel my order",
    "type": "Orders",
    "keywords": ["cancel"],
    "answer": """1. Open App
2. Go to My Orders
3. Select order
4. Click Cancel"""
},

{
    "text": "order delayed delivery late",
    "type": "Orders",
    "keywords": ["delivery", "delay"],
    "answer": "Your order may be delayed due to logistics issues."
},

# PAYMENT
{
    "text": "payment failed refund money deducted",
    "type": "Payment",
    "keywords": ["payment", "refund"],
    "answer": """1. Wait 3-5 working days
2. Check bank/wallet
3. Contact support if needed"""
},

{
    "text": "payment methods available",
    "type": "Payment",
    "keywords": ["payment"],
    "answer": "We support UPI, debit/credit cards, and net banking."
},

# ACCOUNT
{
    "text": "forgot password reset password",
    "type": "Account",
    "keywords": ["password"],
    "answer": """1. Click Forgot Password
2. Verify OTP
3. Set new password"""
},

{
    "text": "update profile change name",
    "type": "Account",
    "keywords": ["profile"],
    "answer": "Go to Account Settings to update your profile."
},

# DELIVERY
{
    "text": "delivery time how long shipping",
    "type": "Delivery",
    "keywords": ["delivery"],
    "answer": "Delivery usually takes 3-7 days."
},

{
    "text": "delivery charges shipping cost",
    "type": "Delivery",
    "keywords": ["shipping"],
    "answer": "Shipping charges depend on your location."
},

# BUY
{
    "text": "buy headphones recommend headphones",
    "type": "Buy",
    "keywords": ["headphones", "buy"],
    "answer": "Recommended Product: Wireless Headphones - ₹999"
},

{
    "text": "buy shoes recommend shoes",
    "type": "Buy",
    "keywords": ["shoes"],
    "answer": "Recommended Product: Sports Shoes - ₹1499"
},

{
    "text": "best mobile under 20000",
    "type": "Buy",
    "keywords": ["mobile"],
    "answer": "Recommended Product: Smartphone under ₹20,000"
}

]

# -------- AUTO EXPAND DATA --------
extra_data = []
topics = ["Orders", "Payment", "Delivery", "Account", "Buy"]

for i in range(100):

    extra_data.append({

        "text": f"customer issue related to {topics[i % 5]} number {i}",

        "type": topics[i % 5],

        "keywords": [topics[i % 5].lower()],

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

# -------- DEFAULT SECTION --------
if "section" not in st.session_state:
    st.session_state.section = "Orders"

# -------- AI FUNCTION --------
def generate_ai(question, history):

    context = ""

    for q, a in history[-3:]:
        context += f"User: {q}\nAI: {a}\n"

    prompt = f"""
You are HACSS, a helpful customer support assistant.

Conversation:
{context}

User: {question}

AI:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=80,
        do_sample=True,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------- UI --------
st.title("💬 Hybrid AI Customer Support System")

st.subheader("Choose Support Section")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📦 Orders"):
        st.session_state.section = "Orders"

with col2:
    if st.button("💳 Payment"):
        st.session_state.section = "Payment"

with col3:
    if st.button("👤 Account"):
        st.session_state.section = "Account"

with col4:
    if st.button("🚚 Delivery"):
        st.session_state.section = "Delivery"

with col5:
    if st.button("🛒 Buy"):
        st.session_state.section = "Buy"

st.success(f"Selected Section: {st.session_state.section}")

# -------- INTRO --------
if len(st.session_state.chat_history) == 0:

    intro = "Hello! I am HACSS, your AI Customer Support Assistant. How can I help you today?"

    st.session_state.chat_history.append(("HACSS", intro))

# -------- SHOW CHAT --------
for q, a in st.session_state.chat_history:

    st.chat_message("user").write(q)

    st.chat_message("assistant").write(a)

# -------- USER INPUT --------
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

        if item["type"] == st.session_state.section:

            if any(word in q for word in item["keywords"]):

                answer = item["answer"]

                st.chat_message("assistant").write(answer)

                st.session_state.chat_history.append((user_input, answer))

                st.stop()

    # -------- VECTOR SEARCH --------
    q_vec = embed_model.encode([user_input])

    D, I = index.search(np.array(q_vec), 1)

    score = D[0][0]

    result = data[I[0][0]]

    if score < 1.5 and result["type"] == st.session_state.section:

        answer = result["answer"]

        st.chat_message("assistant").write(answer)

        st.session_state.chat_history.append((user_input, answer))

        st.stop()

    # -------- AI FALLBACK --------
    with st.spinner("Thinking..."):

        answer = generate_ai(user_input, st.session_state.chat_history)

    st.chat_message("assistant").write(answer)

    st.session_state.chat_history.append((user_input, answer))

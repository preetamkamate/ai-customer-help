import streamlit as st
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# ---------------- PAGE ----------------
st.set_page_config(page_title="HACSS", page_icon="💬")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return tokenizer, model, embed_model

tokenizer, model, embed_model = load_models()

# ---------------- DATASET ----------------
data = [

# ORDER
{
    "text": "track order where is my order order status",
    "keywords": ["order", "track", "status"],
    "answer": """1. Open app
2. Go to My Orders
3. Select your order
4. Click Track Order"""
},

{
    "text": "cancel order remove my order",
    "keywords": ["cancel", "remove"],
    "answer": """1. Open app
2. Go to My Orders
3. Select order
4. Click Cancel"""
},

{
    "text": "delivery delayed late order",
    "keywords": ["delivery", "late", "delay"],
    "answer": "Your delivery may be delayed due to logistics or weather conditions."
},

# PAYMENT
{
    "text": "payment failed refund money deducted",
    "keywords": ["payment", "refund", "money"],
    "answer": """1. Wait 3-5 working days
2. Check bank/wallet status
3. Contact support if needed"""
},

{
    "text": "double payment charged twice",
    "keywords": ["double", "charged"],
    "answer": "If amount was deducted twice, refund will be processed automatically."
},

# ACCOUNT
{
    "text": "forgot password reset password",
    "keywords": ["password", "reset"],
    "answer": """1. Click Forgot Password
2. Enter registered mobile/email
3. Set new password"""
},

{
    "text": "delete account permanently remove account",
    "keywords": ["delete", "account"],
    "answer": "Please contact customer support to permanently delete your account."
},

# RETURN
{
    "text": "return product replace item exchange",
    "keywords": ["return", "replace", "exchange"],
    "answer": """1. Open My Orders
2. Select product
3. Click Return/Replace"""
},

# CUSTOMER CARE
{
    "text": "customer care support contact agent",
    "keywords": ["support", "agent", "customer care"],
    "answer": "You can contact customer support from Help Center section."
}

]

# ---------------- LARGE DATA ----------------
extra_data = []

topics = ["order", "payment", "refund", "delivery", "account"]

for i in range(300):
    extra_data.append({
        "text": f"customer issue about {topics[i % 5]} number {i}",
        "keywords": [topics[i % 5]],
        "answer": f"This is automated response for {topics[i % 5]} issue {i}."
    })

data = data + extra_data

# ---------------- VECTOR DB ----------------
texts = [d["text"] for d in data]

vectors = embed_model.encode(texts)

index = faiss.IndexFlatL2(vectors.shape[1])

index.add(np.array(vectors))

# ---------------- CHAT MEMORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- INTRO MESSAGE ----------------
if "started" not in st.session_state:
    st.session_state.started = True

    intro = "Hello! I am HACSS, your AI Customer Support Assistant. How can I help you today?"

    st.session_state.chat_history.append(("assistant", intro))

# ---------------- AI FUNCTION ----------------
def generate_ai(question, history):

    context = ""

    for role, msg in history[-3:]:
        context += f"{role}: {msg}\n"

    prompt = f"""
You are HACSS, a helpful AI customer support assistant.

Conversation:
{context}

User: {question}

Answer politely and clearly:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=80,
        do_sample=True,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- UI ----------------
st.title("💬 HACSS - Customer Support")

# ---------------- SHOW CHAT ----------------
for role, msg in st.session_state.chat_history:

    if role == "user":
        st.chat_message("user").write(msg)

    else:
        st.chat_message("assistant").write(msg)

# ---------------- USER INPUT ----------------
user_input = st.chat_input("Ask your question...")

if user_input:

    st.chat_message("user").write(user_input)

    st.session_state.chat_history.append(("user", user_input))

    q = user_input.lower()

    # ---------------- GREETING ----------------
    if q in ["hi", "hello", "hey"]:

        answer = "Hello! I am HACSS, your AI Customer Support Assistant. How can I help you today?"

        st.chat_message("assistant").write(answer)

        st.session_state.chat_history.append(("assistant", answer))

        st.stop()

    # ---------------- KEYWORD MATCH ----------------
    for item in data:

        if any(word in q for word in item["keywords"]):

            answer = item["answer"]

            st.chat_message("assistant").write(answer)

            st.session_state.chat_history.append(("assistant", answer))

            st.stop()

    # ---------------- VECTOR SEARCH ----------------
    q_vec = embed_model.encode([user_input])

    D, I = index.search(np.array(q_vec), 1)

    score = D[0][0]

    if score < 1.5:

        result = data[I[0][0]]

        answer = result["answer"]

        st.chat_message("assistant").write(answer)

        st.session_state.chat_history.append(("assistant", answer))

        st.stop()

    # ---------------- AI FALLBACK ----------------
    with st.spinner("Thinking..."):

        answer = generate_ai(user_input, st.session_state.chat_history)

    st.chat_message("assistant").write(answer)

    st.session_state.chat_history.append(("assistant", answer))

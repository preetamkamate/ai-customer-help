import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

# ---------------- DATA ----------------
sections = {

"Orders": [

{
"text": "track order where is my order order status",
"keywords": ["order", "track", "status"],
"answer": """1. Open app
2. Go to My Orders
3. Select your order
4. Click Track Order"""
},

{
"text": "cancel order remove order",
"keywords": ["cancel", "remove"],
"answer": """1. Open app
2. Go to My Orders
3. Click Cancel Order"""
}

],

"Payment": [

{
"text": "payment failed refund money deducted",
"keywords": ["payment", "refund"],
"answer": """1. Wait 3-5 days
2. Check bank/wallet
3. Contact support if needed"""
},

{
"text": "double payment charged twice",
"keywords": ["double", "charged"],
"answer": "Refund will be processed automatically."
}

],

"Account": [

{
"text": "forgot password reset password",
"keywords": ["password", "reset"],
"answer": """1. Click Forgot Password
2. Verify OTP
3. Set new password"""
},

{
"text": "delete account permanently",
"keywords": ["delete", "account"],
"answer": "Please contact support to delete your account."
}

],

"Delivery": [

{
"text": "delivery delayed late order",
"keywords": ["delivery", "delay", "late"],
"answer": "Delivery may be delayed due to logistics or weather."
},

{
"text": "delivery charges shipping cost",
"keywords": ["shipping", "charges"],
"answer": "Shipping charges depend on location."
}

]

}

# ---------------- SELECT SECTION ----------------
selected_section = st.sidebar.selectbox(
"Select Issue Section",
list(sections.keys())
)

data = sections[selected_section]

# ---------------- VECTOR DB ----------------
texts = [d["text"] for d in data]

vectors = embed_model.encode(texts)

index = faiss.IndexFlatL2(vectors.shape[1])

index.add(np.array(vectors))

# ---------------- CHAT MEMORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- INTRO ----------------
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
You are HACSS, a helpful AI Customer Support Assistant.

Conversation:
{context}

User: {question}

Answer politely:
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

st.subheader(f"Selected: {selected_section}")

# ---------------- SHOW CHAT ----------------
for role, msg in st.session_state.chat_history:

    if role == "user":
        st.chat_message("user").write(msg)

    else:
        st.chat_message("assistant").write(msg)

# ---------------- INPUT ----------------
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

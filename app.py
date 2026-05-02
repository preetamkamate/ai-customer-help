import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------- LOAD MODELS --------
@st.cache_resource
def load_models():
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return embed, tokenizer, model

embed_model, tokenizer, model = load_models()

# -------- DATA SECTIONS --------

order_data = [
    {
        "text": "where is my order track order status",
        "keywords": ["track order", "order status", "where is my order"],
        "answer": "You can track your order in the 'My Orders' section."
    },
    {
        "text": "order not delivered late delivery problem",
        "keywords": ["not delivered", "not came", "late delivery"],
        "answer": "Your order may be delayed. Please check 'My Orders' or contact support."
    }
]

buy_data = [
    {
        "text": "how to order buy product purchase item",
        "keywords": ["buy", "purchase", "how to order", "order product"],
        "answer": "Search the product, add it to your cart, and proceed to checkout."
    },
    {
        "text": "after adding to cart next step checkout",
        "keywords": ["after cart", "next step", "checkout"],
        "answer": "Open cart, click checkout, enter details, and place your order."
    }
]

payment_data = [
    {
        "text": "payment failed refund",
        "keywords": ["payment failed", "refund"],
        "answer": "Refund will be processed in 3–5 working days."
    },
    {
        "text": "refund where money come bank wallet",
        "keywords": ["refund where", "money come account"],
        "answer": "Refund goes to original payment method (bank or wallet)."
    }
]

account_data = [
    {
        "text": "forgot password reset password",
        "keywords": ["forgot password", "reset password"],
        "answer": "Use 'Forgot Password' on login page to reset."
    }
]

# -------- MERGE ALL --------
all_data = order_data + buy_data + payment_data + account_data

# -------- VECTOR DB --------
texts = [d["text"] for d in all_data]
vectors = embed_model.encode(texts)
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# -------- MEMORY --------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------- UI --------
st.title("💬 HACSS - Customer Support")

if "started" not in st.session_state:
    st.session_state.started = True
    intro = "Hello! I’m HACSS. How can I help you today?"
    st.chat_message("assistant").write(intro)
    st.session_state.chat_history.append(("system", intro))

for q, a in st.session_state.chat_history:
    if q != "system":
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

# -------- AI FUNCTION --------
def generate_ai(question):
    prompt = f"Answer clearly: {question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=50,
        repetition_penalty=2.5,
        no_repeat_ngram_size=2
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if len(result.split()) > 25:
        return "Please clarify your question."

    return result

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

    # -------- SECTION FILTER --------
    if any(word in q for word in ["refund", "payment"]):
        selected_data = payment_data

    elif any(word in q for word in ["buy", "cart", "checkout"]):
        selected_data = buy_data

    elif any(word in q for word in ["order", "delivery"]):
        selected_data = order_data

    elif any(word in q for word in ["account", "password"]):
        selected_data = account_data

    else:
        selected_data = all_data

    # -------- KEYWORD MATCH --------
    for item in selected_data:
        if any(kw in q for kw in item["keywords"]):
            answer = item["answer"]
            st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append((user_input, answer))
            st.stop()

    # -------- VECTOR MATCH --------
    texts_sel = [d["text"] for d in selected_data]
    vecs_sel = embed_model.encode(texts_sel)
    temp_index = faiss.IndexFlatL2(vecs_sel.shape[1])
    temp_index.add(np.array(vecs_sel))

    q_vec = embed_model.encode([user_input])
    D, I = temp_index.search(np.array(q_vec), 1)

    if D[0][0] < 1.2:
        answer = selected_data[I[0][0]]["answer"]
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # -------- AI FALLBACK --------
    answer = generate_ai(user_input)
    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append((user_input, answer))

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

# -------- DATA --------
data = [

# ORDER
{
"text": "how to order buy product purchase item",
"keywords": ["buy", "purchase", "how to order", "order product"],
"answer": "To place an order, search for the product, add it to your cart, and proceed to checkout."
},

# AFTER CART
{
"text": "after adding to cart what to do next step checkout",
"keywords": ["after cart", "after adding cart", "next step after cart"],
"answer": "After adding items to your cart, open the cart, click 'Checkout', enter your address, choose payment, and place your order."
},

# PAYMENT
{
"text": "payment failed refund",
"keywords": ["payment failed", "refund"],
"answer": "If payment failed but money was deducted, the refund will be processed within 3–5 working days."
},

# REFUND WHERE
{
"text": "refund where will money come bank wallet",
"keywords": ["refund where", "money come account"],
"answer": "The refund will go back to your original payment method. UPI/card goes to bank, wallet returns to wallet."
},

]

# -------- VECTOR DB --------
texts = [d["text"] for d in data]
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
    prompt = f"Answer simply and clearly: {question}"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=50,
        repetition_penalty=2.5,
        no_repeat_ngram_size=2
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # FILTER BAD OUTPUT
    if "person" in result.lower() or len(result.split()) > 25:
        return "Please clarify your question so I can assist you better."

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

    # FOLLOW-UP FIX (IMPORTANT)
    if "next" in q or "then" in q:
        answer = "After adding items to your cart, go to the cart, click 'Checkout', enter details, and place your order."
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # KEYWORD MATCH
    for item in data:
        if any(kw in q for kw in item["keywords"]):
            answer = item["answer"]
            st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append((user_input, answer))
            st.stop()

    # VECTOR MATCH
    q_vec = embed_model.encode([user_input])
    D, I = index.search(np.array(q_vec), 1)

    if D[0][0] < 1.2:
        answer = data[I[0][0]]["answer"]
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append((user_input, answer))
        st.stop()

    # AI FALLBACK
    answer = generate_ai(user_input)
    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append((user_input, answer))

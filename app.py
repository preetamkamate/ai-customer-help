# =========================================================
# CREATE THESE FILES
#
# app.py
# pages/orders.py
# pages/payment.py
# pages/account.py
# pages/delivery.py
# pages/buy.py
# =========================================================


# =========================================================
# FILE: app.py
# =========================================================

import streamlit as st

st.set_page_config(page_title="HACSS Support", page_icon="🤖")

st.title("🤖 HACSS Customer Support System")

st.write("Select Support Category")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📦 Orders"):
        st.switch_page("pages/orders.py")

with col2:
    if st.button("💳 Payment"):
        st.switch_page("pages/payment.py")

with col3:
    if st.button("👤 Account"):
        st.switch_page("pages/account.py")

with col4:
    if st.button("🚚 Delivery"):
        st.switch_page("pages/delivery.py")

with col5:
    if st.button("🛒 Buy"):
        st.switch_page("pages/buy.py")


# =========================================================
# FILE: pages/orders.py
# =========================================================

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="Orders", page_icon="📦")

@st.cache_resource
def load_models():

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small"
    )

    embed_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return tokenizer, model, embed_model


tokenizer, model, embed_model = load_models()

data = [

    {
        "text": "track order where is my order order status",

        "keywords": ["track", "order", "status"],

        "answer": """1. Open App
2. Go to My Orders
3. Select Order
4. Click Track Order"""
    },

    {
        "text": "cancel order remove order",

        "keywords": ["cancel", "remove"],

        "answer": """1. Open App
2. Go to My Orders
3. Select Order
4. Click Cancel"""
    },

    {
        "text": "delivery delayed late order",

        "keywords": ["delivery", "late", "delay"],

        "answer": "Your delivery may be delayed due to logistics or weather conditions."
    }

]

texts = [d["text"] for d in data]

vectors = embed_model.encode(texts)

index = faiss.IndexFlatL2(vectors.shape[1])

index.add(np.array(vectors))

if "orders_chat" not in st.session_state:
    st.session_state.orders_chat = []


def generate_ai(question):

    prompt = f"""
You are HACSS Order Support Assistant.

Question:
{question}

Answer politely:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=80,
        do_sample=True,
        temperature=0.7
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


st.title("📦 Orders Support")

if st.button("⬅ Back"):
    st.switch_page("app.py")

if len(st.session_state.orders_chat) == 0:

    intro = "Hello! I am HACSS Orders Support Assistant. How can I help you today?"

    st.session_state.orders_chat.append(
        ("assistant", intro)
    )

for role, msg in st.session_state.orders_chat:

    if role == "user":

        st.chat_message("user").write(msg)

    else:

        st.chat_message("assistant").write(msg)

user_input = st.chat_input("Ask Orders Issue")

if user_input:

    st.chat_message("user").write(user_input)

    st.session_state.orders_chat.append(
        ("user", user_input)
    )

    q = user_input.lower()

    if q in ["hi", "hello", "hey"]:

        answer = "Hello! I am HACSS Orders Support Assistant."

        st.chat_message("assistant").write(answer)

        st.session_state.orders_chat.append(
            ("assistant", answer)
        )

        st.stop()

    for item in data:

        if any(word in q for word in item["keywords"]):

            answer = item["answer"]

            st.chat_message("assistant").write(answer)

            st.session_state.orders_chat.append(
                ("assistant", answer)
            )

            st.stop()

    q_vec = embed_model.encode([user_input])

    D, I = index.search(np.array(q_vec), 1)

    if D[0][0] < 1.5:

        answer = data[I[0][0]]["answer"]

        st.chat_message("assistant").write(answer)

        st.session_state.orders_chat.append(
            ("assistant", answer)
        )

        st.stop()

    with st.spinner("Thinking..."):

        answer = generate_ai(user_input)

    st.chat_message("assistant").write(answer)

    st.session_state.orders_chat.append(
        ("assistant", answer)
    )


# =========================================================
# FILE: pages/payment.py
# =========================================================

import streamlit as st

st.set_page_config(
    page_title="Payment",
    page_icon="💳"
)

st.title("💳 Payment Support")

if st.button("⬅ Back"):
    st.switch_page("app.py")

question = st.chat_input("Ask Payment Issue")

if question:

    st.chat_message("user").write(question)

    if "refund" in question.lower():

        st.chat_message("assistant").write("""
1. Wait 3-5 working days
2. Check bank status
3. Contact support if needed
""")

    elif "failed" in question.lower():

        st.chat_message("assistant").write(
            "Payment failed. Amount will be refunded automatically."
        )

    else:

        st.chat_message("assistant").write(
            "Payment support response generated."
        )


# =========================================================
# FILE: pages/account.py
# =========================================================

import streamlit as st

st.set_page_config(
    page_title="Account",
    page_icon="👤"
)

st.title("👤 Account Support")

if st.button("⬅ Back"):
    st.switch_page("app.py")

question = st.chat_input("Ask Account Issue")

if question:

    st.chat_message("user").write(question)

    if "password" in question.lower():

        st.chat_message("assistant").write("""
1. Click Forgot Password
2. Verify OTP
3. Set new password
""")

    else:

        st.chat_message("assistant").write(
            "Account support response generated."
        )


# =========================================================
# FILE: pages/delivery.py
# =========================================================

import streamlit as st

st.set_page_config(
    page_title="Delivery",
    page_icon="🚚"
)

st.title("🚚 Delivery Support")

if st.button("⬅ Back"):
    st.switch_page("app.py")

question = st.chat_input("Ask Delivery Issue")

if question:

    st.chat_message("user").write(question)

    if "late" in question.lower():

        st.chat_message("assistant").write(
            "Delivery may be delayed due to logistics or weather."
        )

    else:

        st.chat_message("assistant").write(
            "Delivery support response generated."
        )


# =========================================================
# FILE: pages/buy.py
# =========================================================

import streamlit as st

st.set_page_config(
    page_title="Buy",
    page_icon="🛒"
)

st.title("🛒 Product Recommendation")

if st.button("⬅ Back"):
    st.switch_page("app.py")

question = st.chat_input(
    "Ask Product Recommendation"
)

if question:

    st.chat_message("user").write(question)

    if "mobile" in question.lower():

        st.chat_message("assistant").write(
            "Recommended Smartphone under ₹20,000"
        )

    elif "headphones" in question.lower():

        st.chat_message("assistant").write(
            "Recommended Wireless Headphones - ₹999"
        )

    elif "shoes" in question.lower():

        st.chat_message("assistant").write(
            "Recommended Sports Shoes - ₹1499"
        )

    else:

        st.chat_message("assistant").write(
            "Product recommendation available."
        )

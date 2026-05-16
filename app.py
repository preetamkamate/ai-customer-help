import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from difflib import get_close_matches
import faiss
import numpy as np
import time

# -------- PAGE --------
st.set_page_config(
    page_title="HACSS",
    page_icon="🤖",
    layout="wide"
)

# -------- MODERN PREMIUM UI --------
st.markdown("""
<style>

/* MAIN */
.stApp{
    background: linear-gradient(135deg,#eef2ff,#f8fafc);
    color:#111827;
    overflow-x:hidden;
}

/* REMOVE STREAMLIT */
header{visibility:hidden;}
footer{visibility:hidden;}
section[data-testid="stSidebar"]{display:none;}

/* TEXT */
html, body, [class*="css"]{
    color:#111827 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* HERO CARD */
.hero-card{
    background:white;
    border-radius:35px;
    padding:45px;
    box-shadow:0 10px 35px rgba(0,0,0,0.08);
    margin-bottom:40px;
}

/* SMALL BADGE */
.badge{
    display:inline-block;
    background:#6C63FF;
    color:white;
    padding:12px 22px;
    border-radius:40px;
    font-size:18px;
    font-weight:600;
    margin-bottom:25px;
}

/* TITLE */
.hero-title{
    font-size:72px;
    font-weight:900;
    line-height:1.05;
    margin-bottom:15px;
}

.hero-title span{
    color:#6C63FF;
}

/* SUBTITLE */
.hero-sub{
    font-size:24px;
    color:#6b7280;
}

/* BUTTONS */
.stButton > button{
    width:100%;
    min-height:220px;
    border:none;
    border-radius:35px;
    background:white;
    color:#111827;
    font-size:28px;
    font-weight:700;
    transition:0.3s;
    box-shadow:0 10px 25px rgba(0,0,0,0.07);
}

.stButton > button:hover{
    transform:translateY(-8px);
    box-shadow:0 18px 35px rgba(108,99,255,0.18);
    border:2px solid #6C63FF;
}

/* CHAT */
[data-testid="stChatMessage"]{
    background:white;
    border-radius:25px;
    padding:18px;
    margin-bottom:15px;
    box-shadow:0 4px 15px rgba(0,0,0,0.05);
}

/* CHAT INPUT */
.stChatInput input{
    border-radius:30px !important;
    border:2px solid #d1d5db !important;
    padding:16px !important;
    background:white !important;
    color:#111827 !important;
}

/* SECTION TITLE */
.section-title{
    font-size:40px;
    font-weight:800;
    margin-bottom:20px;
}

/* CHANGE BUTTON */
.change-btn{
    background:#6C63FF;
    color:white;
    padding:12px 24px;
    border-radius:15px;
    font-weight:600;
}

/* FLOATING EFFECT */
.float{
    animation: float 3s ease-in-out infinite;
}

@keyframes float{
    0%{transform:translateY(0px);}
    50%{transform:translateY(-12px);}
    100%{transform:translateY(0px);}
}

</style>
""", unsafe_allow_html=True)

# -------- LOAD EMBEDDING MODEL --------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# -------- LOAD FLAN --------
@st.cache_resource
def load_flan():

    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base"
    )

    return tokenizer, model

tokenizer, model = load_flan()

# -------- DATA --------
sections = {

    "Order": [

        {
            "text": "track order where is my order late delivery status",
            "answer": """1. Open My Orders
2. Check order status
3. Track delivery
4. Contact support if delayed"""
        },

        {
            "text": "cancel order remove order",
            "answer": """1. Open My Orders
2. Select order
3. Click Cancel Order"""
        }
    ],

    "Buy": [

        {
            "text": "buy product purchase food order item",
            "answer": """1. Search product
2. Add to cart
3. Go checkout
4. Confirm order"""
        }
    ],

    "Payment": [

        {
            "text": "refund payment failed money deducted",
            "answer": """1. Wait 3-5 days
2. Check bank status
3. Contact support if needed"""
        }
    ],

    "Account": [

        {
            "text": "forgot password login issue",
            "answer": """1. Open Login
2. Click Forgot Password
3. Reset password"""
        }
    ]
}

# -------- BUILD INDEX --------
def build_index(data):

    texts = [d["text"] for d in data]

    vectors = embed_model.encode(texts)

    index = faiss.IndexFlatL2(vectors.shape[1])

    index.add(np.array(vectors))

    return index

# -------- SEARCH --------
def search(data, question):

    index = build_index(data)

    q_vec = embed_model.encode([question])

    D, I = index.search(np.array(q_vec), 1)

    if D[0][0] < 1.2:

        return data[I[0][0]]["answer"]

    return None

# -------- STRICT FLAN-T5 --------
def generate_ai(question):

    prompt = f"""
You are HACSS customer support AI.

Customer Question:
{question}

Rules:
- Give short professional reply
- Maximum 1 sentence
- No random answers
- If unsure say:
Please contact customer support through Help Center.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=25,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.2
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    answer = answer.replace("\n", " ").strip()

    bad_words = [
        "rules",
        "question",
        "reply",
        "assistant",
        "generate"
    ]

    if (
        len(answer) < 5
        or len(answer) > 180
        or any(word in answer.lower() for word in bad_words)
    ):

        answer = (
            "Please contact customer support through the Help Center."
        )

    return answer

# -------- SESSION --------
if "section" not in st.session_state:
    st.session_state.section = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# -------- HERO --------
st.markdown("""
<div class="hero-card">

<div class="badge">
🤖 HACSS AI SUPPORT
</div>

<div style="
display:flex;
justify-content:space-between;
align-items:center;
gap:40px;
">

<div>

<div class="hero-title">
Smart <span>Customer</span><br>
Support Assistant
</div>

<div class="hero-sub">
Fast AI-powered support for orders, payments, accounts and shopping.
</div>

</div>

<div class="float">

<img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png"
width="260">

</div>

</div>

</div>
""", unsafe_allow_html=True)

# -------- HOME --------
if st.session_state.section is None:

    st.markdown("""
    <div class="section-title">
    Choose Your Support Category
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("📦\n\nOrder\n\nTrack & Manage"):
        st.session_state.section = "Order"
        st.rerun()

    if col2.button("🛒\n\nBuy\n\nPurchase Products"):
        st.session_state.section = "Buy"
        st.rerun()

    if col3.button("💳\n\nPayment\n\nRefund & Billing"):
        st.session_state.section = "Payment"
        st.rerun()

    if col4.button("👤\n\nAccount\n\nProfile Settings"):
        st.session_state.section = "Account"
        st.rerun()

# -------- CHAT --------
else:

    st.markdown(f"""
    <div class="section-title">
    {st.session_state.section} Support
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Change Category"):

        st.session_state.section = None
        st.session_state.chat = []

        st.rerun()

    for q, a in st.session_state.chat:

        st.chat_message("user").write(q)

        st.chat_message("assistant").write(a)

    user_input = st.chat_input(
        "Ask your support question..."
    )

    if user_input:

        st.chat_message("user").write(user_input)

        data = sections[st.session_state.section]

        q = user_input.lower().strip()

        greetings = [
            "hi",
            "hello",
            "hey",
            "hii"
        ]

        thanks_words = [
            "thanks",
            "thank you",
            "thx"
        ]

        if get_close_matches(q, greetings, n=1, cutoff=0.7):

            answer = (
                "Hello! 😊 How can HACSS help you today?"
            )

        elif get_close_matches(q, thanks_words, n=1, cutoff=0.7):

            answer = (
                "You're welcome 😊 Happy to help."
            )

        else:

            answer = search(data, user_input)

            if not answer:

                with st.spinner("HACSS is thinking..."):

                    answer = generate_ai(user_input)

        with st.chat_message("assistant"):

            msg = st.empty()

            full = ""

            for char in answer:

                full += char

                msg.markdown(full + "▌")

                time.sleep(0.01)

            msg.markdown(full)

        st.session_state.chat.append(
            (user_input, answer)
        )

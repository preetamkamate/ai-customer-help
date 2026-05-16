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
    page_icon="💬",
    layout="wide"
)

# -------- MODERN UI --------
st.markdown("""
<style>

.stApp {
    background: #f7f7fb;
    color: #111827 !important;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

section[data-testid="stSidebar"] {
    display: none;
}

/* ALL TEXT */
html, body, [class*="css"] {
    color: #111827 !important;
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #111827 !important;
}

/* PARAGRAPH */
p {
    color: #6b7280 !important;
}

/* BUTTON */
.stButton > button {

    width: 100%;
    min-height: 220px;

    border-radius: 30px;

    border: none;

    background: white;

    color: #111827 !important;

    font-size: 24px;

    font-weight: 700;

    box-shadow: 0 8px 25px rgba(0,0,0,0.08);

    transition: 0.3s;

    padding: 20px;
}

.stButton > button:hover {

    transform: translateY(-5px);

    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
}

/* CHAT */
[data-testid="stChatMessage"] {

    background: white;

    border-radius: 20px;

    padding: 15px;

    margin-bottom: 12px;

    box-shadow: 0 2px 10px rgba(0,0,0,0.05);

    color: #111827 !important;
}

/* FORCE CHAT TEXT BLACK */
[data-testid="stChatMessage"] * {

    color: #111827 !important;

    opacity: 1 !important;
}

/* MARKDOWN */
.stMarkdown,
.stMarkdown p,
.stMarkdown li {

    color: #111827 !important;
}

/* INPUT */
.stChatInput input {

    border-radius: 20px !important;

    border: 2px solid #d1d5db !important;

    padding: 15px !important;

    background: white !important;

    color: #111827 !important;
}

.stChatInput input::placeholder {
    color: #6b7280 !important;
}

</style>
""", unsafe_allow_html=True)

# -------- LOAD EMBEDDING MODEL --------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# -------- LOAD FLAN-T5 --------
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
            "text": "track order where is my order not delivered late order status",
            "answer": """1. Open My Orders
2. Check order status
3. Track delivery
4. Contact support if delayed"""
        },

        {
            "text": "cancel order remove order",
            "answer": """1. Open My Orders
2. Select your order
3. Click Cancel Order"""
        },

        {
            "text": "support contact customer care help",
            "answer": """1. Open Help Center
2. Contact customer support
3. Explain your issue"""
        }
    ],

    "Buy": [

        {
            "text": "how to order buy product purchase food ice cream",
            "answer": """1. Search product
2. Add to cart
3. Go to checkout
4. Enter details
5. Place order"""
        },

        {
            "text": "after adding cart what next",
            "answer": """1. Open cart
2. Click checkout
3. Enter delivery details
4. Select payment
5. Confirm order"""
        },

        {
            "text": "support contact customer care help",
            "answer": """1. Open Help Center
2. Contact customer support
3. Explain your issue"""
        }
    ],

    "Payment": [

        {
            "text": "payment failed refund money deducted refund not received when i get my refund",
            "answer": """1. Wait 3–5 working days
2. Check bank/wallet
3. Contact support if needed"""
        },

        {
            "text": "double payment charged twice",
            "answer": "Refund for extra payment will be processed automatically."
        },

        {
            "text": "support contact customer care help",
            "answer": """1. Open Help Center
2. Contact customer support
3. Explain your issue"""
        }
    ],

    "Account": [

        {
            "text": "forgot password reset login issue",
            "answer": """1. Go to login page
2. Click Forgot Password
3. Enter details
4. Reset password"""
        },

        {
            "text": "delete account remove account",
            "answer": "Please contact customer support to permanently delete your account."
        },

        {
            "text": "support contact customer care help",
            "answer": """1. Open Help Center
2. Contact customer support
3. Explain your issue"""
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

# -------- FLAN-T5 --------
def generate_ai(question):

    prompt = f"""
Customer support question: {question}
Give a short helpful reply.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=0.2,
        do_sample=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    if len(answer) < 3:

        answer = "Please contact customer support through the Help Center section."

    return answer

# -------- SESSION --------
if "section" not in st.session_state:
    st.session_state.section = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# -------- HERO SECTION --------
st.markdown("""
<div style="padding-top:20px;">

<div style="
background:white;
padding:12px 24px;
border-radius:20px;
width:fit-content;
box-shadow:0 4px 15px rgba(0,0,0,0.08);
font-size:20px;
font-weight:600;
color:#6C63FF;
margin-bottom:25px;
">

Hello! I am HACSS 😊

</div>

<div style="
display:flex;
justify-content:space-between;
align-items:center;
">

<div>

<h1 style="
font-size:72px;
font-weight:800;
color:#111827;
line-height:1.1;
margin-bottom:10px;
">

Your <span style="color:#6C63FF;">AI</span><br>
Customer Support Assistant

</h1>

<p style="
font-size:24px;
color:#6b7280;
">
How can I help you today?
</p>

</div>

<div style="position:relative;">

<img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png"
width="250">

<div style="
position:absolute;
top:-20px;
right:0;
background:#6C63FF;
padding:14px 20px;
border-radius:20px;
color:white;
font-size:18px;
font-weight:600;
box-shadow:0 4px 15px rgba(0,0,0,0.12);
">

💬 Messages

</div>

<div style="
position:absolute;
top:80px;
right:-60px;
background:#60A5FA;
padding:14px 20px;
border-radius:20px;
color:white;
font-size:18px;
font-weight:600;
box-shadow:0 4px 15px rgba(0,0,0,0.12);
transform:rotate(-8deg);
">

📩 Support

</div>

<img src="https://cdn-icons-png.flaticon.com/512/628/628324.png"
width="120"
style="
position:absolute;
bottom:-40px;
right:-20px;
">

</div>

</div>

</div>
""", unsafe_allow_html=True)

# -------- HOME --------
if st.session_state.section is None:

    st.markdown("## Choose your issue")

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("📦\n\nOrder\n\nTrack and manage orders"):
        st.session_state.section = "Order"
        st.rerun()

    if col2.button("🛒\n\nBuy\n\nPurchase products easily"):
        st.session_state.section = "Buy"
        st.rerun()

    if col3.button("💳\n\nPayment\n\nRefunds and transactions"):
        st.session_state.section = "Payment"
        st.rerun()

    if col4.button("👤\n\nAccount\n\nManage your profile"):
        st.session_state.section = "Account"
        st.rerun()

# -------- CHAT --------
else:

    st.markdown(
        f"<h2>Selected: {st.session_state.section}</h2>",
        unsafe_allow_html=True
    )

    if st.button("🔄 Change Issue"):

        st.session_state.section = None

        st.session_state.chat = []

        st.rerun()

    for q, a in st.session_state.chat:

        st.chat_message("user").write(q)

        st.chat_message("assistant").write(a)

    user_input = st.chat_input("Ask your question...")

    if user_input:

        st.chat_message("user").write(user_input)

        data = sections[st.session_state.section]

        q = user_input.lower().strip()

        greetings = [
            "hi",
            "hello",
            "hey",
            "hlo",
            "hii"
        ]

        thanks_words = [
            "thanks",
            "thank you",
            "thx"
        ]

        close_greeting = get_close_matches(
            q,
            greetings,
            n=1,
            cutoff=0.7
        )

        close_thanks = get_close_matches(
            q,
            thanks_words,
            n=1,
            cutoff=0.7
        )

        if close_greeting:

            answer = "Hello! I am HACSS 😊 How can I help you today?"

        elif close_thanks:

            answer = "You're welcome! 😊 Happy to help."

        else:

            answer = search(data, user_input)

            if not answer:

                with st.spinner("HACSS is thinking..."):

                    answer = generate_ai(user_input)

        with st.chat_message("assistant"):

            message = st.empty()

            full_text = ""

            for char in answer:

                full_text += char

                message.markdown(full_text + "▌")

                time.sleep(0.01)

            message.markdown(full_text)

        st.session_state.chat.append((user_input, answer))

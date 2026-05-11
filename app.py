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

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small"
    )

    embed_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return tokenizer, model, embed_model

tokenizer, model, embed_model = load_models()

# ---------------- DATA ----------------
data = {

    "Payment": [

        {
            "keywords": ["refund", "money", "payment failed"],
            "answer": """1. Wait 3-5 working days
2. Check bank/wallet
3. Contact support if needed"""
        },

        {
            "keywords": ["charged twice", "double payment"],
            "answer": "Refund for extra payment will be processed automatically."
        }

    ],

    "Order": [

        {
            "keywords": ["track order", "where is my order"],
            "answer": """1. Open app
2. Go to My Orders
3. Click Track Order"""
        },

        {
            "keywords": ["cancel order"],
            "answer": """1. Open My Orders
2. Select Order
3. Click Cancel"""
        }

    ],

    "Account": [

        {
            "keywords": ["reset password", "forgot password"],
            "answer": """1. Click Forgot Password
2. Verify mobile/email
3. Set new password"""
        }

    ]

}

# ---------------- CREATE VECTOR DATA ----------------
texts = []
answers = []

for category in data:

    for item in data[category]:

        for k in item["keywords"]:

            texts.append(k)

            answers.append(item["answer"])

# ---------------- EMBEDDINGS ----------------
vectors = embed_model.encode(texts)

index = faiss.IndexFlatL2(vectors.shape[1])

index.add(np.array(vectors))

# ---------------- CHAT MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- INTRO ----------------
if "intro" not in st.session_state:

    st.session_state.intro = True

    st.session_state.messages.append(
        (
            "assistant",
            "Hello! I am HACSS, your AI Customer Support Assistant. How can I help you today?"
        )
    )

# ---------------- FLAN-T5 FUNCTION ----------------
def generate_ai(question):

    prompt = f"""
You are HACSS, an AI customer support assistant.

Answer the user politely and clearly.

User Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.7,
        do_sample=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer

# ---------------- UI ----------------
st.title("💬 HACSS - Customer Support")

# ---------------- CATEGORY SELECT ----------------
selected = st.selectbox(
    "Select Issue Category",
    list(data.keys())
)

st.write(f"### Selected: {selected}")

# ---------------- SHOW CHAT ----------------
for role, msg in st.session_state.messages:

    st.chat_message(role).write(msg)

# ---------------- INPUT ----------------
user_input = st.chat_input("Ask your question...")

if user_input:

    st.session_state.messages.append(
        ("user", user_input)
    )

    st.chat_message("user").write(user_input)

    q = user_input.lower()

    found = False

    # ---------------- KEYWORD MATCH ----------------
    for item in data[selected]:

        for k in item["keywords"]:

            if k in q:

                answer = item["answer"]

                st.chat_message("assistant").write(answer)

                st.session_state.messages.append(
                    ("assistant", answer)
                )

                found = True

                break

        if found:
            break

    # ---------------- FAISS SEARCH ----------------
    if not found:

        q_vec = embed_model.encode([q])

        D, I = index.search(np.array(q_vec), 1)

        score = D[0][0]

        # LOWER = MORE SIMILAR
        if score < 1.0:

            answer = answers[I[0][0]]

            st.chat_message("assistant").write(answer)

            st.session_state.messages.append(
                ("assistant", answer)
            )

            found = True

    # ---------------- FLAN-T5 AI ----------------
    if not found:

        with st.spinner("HACSS is thinking..."):

            ai_answer = generate_ai(user_input)

        st.chat_message("assistant").write(ai_answer)

        st.session_state.messages.append(
            ("assistant", ai_answer)
        )

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

# -------- DATA (YOUR APP KNOWLEDGE) --------
data = [
{
"text": "track order",
"type": "navigation",
"steps": ["Open app", "Go to My Orders", "Select order", "Click Track Order"]
},

{
"text": "cancel order",
"type": "navigation",
"steps": ["Open app", "Go to My Orders", "Select order", "Click Cancel Order"]
},

{
"text": "payment failed",
"type": "general",
"reply": "If payment failed but money deducted, refund will happen in 3-5 days."
}
]

# -------- VECTOR SEARCH --------
texts = [d["text"] for d in data]
vectors = embed_model.encode(texts)

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# -------- FUNCTION --------
def generate_answer(question, context=""):
    prompt = f"Answer clearly: {question}. Context: {context}"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs, max_length=50)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------- UI --------
st.title("AI Customer Help Assistant")

question = st.text_input("Ask your problem")

if question:

    q_vec = embed_model.encode([question])
    D,I = index.search(np.array(q_vec),1)

    result = data[I[0][0]]

    # -------- NAVIGATION --------
    if result["type"] == "navigation":

        st.subheader("Steps")

        for step in result["steps"]:
            st.write("-", step)

    # -------- GENERAL --------
    else:

        answer = generate_answer(question, result["reply"])
        st.write(answer)

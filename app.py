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

# -------- DATA --------
data = [
{
"text": "track order",
"type": "navigation",
"steps": ["Open app", "Go to My Orders", "Click Track Order"],
"images": ["orders_page.png", "track_button.png"]
},

{
"text": "cancel order",
"type": "navigation",
"steps": ["Open app", "Go to My Orders", "Click Cancel Order"],
"images": ["cancel_page.png"]
},

{
"text": "payment failed",
"type": "general",
"answer": "Refund will be processed in 3-5 days"
}
]

# -------- VECTOR DB --------
texts = [d["text"] for d in data]
vectors = embed_model.encode(texts)

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))

# -------- AI FUNCTION --------
def generate_ai(question):
    prompt = f"Answer clearly: {question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------- UI --------
st.title("AI Customer Support Assistant")

question = st.text_input("Ask your problem")

if question:

    q_vec = embed_model.encode([question])
    D, I = index.search(np.array(q_vec), 1)

    score = D[0][0]

    # -------- FIXED ANSWER (WITH IMAGE) --------
    if score < 0.8:

        result = data[I[0][0]]

        # Navigation → show steps + images
        if result["type"] == "navigation":

            st.success("⚡ Navigation Guide")

            st.subheader("Steps")
            for step in result["steps"]:
                st.write("-", step)

            st.subheader("Images")
            for img in result["images"]:
                st.image(f"images/{img}")

        # General → fixed text
        else:
            st.success("⚡ Fast Answer")
            st.write(result["answer"])

    # -------- AI GENERATION --------
    else:
        st.info("🤖 AI Generated Answer")
        answer = generate_ai(question)
        st.write(answer)

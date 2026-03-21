import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

st.title("AI Customer Help Assistant")

st.write("Loading model... ⏳ (first time takes time)")

tokenizer, model = load_model()

st.success("Model loaded ✅")

# -------- FUNCTION --------
def get_answer(question):
    # better prompt
    prompt = f"Answer shortly and clearly: {question}"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=50,
        num_beams=2,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# -------- UI --------
st.write("Ask your problem below 👇")

question = st.text_input("Type your question")

if not question:
    st.info("Waiting for your question...")
else:
    start = time.time()

    with st.spinner("Processing..."):
        answer = get_answer(question)

    end = time.time()

    st.success(answer)
    st.caption(f"⏱ Response time: {round(end-start,2)} sec")

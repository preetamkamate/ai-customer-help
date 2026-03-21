import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

tokenizer, model = load_model()

# Function
def get_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI
st.title("AI Customer Help Assistant")
st.write("Ask your problem below 👇")

question = st.text_input("Type your question")

if question:
    st.write("Processing...")
    answer = get_answer(question)
    st.success(answer)

HACSS – AI Customer Support Assistant

HACSS is an AI-based customer support system designed to understand customer queries and provide relevant responses using multiple AI and NLP techniques.

How It Works

User Query
    ↓
Keyword Search
    ↓
If no match
    ↓
Sentence Transformer
    ↓
FAISS Semantic Search
    ↓
If no match
    ↓
FLAN-T5 Generative AI
    ↓
Response

Key Features

* Keyword Search – Provides fast responses for predefined customer queries.
* Semantic Search – Uses Sentence Transformers and FAISS to find relevant answers even when the wording is different.
* Generative AI – Uses FLAN-T5 for generating responses to unseen queries.
* Fuzzy Matching – Handles common spelling variations in greetings and user inputs.
* Interactive UI – Built using Streamlit for real-time customer interaction.

Technologies Used

* Python
* Streamlit
* NLP
* Sentence Transformers
* FAISS
* FLAN-T5
* Hugging Face Transformers
* NumPy

System Workflow

1. The user enters a customer-support query.
2. Keyword Search checks for a predefined matching query.
3. If no keyword match is found, Sentence Transformer converts the query into an embedding.
4. FAISS performs semantic similarity search against the knowledge base.
5. If no relevant result is found, FLAN-T5 generates a response.
6. HACSS displays the final response through the Streamlit interface.

Project Structure

HACSS/
├── app.py
├── requirements.txt
├── README.md
└── assets/

Objective

To develop a fast and user-friendly AI customer support system that combines keyword matching, semantic search, and generative AI to improve automated customer assistance.

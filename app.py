
# -------- DATA --------
sections = {
    "Order / Delivery": [
        {
            "text": "track order where is my order not delivered late",
            "answer": """1. Open My Orders
    "Order": [
        {"text": "track order where is my order not delivered late",
         "answer": """1. Open My Orders
2. Check order status
3. Track delivery
4. Contact support if delayed"""
        }
4. Contact support if delayed"""}
],

    "Buy / Product": [
        {
            "text": "how to order buy product purchase food",
            "answer": """1. Search the product
    "Buy": [
        {"text": "how to order buy product food ice cream",
         "answer": """1. Search product
2. Add to cart
3. Go to checkout
4. Enter details
5. Place your order"""
        },
        {
            "text": "after adding cart what next",
            "answer": """1. Open cart
2. Click checkout
3. Enter delivery details
4. Select payment
5. Confirm order"""
        }
5. Place order"""}
],

    "Payment / Refund": [
        {
            "text": "payment failed refund",
            "answer": """1. Wait 3–5 working days
    "Payment": [
        {"text": "payment failed refund",
         "answer": """1. Wait 3–5 days
2. Check bank/wallet
3. Contact support if needed"""
        }
3. Contact support if needed"""}
],

"Account": [
        {
            "text": "forgot password reset login issue",
            "answer": """1. Go to login page
        {"text": "forgot password login issue",
         "answer": """1. Go to login
2. Click Forgot Password
3. Enter details
4. Set new password"""
        }
4. Reset password"""}
]
}

# -------- BUILD INDEX --------
def build_index(data):
# -------- FUNCTION --------
def search(data, question):
texts = [d["text"] for d in data]
vectors = embed_model.encode(texts)

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(np.array(vectors))
    return index

    q_vec = embed_model.encode([question])
    D, I = index.search(np.array(q_vec), 1)

    if D[0][0] < 1.2:
        return data[I[0][0]]["answer"]
    return None

# -------- SESSION --------
if "section" not in st.session_state:
    st.session_state.section = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# -------- UI --------
st.title("💬 HACSS - Customer Support")

# SECTION SELECT
selected_section = st.selectbox(
    "Select your issue type:",
    list(sections.keys())
)
# -------- STEP 1: SELECT SECTION --------
if st.session_state.section is None:

# INIT CHAT
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.write("### Choose your issue:")

# SHOW CHAT
for q, a in st.session_state.chat_history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)
    col1, col2 = st.columns(2)

# INPUT
user_input = st.chat_input("Ask your question...")
    if col1.button("📦 Order"):
        st.session_state.section = "Order"

if user_input:
    st.chat_message("user").write(user_input)
    if col2.button("🛒 Buy"):
        st.session_state.section = "Buy"

    data = sections[selected_section]
    index = build_index(data)
    if col1.button("💳 Payment"):
        st.session_state.section = "Payment"

    q_vec = embed_model.encode([user_input])
    D, I = index.search(np.array(q_vec), 1)
    if col2.button("👤 Account"):
        st.session_state.section = "Account"

    if D[0][0] < 1.2:
        answer = data[I[0][0]]["answer"]
    else:
        answer = """I didn't understand clearly.
# -------- STEP 2: CHAT --------
else:
    st.write(f"### Selected: {st.session_state.section}")

    # CHANGE BUTTON
    if st.button("🔄 Change Issue"):
        st.session_state.section = None
￼
        st.session_state.chat = []
        st.rerun()

    # SHOW CHAT
    for q, a in st.session_state.chat:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    # INPUT
    user_input = st.chat_input("Ask your question...")

    if user_input:
        st.chat_message("user").write(user_input)

        data = sections[st.session_state.section]
        answer = search(data, user_input)

        if not answer:
            answer = """I didn't understand clearly.

Try asking about:
• Order
• Payment
• Account"""
Try asking related to this section."""

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append((user_input, answer))
        st.chat_message("assistant").write(answer)
        st.session_state.chat.append((user_input, answer))

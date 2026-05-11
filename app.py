# =========================
# FILE 1 : app.py
# =========================

import streamlit as st

st.set_page_config(page_title="HACSS", page_icon="💬")

st.title("💬 HACSS")
st.subheader("Hybrid AI Customer Support System")

st.write("Choose Support Section")

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

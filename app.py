import streamlit as st

pages = [
    st.Page("pages/binomiale_normale.py", title="Binomiale → Normale"),
    st.Page("pages/test_adequation.py", title="Test d'adéquation"),
]

pg = st.navigation(pages)
pg.run()
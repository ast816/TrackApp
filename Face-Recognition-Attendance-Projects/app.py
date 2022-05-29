# app.py

# Import all apps part of the multipage app
from apps import Home,Hello,Test
import streamlit as st


# Naming all the three pages

PAGES = {
    "Home": Home,
    "Hello New User": Hello,
    "Test Model":Test
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

# app.py — Streamlit launcher (safe version)
import streamlit as st
from importlib import import_module
import uuid

st.set_page_config(page_title="EMIPredict AI", layout="wide")

# Import from "pages" instead of "app"
PAGES = {
    "Home": "pages.Home",
    "EMI Eligibility": "pages.EMI_Eligibility",
    "Max EMI Predictor": "pages.Max_EMI_Predictor"
}

if "page" not in st.session_state:
    st.session_state.page = "Home"

# Generate unique key for this session (avoids duplicate element key errors)
if "nav_key" not in st.session_state:
    st.session_state.nav_key = str(uuid.uuid4())

st.sidebar.title("Navigation")
options = list(PAGES.keys())

try:
    default_index = options.index(st.session_state.page)
except ValueError:
    default_index = 0

selection = st.sidebar.selectbox(
    "Go to",
    options,
    index=default_index,
    key=f"main_nav_{st.session_state.nav_key}"
)

st.session_state.page = selection

module = import_module(PAGES[selection])
if hasattr(module, "run"):
    module.run()
else:
    st.warning("⚠️ Page not implemented")

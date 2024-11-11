import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Use model", layout="wide")

components.iframe("http://127.0.0.1:5000", width=1200, height=2000)

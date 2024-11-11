# pages/page1.py
import streamlit as st


from contextlib import contextmanager, redirect_stdout
from io import StringIO

from ds.main import train_model

st.set_page_config(page_title="Train model", layout="wide")
st.title("Train mninist mlp model")


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


col1, col2 = st.columns([1, 3])

with col1:
    hidden_size = st.number_input(label="hidden_size", value=256)
    epochs = st.number_input(label="epochs", value=100)
    batch_size = st.number_input(label="batch_size", value=32)
    initial_lr = st.number_input(label="initial_lr", value=0.01)
    decay = st.number_input(label="decay", value=0.001, format="%0.3f")
    patience = st.number_input(label="patience", value=5)

with col2:
    if st.session_state.get("but_a", False):
        st.session_state.disabled = True

    button_a = st.button(
        "Train model", key="but_a", disabled=st.session_state.get("disabled", False)
    )

    if button_a:
        st.session_state.button_disabled = True
        st.write("the model is training")
        output = st.empty()
        with st_capture(output.code):
            train_model(
                hidden_size=hidden_size,
                epochs=epochs,
                batch_size=batch_size,
                initial_lr=initial_lr,
                decay=decay,
                patience=patience,
            )
        st.session_state.disabled = False

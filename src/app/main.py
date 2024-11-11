from typing import Dict, List, Any, Tuple
import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2
import mlflow
from mlflow.client import MlflowClient
from mlflow.pyfunc import PyFuncModel
import plotly.graph_objects as go
from numpy.typing import NDArray


def get_models() -> List[Dict[str, Any]]:
    """
    Retrieve registered models from MLflow.

    Returns:
        List[Dict[str, Any]]: List of model information dictionaries
    """
    client: MlflowClient = mlflow.MlflowClient()
    data = client.search_registered_models()
    models: List[str] = []

    for model in data:
        models.append(model.name)

    result: List[Dict[str, Any]] = []
    for model in models:
        model_versions: Dict[str, Any] = {"name": model}
        data = client.search_model_versions(
            filter_string=f"name='{model}'", order_by=["version_number DESC"]
        )
        versions: List[Dict] = list(map(lambda x: dict(x), data))
        model_versions["latest_versions"] = versions
        result.append(model_versions)

    return result


def process_image(image_data: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Process the drawn image for model prediction.

    Args:
        image_data: Raw image data from canvas

    Returns:
        NDArray[np.float32]: Processed image ready for model prediction,
        NDArray[np.float32]: alpha channel of image redy for preview
    """
    blurred_image: NDArray[np.float32] = cv2.GaussianBlur(image_data, (23, 23), 2)
    alpha_channel: NDArray[np.float32] = blurred_image[:, :, 3]
    resized_image: NDArray[np.float32] = cv2.resize(
        alpha_channel, (28, 28), interpolation=cv2.INTER_AREA
    )
    return resized_image.reshape(-1, 28 * 28).astype("float32") / 255, alpha_channel


def create_probability_plot(probabilities: NDArray[np.float32]) -> go.Figure:
    """
    Create a probability distribution plot.

    Args:
        probabilities: Model prediction probabilities

    Returns:
        go.Figure: Plotly figure object
    """
    categories: List[str] = [f"{i}" for i in range(len(probabilities))]
    fig: go.Figure = go.Figure(
        data=[go.Bar(x=categories, y=probabilities, marker_color="skyblue")]
    )
    # Process the drawn image for model prediction.
    fig.update_layout(
        title="Probability Distribution",
        xaxis_title="Categories",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
    )
    return fig


def main() -> None:
    """Main application function."""
    st.set_page_config(page_title="Use model", layout="wide")
    st.title("Use model, draw the number")
    st.info(
        "Model was trained on scanned handwriting, not digital input. "
        "Model accuracy will likely be lower than measured accuracy."
    )

    # Sidebar settings
    drawing_mode: str = "freedraw"
    realtime_update: bool = st.sidebar.checkbox("Update in realtime", True)

    # Model selection
    registered_models: List[Dict[str, Any]] = get_models()
    model_names: List[str] = [
        (
            x["name"],
            x["latest_versions"][0]["version"],
        )
        for x in registered_models
    ]
    selected_model: Tuple[str, int] = st.selectbox("models", model_names)

    # Load model
    model: PyFuncModel = mlflow.pyfunc.load_model(
        f"models:/{selected_model[0]}/{selected_model[1]}"
    )

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        canvas_result = st_canvas(
            fill_color="#eee",
            stroke_width=9,
            stroke_color="rgba(0, 0, 0, 0.9)",
            background_color=None,
            update_streamlit=realtime_update,
            height=28 * 4,
            width=28 * 4,
            drawing_mode=drawing_mode,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="full_app",
        )

        is_draft: bool = not (
            canvas_result.image_data == np.zeros([112, 112, 4])
        ).all() and (canvas_result.image_data is not None)

        if is_draft:
            image: NDArray[np.float32] = canvas_result.image_data
            processed_image, alpha_channel = process_image(image)
            st.image(alpha_channel)

            probabilities: NDArray[np.float32] = model.predict(
                processed_image, {"predict_method": "predict_proba"}
            )[0]

            with col2:
                if is_draft:
                    fig: go.Figure = create_probability_plot(probabilities)
                    st.title("Probability Distribution")
                    st.plotly_chart(fig)


if __name__ == "__main__":
    main()

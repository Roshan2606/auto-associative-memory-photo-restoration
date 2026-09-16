import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from experiment import run_damage_experiment
from image_utils import load_image


st.set_page_config(
    page_title="Photo Restoration Using Auto-associative Memory Network",
    page_icon="🖼️",
    layout="wide",
)


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


def reset_state():
    for key in [
        "uploaded_image",
        "network_iterations",
        "image_size",
        "experiment_run",
        "results_df",
        "failure_point",
        "csv_path",
        "chart_path",
    ]:
        st.session_state.pop(key, None)


st.title("Photo Restoration Using Auto-associative Memory Network")

st.markdown("""
This project demonstrates a small Hopfield-style auto-associative memory network for image restoration.
It is an educational prototype for a college assignment and is designed to show how a damaged photograph can be reconstructed from a compact grayscale memory pattern.

Important: a simple Hopfield network is not a high-quality modern photo restoration model. It is useful for understanding associative recall, but natural photographs may fail at higher damage levels.
""")

with st.sidebar:
    st.header("Network Configuration")
    st.session_state.network_iterations = st.slider(
        "Number of recall iterations",
        min_value=10,
        max_value=200,
        value=60,
        step=10,
    )
    st.session_state.image_size = st.selectbox(
        "Low-resolution grayscale image size",
        options=[(16, 16), (24, 24), (32, 32)],
        index=2,
    )

    st.header("Experiment")
    damage_levels = [10, 20, 30, 40, 50, 60, 70]
    st.session_state.experiment_run = st.button("Run damage experiment")
    st.session_state.reset_button = st.button("Reset", on_click=reset_state)

st.header("1. Upload Photograph")
uploaded_file = st.file_uploader(
    "Upload a photograph (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    help="Upload a real photo. The application converts it to a small grayscale memory pattern.",
)

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.session_state.uploaded_image = image
    st.image(image, caption="Uploaded photograph", use_container_width=True)

    if st.session_state.experiment_run:
        with st.spinner("Processing image and running the recall experiment..."):
            results_df, failure_point, csv_path, chart_path = run_damage_experiment(
                original_image=image,
                damage_levels=damage_levels,
                output_dir=str(RESULTS_DIR),
                iterations=st.session_state.network_iterations,
                image_size=st.session_state.image_size,
                output_prefix="experiment",
            )

            st.session_state.results_df = results_df
            st.session_state.failure_point = failure_point
            st.session_state.csv_path = csv_path
            st.session_state.chart_path = chart_path

if "results_df" in st.session_state:
    result_df = st.session_state.results_df
    failure_point = st.session_state.failure_point

    st.header("2. Original vs Damaged vs Recalled Images")
    for _, row in result_df.iterrows():
        damage = int(row["damage_percent"])
        st.subheader(f"Damage Level: {damage}%")
        col1, col2, col3 = st.columns(3)

        original_path = str(RESULTS_DIR / "experiment_original.png")
        damaged_path = str(RESULTS_DIR / f"experiment_damage_{damage}_damaged.png")
        recalled_path = str(RESULTS_DIR / f"experiment_damage_{damage}_recalled.png")

        with col1:
            st.image(original_path, caption="Original image")
        with col2:
            st.image(damaged_path, caption=f"Damaged image ({damage}%)")
        with col3:
            st.image(recalled_path, caption=f"Recalled image ({damage}%)")

    st.header("3. Reconstruction Accuracy Table")
    st.dataframe(result_df[["damage_percent", "accuracy", "reliable_recall"]], use_container_width=True)

    st.header("4. Performance Chart")
    st.image(str(st.session_state.chart_path), caption="Damage percentage versus reconstruction accuracy")

    st.header("5. Failure Point Analysis")
    if failure_point is None:
        st.success("No failure point was detected within the tested damage range. The network remained reasonably reliable for all tested levels.")
    else:
        st.warning(
            f"The first damage level where reliable recall appears to fail is {failure_point}%. "
            "This indicates the network begins to lose stable associative recall beyond this damage threshold."
        )

    st.header("6. Download Results")
    with open(st.session_state.csv_path, "rb") as file:
        st.download_button(
            label="Download experiment CSV",
            data=file,
            file_name="experiment_results.csv",
            mime="text/csv",
        )

st.markdown("""
## How the network works

- The image is converted into a compact grayscale pattern.
- A bipolar representation is used: bright pixels become +1 and dark pixels become -1.
- Hebbian learning stores the pattern in a weight matrix using the relation:
  W = (1/N) sum x x^T, with diagonal weights set to zero.
- Recall works by iteratively updating each neuron using the weighted sum of neighboring neurons.
- As damage increases, the network receives less correct information and may settle into a wrong attractor state.

A small Hopfield network is suitable for an educational demonstration, but it is not a modern high-quality photo restoration model.
""")

st.markdown("""
## Project Explanation

The auto-associative memory network stores a low-resolution version of the photograph as a memory pattern. When a damaged version is presented, the network updates its internal state iteratively until it converges to the nearest stored memory. This process is known as associative recall.
""")

"""Main entry point for the visualizations."""
import streamlit as st

st.set_page_config(
    page_title="Experiment Visualizations",
    page_icon="🧠",
)

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    # Experiments
    This project contains various demos for brain-inspired systems.

    Click any of the interactive visualizations on the left panel to get started.
    """
)

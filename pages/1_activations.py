
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from experiments.simplelayer import SimpleLayer


SEED = 0


if __name__ == "__main__":
    st.set_page_config(layout="centered")
    st.title("Activations")
    st.markdown(
        """
        The goal of this experiment is to validate a model using `SimpleLayer` where the activity
        can remain somewhat constant with a variety of neuron activations.
        """
    )

    with st.sidebar:
        n = st.slider("n", min_value=1, max_value=100, value=50)
        steps = st.slider("steps", min_value=10, max_value=100, value=50)

        potential_decay = st.slider("potential_decay", min_value=0.0, max_value=1.0, value=0.9)
        activation_threshold = st.slider("activation_threshold", min_value=0.0, max_value=1.0, value=0.8)
        refractory_value = st.slider("refractory_value", min_value=-10.0, max_value=0.0, value=-0.2)

    np.random.seed(SEED)
    l = SimpleLayer(
        n=n,
        potential_decay=potential_decay,
        activation_threshold=activation_threshold,
        refractory_value=refractory_value,
    )
    activation_records = []
    potential_records = []
    for i in range(steps):
        activations = l.activations()
        potential_records.append(l.potential)
        activation_records.append(activations)
        l.tick(activations=activations)

    all_activations = [
        (idx[0], t)
        for t, slice in enumerate(activation_records)
        for idx in np.transpose(np.nonzero(slice))
    ]
    df = pd.DataFrame(
        data=all_activations,
        columns=["idx", "t"],
    )
    
    act_fig = px.density_heatmap(df, x="t", y="idx", marginal_x="histogram", marginal_y="histogram", nbinsx=steps, nbinsy=n)
    st.header("Current Results")
    firing_period = steps * n / len(all_activations) if all_activations else float("inf")
    average_activation = len(all_activations) / (n * steps) * 100
    st.subheader(f"Firing period: {firing_period:.2f} cycles")
    st.subheader(f"Average activation: {average_activation:.2f}%")
    st.plotly_chart(act_fig, theme="streamlit", use_container_width=True)
    st.caption("Neuron activations across time")
    pot_fig = px.imshow(np.transpose(potential_records))
    st.plotly_chart(pot_fig, theme="streamlit", use_container_width=True)
    st.caption("Neuron potential across time")
    con_fig = px.imshow(np.transpose(l.weights.excitatory_connections - l.weights.inhibitory_connections), origin="lower")
    st.plotly_chart(con_fig, theme="streamlit", use_container_width=True)
    st.caption("Connections")

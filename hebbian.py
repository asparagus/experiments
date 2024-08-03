
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from experiments.learning import HebbianLearning
from experiments.simplelayer import SimpleLayer


SEED = 0


if __name__ == "__main__":
    st.title("Hebbian")
    st.markdown(
        """
        The goal of this experiment is to validate a model using `Hebbian` learning
        reaches somewhat stable patterns after a few iterations.
        """
    )

    with st.sidebar:
        n = st.slider("n", min_value=1, max_value=100, value=50)
        steps = st.slider("steps", min_value=10, max_value=100, value=50)

        potential_decay = st.slider("potential_decay", min_value=0.0, max_value=1.0, value=0.9)
        activation_threshold = st.slider("activation_threshold", min_value=0.0, max_value=1.0, value=0.8)
        refractory_value = st.slider("refractory_value", min_value=-10.0, max_value=0.0, value=-0.2)

        learning_rate = st.slider("learning_rate", min_value=0.00, max_value=1.00, value=0.01)

    np.random.seed(SEED)
    l = SimpleLayer(
        n=n,
        potential_decay=potential_decay,
        activation_threshold=activation_threshold,
        refractory_value=refractory_value,
    )
    h = HebbianLearning(learning_rate=learning_rate)
    activation_records = []
    potential_records = []
    activations = l.activations()
    for i in range(steps):
        potential_records.append(l.potential)
        activation_records.append(activations)
        l.tick(activations=activations)
        next_activations = l.activations()
        l.excitatory_connections = h.excitatory_update(old_activations=activations, new_activations=next_activations, excitatory_connections=l.excitatory_connections)
        l.inhibitory_connections = h.inhibitory_update(old_activations=activations, new_activations=next_activations, inhibitory_connections=l.inhibitory_connections)
        activations = next_activations

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
    pot_fig = px.imshow(np.transpose(potential_records), origin="lower")
    st.plotly_chart(pot_fig, theme="streamlit", use_container_width=True)
    st.caption("Neuron potential across time")
    con_fig = px.imshow(np.transpose(l.excitatory_connections - l.inhibitory_connections), origin="lower")
    st.plotly_chart(con_fig, theme="streamlit", use_container_width=True)
    st.caption("Connections at the end")


import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from experiments.io import CyclicEncoder
from experiments.learning import HebbianLearning
from experiments.simplelayer import SimpleLayer, Weights


SEED = 0


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Encoder")
    st.markdown(
        """
        The goal of this experiment is to provide inputs to the neurons and observe their learning.
        """
    )

    with st.sidebar:
        with st.expander(label="Network", expanded=False):
            n = st.slider("n", min_value=1, max_value=100, value=50)
            steps = st.slider("steps", min_value=10, max_value=100, value=50)

            potential_decay = st.slider("potential_decay", min_value=0.0, max_value=1.0, value=0.85)
            activation_threshold = st.slider("activation_threshold", min_value=0.0, max_value=1.0, value=0.85)
            refractory_value = st.slider("refractory_value", min_value=-10.0, max_value=0.0, value=-0.2)

            learning_rate = st.slider("learning_rate", min_value=0.00, max_value=1.00, value=0.20)
            norm_inputs = st.checkbox("normalize_inputs", value=False)
            norm_outputs = st.checkbox("normalize_outputs", value=True)

        st.divider()

        with st.expander(label="Input encoder", expanded=True):
            input_dim = st.slider("input_dim", min_value=3, max_value=100, value=20)
            encoding_width = st.slider("encoding_width", min_value=1, max_value=n, value=5)
            options = st.multiselect(
                "Configure the input",
                np.tile(np.arange(input_dim), 10),
                np.tile(np.arange(input_dim), 10),
            )

    np.random.seed(SEED)
    l = SimpleLayer(
        n=n,
        potential_decay=potential_decay,
        activation_threshold=activation_threshold,
        refractory_value=refractory_value,
    )
    h = HebbianLearning(learning_rate=learning_rate)
    ce = CyclicEncoder(dim_in=input_dim, dim_out=n, width=encoding_width)
    encoder_weights = Weights(n=n)

    input_values = []
    output_potentials = []
    output_activations = []
    activation_records = []
    potential_records = []
    activations = l.activations()
    for i in range(steps):
        current_input = options[i % len(options)]
        encoding = ce.encode(0 if np.isnan(current_input) else current_input)
        input_values.append(encoding)

        potential_records.append(l.potential)
        activation_records.append(activations)
        l.tick(activations=activations, external_update=10 * encoder_weights.compute(encoding))
        next_activations = l.activations()
        h.update(old_activations=activations, new_activations=next_activations, weights=l.weights)
        h.update(old_activations=encoding, new_activations=next_activations, weights=encoder_weights)
        if norm_inputs:
            l.weights.normalize_inputs()
            encoder_weights.normalize_inputs()
        if norm_outputs:
            l.weights.normalize_outputs()
            encoder_weights.normalize_outputs()
        activations = next_activations
        output_potential = encoder_weights.compute_transposed(activations)
        output_potentials.append(output_potential)
        output_activations.append(output_potential > activation_threshold)

    all_activations = np.roll(np.transpose(np.nonzero(activation_records)), shift=1, axis=1)
    
    st.header("Current Results")
    left, mid, rght = st.columns(3)
    with left:
        metrics_tab, = st.tabs(["Metrics"])
        with metrics_tab:
            firing_period = steps * n / len(all_activations) if len(all_activations) > 0 else float("inf")
            average_activation = len(all_activations) / (n * steps) * 100
            st.metric("Firing period", value=f"{firing_period:.2f} seconds")
            st.metric("Average activation", value=f"{average_activation:.2f}%")
    with mid:
        inputs_tab, output_potentials_tab, outputs_tab = st.tabs(["Inputs", "Output Potentials", "Outputs"])
        with inputs_tab:
            inputs_fig = px.imshow(np.transpose(input_values), origin="lower")
            inputs_fig.update_xaxes(title="time")
            inputs_fig.update_yaxes(title="neuron_index")
            st.plotly_chart(inputs_fig, theme="streamlit", use_container_width=True)
        with output_potentials_tab:
            output_potentials_fig = px.imshow(np.transpose(output_potentials), origin="lower")
            output_potentials_fig.update_xaxes(title="time")
            output_potentials_fig.update_yaxes(title="neuron_index")
            st.plotly_chart(output_potentials_fig, theme="streamlit", use_container_width=True)
        with outputs_tab:
            outputs_fig = px.imshow(np.transpose(output_activations), origin="lower")
            outputs_fig.update_xaxes(title="time")
            outputs_fig.update_yaxes(title="neuron_index")
            st.plotly_chart(outputs_fig, theme="streamlit", use_container_width=True)

    with rght:
        activations_tab, potential_tab, connections_tab = st.tabs(["Activations", "Potential", "Connections"])
        with activations_tab:
            activations_df = pd.DataFrame(
                data=all_activations,
                columns=["neuron_index", "time"],
            )
            act_fig = px.density_heatmap(activations_df, x="time", y="neuron_index", marginal_x="histogram", marginal_y="histogram", nbinsx=steps, nbinsy=n)
            st.plotly_chart(act_fig, theme="streamlit", use_container_width=True)
        with potential_tab:
            pot_fig = px.imshow(np.transpose(potential_records), origin="lower")
            pot_fig.update_xaxes(title="time")
            pot_fig.update_yaxes(title="neuron_index")
            st.plotly_chart(pot_fig, theme="streamlit", use_container_width=True)
        with connections_tab:
            con_fig = px.imshow(np.transpose(l.weights.excitatory_connections - l.weights.inhibitory_connections), origin="lower")
            con_fig.update_xaxes(title="time")
            con_fig.update_yaxes(title="neuron_index")
            st.plotly_chart(con_fig, theme="streamlit", use_container_width=True)

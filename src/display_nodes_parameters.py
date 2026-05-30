import streamlit as st
from src.config import SIMULATION_CONFIG
import wntr
import plotly.graph_objects as go


def display_nodes_parameters(node_name_list, results_real, results_base):

	with st.expander("Analiza parametrów węzłów"):

	    nodes_str = [x for x in node_name_list if x.isdigit()]

	    col1, col2 = st.columns([1, 4])

	    with col1:
	        st.write("### Filtry")
	        
	        selected_node = st.selectbox(
	            "Node", 
	            options=nodes_str,
	            key="node_selector"
	        )
	        
	        selected_param = st.selectbox(
	            "Parametr:",
	            options=["demand", "pressure"],
	            format_func=lambda x: "Demand" if x == "demand" else "Pressure",
	            key="param_selector"
	        )

	    with col2:
	        data_real = results_real.node[selected_param]
	        data_base = results_base.node[selected_param]
	        
	        if not data_real.index.equals(data_base.index):
	            data_base = data_base.reindex(data_real.index)

	        x_axis = data_real.index / 3600
	        y_mod = data_real[selected_node]
	        y_orig = data_base[selected_node]

	        fig = go.Figure()

	        fig.add_trace(go.Scatter(
	            x=x_axis, 
	            y=y_mod,
	            mode='lines',
	            name=f'Modified ({selected_param})',
	            line=dict(color='#EF553B', width=2)
	        ))

	        fig.add_trace(go.Scatter(
	            x=x_axis, 
	            y=y_orig,
	            mode='lines',
	            name=f'Original ({selected_param})',
	            line=dict(color='#636EFA', width=2, dash='dash')
	        ))

	        unit = " [m]" if selected_param == "pressure" else "" 
	        fig.update_layout(
	            title=f"Node {selected_node}: {selected_param.capitalize()}",
	            xaxis_title="Czas [h]",
	            yaxis_title=f"{selected_param.capitalize()}{unit}",
	            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
	            margin=dict(l=20, r=20, t=60, b=20),
	            hovermode="x unified",
	            template="plotly_white"
	        )

	        st.plotly_chart(fig, use_container_width=True)


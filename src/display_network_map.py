import streamlit as st
from src.config import SIMULATION_CONFIG
import wntr
from pathlib import Path
import pickle
import plotly.graph_objects as go

def display_network_map(scenario_metadata, wn):

	with st.expander("Mapa sieci"):

	    edge_x_normal, edge_y_normal, edge_text_normal = [], [], []
	    edge_x_special, edge_y_special, edge_text_special = [], [], []

	    outlier_scenarios = scenario_metadata.loc[scenario_metadata.is_outlier==True]['leak_location'].unique().tolist()

	    for name, link in wn.links():
	        x0, y0 = wn.get_node(link.start_node_name).coordinates
	        x1, y1 = wn.get_node(link.end_node_name).coordinates

	        x_mid = (x0 + x1) / 2
	        y_mid = (y0 + y1) / 2
	        
	        segment_x = [x0, x_mid, x1, None]
	        segment_y = [y0, y_mid, y1, None]
	        segment_text = [name, name, name, None] 

	        if str(name) in outlier_scenarios:
	            edge_x_special.extend(segment_x)
	            edge_y_special.extend(segment_y)
	            edge_text_special.extend(segment_text)
	        else:
	            edge_x_normal.extend(segment_x)
	            edge_y_normal.extend(segment_y)
	            edge_text_normal.extend(segment_text)

	    node_groups = {}
	    for name, node in wn.nodes():
	        ntype = node.node_type
	        if ntype not in node_groups:
	            colors = {"Junction": "gray", "Reservoir": "blue", "Tank": "green"}
	            node_groups[ntype] = {'x': [], 'y': [], 'text': [], 'color': colors.get(ntype, "red")}

	        x, y = node.coordinates
	        node_groups[ntype]['x'].append(x)
	        node_groups[ntype]['y'].append(y)
	        node_groups[ntype]['text'].append(f"<b>Węzeł:</b> {name}<br><b>Typ:</b> {ntype}")

	    fig = go.Figure()

	    fig.add_trace(go.Scatter(
	        x=edge_x_normal, y=edge_y_normal,
	        text=edge_text_normal,
	        mode='lines',
	        line=dict(width=1.5, color='black'),
	        hoverinfo='text',      
	        hoverlabel=dict(namelength=0), 
	        name='Rury'
	    ))

	    if edge_x_special:
	        fig.add_trace(go.Scatter(
	            x=edge_x_special, y=edge_y_special,
	            text=edge_text_special,
	            mode='lines',
	            line=dict(width=4, color='red'),
	            hoverinfo='text',
	            hoverlabel=dict(namelength=0),
	            name='Ignored leak'
	        ))

	    for ntype, data in node_groups.items():
	        fig.add_trace(go.Scatter(
	            x=data['x'], y=data['y'], 
	            mode='markers', 
	            name=ntype, 
	            text=data['text'], 
	            hoverinfo='text',
	            marker=dict(size=10 if "Junction" in ntype else 14, color=data['color'], line=dict(width=1, color='white'))
	        ))

	    fig.update_layout(
	        margin=dict(l=0, r=0, t=0, b=0),
	        showlegend=True,
	        legend=dict(
	            yanchor="top", y=0.98, xanchor="right", x=0.98,
	            bgcolor="rgba(255, 255, 255, 0.7)", bordercolor="Black", borderwidth=1,
	            font=dict(color="black")
	        ),
	        hovermode='closest',
	        plot_bgcolor='white',
	        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
	        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
	        width=969,
	        height=698
	    )

	    col1, col2, col3 = st.columns([1, 2, 1])
	    with col2:
	        st.plotly_chart(fig, use_container_width=False)

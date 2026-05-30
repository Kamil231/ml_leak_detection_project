import streamlit as st
import pickle
import matplotlib.pyplot as plt
from src.config import SIMULATION_CONFIG
import wntr
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics import auc
from display_XGBoost import display_XGBoost
from display_LightGBM import display_LightGBM
from display_NN import display_NN
from display_Chama import display_Chama
from pathlib import Path





PICKLE_DIR = Path('/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/code/leak_simulation/output_folder/pickle/')

@st.cache_data(show_spinner="Ładowanie wszystkich danych wejściowych...")
def load_all_data(base_dir: Path):
    files_to_load = {
        'chama_outputs': 'chama_outputs.pkl',
        'scenario_metadata': 'scenario_metadata.pkl',
        'sensors_wn_dict': 'sensors_wn_dict.pkl',
        'precision_recall_data': 'precision_recall_data_chama.pkl',
        'cm_xgb': 'confusion_matrix_df.pkl',
        'cm_best_xgb': 'confusion_matrix_best_nodes_df.pkl',
        'cm_lgb': 'confusion_matrix_lgb.pkl',
        'cm_best_lgb': 'confusion_matrix_best_nodes_lgb.pkl',
        'signals': 'signals_with_bp.pkl',
        'cm_nn': 'confusion_matrix_df_nn.pkl',
        'cm_best_nn': 'confusion_matrix_df_nn_top_nodes.pkl'
    }
    
    loaded_data = {}
    
    for key, filename in files_to_load.items():
        with open(base_dir / filename, 'rb') as file:
            loaded_data[key] = pickle.load(file)
            
    return loaded_data

@st.cache_resource(show_spinner="Uruchamianie i cachowanie symulacji WNTR...")
def load_simulation_results():
    wn_base = SIMULATION_CONFIG.create_network_base()
    wn_real = SIMULATION_CONFIG.create_network_real()
    
    sim_real = wntr.sim.WNTRSimulator(wn_real)
    results_real = sim_real.run_sim()
    
    sim_base = wntr.sim.WNTRSimulator(wn_base)
    results_base = sim_base.run_sim()

    node_name_list = wn_base.node_name_list
    
    return results_real, results_base, node_name_list, wn_real

data = load_all_data(PICKLE_DIR)

chama_outputs = data['chama_outputs']
scenario_metadata = data['scenario_metadata']
sensors_wn_dict = data['sensors_wn_dict']
precision_recall_data = data['precision_recall_data']
confusion_matrix_df_XGB = data['cm_xgb']
confusion_matrix_best_nodes_df_XGB = data['cm_best_xgb']
confusion_matrix_df_LGB = data['cm_lgb']
confusion_matrix_best_nodes_df_LGB = data['cm_best_lgb']
df_signals = data['signals']
confusion_matrix_df = data['cm_nn']
confusion_matrix_best_nodes_df = data['cm_best_nn']

results_real, results_base, node_name_list, wn = load_simulation_results()

st.set_page_config(layout="wide")

nodal_thresholds = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'nodal_thresholds_std.pkl')

with st.expander("Analiza przebiegów sygnałów"):

    col_params_sig, col_plots_sig = st.columns([1, 4], vertical_alignment="top")

    with col_params_sig:
        st.markdown("### Filtry")

        view_mode = st.toggle("Widok: Scenariusz dla wszystkich węzłów", value=False, key="toggle_view_mode_1")

        st.divider()

        leak_options_sig = sorted(scenario_metadata['leak_diameter_parameter'].unique())
        time_options_sig = sorted(scenario_metadata['time_of_failure_h'].unique())
        threshold_col_options = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

        
        exclude = ['T', 'Node', 'leak_diameter_parameter', 'time_of_failure_h']
        scenario_cols = [col for col in df_signals.columns if col not in exclude]

        selected_leak_sig = st.selectbox("Leak Diameter", leak_options_sig, key="sig_leak_select")
        selected_time_sig = st.selectbox("Time of Failure (h)", time_options_sig, key="sig_time_select")

        scenarios_picked = scenario_metadata[(scenario_metadata['leak_diameter_parameter'] == selected_leak_sig) & (scenario_metadata['time_of_failure_h'] == selected_time_sig)]
        scenarios_picked = scenarios_picked['Scenario_Name'].tolist()
        scenario_node_dict = dict(zip(scenario_metadata['Scenario_Name'], scenario_metadata['leak_location']))

        if not view_mode:
            node_options = sorted(df_signals['Node'].unique())
            selected_node = st.selectbox("Node", node_options, key="sig_node_select")
            selected_thresh_param = st.selectbox("Threshold Parametr", threshold_col_options, key="sig_thresh_select")
            outlier_show = st.toggle("Pokaz scenariusze ktore\nnie pokrywaja sie z BP przed wyciekiem", value=False, key="toggle_view_mode_2")
        else:
            selected_scenario = st.selectbox("Wybierz Scenariusz", scenarios_picked, key="sig_scenario_select")

    with col_plots_sig:

        fig_sig = go.Figure()

        if not view_mode:

            scenarios_picked.insert(0, 'blueprint_scenario')
            scenario_node_dict['blueprint_scenario'] = 'blueprint_scenario'

            if not outlier_show:
                outlier_scenarios = scenario_metadata.loc[scenario_metadata.is_outlier==True].Scenario_Name.tolist()
                for i in range(len(scenarios_picked) - 1, -1, -1):
                    if scenarios_picked[i] in outlier_scenarios:
                        del scenarios_picked[i]

            filtered_signals = df_signals[['T', 'Node'] + scenarios_picked].sort_values('T')

            filtered_signals = filtered_signals[
                (filtered_signals['Node'] == selected_node) 
            ].sort_values('T')

            if not filtered_signals.empty:
                for col in scenarios_picked:
                    fig_sig.add_trace(go.Scatter(
                        x=filtered_signals['T']/3600, y=filtered_signals[col],
                        mode='lines', name=scenario_node_dict[col], line=dict(width=1.5)
                    ))
                try:
                    m_node = nodal_thresholds.loc[selected_node, 'mean']
                    s_node = nodal_thresholds.loc[selected_node, 'std'] * selected_thresh_param

                    fig_sig.add_hline(y=m_node+s_node, line_dash="dash", line_color="red", annotation_text="Th +")
                    fig_sig.add_hline(y=m_node-s_node, line_dash="dash", line_color="red", annotation_text="Th -")
                    fig_sig.add_hline(y=m_node, line_dash="dash", line_color="blue", annotation_text="Th -")

                except Exception as e:
                    print(f"Wystąpił błąd: {e}") 
        else:
            filtered_mode2 = df_signals[['T', 'Node'] + scenarios_picked].sort_values('T')
            filtered_mode2 = filtered_mode2[filtered_mode2['Node'].isin(wn.node_name_list)]

            if not filtered_mode2.empty:
                for node_name, group in filtered_mode2.groupby('Node'):
                    fig_sig.add_trace(go.Scatter(
                        x=group['T']/3600, y=group[selected_scenario],
                        mode='lines', name=f"Node: {node_name}", line=dict(width=1)
                    ))

        fig_sig.add_vline(
            x=selected_time_sig, line_dash="dot", line_color="green", 
            line_width=2, annotation_text=f"Awaria: {selected_time_sig}h", annotation_position="top left"
        )

        fig_sig.update_layout(
            height=600, template="plotly_white",
            xaxis_title="Czas [h]", yaxis_title="Wartość Sygnału",
            hovermode="closest", margin=dict(t=30, b=50, r=150),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )

        if (not view_mode and not filtered_signals.empty) or (view_mode and not filtered_mode2.empty):
            st.plotly_chart(fig_sig, use_container_width=True)
        else:
            st.warning("Brak danych dla wybranych parametrów.")

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

display_Chama(chama_outputs, sensors_wn_dict, wn, precision_recall_data)

display_XGBoost(confusion_matrix_best_nodes_df_XGB, confusion_matrix_df_XGB, wn)

display_LightGBM(confusion_matrix_best_nodes_df_LGB, confusion_matrix_df_LGB, wn)

display_NN(confusion_matrix_df, confusion_matrix_best_nodes_df, wn)
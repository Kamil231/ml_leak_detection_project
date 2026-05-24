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

pickle_path = '/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/code/leak_simulation/output_folder/pickle/'

with open(pickle_path + 'chama_outputs.pkl', 'rb') as file:
    chama_outputs = pickle.load(file)

with open(pickle_path + 'scenario_metadata.pkl', 'rb') as file:
    scenario_metadata = pickle.load(file)

with open(pickle_path + 'sensors_wn_dict.pkl', 'rb') as file:
    sensors_wn_dict = pickle.load(file)

with open(pickle_path + 'precision_recall_data_chama.pkl', 'rb') as file:
    precision_recall_data = pickle.load(file)

with open(pickle_path + 'confusion_matrix_df.pkl', 'rb') as file:
    confusion_matrix_df_XGB = pickle.load(file)

@st.cache_data 
def load_data_signals():  
    df = pd.read_pickle(pickle_path + 'signals_with_bp.pkl')   
    return df

@st.cache_data 
def load_simulation_results():

    wn_base = SIMULATION_CONFIG.create_network_base()
    wn_real = SIMULATION_CONFIG.create_network_real()
    
    sim_real = wntr.sim.WNTRSimulator(wn_real)
    results_real = sim_real.run_sim()
    
    sim_base = wntr.sim.WNTRSimulator(wn_base)
    results_base = sim_base.run_sim()

    node_name_list = wn_base.node_name_list

    return results_real, results_base, node_name_list

df_signals = load_data_signals()

results_real, results_base, node_name_list = load_simulation_results()

wn = SIMULATION_CONFIG.create_network_real()

st.set_page_config(layout="wide")

nodal_thresholds = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'nodal_thresholds_std.pkl')

with st.expander("Analiza przebiegów sygnałów", expanded=True):

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

with st.expander("Analiza parametrów węzłów", expanded=True):

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

with st.expander("Mapa sieci", expanded=True):

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

with st.expander("Optymalizacja Chama - Wyniki", expanded=True):

    with st.expander("Chama - CoverageFormulation & ImpactFormulation", expanded=True):
    
        budget_list = chama_outputs['Budget'].unique().tolist()

        budget_picked = st.selectbox("Sensor budget", budget_list)

        impact_row = chama_outputs[(chama_outputs['Budget'] == budget_picked) & (chama_outputs['Formulation'] == 'ImpactFormulation')]
        coverage_row = chama_outputs[(chama_outputs['Budget'] == budget_picked) & (chama_outputs['Formulation'] == 'CoverageFormulation')]
        impact_row_dict = impact_row['Result'].item()
        coverage_row_dict = coverage_row['Result'].item()

        impact_row_data = chama_outputs[
            (chama_outputs['Budget'] == budget_picked) &
            (chama_outputs['Formulation'] == 'ImpactFormulation')
        ]

        coverage_row_data = chama_outputs[
            (chama_outputs['Budget'] == budget_picked) &
            (chama_outputs['Formulation'] == 'CoverageFormulation')
        ]

        if not impact_row_data.empty and not coverage_row_data.empty:
            with st.expander("Impact Formulation", expanded=True):

                m1, m2, m3, m4 = st.columns(4)
                m1.metric('Objective: ', f"{impact_row_dict['Objective']:.4f}")
                m2.metric('FractionDetected: ', f"{impact_row_dict['FractionDetected']:.4f}")
                m3.metric('Solved: ', f"{impact_row_dict['Solved']:.4f}")
                m4.metric('TotalSensorCost: ', f"{impact_row_dict['TotalSensorCost']:.4f}")

                st.divider()

                st.markdown("Sensors")
                st.write(", ".join(map(str, impact_row_dict['Sensors'])))

                st.divider()

                fig, ax = plt.subplots(figsize=(6, 4))

                sensor_results_wn = []
                for sensor in impact_row_dict['Sensors']:
                    sensor_results_wn.append(sensors_wn_dict[sensor][0])

                node_colors = {name: 'red' if name in sensor_results_wn else 'lightgrey' for name in wn.node_name_list}
                wntr.graphics.plot_network(
                    wn, 
                    ax=ax, 
                    node_attribute=node_colors, 
                    node_size=10,               
                    add_colorbar=False,
                    title="Sensory zaznaczone na czerwono"
                )

                for node_name in impact_row_dict['Sensors']: 
                    node_name = sensors_wn_dict[node_name][0]
                    coord = wn.get_node(node_name).coordinates
                    ax.text(coord[0] + .3, coord[1] + .7, node_name, 
                        fontsize=6, 
                        fontweight='bold'
                        )
                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    st.pyplot(fig, use_container_width=False)

        if not coverage_row.empty and not coverage_row.empty:
            with st.expander("Coverage Formulation", expanded=True):

                m1, m2, m3, m4 = st.columns(4)
                m1.metric('Objective: ', f"{coverage_row_dict['Objective']:.4f}")
                m2.metric('FractionDetected: ', f"{coverage_row_dict['FractionDetected']:.4f}")
                m3.metric('Solved: ', f"{coverage_row_dict['Solved']:.4f}")
                m4.metric('TotalSensorCost: ', f"{coverage_row_dict['TotalSensorCost']:.4f}")

                st.divider()

                st.markdown("Sensors")
                st.write(", ".join(map(str, coverage_row_dict['Sensors'])))

                st.divider()

                fig, ax = plt.subplots(figsize=(6, 4))

                sensor_results_wn = []
                for sensor in coverage_row_dict['Sensors']:
                    sensor_results_wn.append(sensors_wn_dict[sensor][0])

                node_colors = {name: 'red' if name in sensor_results_wn else 'lightgrey' for name in wn.node_name_list}
                wntr.graphics.plot_network(
                    wn, 
                    ax=ax, 
                    node_attribute=node_colors, 
                    node_size=10,               
                    add_colorbar=False,
                    title="Sensory zaznaczone na czerwono"
                )

                for node_name in coverage_row_dict['Sensors']: 
                    node_name = sensors_wn_dict[node_name][0]
                    coord = wn.get_node(node_name).coordinates
                    ax.text(coord[0] + .3, coord[1] + .7, node_name, 
                        fontsize=6, 
                        fontweight='bold'
                        )
                col1, col2, col3 = st.columns([1, 2, 1])

                with col2:
                    st.pyplot(fig, use_container_width=False)

    with st.expander("Analiza wyników symulacji wycieków", expanded=True):

        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=("Impact Formulation", "Coverage Formulation"),
            shared_xaxes=True,
            specs=[[{"secondary_y": True}, {"secondary_y": True}]]
        )

        impact_objective_list = []
        coverage_objective_list = []
        impact_FractionDetected_list = []
        coverage_FractionDetected_list = []

        for budget in budget_list:
            impact_row = chama_outputs[(chama_outputs['Budget'] == budget) & (chama_outputs['Formulation'] == 'ImpactFormulation')]
            coverage_row = chama_outputs[(chama_outputs['Budget'] == budget) & (chama_outputs['Formulation'] == 'CoverageFormulation')]
            
            if not impact_row.empty:
                impact_res = impact_row['Result'].item()
                impact_objective_list.append(impact_res['Objective']/3600)
                impact_FractionDetected_list.append(impact_res['FractionDetected'])
            
            if not coverage_row.empty:
                coverage_res = coverage_row['Result'].item()
                coverage_objective_list.append(coverage_res['Objective'])
                coverage_FractionDetected_list.append(coverage_res['FractionDetected'])


        if len(impact_FractionDetected_list) > 0 and len(coverage_objective_list) > 0:
            
            fig.add_trace(
                go.Scatter(x=budget_list, y=impact_objective_list,
                           mode='lines+markers', name="Impact (Objective)", 
                           legend="legend", 
                           line=dict(color='red', width=3),
                           hovertemplate="<b>Objective (Time)</b><br>X: %{x}<br>Y: %{y:.2f}<extra></extra>"),
                row=1, col=1, secondary_y=False
            )

            fig.add_trace(
                go.Scatter(x=budget_list, y=impact_FractionDetected_list,
                           mode='lines+markers', name="Impact (Detected %)", 
                           legend="legend",  
                           line=dict(color='green', width=2, dash='dot'),
                           hovertemplate="<b>Fraction Detected</b><br>X: %{x}<br>Y: %{y:.2%}<extra></extra>"),
                row=1, col=1, secondary_y=True
            )

            fig.add_trace(
                go.Scatter(x=budget_list, y=coverage_objective_list,
                           mode='lines+markers', name="Coverage (Objective)", 
                           legend="legend2", 
                           line=dict(color='blue', width=3),
                           hovertemplate="<b>Objective</b><br>X: %{x}<br>Y: %{y:.2f}<extra></extra>"),
                row=1, col=2, secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=budget_list, y=coverage_FractionDetected_list,
                           mode='lines+markers', name="Coverage (Detected %)", 
                           legend="legend2",  
                           line=dict(color='orange', width=2, dash='dot'),
                           hovertemplate="<b>Fraction Detected</b><br>X: %{x}<br>Y: %{y:.2%}<extra></extra>"),
                row=1, col=2, secondary_y=True
            )

        fig.update_layout(
            height=600,
            template="plotly_white",
            margin=dict(t=50, b=100), 
            hovermode='closest',
            
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.15,  
                xanchor="center", x=0.22, 
                title=dict(text="") 
            ),

            legend2=dict(
                orientation="h",
                yanchor="top", y=-0.15,   
                xanchor="center", x=0.78, 
                title=dict(text="")
            )
        )

        fig.update_xaxes(title_text="Number of sensors")
        fig.update_yaxes(title_text="Impact Time (h)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Coverage Obj", row=1, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Fraction Detected", row=1, col=1, secondary_y=True, showgrid=False)
        fig.update_yaxes(title_text="Fraction Detected", row=1, col=2, secondary_y=True, showgrid=False)
        
        fig.update_yaxes(range=[0, 1.1], secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Precision Recall", expanded=True):

        budget_list = chama_outputs['Budget'].unique().tolist()
        default_budget = budget_list[len(budget_list) // 2]
        
        col_params, col_plots = st.columns([1, 4], vertical_alignment="top")

        with col_params:
            st.markdown("### Filtry")

            selected_budget = st.select_slider(
                "Budget",
                options=budget_list,
                value=default_budget,
                key="pr_budget"
            )
            selected_form = st.selectbox("Formulation", precision_recall_data['formulation'].unique())

            st.markdown("**Wybierz Threshold (THP):**")

            unique_thp = sorted(precision_recall_data['thp'].dropna().unique())
            df_thp = pd.DataFrame({'Thp': unique_thp})

            selection_event = st.dataframe(
                df_thp,
                hide_index=True,           
                on_select="rerun",         
                selection_mode="multi-row",
                height=250                 
            )

            selected_indices = selection_event.selection.rows

            if not selected_indices:
                selected_thp = unique_thp
                st.info("Brak zaznaczenia: Wyświetlam wszystkie progi.")
            else:
                selected_thp = df_thp.iloc[selected_indices]['Thp'].tolist()

        filtered_precision_recall_data = precision_recall_data[
            (precision_recall_data['budget'] == selected_budget) & 
            (precision_recall_data['formulation'] == selected_form) &
            (precision_recall_data['thp'].isin(selected_thp)) # Dodany warunek dla THP
        ]

        with col_plots:
            fig_pr = px.line(
                filtered_precision_recall_data, 
                x="recall", 
                y="precision", 
                color="leak_diameters",
                hover_data=["thp"],
                markers=True,
                title=f"Krzywa P-R dla Budżetu {selected_budget} ({selected_form})"
            )
            st.plotly_chart(fig_pr, use_container_width=True)

            fig_f1 = px.line(
                filtered_precision_recall_data,
                x="thp",
                y="f1_score",
                color="leak_diameters",
                title="Optymalizacja Progu (F1-Score vs Threshold)"
            )
            st.plotly_chart(fig_f1, use_container_width=True)

            print('filtered_precision_recall_data: \n', filtered_precision_recall_data)

    with st.expander("Confusion Matrix", expanded=True): 
        
        col_params_cm, col_plots_cm = st.columns([1, 4], vertical_alignment="top")

        budget_list = chama_outputs['Budget'].unique().tolist()
        default_budget = budget_list[len(budget_list) // 2]

        with col_params_cm:
            st.markdown("### Filtry Macierzy")

            selected_budget_cm = st.select_slider(
                "Budget",
                options=budget_list,
                value=default_budget,
                key="cm_budget"
            )

            selected_form_cm = st.selectbox("Formulation", precision_recall_data['formulation'].unique(), key="cm_form")
            
            unique_thp_cm = sorted(precision_recall_data['thp'].dropna().unique())
            selected_thp_cm = st.selectbox(
                "Wybierz Threshold (THP)", 
                unique_thp_cm, 
                index=len(unique_thp_cm)//2, 
                format_func=lambda x: f"{x:.2f}", 
                key="cm_thp"
            )

            unique_leaks = precision_recall_data['leak_diameters'].unique()
            selected_leak_cm = st.selectbox("Wybierz Analizowany Wyciek", unique_leaks, key="cm_leak")

        filtered_cm_data = precision_recall_data[
            (precision_recall_data['budget'] == selected_budget_cm) & 
            (precision_recall_data['formulation'] == selected_form_cm) &
            (precision_recall_data['thp'] == selected_thp_cm) &
            (precision_recall_data['leak_diameters'] == selected_leak_cm)
        ]

        with col_plots_cm:
            if filtered_cm_data.empty:
                st.warning("Brak danych do wyświetlenia dla wybranych parametrów.")
            else:
                row = filtered_cm_data.iloc[0]
                tp, fp = int(row['TP']), int(row['FP'])
                tn, fn = int(row['TN']), int(row['FN'])

                z_values = [[tn, fp], 
                            [fn, tp]]
                
                x_labels = ['Przewidywany Szum (-)', 'Przewidywany Wyciek (+)']
                y_labels = ['Faktyczny Szum (-)', 'Faktyczny Wyciek (+)']

                fig_cm = px.imshow(
                    z_values, 
                    x=x_labels, 
                    y=y_labels, 
                    text_auto=True, 
                    color_continuous_scale=[[0, 'white'], [1, 'white']], 
                    title=f"Macierz Pomyłek (Budżet: {selected_budget_cm}, THP: {selected_thp_cm:.2f}, Wyciek: {selected_leak_cm})"
                )
                
                fig_cm.update_layout(
                    xaxis_title="Decyzja Systemu (Algorytm)", 
                    yaxis_title="Stan Rzeczywisty (Fizyka Sieci)",
                    coloraxis_showscale=False 
                )

                st.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("ROC curve", expanded=True):

        precision_recall_data['TPR'] = precision_recall_data['TP'] / (precision_recall_data['TP'] + precision_recall_data['FN'] + 1e-9)
        precision_recall_data['FPR'] = precision_recall_data['FP'] / (precision_recall_data['FP'] + precision_recall_data['TN'] + 1e-9)

        st.set_page_config(layout="wide")

        col_filters, col_chart = st.columns([1, 3])

        with col_filters:
            st.markdown("### Filtry modelu")
            selected_formulation = st.selectbox("Formulation", options=sorted(precision_recall_data['formulation'].unique()))
            selected_budget = st.selectbox("Budget", options=sorted(precision_recall_data['budget'].unique()))

        filtered_precision_recall_data = precision_recall_data[
            (precision_recall_data['formulation'] == selected_formulation) &
            (precision_recall_data['budget'] == selected_budget)
        ].copy()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='#FF4B4B'),
            name='Losowy klasyfikator',
            showlegend=False
        ))

        auc_results = []

        if not filtered_precision_recall_data.empty:
            unique_leaks = sorted(
                filtered_precision_recall_data['leak_diameters'].astype(str).unique(),
                key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
            )
            
            for leak in unique_leaks:
                group = filtered_precision_recall_data[filtered_precision_recall_data['leak_diameters'].astype(str) == leak]
                group_sorted = group.sort_values(by='thp', ascending=False)
                
                fpr_points = group_sorted['FPR'].tolist()
                tpr_points = group_sorted['TPR'].tolist()

                roc_points = sorted(zip(fpr_points, tpr_points))
                
                if roc_points[0][0] != 0:
                    roc_points.insert(0, (0.0, 0.0))
                if roc_points[-1][0] != 1:
                    roc_points.append((1.0, 1.0))
                    
                x_auc, y_auc = zip(*roc_points)

                calculated_auc = auc(x_auc, y_auc)
                
                auc_results.append({
                    'Formulation': selected_formulation,
                    'Budget': selected_budget,
                    'Leak Diameter': leak,
                    'AUC Score': round(calculated_auc, 4)
                })

                fig.add_trace(go.Scatter(
                    x=group_sorted['FPR'],
                    y=group_sorted['TPR'],
                    mode='lines+markers',
                    name=f"{leak}",
                    marker=dict(size=6),
                    text=group_sorted['thp'],
                    hovertemplate=(
                        f"<b>Średnica wycieku:</b> {leak}<br>" +
                        "<b>FPR:</b> %{x:.4f}<br>" +
                        "<b>TPR:</b> %{y:.4f}<br>" +
                        "<b>Próg (thp):</b> %{text}<extra></extra>"
                    )
                ))

        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            xaxis_title="False Positive Rate (FPR)",
            xaxis=dict(range=[-0.02, 1.05], gridcolor='#262730', zerolinecolor='#41444C'),
            yaxis_title="True Positive Rate (TPR)",
            yaxis=dict(range=[-0.02, 1.05], gridcolor='#262730', zerolinecolor='#41444C'),
            height=600,
            margin=dict(l=40, r=20, t=40, b=40),
            showlegend=True,
            legend=dict(
                title=dict(text="Leaks", font=dict(color='white')),
                font=dict(color='#A3A8B4', size=12),
                orientation="v",
                x=1.02, y=1,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(0,0,0,0)"
            ),
            legend_itemclick="toggle",
            legend_itemdoubleclick="toggleothers"
        )

        with col_chart:
            if not filtered_precision_recall_data.empty:
                st.plotly_chart(fig, use_container_width=True)
                
                auc_precision_recall_data = pd.DataFrame(auc_results)
                
                if not auc_precision_recall_data.empty:
                    auc_precision_recall_data = auc_precision_recall_data.sort_values(by='AUC Score', ascending=False)
                
                with st.expander("Tabela AUC (Area Under Curve)", expanded=True):
                    st.dataframe(
                        auc_precision_recall_data, 
                        use_container_width=True,
                        hide_index=True 
                    )
            else:
                st.info("Brak danych spełniających kryteria wybranych filtrów.")

with st.expander("Optymalizacja XGBoost - Wyniki", expanded=True):

    with st.expander("Model XGBoost dla wszystkich węzłów", expanded=True):

        confusion_matrix_df_XGB['TPR'] = confusion_matrix_df_XGB['TP'] / (confusion_matrix_df_XGB['TP'] + confusion_matrix_df_XGB['FN'] + 1e-9)          
        confusion_matrix_df_XGB['FPR'] = confusion_matrix_df_XGB['FP'] / (confusion_matrix_df_XGB['FP'] + confusion_matrix_df_XGB['TN'] + 1e-9)         
        confusion_matrix_df_XGB['Precision'] = confusion_matrix_df_XGB['TP'] / (confusion_matrix_df_XGB['TP'] + confusion_matrix_df_XGB['FP'] + 1e-9)    

        confusion_matrix_df_XGB['F1'] = 2 * (confusion_matrix_df_XGB['Precision'] * confusion_matrix_df_XGB['TPR']) / (confusion_matrix_df_XGB['Precision'] + confusion_matrix_df_XGB['TPR'] + 1e-9)

        st.set_page_config(layout="wide")

        unique_leaks = sorted(
            confusion_matrix_df_XGB['leak_diameter_parameter'].astype(str).unique(),
            key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
        )

        with st.expander("Krzywa ROC", expanded=True):

            fig_roc = go.Figure()
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                line=dict(dash='dash', color='#FF4B4B'), name='Losowy', showlegend=False
            ))

            auc_results = []
            
            for leak in unique_leaks:
                group = confusion_matrix_df_XGB[confusion_matrix_df_XGB['leak_diameter_parameter'].astype(str) == leak]
                group_sorted = group.sort_values(by='decision_threshold', ascending=True)

                fpr_points = group_sorted['FPR'].tolist()
                tpr_points = group_sorted['TPR'].tolist()

                roc_points = sorted(zip(fpr_points, tpr_points))

                if roc_points and roc_points[0][0] != 0:
                    roc_points.insert(0, (0.0, 0.0))
                if roc_points and roc_points[-1][0] != 1:
                    roc_points.append((1.0, 1.0))

                x_auc, y_auc = zip(*roc_points)
                calculated_auc = auc(x_auc, y_auc)
                auc_results.append({
                    'Średnica wycieku (Leak)': leak,
                    'AUC Score': round(calculated_auc, 4)
                })
                
                fig_roc.add_trace(go.Scatter(
                    x=group_sorted['FPR'],
                    y=group_sorted['TPR'],
                    mode='lines+markers',
                    name=f"{leak}",
                    marker=dict(size=6),
                    text=group_sorted['decision_threshold'],
                    hovertemplate=(
                        f"<b>Średnica wycieku: {leak}</b><br>" +
                        "FPR: %{x:.4f}<br>" +
                        "TPR (Recall): %{y:.4f}<br>" +
                        "Próg: %{text}<extra></extra>"
                    )
                ))
                
            fig_roc.update_layout(
                template='plotly_dark', plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                xaxis_title="False Positive Rate (FPR)", yaxis_title="True Positive Rate (TPR)",
                xaxis=dict(range=[-0.02, 1.05], gridcolor='#262730', zerolinecolor='#41444C'),
                yaxis=dict(range=[-0.02, 1.05], gridcolor='#262730', zerolinecolor='#41444C'),
                height=650, showlegend=True,
                legend=dict(
                    title=dict(text="Leaks"),
                    orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"
                ),
                legend_itemclick="toggle", legend_itemdoubleclick="toggleothers"
            )

            st.plotly_chart(fig_roc, use_container_width=True)

            auc_df = pd.DataFrame(auc_results)
            
            if not auc_df.empty:
                auc_df = auc_df.sort_values(by='AUC Score', ascending=False)
                
                with st.expander("Tabela AUC (Area Under Curve)", expanded=True):
                    st.dataframe(
                        auc_df, 
                        use_container_width=True,
                        hide_index=True 
                    )

        with st.expander("Krzywa Precision-Recall oraz F1 Score", expanded=True):

            st.markdown("### Krzywa Precision-Recall")

            fig_pr = go.Figure()
            
            for leak in unique_leaks:
                group = confusion_matrix_df_XGB[confusion_matrix_df_XGB['leak_diameter_parameter'].astype(str) == leak]
                group_sorted = group.sort_values(by='decision_threshold', ascending=True)
                
                fig_pr.add_trace(go.Scatter(
                    x=group_sorted['TPR'], # Recall na osi X
                    y=group_sorted['Precision'], # Precision na osi Y
                    mode='lines+markers',
                    name=f"{leak}",
                    marker=dict(size=6),
                    text=group_sorted['decision_threshold'],
                    hovertemplate=(
                        f"<b>Średnica wycieku: {leak}</b><br>" +
                        "Recall (Czułość): %{x:.4f}<br>" +
                        "Precision (Precyzja): %{y:.4f}<br>" +
                        "Próg: %{text}<extra></extra>"
                    )
                ))
                
            fig_pr.update_layout(
                template='plotly_dark', plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                xaxis_title="Recall (Czułość)", yaxis_title="Precision (Precyzja)",
                xaxis=dict(range=[-0.02, 1.05], gridcolor='#262730', zerolinecolor='#41444C'),
                yaxis=dict(range=[-0.02, 1.05], gridcolor='#262730', zerolinecolor='#41444C'),
                height=650, showlegend=True,
                legend=dict(
                    title=dict(text="Leaks"),
                    orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"
                ),
                legend_itemclick="toggle", legend_itemdoubleclick="toggleothers"
            )
            st.plotly_chart(fig_pr, use_container_width=True)


            st.markdown("### Wykres F1-Score vs Próg Odcięcia (Threshold)")
            fig_f1 = go.Figure()

            for leak in unique_leaks:
                group = confusion_matrix_df_XGB[confusion_matrix_df_XGB['leak_diameter_parameter'].astype(str) == leak]
                group_sorted = group.sort_values(by='decision_threshold', ascending=True)
                
                fig_f1.add_trace(go.Scatter(
                    x=group_sorted['decision_threshold'], # Threshold na osi X
                    y=group_sorted['F1'],                 # F1-score na osi Y
                    mode='lines+markers',
                    name=f"{leak}",
                    marker=dict(size=6),
                    text=group_sorted['decision_threshold'],
                    hovertemplate=(
                        f"<b>Średnica wycieku: {leak}</b><br>" +
                        "Próg (Threshold): %{x:.2f}<br>" +
                        "F1-Score: %{y:.4f}<extra></extra>"
                    )
                ))

            fig_f1.update_layout(
                template='plotly_dark', plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                xaxis_title="Próg decyzyjny (Decision Threshold)", yaxis_title="F1-Score",
                xaxis=dict(range=[-0.02, 1.05], gridcolor='#262730', zerolinecolor='#41444C', tickmode='linear', dtick=0.1),
                yaxis=dict(range=[-0.02, 1.05], gridcolor='#262730', zerolinecolor='#41444C'),
                height=500, showlegend=True,
                legend=dict(
                    title=dict(text="Leaks"),
                    orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"
                ),
                legend_itemclick="toggle", legend_itemdoubleclick="toggleothers"
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        with st.expander("Macierz Pomyłek (Confusion Matrix)", expanded=True): 

                col_params_cm, col_plots_cm = st.columns([1, 4], vertical_alignment="top")

                with col_params_cm:
                    st.markdown("### Filtry Macierzy")

                    unique_leaks_cm = sorted(
                        confusion_matrix_df_XGB['leak_diameter_parameter'].astype(str).unique(),
                        key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
                    )
                    selected_leak_cm = st.selectbox(
                        "Wybierz Analizowany Wyciek", 
                        options=unique_leaks_cm, 
                        key="xgb_cm_leak"
                    )
                    
                    unique_thp_cm = sorted(confusion_matrix_df_XGB['decision_threshold'].dropna().unique())
                    selected_thp_cm = st.selectbox(
                        "Wybierz Threshold (THP)", 
                        options=unique_thp_cm, 
                        index=len(unique_thp_cm)//2, 
                        format_func=lambda x: f"{x:.2f}", 
                        key="xgb_cm_thp"
                    )

                filtered_cm_data = confusion_matrix_df_XGB[
                    (confusion_matrix_df_XGB['leak_diameter_parameter'].astype(str) == selected_leak_cm) &
                    (confusion_matrix_df_XGB['decision_threshold'] == selected_thp_cm)
                ]

                with col_plots_cm:
                    if filtered_cm_data.empty:
                        st.warning("Brak danych do wyświetlenia dla wybranych parametrów.")
                    else:
                        # Pobranie wartości TP, FP, TN, FN
                        row = filtered_cm_data.iloc[0]
                        tp, fp = int(row['TP']), int(row['FP'])
                        tn, fn = int(row['TN']), int(row['FN'])

                        z_values = [[tn, fp], 
                                    [fn, tp]]
                        
                        x_labels = ['Przewidywany Szum (-)', 'Przewidywany Wyciek (+)']
                        y_labels = ['Faktyczny Szum (-)', 'Faktyczny Wyciek (+)']

                        fig_cm = px.imshow(
                            z_values, 
                            x=x_labels, 
                            y=y_labels, 
                            text_auto=True, 
                            color_continuous_scale=[[0, 'white'], [1, 'white']], 
                            title=f"Macierz Pomyłek (THP: {selected_thp_cm:.2f}, Wyciek: {selected_leak_cm})"
                        )
                        
                        fig_cm.update_traces(textfont=dict(color='black', size=14))

                        fig_cm.update_layout(
                            xaxis_title="Decyzja Systemu (Algorytm)", 
                            yaxis_title="Stan Rzeczywisty (Fizyka Sieci)",
                            coloraxis_showscale=False
                        )

                        st.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("XGBoost: wybor najlepszych wezlow i thresholdow", expanded=True):

        top_nodes_path = SIMULATION_CONFIG.output_folder / 'pickle' / 'top_nodes_xgb.pkl'

        with open(top_nodes_path, 'rb') as f:
            loaded_top_nodes = pickle.load(f)

        max_budget = len(loaded_top_nodes)

        budget = st.slider(
            "Wybierz liczbę najważniejszych węzłów do przeanalizowania:",
            min_value=1,
            max_value=max_budget,
            value=1,  
            step=1
        )

        selected_sensors = loaded_top_nodes[:budget]

        print('selected_sensors')
        print(selected_sensors)

        df_selected_sensors = pd.DataFrame(selected_sensors, columns=['Identyfikator Węzła', 'Optymalny Próg (ROC)'])

        df_selected_sensors.index = df_selected_sensors.index + 1
        df_selected_sensors.index.name = 'Importance No'

        st.dataframe(
            df_selected_sensors.style.format({'Optymalny Próg (ROC)': '{:.4f}'}),
            use_container_width=True
        )

        # wybrane_sensory = [node for node, thresh in wybrane_tuple]
import streamlit as st
from pathlib import Path
import pickle
import plotly.graph_objects as go

def display_signals(scenario_metadata, df_signals, wn, nodal_thresholds):

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

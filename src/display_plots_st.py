import streamlit as st
import pickle
import matplotlib.pyplot as plt
from src.config import SIMULATION_CONFIG
import wntr
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

#pickle_path = '/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/Code/Ex15/output_folder/pickle/'
pickle_path = '/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/code/leak_simulation/output_folder/pickle/'
#Śpickle_path = '/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/Code/Ex15/output_folder_big_data/pickle/'


with open(pickle_path + 'chama_outputs.pkl', 'rb') as file:
    chama_outputs = pickle.load(file)

with open(pickle_path + 'scenario_metadata.pkl', 'rb') as file:
    scenario_metadata = pickle.load(file)

with open(pickle_path + 'sensors_wn_dict.pkl', 'rb') as file:
    sensors_wn_dict = pickle.load(file)

with open(pickle_path + 'sensors_wn_dict.pkl', 'rb') as file:
    sensors_wn_dict = pickle.load(file)

@st.cache_data 
def load_data_signals():  
    df = pd.read_pickle(pickle_path + 'signals.pkl')   
    return df

@st.cache_data 
def load_data_thresholds():
    df = pd.read_pickle(pickle_path + 'nodal_thresholds.pkl')   
    return df

df_signals = load_data_signals()
df_thresholds = load_data_thresholds()

budget_list = chama_outputs['Budget'].unique().tolist()
formulation = chama_outputs['Formulation'].unique().tolist()

st.set_page_config(layout="wide")

wn = SIMULATION_CONFIG.create_network_real()

with st.container(border=True):

    st.markdown("<h2 style='text-align: center;'>Optymalizacja Chama - Wyniki</h2>", unsafe_allow_html=True)
    
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
                sensor_results_wn.append(sensors_wn_dict[sensor])

            #for name in wn.node_name_list:
                #print(name)
            #print(sensor_results_wn)

            node_colors = {name: 'red' if name in sensor_results_wn else 'lightgrey' for name in wn.node_name_list}
            wntr.graphics.plot_network(
                wn, 
                ax=ax, 
                node_attribute=node_colors, # TO ustawia kolor kropki
                node_size=10,               # Powiększone kropki sensorów
                add_colorbar=False,
                title="Sensory zaznaczone na czerwono"
            )

            for node_name in impact_row_dict['Sensors']: 
                node_name = sensors_wn_dict[node_name]
                coord = wn.get_node(node_name).coordinates
                ax.text(coord[0] + .3, coord[1] + .7, node_name, 
                    fontsize=6, 
                    fontweight='bold'
                    )
            col1, col2, col3 = st.columns([1, 2, 1])

            # 2. Wyświetlamy wykres tylko w środkowej kolumnie
            with col2:
                st.pyplot(fig, use_container_width=False)
            # st.pyplot(fig, use_container_width=False)

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
                sensor_results_wn.append(sensors_wn_dict[sensor])

            # for name in wn.node_name_list:
            #     print(name)
            # print(sensor_results_wn)

            node_colors = {name: 'red' if name in sensor_results_wn else 'lightgrey' for name in wn.node_name_list}
            wntr.graphics.plot_network(
                wn, 
                ax=ax, 
                node_attribute=node_colors, # TO ustawia kolor kropki
                node_size=10,               # Powiększone kropki sensorów
                add_colorbar=False,
                title="Sensory zaznaczone na czerwono"
            )

            for node_name in coverage_row_dict['Sensors']: 
                node_name = sensors_wn_dict[node_name]
                coord = wn.get_node(node_name).coordinates
                ax.text(coord[0] + .3, coord[1] + .7, node_name, 
                    fontsize=6, 
                    fontweight='bold'
                    )
            col1, col2, col3 = st.columns([1, 2, 1])

            # 2. Wyświetlamy wykres tylko w środkowej kolumnie
            with col2:
                st.pyplot(fig, use_container_width=False)
            # st.pyplot(fig, use_container_width=False)

with st.container(border=True):

    st.markdown(
        "<h2 style='text-align: center;'>Analiza wyników symulacji wycieków</h2>", 
        unsafe_allow_html=True
    )

    # Tworzymy subplots z dwiema osiami Y dla każdego wykresu
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Impact Formulation", "Coverage Formulation"),
        shared_xaxes=True,
        specs=[[{"secondary_y": True}, {"secondary_y": True}]]
    )

    budget_list = chama_outputs['Budget'].unique().tolist()

    impact_objective_list = []
    coverage_objective_list = []
    impact_FractionDetected_list = []
    coverage_FractionDetected_list = []

    # Pobieranie danych
    for budget in budget_list:
        impact_row = chama_outputs[(chama_outputs['Budget'] == budget) & (chama_outputs['Formulation'] == 'ImpactFormulation')]
        coverage_row = chama_outputs[(chama_outputs['Budget'] == budget) & (chama_outputs['Formulation'] == 'CoverageFormulation')]
        
        if not impact_row.empty:
            impact_res = impact_row['Result'].item()
            impact_objective_list.append(impact_res['Objective'])
            impact_FractionDetected_list.append(impact_res['FractionDetected'])
        
        if not coverage_row.empty:
            coverage_res = coverage_row['Result'].item()
            coverage_objective_list.append(coverage_res['Objective'])
            coverage_FractionDetected_list.append(coverage_res['FractionDetected'])


    if len(impact_FractionDetected_list) > 0 and len(coverage_objective_list) > 0:
        
        # --- LEWY WYKRES (Impact) ---
        # Przypisujemy do 'legend="legend"' (domyślna pierwsza legenda)
        
        fig.add_trace(
            go.Scatter(x=budget_list, y=impact_objective_list,
                       mode='lines+markers', name="Impact (Objective)", 
                       legend="legend",  # <--- PRZYPISANIE DO PIERWSZEJ LEGENDY
                       line=dict(color='red', width=3),
                       hovertemplate="<b>Objective (Time)</b><br>X: %{x}<br>Y: %{y:.2f}<extra></extra>"),
            row=1, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=budget_list, y=impact_FractionDetected_list,
                       mode='lines+markers', name="Impact (Detected %)", 
                       legend="legend",  # <--- PRZYPISANIE DO PIERWSZEJ LEGENDY
                       line=dict(color='green', width=2, dash='dot'),
                       hovertemplate="<b>Fraction Detected</b><br>X: %{x}<br>Y: %{y:.2%}<extra></extra>"),
            row=1, col=1, secondary_y=True
        )

        # --- PRAWY WYKRES (Coverage) ---
        # Przypisujemy do 'legend="legend2"' (nowa, druga legenda)

        fig.add_trace(
            go.Scatter(x=budget_list, y=coverage_objective_list,
                       mode='lines+markers', name="Coverage (Objective)", 
                       legend="legend2",  # <--- PRZYPISANIE DO DRUGIEJ LEGENDY
                       line=dict(color='blue', width=3),
                       hovertemplate="<b>Objective</b><br>X: %{x}<br>Y: %{y:.2f}<extra></extra>"),
            row=1, col=2, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=budget_list, y=coverage_FractionDetected_list,
                       mode='lines+markers', name="Coverage (Detected %)", 
                       legend="legend2",  # <--- PRZYPISANIE DO DRUGIEJ LEGENDY
                       line=dict(color='orange', width=2, dash='dot'),
                       hovertemplate="<b>Fraction Detected</b><br>X: %{x}<br>Y: %{y:.2%}<extra></extra>"),
            row=1, col=2, secondary_y=True
        )

    # --- KONFIGURACJA UKŁADU I LEGEND ---
    fig.update_layout(
        height=600,
        template="plotly_white",
        margin=dict(t=50, b=100), # Margines dolny na legendy
        hovermode='closest',
        
        # Konfiguracja PIERWSZEJ legendy (dla lewego wykresu)
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.15,  # Pozycja Y (pod wykresem)
            xanchor="center", x=0.22, # Pozycja X (wyśrodkowana pod lewym wykresem)
            title=dict(text="") # Opcjonalny tytuł
        ),
        
        # Konfiguracja DRUGIEJ legendy (dla prawego wykresu)
        legend2=dict(
            orientation="h",
            yanchor="top", y=-0.15,   # Ta sama wysokość co pierwsza
            xanchor="center", x=0.78, # Pozycja X (wyśrodkowana pod prawym wykresem)
            title=dict(text="")
        )
    )

    fig.update_xaxes(title_text="Number of sensors")

    # Opisy osi Y
    fig.update_yaxes(title_text="Impact Time (h)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Coverage Obj", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Fraction Detected", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text="Fraction Detected", row=1, col=2, secondary_y=True, showgrid=False)
    
    fig.update_yaxes(range=[0, 1.1], secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

with st.container(border=True):
    st.markdown(
        "<h2 style='text-align: center;'>Analiza przebiegów sygnałów (Scenariusze)</h2>", 
        unsafe_allow_html=True
    )
    st.write("") 

    col_params_sig, col_plots_sig = st.columns([1, 4], vertical_alignment="top")

    # --- KOLUMNA FILTRÓW ---
    with col_params_sig:
        st.markdown("### Filtry")
        
        # Toggle pod napisem "Filtry"
        view_mode = st.toggle("Widok: Scenariusz dla wszystkich węzłów", value=False, key="toggle_view_mode")
        st.divider()

        #leak_options_sig = sorted(df_signals['leak_diameter_parameter'].unique())
        leak_options_sig = sorted(scenario_metadata['leak_diameter_parameter'].unique())
        #time_options_sig = sorted(df_signals['time_of_failure_h'].unique())
        time_options_sig = sorted(scenario_metadata['time_of_failure_h'].unique())
        threshold_col_options = sorted(df_thresholds.columns)
        
        exclude = ['T', 'Node', 'leak_diameter_parameter', 'time_of_failure_h']
        scenario_cols = [col for col in df_signals.columns if col not in exclude]

        selected_leak_sig = st.selectbox("Leak Diameter", leak_options_sig, key="sig_leak_select")
        selected_time_sig = st.selectbox("Time of Failure (h)", time_options_sig, key="sig_time_select")
        selected_thresh_param = st.selectbox("Parametr Threshold", threshold_col_options, key="sig_thresh_select")

        scenarios_picked = scenario_metadata[(scenario_metadata['leak_diameter_parameter'] == selected_leak_sig) & (scenario_metadata['time_of_failure_h'] == selected_time_sig)]
        scenarios_picked = scenarios_picked['Scenario_Name'].tolist()
        #print('scenario_metadata: ', scenario_metadata)
        #scenarios_picked = ['T', 'Node'] + scenarios_picked

        scenario_node_dict = dict(zip(scenario_metadata['Scenario_Name'], scenario_metadata['leak_location']))

        if not view_mode:
            node_options = sorted(df_signals['Node'].unique())
            selected_node = st.selectbox("Wybierz Węzeł", node_options, key="sig_node_select")
        else:
            selected_scenario = st.selectbox("Wybierz Scenariusz", scenarios_picked, key="sig_scenario_select")

    with col_plots_sig:
        fig_sig = go.Figure()

        if not view_mode:

            filtered_signals = df_signals[['T', 'Node'] + scenarios_picked].sort_values('T')

            filtered_signals = filtered_signals[
                (filtered_signals['Node'] == selected_node) 
            ].sort_values('T')

            if not filtered_signals.empty:
                for col in scenarios_picked:
                    fig_sig.add_trace(go.Scatter(
                        x=filtered_signals['T'], y=filtered_signals[col],
                        mode='lines', name=scenario_node_dict[col], line=dict(width=1.5)
                    ))

                try:
                    val = df_thresholds.loc[selected_node, selected_thresh_param]
                    fig_sig.add_hline(y=val, line_dash="dash", line_color="red", annotation_text="Th +")
                    fig_sig.add_hline(y=-val, line_dash="dash", line_color="red", annotation_text="Th -")
                except: pass
        else:
            # TRYB 2: Jeden scenariusz -> wiele węzłów

            #filtered_mode2 = df_signals[['T', 'Node'] + scenarios_picked].sort_values('T')
            filtered_mode2 = df_signals[['T', 'Node'] + scenarios_picked].sort_values('T')
            filtered_mode2 = filtered_mode2[filtered_mode2['Node'].isin(wn.node_name_list)]

            if not filtered_mode2.empty:
                for node_name, group in filtered_mode2.groupby('Node'):
                    fig_sig.add_trace(go.Scatter(
                        x=group['T'], y=group[selected_scenario],
                        mode='lines', name=f"Node: {node_name}", line=dict(width=1)
                    ))

        fig_sig.update_layout(
            height=600, template="plotly_white",
            xaxis_title="Czas [T]", yaxis_title="Wartość Sygnału",
            hovermode="closest", margin=dict(t=30, b=50, r=150),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )

        # Wyświetlanie
        if (not view_mode and not filtered_signals.empty) or (view_mode and not filtered_mode2.empty):
            st.plotly_chart(fig_sig, use_container_width=True)
        else:
            st.warning("Brak danych dla wybranych parametrów.")
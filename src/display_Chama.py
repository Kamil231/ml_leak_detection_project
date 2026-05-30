from pathlib import Path
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from src.config import SIMULATION_CONFIG
import wntr
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from sklearn.metrics import auc

def display_Chama(chama_outputs, sensors_wn_dict, wn, precision_recall_data):

	with st.expander("Optymalizacja Chama - Wyniki"):

	    with st.expander("Chama - CoverageFormulation & ImpactFormulation"):
	    
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
	            with st.expander("Impact Formulation"):

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
	            with st.expander("Coverage Formulation"):

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

	    with st.expander("Analiza wyników symulacji wycieków"):

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

	    with st.expander("Precision Recall"):

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

	            # print('filtered_precision_recall_data: \n', filtered_precision_recall_data)

	    with st.expander("Confusion Matrix"): 
	        
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

	    with st.expander("ROC curve"):

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
	                
	                with st.expander("Tabela AUC (Area Under Curve)"):
	                    st.dataframe(
	                        auc_precision_recall_data, 
	                        use_container_width=True,
	                        hide_index=True 
	                    )
	            else:
	                st.info("Brak danych spełniających kryteria wybranych filtrów.")


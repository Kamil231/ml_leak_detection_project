import streamlit as st
import pickle
from src.config import SIMULATION_CONFIG
import wntr
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics import auc

def dipslay_roc_curve(unique_leaks, cm, description):

	with st.expander("Krzywa ROC"):

		fig_roc = go.Figure()
		        
		fig_roc.add_trace(go.Scatter(
		    x=[0, 1], y=[0, 1], mode='lines',
		    line=dict(dash='dash', color='#FF4B4B'), name='Losowy', showlegend=False
		))

		auc_results = []

		for leak in unique_leaks:
		    group = cm[cm['leak_diameter_parameter'].astype(str) == leak]
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

		st.plotly_chart(fig_roc, use_container_width=True, key=description+"roc")

		auc_df = pd.DataFrame(auc_results)

		if not auc_df.empty:
		    auc_df = auc_df.sort_values(by='AUC Score', ascending=False)
		    
		    with st.expander("Tabela AUC (Area Under Curve)"):
		        st.dataframe(
		            auc_df, 
		            use_container_width=True,
		            hide_index=True 
		        )

def display_precision_recall_f1(unique_leaks, cm):
	
	with st.expander("Krzywa Precision-Recall oraz F1 Score"):

		st.markdown("### Krzywa Precision-Recall")

		fig_pr = go.Figure()

		for leak in unique_leaks:
		    group = cm[cm['leak_diameter_parameter'].astype(str) == leak]
		    group_sorted = group.sort_values(by='decision_threshold', ascending=True)
		    
		    fig_pr.add_trace(go.Scatter(
		        x=group_sorted['TPR'], 
		        y=group_sorted['Precision'], 
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
		    group = cm[cm['leak_diameter_parameter'].astype(str) == leak]
		    group_sorted = group.sort_values(by='decision_threshold', ascending=True)
		    
		    fig_f1.add_trace(go.Scatter(
		        x=group_sorted['decision_threshold'], 
		        y=group_sorted['F1'],                 
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

def display_cm(cm, description):
	with st.expander("Macierz Pomyłek (Confusion Matrix)"): 

		col_params_cm, col_plots_cm = st.columns([1, 4], vertical_alignment="top")

		with col_params_cm:
		    st.markdown("### Filtry Macierzy")

		    unique_leaks_cm = sorted(
		        cm['leak_diameter_parameter'].astype(str).unique(),
		        key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
		    )
		    selected_leak_cm = st.selectbox(
		        "Wybierz Analizowany Wyciek", 
		        options=unique_leaks_cm, 
		        key=description+"_cm_leak"
		    )
		    
		    unique_thp_cm = sorted(cm['decision_threshold'].dropna().unique())
		    selected_thp_cm = st.selectbox(
		        "Wybierz Threshold (THP)", 
		        options=unique_thp_cm, 
		        index=len(unique_thp_cm)//2, 
		        format_func=lambda x: f"{x:.2f}", 
		        key=description+"_cm_thp"
		    )

		filtered_cm_data = cm[
		    (cm['leak_diameter_parameter'].astype(str) == selected_leak_cm) &
		    (cm['decision_threshold'] == selected_thp_cm)
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

def display_results_table(best_nodes, unique_leaks_dia, description, budget):

	with st.expander("Tabea wynikow"):

		selected_leak_dia = st.selectbox("Wybierz Średnicę Wycieku", options=unique_leaks_dia,
			    key=description+"_leak_dia_table")

		if 'budget' in best_nodes.columns:
			df_selected_sensors = best_nodes.loc[(best_nodes['leak_diameter_parameter'] == selected_leak_dia) & (best_nodes['budget'] == budget)]
			df_selected_sensors = df_selected_sensors.sort_values(by="Importance", ascending=False)
			df_selected_sensors = df_selected_sensors.reset_index(drop=True)

			df_selected_sensors['Importance No'] = range(1, len(df_selected_sensors) + 1)
			df_selected_sensors = df_selected_sensors.set_index('Importance No')
		else:
			df_selected_sensors = best_nodes.loc[(best_nodes['leak_diameter_parameter'] == selected_leak_dia)]
			df_selected_sensors = df_selected_sensors.sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)
			df_selected_sensors = df_selected_sensors.head(budget)
			df_selected_sensors = df_selected_sensors.set_index('Importance_Mean')

		st.dataframe(
		    df_selected_sensors.style.format({'Optymalny Próg (ROC)': '{:.4f}'}),
		    use_container_width=True
		)

def display_sensors_map(best_nodes, unique_leaks_dia, description, budget, wn):
	
	with st.expander("Mapa sensorów"): 

		selected_leak_dia = st.selectbox("Wybierz Średnicę Wycieku", options=unique_leaks_dia,
			    key=description+"_leak_dia_map")

		fig, ax = plt.subplots(figsize=(6, 4))


		if 'budget' in best_nodes.columns:
			df_selected_sensors = best_nodes.loc[(best_nodes['leak_diameter_parameter'] == selected_leak_dia) & (best_nodes['budget'] == budget)]
			sensor_results_wn = df_selected_sensors['Nodes'].unique().tolist()
		else:
			df_selected_sensors = best_nodes.loc[(best_nodes['leak_diameter_parameter'] == selected_leak_dia)]
			df_selected_sensors = df_selected_sensors.sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)
			df_selected_sensors = df_selected_sensors.head(budget)
			df_selected_sensors = df_selected_sensors.set_index('Importance_Mean')
			sensor_results_wn = df_selected_sensors['Node'].unique().tolist()

		# df_selected_sensors = best_nodes.loc[(best_nodes['leak_diameter_parameter'] == selected_leak_dia) & (best_nodes['budget'] == budget)]
		# sensor_results_wn = df_selected_sensors['Nodes'].unique().tolist()

		node_colors = {name: 'red' if name in sensor_results_wn else 'lightgrey' for name in wn.node_name_list}
		wntr.graphics.plot_network(
		    wn, 
		    ax=ax, 
		    node_attribute=node_colors, 
		    node_size=10,               
		    add_colorbar=False,
		    title="Sensory zaznaczone na czerwono"
		)

		for node_name in sensor_results_wn: 
		    coord = wn.get_node(node_name).coordinates
		    ax.text(coord[0] + .3, coord[1] + .7, node_name, 
		        fontsize=6, 
		        fontweight='bold'
		        )
		col1, col2, col3 = st.columns([1, 2, 1])

		with col2:
		    st.pyplot(fig, use_container_width=False)

def get_precision_recall_data(cm):

	cm['TPR'] = cm['TP'] / (cm['TP'] + cm['FN'] + 1e-9)          
	cm['FPR'] = cm['FP'] / (cm['FP'] + cm['TN'] + 1e-9)         
	cm['Precision'] = cm['TP'] / (cm['TP'] + cm['FP'] + 1e-9)    

	cm['F1'] = 2 * (cm['Precision'] * cm['TPR']) / (cm['Precision'] + cm['TPR'] + 1e-9)

	unique_leaks = sorted(
	    cm['leak_diameter_parameter'].astype(str).unique(),
	    key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
	)

	return cm, unique_leaks

def display_ml_results(cm_all_nodes, cm_best_nodes, wn, description, best_nodes, cm_global=None, best_nodes_global=None):

	with st.expander(f"Optymalizacja {description} - Wyniki"):

		# with st.expander("Model dla wszystkich węzłów"):

		#     cm_all_nodes['TPR'] = cm_all_nodes['TP'] / (cm_all_nodes['TP'] + cm_all_nodes['FN'] + 1e-9)          
		#     cm_all_nodes['FPR'] = cm_all_nodes['FP'] / (cm_all_nodes['FP'] + cm_all_nodes['TN'] + 1e-9)         
		#     cm_all_nodes['Precision'] = cm_all_nodes['TP'] / (cm_all_nodes['TP'] + cm_all_nodes['FP'] + 1e-9)    

		#     cm_all_nodes['F1'] = 2 * (cm_all_nodes['Precision'] * cm_all_nodes['TPR']) / (cm_all_nodes['Precision'] + cm_all_nodes['TPR'] + 1e-9)

		#     unique_leaks = sorted(
		#         cm_all_nodes['leak_diameter_parameter'].astype(str).unique(),
		#         key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
		#     )

		#     # cm_all_nodes, unique_leaks = get_precision_recall_data(cm_all_nodes)

		#     st.set_page_config(layout="wide")

		#     display_precision_recall_f1(unique_leaks, cm_all_nodes)

		#     dipslay_roc_curve(unique_leaks, cm_all_nodes, description+'_all')

		#     display_cm(cm_all_nodes, description+'_all')

		with st.expander("Osobna optymalizacja dla każdej wielkości wycieku"):

			top_nodes_path = SIMULATION_CONFIG.output_folder / 'pickle' / 'best_nodes_xgb.pkl'

			# max_budget = best_nodes['budget'].max()
			# min_budget = best_nodes['budget'].min()
			max_budget = cm_best_nodes['budget'].max()
			min_budget = cm_best_nodes['budget'].min()

			budget = st.slider(
			    "Wybierz liczbę najważniejszych węzłów do przeanalizowania:",
			    min_value=min_budget,
			    max_value=max_budget,
			    value=1,  
			    step=1,
			    key=description+"_best_nodes_number_leak"
			)

			cm_best_nodes_budget = cm_best_nodes.loc[cm_best_nodes['budget'] == budget]

			cm_best_nodes_budget['TPR'] = cm_best_nodes_budget['TP'] / (cm_best_nodes_budget['TP'] + cm_best_nodes_budget['FN'] + 1e-9)          
			cm_best_nodes_budget['FPR'] = cm_best_nodes_budget['FP'] / (cm_best_nodes_budget['FP'] + cm_best_nodes_budget['TN'] + 1e-9)         
			cm_best_nodes_budget['Precision'] = cm_best_nodes_budget['TP'] / (cm_best_nodes_budget['TP'] + cm_best_nodes_budget['FP'] + 1e-9)    

			cm_best_nodes_budget['F1'] = 2 * (cm_best_nodes_budget['Precision'] * cm_best_nodes_budget['TPR']) / (cm_best_nodes_budget['Precision'] + cm_best_nodes_budget['TPR'] + 1e-9)

			unique_leaks = sorted(
				cm_best_nodes_budget['leak_diameter_parameter'].astype(str).unique(),
				key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
			)

			# cm_best_nodes_budget, unique_leaks = get_precision_recall_data(cm_best_nodes_budget)

			unique_leaks_dia = sorted(
				best_nodes['leak_diameter_parameter'].astype(str).unique(),
				key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
			)

			# selected_leak_dia = st.selectbox("Wybierz Średnicę Wycieku", options=unique_leaks_dia,
			#     key=description+"_leak_dia")

			best_nodes['leak_diameter_parameter'] = best_nodes['leak_diameter_parameter'].apply(
				lambda x: str(round(x, 2)) if isinstance(x, (float, int)) and pd.notna(x) else str(x)
			)

			display_results_table(best_nodes, unique_leaks_dia, description, budget)

			display_precision_recall_f1(unique_leaks, cm_best_nodes_budget)

			dipslay_roc_curve(unique_leaks, cm_best_nodes_budget, description+'_best')

			display_cm(cm_best_nodes_budget, description+'_best')

			display_sensors_map(best_nodes, unique_leaks_dia, description, budget, wn)


		if cm_global is not None and best_nodes_global is not None:
			with st.expander("Optymalizacja dla wszystkich rozmiarów wycieków"):
				
				max_budget_glob = cm_global['budget'].max()
				min_budget_glob = cm_global['budget'].min()

				budget_glob = st.slider(
					"Wybierz budżet (liczbę czujników) dla modelu globalnego:",
					min_value=min_budget_glob,
					max_value=max_budget_glob,
					value=1,  
					step=1,
					key=description+"_global_budget"
				)

				cm_global_budget = cm_global.loc[cm_global['budget'] == budget_glob].copy()

				cm_global_budget['TPR'] = cm_global_budget['TP'] / (cm_global_budget['TP'] + cm_global_budget['FN'] + 1e-9)          
				cm_global_budget['FPR'] = cm_global_budget['FP'] / (cm_global_budget['FP'] + cm_global_budget['TN'] + 1e-9)         
				cm_global_budget['Precision'] = cm_global_budget['TP'] / (cm_global_budget['TP'] + cm_global_budget['FP'] + 1e-9)    
				cm_global_budget['F1'] = 2 * (cm_global_budget['Precision'] * cm_global_budget['TPR']) / (cm_global_budget['Precision'] + cm_global_budget['TPR'] + 1e-9)

				unique_leaks_glob = sorted(
					cm_global_budget['leak_diameter_parameter'].astype(str).unique(),
					key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
				)

				display_results_table(best_nodes_global, ['Global'], description + '_global_table', budget_glob) 
				display_precision_recall_f1(unique_leaks_glob, cm_global_budget)
				dipslay_roc_curve(unique_leaks_glob, cm_global_budget, description+'_global')
				display_cm(cm_global_budget, description+'_global')
				display_sensors_map(best_nodes_global, ['Global'], description + '_global_map', budget_glob, wn)
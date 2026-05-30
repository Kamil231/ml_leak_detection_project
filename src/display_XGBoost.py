import streamlit as st
import pickle
import matplotlib.pyplot as plt
from src.config import SIMULATION_CONFIG
import wntr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics import auc

def display_XGBoost(confusion_matrix_best_nodes_df_XGB, confusion_matrix_df_XGB, wn):
    with st.expander("Optymalizacja XGBoost - Wyniki"):

        with st.expander("Model XGBoost dla wszystkich węzłów"):

            confusion_matrix_df_XGB['TPR'] = confusion_matrix_df_XGB['TP'] / (confusion_matrix_df_XGB['TP'] + confusion_matrix_df_XGB['FN'] + 1e-9)          
            confusion_matrix_df_XGB['FPR'] = confusion_matrix_df_XGB['FP'] / (confusion_matrix_df_XGB['FP'] + confusion_matrix_df_XGB['TN'] + 1e-9)         
            confusion_matrix_df_XGB['Precision'] = confusion_matrix_df_XGB['TP'] / (confusion_matrix_df_XGB['TP'] + confusion_matrix_df_XGB['FP'] + 1e-9)    

            confusion_matrix_df_XGB['F1'] = 2 * (confusion_matrix_df_XGB['Precision'] * confusion_matrix_df_XGB['TPR']) / (confusion_matrix_df_XGB['Precision'] + confusion_matrix_df_XGB['TPR'] + 1e-9)

            st.set_page_config(layout="wide")

            unique_leaks = sorted(
                confusion_matrix_df_XGB['leak_diameter_parameter'].astype(str).unique(),
                key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
            )

            with st.expander("Krzywa ROC"):

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

                st.plotly_chart(fig_roc, use_container_width=True, key="xgb_roc")

                auc_df = pd.DataFrame(auc_results)
                
                if not auc_df.empty:
                    auc_df = auc_df.sort_values(by='AUC Score', ascending=False)
                    
                    with st.expander("Tabela AUC (Area Under Curve)"):
                        st.dataframe(
                            auc_df, 
                            use_container_width=True,
                            hide_index=True 
                        )

            with st.expander("Krzywa Precision-Recall oraz F1 Score"):

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

            with st.expander("Macierz Pomyłek (Confusion Matrix)"): 

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

        with st.expander("XGBoost: wybor najlepszych wezlow i thresholdow"):

            top_nodes_path = SIMULATION_CONFIG.output_folder / 'pickle' / 'top_nodes_xgb.pkl'

            with open(top_nodes_path, 'rb') as f:
                loaded_top_nodes = pickle.load(f)

            max_budget = loaded_top_nodes['budget'].max()

            budget = st.slider(
                "Wybierz liczbę najważniejszych węzłów do przeanalizowania:",
                min_value=1,
                max_value=max_budget,
                value=1,  
                step=1
            )

            confusion_matrix_best_nodes_df_XGB_local = confusion_matrix_best_nodes_df_XGB.loc[confusion_matrix_best_nodes_df_XGB['budget'] == budget]

            confusion_matrix_best_nodes_df_XGB_local['TPR'] = confusion_matrix_best_nodes_df_XGB_local['TP'] / (confusion_matrix_best_nodes_df_XGB_local['TP'] + confusion_matrix_best_nodes_df_XGB_local['FN'] + 1e-9)          
            confusion_matrix_best_nodes_df_XGB_local['FPR'] = confusion_matrix_best_nodes_df_XGB_local['FP'] / (confusion_matrix_best_nodes_df_XGB_local['FP'] + confusion_matrix_best_nodes_df_XGB_local['TN'] + 1e-9)         
            confusion_matrix_best_nodes_df_XGB_local['Precision'] = confusion_matrix_best_nodes_df_XGB_local['TP'] / (confusion_matrix_best_nodes_df_XGB_local['TP'] + confusion_matrix_best_nodes_df_XGB_local['FP'] + 1e-9)    

            confusion_matrix_best_nodes_df_XGB_local['F1'] = 2 * (confusion_matrix_best_nodes_df_XGB_local['Precision'] * confusion_matrix_best_nodes_df_XGB_local['TPR']) / (confusion_matrix_best_nodes_df_XGB_local['Precision'] + confusion_matrix_best_nodes_df_XGB_local['TPR'] + 1e-9)

            unique_leaks = sorted(
                confusion_matrix_best_nodes_df_XGB_local['leak_diameter_parameter'].astype(str).unique(),
                key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
            )

            with st.expander("Tabea wynikow"):

                unique_leaks_dia = sorted(
                    loaded_top_nodes['leak_diameter_parameter'].astype(str).unique(),
                    key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
                    )

                selected_leak_dia = st.selectbox("Wybierz Średnicę Wycieku", options=unique_leaks_dia)

                loaded_top_nodes['leak_diameter_parameter'] = loaded_top_nodes['leak_diameter_parameter'].apply(
                    lambda x: str(round(x, 2)) if isinstance(x, (float, int)) and pd.notna(x) else str(x)
                    )

                df_selected_sensors = loaded_top_nodes.loc[(loaded_top_nodes['leak_diameter_parameter'] == selected_leak_dia) & (loaded_top_nodes['budget'] == budget)]


                df_selected_sensors = df_selected_sensors.sort_values(by="Importance", ascending=False)
                df_selected_sensors = df_selected_sensors.reset_index(drop=True)

                df_selected_sensors['Importance No'] = range(1, len(df_selected_sensors) + 1)
                df_selected_sensors = df_selected_sensors.set_index('Importance No')

                st.dataframe(
                    df_selected_sensors.style.format({'Optymalny Próg (ROC)': '{:.4f}'}),
                    use_container_width=True
                )

            with st.expander("Krzywa ROC"):

                fig_roc = go.Figure()
                
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines',
                    line=dict(dash='dash', color='#FF4B4B'), name='Losowy', showlegend=False
                ))

                auc_results = []
                
                for leak in unique_leaks:
                    group = confusion_matrix_best_nodes_df_XGB_local[confusion_matrix_best_nodes_df_XGB_local['leak_diameter_parameter'].astype(str) == leak]
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

                # st.plotly_chart(fig_roc, use_container_width=True)
                st.plotly_chart(fig_roc, use_container_width=True, key=f"roc_chart_1")

                auc_df = pd.DataFrame(auc_results)
                
                if not auc_df.empty:
                    auc_df = auc_df.sort_values(by='AUC Score', ascending=False)
                    
                    with st.expander("Tabela AUC (Area Under Curve)"):
                        st.dataframe(
                            auc_df, 
                            use_container_width=True,
                            hide_index=True 
                        )

            with st.expander("Krzywa Precision-Recall oraz F1 Score"):

                st.markdown("### Krzywa Precision-Recall")

                fig_pr = go.Figure()
                
                for leak in unique_leaks:
                    group = confusion_matrix_best_nodes_df_XGB_local[confusion_matrix_best_nodes_df_XGB_local['leak_diameter_parameter'].astype(str) == leak]
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
                    group = confusion_matrix_best_nodes_df_XGB_local[confusion_matrix_best_nodes_df_XGB_local['leak_diameter_parameter'].astype(str) == leak]
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

            with st.expander("Macierz Pomyłek (Confusion Matrix)"): 

                    col_params_cm, col_plots_cm = st.columns([1, 4], vertical_alignment="top")

                    with col_params_cm:
                        st.markdown("### Filtry Macierzy")

                        unique_leaks_cm = sorted(
                            confusion_matrix_best_nodes_df_XGB_local['leak_diameter_parameter'].astype(str).unique(),
                            key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf')
                        )
                        selected_leak_cm = st.selectbox(
                            "Wybierz Analizowany Wyciek", 
                            options=unique_leaks_cm, 
                            key="xgb_cm_leak_bn"
                        )
                        
                        unique_thp_cm = sorted(confusion_matrix_best_nodes_df_XGB_local['decision_threshold'].dropna().unique())
                        selected_thp_cm = st.selectbox(
                            "Wybierz Threshold (THP)", 
                            options=unique_thp_cm, 
                            index=len(unique_thp_cm)//2, 
                            format_func=lambda x: f"{x:.2f}", 
                            key="xgb_cm_thp_bn"
                        )

                    filtered_cm_data = confusion_matrix_best_nodes_df_XGB_local[
                        (confusion_matrix_best_nodes_df_XGB_local['leak_diameter_parameter'].astype(str) == selected_leak_cm) &
                        (confusion_matrix_best_nodes_df_XGB_local['decision_threshold'] == selected_thp_cm)
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

                            st.plotly_chart(fig_cm, use_container_width=True, key="unique_cm_bn")

            with st.expander("Mapa sensorów"): 

                fig, ax = plt.subplots(figsize=(6, 4))

                sensor_results_wn = df_selected_sensors['Nodes'].unique().tolist()

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
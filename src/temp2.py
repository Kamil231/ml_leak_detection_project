import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc

pickle_path = '/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/code/leak_simulation/output_folder/pickle/'

with open(pickle_path + 'precision_recall_data_chama.pkl', 'rb') as file:
    precision_recall_data = pickle.load(file)



precision_recall_data['TPR'] = precision_recall_data['TP'] / (precision_recall_data['TP'] + precision_recall_data['FN'] + 1e-9)
precision_recall_data['FPR'] = precision_recall_data['FP'] / (precision_recall_data['FP'] + precision_recall_data['TN'] + 1e-9)

st.set_page_config(layout="wide")
st.title("Wizualizacja Krzywych ROC — Analiza Średnic Wycieków")

col_chart, col_filters = st.columns([3, 1])

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
            name=f"Średnica: {leak}",
            marker=dict(size=6),
            text=group_sorted['thp'],
            hovertemplate=(
                f"<b>Średnica (Leak):</b> {leak}<br>" +
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
        title=dict(text="<b>Średnice wycieku (Leak)</b><br><i>(Kliknij=ukryj | Dwuklik=izoluj)</i>", font=dict(color='white')),
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
        
        st.markdown("### Podsumowanie wskaźników AUC (Area Under Curve)")
        auc_precision_recall_data = pd.DataFrame(auc_results)
        
        if not auc_precision_recall_data.empty:
            auc_precision_recall_data = auc_precision_recall_data.sort_values(by='AUC Score', ascending=False)
        
        st.dataframe(
            auc_precision_recall_data, 
            use_container_width=True,
            hide_index=True 
        )
    else:
        st.info("Brak danych spełniających kryteria wybranych filtrów.")
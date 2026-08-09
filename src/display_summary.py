import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def display_summary_plot(df_original, description):

    df = df_original.copy()
    
    if 'budget' not in df.columns:
        df = df.reset_index()

    models = []
    if isinstance(df.columns, pd.MultiIndex):
        models = list(set([col[1] for col in df.columns if col[1] != '']))
        new_cols = []
        for col in df.columns:
            if col[1]:
                new_cols.append(f"{col[1]}_{col[0]}")
            else:
                new_cols.append(col[0])
        df.columns = new_cols
    else:
        st.error("Błąd odczytu: Brak struktury MultiIndex. Odśwież dane z plików .pkl.")
        return

    col_select, col_plot = st.columns([1, 4], vertical_alignment="top")
    
    with col_select:
        
        use_budget_on_x = st.toggle(
            "Budżet na osi X", 
            value=False,
            key=f"{description}_xaxis_toggle",
        )

        fixed_xaxis_budget = False
        if use_budget_on_x:
            fixed_xaxis_budget = st.toggle(
                "Zakres osi X: 0 - 20",
                value=False,
                key=f"{description}_fixed_x_toggle",
            )
        
        x_axis_mode = "Budżet (Sensory)" if use_budget_on_x else "Wielkość wycieku"
        
        if x_axis_mode == "Wielkość wycieku":
            budgets = sorted(df['budget'].unique())
            selected_filter = st.selectbox(
                "Wykresy dla budżetu równego:", 
                options=budgets, 
                key=f"{description}_plot_filter_budget"
            )
            
            df_filtered = df[df['budget'] == selected_filter].copy()
            df_filtered['leak_num'] = pd.to_numeric(df_filtered['leak_diameter_parameter'], errors='coerce')
            df_filtered = df_filtered.sort_values('leak_num')
            
            x_values = df_filtered['leak_diameter_parameter'].astype(str).tolist()
            x_title = "Rozmiar awarii (Leak Diameter)"
            plot_title = f"Skuteczność detekcji dla budżetu = {selected_filter}"
            
        else:
            leaks = sorted(df['leak_diameter_parameter'].unique(), key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf'))
            selected_filter = st.selectbox(
                "Wielkość wycieku:", 
                options=leaks, 
                key=f"{description}_plot_filter_leak"
            )
            
            df_filtered = df[df['leak_diameter_parameter'] == selected_filter].copy()
            df_filtered = df_filtered.sort_values('budget', ascending=True) 
            
            x_values = df_filtered['budget'].tolist()
            x_title = "Liczba sensorów (Budżet)"
            plot_title = f"Skuteczność detekcji dla wycieku = {selected_filter}"
        
    with col_plot:

        colors = {
            'Chama': '#00CC96',
            'XGBoost': '#EF553B',
            'LightGBM': '#636EFA',
            'NeuralNet': '#FFA15A'
        }

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for model in models:

            color = colors.get(model, '#AB63FA')
            
            col_roc_auc = f"{model}_AUC_ROC"
            col_pr_auc = f"{model}_PR_AUC"
            col_f1 = f"{model}_Max_F1"
            col_partial_pr_auc = f"{model}_Partial_PR_AUC_0.6" 
            
            if col_roc_auc in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_values, y=df_filtered[col_roc_auc].tolist(), 
                        name=f"{model} (ROC AUC)",
                        mode='lines+markers', 
                        line=dict(color=color, dash='solid', width=2)
                    ),
                    secondary_y=False,
                )

            if col_pr_auc in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_values, y=df_filtered[col_pr_auc].tolist(), 
                        name=f"{model} (PR AUC)",
                        mode='lines+markers', 
                        line=dict(color=color, dash='solid', width=2)
                    ),
                    secondary_y=False,
                )

            if col_partial_pr_auc in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_values, y=df_filtered[col_partial_pr_auc].tolist(), 
                        name=f"{model} (Partial PR AUC ≥ 0.6)",
                        mode='lines+markers', 
                        line=dict(color=color, dash='solid', width=2)
                    ),
                    secondary_y=False, # Zostawiamy na głównej osi Y z resztą AUC
                )

            if col_f1 in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_values, y=df_filtered[col_f1].tolist(), 
                        name=f"{model} (Operating F1-Score)",
                        mode='lines+markers', 
                        line=dict(color=color, dash='dash', width=2.5)
                    ),
                    secondary_y=True,
                )

        fig.update_layout(
            title=plot_title,
            xaxis_title=x_title,
            template='plotly_dark',
            hovermode="x unified",
            legend=dict(
                orientation="v", 
                yanchor="top", y=1, 
                xanchor="left", x=1.02
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=650
        )

        fig.update_yaxes(title_text="<b>AUC (ROC, PR, Partial PR)</b>", secondary_y=False, color="white", gridcolor='#262730', range=[-0.05, 1.05])
        fig.update_yaxes(title_text="<b>Operating F1 Score</b>", secondary_y=True, color="white", showgrid=False, range=[-0.05, 1.05])

        if use_budget_on_x and fixed_xaxis_budget:
            fig.update_xaxes(range=[0, 20])

        st.plotly_chart(fig, use_container_width=True)

def filter_and_sort_dataframe(df, description):

    df_table = df.copy()
    if 'budget' not in df_table.columns:
        df_table = df_table.reset_index()
    
    if isinstance(df_table.columns, pd.MultiIndex):
        new_cols = []
        for col in df_table.columns:
            if col[1]:
                new_cols.append(f"{col[1]} ({col[0]})")
            else:
                new_cols.append(col[0])
        df_table.columns = new_cols

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Tabela")
        budgets = sorted(df_table['budget'].unique())
        selected_budget = st.selectbox(
            "Budzet na osi X", 
            options=budgets, 
            key=f"{description}_table_budget"
        )
        
    with col2:
        st.markdown("### &nbsp;") # Puste miejsce dla wyrównania wizualnego
        leaks = sorted(df_table['leak_diameter_parameter'].unique(), key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else float('inf'))
        selected_leak = st.selectbox(
            "Pokaż dane dla średnicy wycieku:", 
            options=leaks, 
            key=f"{description}_table_leak"
        )

    if selected_budget is not None and selected_leak is not None:
        mask = (df_table['budget'] == selected_budget) & (df_table['leak_diameter_parameter'] == selected_leak)
        filtered_df = df_table[mask].copy()
    else:
        filtered_df = df_table.copy() 

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

def display_summary(df_uniwersalne, df_wycieki):
    with st.expander("Podsumowanie", expanded=False):

        if df_wycieki is not None:
            with st.expander("Osobna optymalizacja dla każdej wielkości wycieku", expanded=True):
                display_summary_plot(df_wycieki, "dedykowane")
                st.divider()
                filter_and_sort_dataframe(df_wycieki, "dedykowane")

        if df_uniwersalne is not None:
            with st.expander("Optymalizacja dla wszystkich rozmiarów wycieków", expanded=True):
                display_summary_plot(df_uniwersalne, "globalne")
                st.divider()
                filter_and_sort_dataframe(df_uniwersalne, "globalne")
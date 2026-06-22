import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import auc
from src.config import SIMULATION_CONFIG

def load_and_standardize_data():

    pickle_dir = SIMULATION_CONFIG.output_folder / 'pickle'
    dataframes = []

    try:
        with open(pickle_dir / 'precision_recall_data_chama.pkl', 'rb') as f:
            df = pickle.load(f)
            if 'Formulation' in df.columns:
                df = df[df['Formulation'] == 'ImpactFormulation']
                
            df = df.rename(columns={'thp': 'decision_threshold', 'leak_diameters': 'leak_diameter_parameter', 'Budget': 'budget'})
            df['Model'] = 'Chama'
            df['Model_Type'] = 'Uniwersalny'
            dataframes.append(df)
    except FileNotFoundError: pass

    try:
        with open(pickle_dir / 'precision_recall_data_chama_seperate.pkl', 'rb') as f:
            df = pickle.load(f)

            if 'Formulation' in df.columns:
                df = df[df['Formulation'] == 'ImpactFormulation']
                
            df = df.rename(columns={'thp': 'decision_threshold', 'leak_diameters': 'leak_diameter_parameter', 'Budget': 'budget'})
            df['Model'] = 'Chama'
            df['Model_Type'] = 'Dla_wyciekow'
            dataframes.append(df)
    except FileNotFoundError: pass

    ml_mapping = {
        'XGBoost': ('confusion_matrix_global_xgb.pkl', 'confusion_matrix_best_nodes_df_xgb.pkl'),
        'LightGBM': ('confusion_matrix_global_lgb.pkl', 'confusion_matrix_best_nodes_df_lgb.pkl'),
        'NeuralNet': ('confusion_matrix_global_nn.pkl', 'confusion_matrix_best_nodes_df_nn.pkl')
    }

    for model_name, (file_global, file_specific) in ml_mapping.items():
        try:
            with open(pickle_dir / file_global, 'rb') as f:
                df = pickle.load(f)
                df['Model'] = model_name
                df['Model_Type'] = 'Uniwersalny'
                dataframes.append(df)
        except FileNotFoundError: pass

        try:
            with open(pickle_dir / file_specific, 'rb') as f:
                df = pickle.load(f)
                df['Model'] = model_name
                df['Model_Type'] = 'Dla_wyciekow'
                dataframes.append(df)
        except FileNotFoundError: pass

    if not dataframes:
        raise ValueError("Nie załadowano żadnych danych! Sprawdź ścieżki do plików .pkl.")

    df_all = pd.concat(dataframes, ignore_index=True)
    df_all.rename(columns={'Budget': 'budget'}, inplace=True)
    
    df_all = df_all[~df_all['leak_diameter_parameter'].astype(str).str.lower().isin(['all', 'any'])]

    df_all['leak_diameter_parameter'] = pd.to_numeric(df_all['leak_diameter_parameter'], errors='coerce').round(4).astype(str)

    cols_to_keep = ['Model', 'Model_Type', 'budget', 'leak_diameter_parameter', 'decision_threshold', 'TP', 'FP', 'TN', 'FN']
    return df_all[cols_to_keep]

def calculate_metrics(group):

    group_sorted = group.sort_values(by='decision_threshold', ascending=True).copy()
    group_sorted['TPR'] = group_sorted['TP'] / (group_sorted['TP'] + group_sorted['FN'] + 1e-9)
    group_sorted['FPR'] = group_sorted['FP'] / (group_sorted['FP'] + group_sorted['TN'] + 1e-9)
    group_sorted['Precision'] = group_sorted['TP'] / (group_sorted['TP'] + group_sorted['FP'] + 1e-9)
    group_sorted['F1'] = 2 * (group_sorted['Precision'] * group_sorted['TPR']) / (group_sorted['Precision'] + group_sorted['TPR'] + 1e-9)

    fpr_points, tpr_points = group_sorted['FPR'].tolist(), group_sorted['TPR'].tolist()
    roc_points = sorted(zip(fpr_points, tpr_points))
    if roc_points and roc_points[0][0] != 0: roc_points.insert(0, (0.0, 0.0))
    if roc_points and roc_points[-1][0] != 1: roc_points.append((1.0, 1.0))
    
    auc_val = auc(*zip(*roc_points)) if len(roc_points) > 1 else 0.0

    max_f1 = group_sorted['F1'].max()

    return pd.Series({'AUC_ROC': round(auc_val, 4), 'Max_F1': round(max_f1, 4)})

def generate_comparative_tables():

    df_all = load_and_standardize_data()
    
    df_all = df_all.dropna(subset=['budget'])
    df_all['budget'] = df_all['budget'].astype(int)

    summary_df = df_all.groupby(['Model', 'Model_Type', 'budget', 'leak_diameter_parameter']).apply(calculate_metrics).reset_index()

    output_dir = SIMULATION_CONFIG.output_folder / 'csv'
    output_dir.mkdir(exist_ok=True)

    df_uniwersalne = summary_df[summary_df['Model_Type'] == 'Uniwersalny']
    
    if not df_uniwersalne.empty:
        pivot_uniwersalne = df_uniwersalne.pivot_table(
            index=['budget', 'leak_diameter_parameter'], 
            columns='Model', 
            values=['AUC_ROC', 'Max_F1']
        )
        
        csv_path_uni = output_dir / 'global_models_table.csv'
        pivot_uniwersalne.to_csv(csv_path_uni)

        pickle_output_uni = SIMULATION_CONFIG.output_folder / 'pickle' / 'global_models_table.pkl'
        pivot_uniwersalne.to_pickle(pickle_output_uni)


    df_wycieki = summary_df[summary_df['Model_Type'] == 'Dla_wyciekow']
    
    if not df_wycieki.empty:
        pivot_wycieki = df_wycieki.pivot_table(
            index=['budget', 'leak_diameter_parameter'], 
            columns='Model', 
            values=['AUC_ROC', 'Max_F1']
        )
        
        csv_path_wyc = output_dir / 'seperate_leaks_models_table.csv'
        pivot_wycieki.to_csv(csv_path_wyc)

        pickle_output_uni = SIMULATION_CONFIG.output_folder / 'pickle' / 'seperate_leaks_models_table.pkl'
        pivot_wycieki.to_pickle(pickle_output_uni)


generate_comparative_tables()
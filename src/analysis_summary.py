import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import auc
from src.config import SIMULATION_CONFIG


#max false positive rate allowed: 5%
MAX_ACCEPTABLE_FPR = 0.05  

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
    
    df_all['leak_diameter_parameter'] = df_all['leak_diameter_parameter'].apply(
        lambda x: str(x) if str(x).lower() in ['all', 'any'] else str(pd.to_numeric(x, errors='coerce').round(4))
    )

    cols_to_keep = ['Model', 'Model_Type', 'budget', 'leak_diameter_parameter', 'decision_threshold', 'TP', 'FP', 'TN', 'FN']
    return df_all[cols_to_keep]


def calculate_metrics(group, target_threshold=None):
    group_sorted = group.sort_values(by='decision_threshold', ascending=True).copy()
    
    group_sorted['TPR'] = group_sorted['TP'] / (group_sorted['TP'] + group_sorted['FN'] + 1e-9) 
    group_sorted['FPR'] = group_sorted['FP'] / (group_sorted['FP'] + group_sorted['TN'] + 1e-9)
    group_sorted['Precision'] = group_sorted['TP'] / (group_sorted['TP'] + group_sorted['FP'] + 1e-9)
    group_sorted['F1'] = 2 * (group_sorted['Precision'] * group_sorted['TPR']) / (group_sorted['Precision'] + group_sorted['TPR'] + 1e-9)

    fpr_points, tpr_points = group_sorted['FPR'].tolist(), group_sorted['TPR'].tolist()
    roc_points = sorted(zip(fpr_points, tpr_points))
    if roc_points and roc_points[0][0] != 0: 
        roc_points.insert(0, (0.0, 0.0))
    auc_roc_val = auc(*zip(*roc_points)) if len(roc_points) > 1 else 0.0

    recall_points = group_sorted['TPR'].tolist()
    precision_points = group_sorted['Precision'].tolist()
    pr_points = sorted(zip(recall_points, precision_points)) 
    
    if pr_points and pr_points[0][0] != 0: 
        pr_points.insert(0, (0.0, pr_points[0][1]))
    
    pr_auc_val = auc(*zip(*pr_points)) if len(pr_points) > 1 else 0.0

    min_recall = 0.6
    valid_points = group_sorted[group_sorted['TPR'] >= min_recall].sort_values(by='TPR')
    if not valid_points.empty:
        recalls = valid_points['TPR'].tolist()
        precisions = valid_points['Precision'].tolist()
        if recalls[0] > min_recall:
            recalls.insert(0, min_recall)
            precisions.insert(0, precisions[0])
        raw_partial_auc = auc(recalls, precisions)
        p_auc_06 = raw_partial_auc / (1.0 - min_recall) 
    else:
        p_auc_06 = 0.0

    if target_threshold is not None:
        idx = (group_sorted['decision_threshold'] - target_threshold).abs().idxmin()
        operational_f1 = group_sorted.loc[idx, 'F1']
        operational_thp = group_sorted.loc[idx, 'decision_threshold']
    else:
        valid_thps = group_sorted[group_sorted['FPR'] <= MAX_ACCEPTABLE_FPR]
        if not valid_thps.empty:
            operational_f1 = valid_thps.iloc[0]['F1']
            operational_thp = valid_thps.iloc[0]['decision_threshold']
        else:
            operational_f1 = group_sorted['F1'].max()  # Fallback
            operational_thp = group_sorted.loc[group_sorted['F1'].idxmax(), 'decision_threshold']

    return pd.Series({
        'AUC_ROC': round(auc_roc_val, 4), 
        'PR_AUC': round(pr_auc_val, 4), 
        'Max_F1': round(operational_f1, 4), 
        'Partial_PR_AUC_0.6': round(p_auc_06, 4),
        'Operational_THP': round(operational_thp, 4) # <-- NOWA KOLUMNA
    })

def generate_comparative_tables():
    df_raw = load_and_standardize_data()
    
    df_raw = df_raw.dropna(subset=['budget'])
    df_raw['budget'] = df_raw['budget'].astype(int)

    results = []

    for (model, model_type, budget), group in df_raw.groupby(['Model', 'Model_Type', 'budget']):
        
        target_thp = None
        
        if model_type == 'Uniwersalny':
            df_all_leaks = group[group['leak_diameter_parameter'].astype(str).str.lower() == 'all']
            if not df_all_leaks.empty:
                df_sorted = df_all_leaks.sort_values('decision_threshold')
                df_sorted['FPR'] = df_sorted['FP'] / (df_sorted['FP'] + df_sorted['TN'] + 1e-9)
                
                valid_thps = df_sorted[df_sorted['FPR'] <= MAX_ACCEPTABLE_FPR]
                if not valid_thps.empty:
                    target_thp = valid_thps.iloc[0]['decision_threshold']
                else:
                    target_thp = df_sorted.iloc[-1]['decision_threshold'] 

        group_no_all = group[~group['leak_diameter_parameter'].astype(str).str.lower().isin(['all', 'any'])]

        for leak_dia, sub_group in group_no_all.groupby('leak_diameter_parameter'):
            metrics = calculate_metrics(sub_group, target_threshold=target_thp)
            
            metrics['Model'] = model
            metrics['Model_Type'] = model_type
            metrics['budget'] = budget
            metrics['leak_diameter_parameter'] = leak_dia
            
            results.append(metrics)

    summary_df = pd.DataFrame(results)

    output_dir = SIMULATION_CONFIG.output_folder / 'csv'
    output_dir.mkdir(exist_ok=True)

    df_uniwersalne = summary_df[summary_df['Model_Type'] == 'Uniwersalny']
    
    if not df_uniwersalne.empty:
        pivot_uniwersalne = df_uniwersalne.pivot_table(
            index=['budget', 'leak_diameter_parameter'], 
            columns='Model', 
            values=['AUC_ROC', 'PR_AUC', 'Max_F1', 'Partial_PR_AUC_0.6', 'Operational_THP'] 
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
            values=['AUC_ROC', 'PR_AUC', 'Max_F1', 'Partial_PR_AUC_0.6', 'Operational_THP'] 
        )
        
        csv_path_wyc = output_dir / 'seperate_leaks_models_table.csv'
        pivot_wycieki.to_csv(csv_path_wyc)

        pickle_output_wyc = SIMULATION_CONFIG.output_folder / 'pickle' / 'seperate_leaks_models_table.pkl'
        pivot_wycieki.to_pickle(pickle_output_wyc)


if __name__ == "__main__":
    generate_comparative_tables()
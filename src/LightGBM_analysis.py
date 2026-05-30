import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np
from src.config import SIMULATION_CONFIG
from tqdm import tqdm
import pickle

def LightGBM_analysis_all_nodes():

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_dataset_XGB.pkl')
    y = df_signals['Is_Leak']

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]
    
    unique_leaks = [x for x in df_signals['leak_diameter_parameter'].unique() if pd.notna(x)]
    leak_diameter_parameters = unique_leaks + ['All']

    results_list = []

    for leak_diameter_parameter in tqdm(leak_diameter_parameters, desc="LGBM: Analiza wszystkich węzłów"):
        if leak_diameter_parameter == 'All':
            mask = pd.Series(True, index=df_signals.index)
        else:
            mask = (df_signals['leak_diameter_parameter'] == leak_diameter_parameter) | (df_signals['leak_diameter_parameter'].isna())

        X_filtered = df_signals.loc[mask].copy().drop(columns=metadata_cols, errors='ignore')
        y_filtered = y.loc[mask].copy()
        
        # Podział na zbiory
        unique_scenarios = df_signals.loc[mask, 'Scenario_Name'].unique()
        train_scenarios, test_scenarios = train_test_split(unique_scenarios, test_size=0.3, random_state=42)
        
        train_keys = df_signals.loc[mask, 'Scenario_Name'].isin(train_scenarios)
        test_keys = df_signals.loc[mask, 'Scenario_Name'].isin(test_scenarios)

        X_train, y_train = X_filtered.loc[train_keys], y_filtered.loc[train_keys]
        X_test, y_test = X_filtered.loc[test_keys], y_filtered.loc[test_keys]
        
        class_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1

        model_lgb = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            num_leaves=31, # Kluczowe dla LGBM przy max_depth=5
            learning_rate=0.1,
            scale_pos_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )

        model_lgb.fit(X_train, y_train)
        probs = model_lgb.predict_proba(X_test)[:, 1]

        for threshold in [round(x * 0.1, 1) for x in range(1, 10)]:
            preds = (probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            results_list.append({
                'leak_diameter_parameter': leak_diameter_parameter,
                'decision_threshold': threshold,
                'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)
            })

    results_df = pd.DataFrame(results_list)
    results_df.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_lgb.pkl')
    results_df.to_csv(SIMULATION_CONFIG.output_folder / 'csv' / 'confusion_matrix_lgb.csv')

def LightGBM_analysis_best_nodes():

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_dataset_XGB.pkl')
    y = df_signals['Is_Leak']
    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']

    unique_leaks = [x for x in df_signals['leak_diameter_parameter'].unique() if pd.notna(x)]
    leak_diameter_parameters = unique_leaks + ['All']
    
    initial_nodes = [c for c in df_signals.columns if c not in metadata_cols]
    nodes_number = len(initial_nodes)

    results_list = []
    importances_list = []

    for leak_diameter in tqdm(leak_diameter_parameters, desc="LGBM RFE: Średnice"):
        if leak_diameter == 'All':
            mask = pd.Series(True, index=df_signals.index)
        else:
            mask = (df_signals['leak_diameter_parameter'] == leak_diameter) | (df_signals['leak_diameter_parameter'].isna())

        X_base = df_signals.loc[mask].copy().drop(columns=metadata_cols, errors='ignore')
        y_base = y.loc[mask].copy()
        
        train_scenarios, test_scenarios = train_test_split(df_signals.loc[mask, 'Scenario_Name'].unique(), test_size=0.3, random_state=42)
        train_keys = df_signals.loc[mask, 'Scenario_Name'].isin(train_scenarios)
        test_keys = df_signals.loc[mask, 'Scenario_Name'].isin(test_scenarios)

        X_train_base, y_train = X_base.loc[train_keys], y_base.loc[train_keys]
        X_test_base, y_test = X_base.loc[test_keys], y_base.loc[test_keys]
        class_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1

        dropped_nodes = []
        for budget in tqdm(range(nodes_number, 0, -1), desc=f"Budgeting {leak_diameter}", leave=False):
            X_tr = X_train_base.drop(columns=dropped_nodes, errors='ignore')
            X_ts = X_test_base.drop(columns=dropped_nodes, errors='ignore')

            model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, num_leaves=31, scale_pos_weight=class_weight, random_state=42, n_jobs=-1, verbosity=-1)
            model.fit(X_tr, y_train)
            
            probs = model.predict_proba(X_ts)[:, 1]
            fpr, tpr, thresholds_roc = roc_curve(y_test, probs) # <--- POPRAWIONE
            opt_thresh = float(thresholds_roc[(tpr - fpr).argmax()]) if len(fpr) > 0 else 0.5

            importances_list.append(pd.DataFrame({
                'leak_diameter_parameter': leak_diameter, 'Nodes': X_tr.columns, 
                'Importance': model.feature_importances_, 'optimal_decision_threshold': round(opt_thresh, 4), 'budget': budget
            }))

            # Usuwanie najmniej ważnego węzła
            least_node = X_tr.columns[np.argmin(model.feature_importances_)]
            dropped_nodes.append(least_node)

            for th in [round(x * 0.1, 1) for x in range(1, 10)]:
                tn, fp, fn, tp = confusion_matrix(y_test, (probs >= th).astype(int)).ravel()
                results_list.append({'leak_diameter_parameter': leak_diameter, 'decision_threshold': th, 'budget': budget, 'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)})

    pd.DataFrame(results_list).to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_best_nodes_lgb.pkl')
    pd.concat(importances_list).to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'top_nodes_lgb.pkl')

if __name__ == "__main__":
    LightGBM_analysis_all_nodes()
    LightGBM_analysis_best_nodes()
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from src.config import SIMULATION_CONFIG
from tqdm import tqdm
from sklearn.metrics import roc_curve
import pickle
from joblib import Parallel, delayed

def XGBoost_analysis_all_nodes():

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')

    y = df_signals['Is_Leak']

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]

    leak_diameter_parameters = df_signals['leak_diameter_parameter'].unique().tolist()
    leak_diameter_parameters.append('All')

    results_list = []

    for leak_diameter_parameter in tqdm(leak_diameter_parameters):

        if pd.isna(leak_diameter_parameter):
            continue 
        elif leak_diameter_parameter == 'All':
            mask = pd.Series(True, index=df_signals.index) 
        else:
            mask = (df_signals['leak_diameter_parameter'] == leak_diameter_parameter) | (df_signals['leak_diameter_parameter'].isna())

        X_filtered = df_signals.loc[mask].copy()
        y_filtered = y.loc[mask].copy()
        metadata_filtered = metadata.loc[mask].copy()

        X_filtered = X_filtered.drop(columns=metadata_cols, errors='ignore')

        unique_scenarios = metadata_filtered['Scenario_Name'].unique()

        train_scenarios, test_scenarios = train_test_split(
            unique_scenarios, 
            test_size=0.3,         
            random_state=42        
        )

        train_keys = metadata_filtered['Scenario_Name'].isin(train_scenarios)
        test_keys = metadata_filtered['Scenario_Name'].isin(test_scenarios)

        X_train, y_train = X_filtered.loc[train_keys], y_filtered.loc[train_keys]
        X_test, y_test = X_filtered.loc[test_keys], y_filtered.loc[test_keys]
        
        metadata_test = metadata_filtered.loc[test_keys].copy()

        noise_number = np.sum(y_train == 0)
        leak_number = np.sum(y_train == 1)
        class_weight = noise_number / leak_number

        model_xgb = xgb.XGBClassifier(
            n_estimators=100,         
            max_depth=5,              
            learning_rate=0.1,        
            scale_pos_weight=class_weight,
            random_state=42,
            eval_metric='logloss',
            n_jobs=1
        )

        model_xgb.fit(X_train, y_train)

        probabilities = model_xgb.predict_proba(X_test)[:, 1]

        metadata_test['Leak_Probability'] = probabilities
        metadata_test['True_Is_Leak'] = y_test

        decision_thresholds = [round(x * 0.1, 1) for x in range(1, 10)]

        for decision_threshold in decision_thresholds:

            metadata_test['Final_Prediction'] = (metadata_test['Leak_Probability'] >= decision_threshold).astype(int)
            metadata_test['Decision_Threshold'] = decision_threshold

            tn, fp, fn, tp = confusion_matrix(
                metadata_test['True_Is_Leak'], 
                metadata_test['Final_Prediction']
            ).ravel()

            results_list.append({
                'leak_diameter_parameter': leak_diameter_parameter,
                'decision_threshold': decision_threshold,
                'TP': int(tp),
                'FP': int(fp),
                'TN': int(tn),
                'FN': int(fn)
                })

    pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_df_xgb.pkl'
    csv_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'csv' / 'confusion_matrix_df.csv'

    results_df = pd.DataFrame(results_list)

    with open(pickle_output_confusion_matrix_df, 'wb') as file:
        pickle.dump(results_df, file)

    # results_df.to_csv(csv_output_confusion_matrix_df)

def XGBoost_analysis_best_nodes():

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')

    y = df_signals['Is_Leak']

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]
    
    unique_leaks = [x for x in df_signals['leak_diameter_parameter'].unique() if pd.notna(x)]
    leak_diameter_parameters = unique_leaks + ['All']

    initial_nodes = [c for c in df_signals.columns if c not in metadata_cols]
    nodes_number = len(initial_nodes) 

    results_list = []
    importances_list = []

    for leak_diameter_parameter in tqdm(leak_diameter_parameters, desc="Iterowanie po sreednicach wyciekow"):
        
        if leak_diameter_parameter == 'All':
            mask = pd.Series(True, index=df_signals.index)
        else:
            mask = (df_signals['leak_diameter_parameter'] == leak_diameter_parameter) | (df_signals['leak_diameter_parameter'].isna())

        X_base = df_signals.loc[mask].copy()
        y_base = y.loc[mask].copy()
        metadata_base = metadata.loc[mask].copy()

        X_base = X_base.drop(columns=metadata_cols, errors='ignore')

        unique_scenarios = metadata_base['Scenario_Name'].unique()
        train_scenarios, test_scenarios = train_test_split(
            unique_scenarios, test_size=0.3, random_state=42
        )

        train_keys = metadata_base['Scenario_Name'].isin(train_scenarios)
        test_keys = metadata_base['Scenario_Name'].isin(test_scenarios)

        X_train_base, y_train = X_base.loc[train_keys], y_base.loc[train_keys]
        X_test_base, y_test = X_base.loc[test_keys], y_base.loc[test_keys]
        metadata_test_base = metadata_base.loc[test_keys].copy()

        noise_number = np.sum(y_train == 0)
        leak_number = np.sum(y_train == 1)
        class_weight = noise_number / leak_number if leak_number > 0 else 1

        unimportant_nodes_dropped = []

        for budget in tqdm(range(nodes_number, 0, -1), desc=f"Iterowanie po budzetach dla leka diameter: {leak_diameter_parameter}", leave=False):

            X_train = X_train_base.drop(columns=unimportant_nodes_dropped, errors='ignore')
            X_test = X_test_base.drop(columns=unimportant_nodes_dropped, errors='ignore')
            metadata_test = metadata_test_base.copy()

            model_xgb = xgb.XGBClassifier(
                n_estimators=100,         
                max_depth=5,              
                learning_rate=0.1,        
                scale_pos_weight=class_weight,
                random_state=42,
                eval_metric='logloss',
                n_jobs=1
            )

            model_xgb.fit(X_train, y_train)

            probabilities = model_xgb.predict_proba(X_test)[:, 1]
            metadata_test['Leak_Probability'] = probabilities
            metadata_test['True_Is_Leak'] = y_test

            fpr, tpr, thresholds_roc = roc_curve(
                metadata_test['True_Is_Leak'], metadata_test['Leak_Probability']
            )
            idx_optymalne = (tpr - fpr).argmax()
            optymalny_threshold = float(thresholds_roc[idx_optymalne])

            node_importance_df = pd.DataFrame({
                'leak_diameter_parameter': leak_diameter_parameter,
                'Nodes': X_train.columns,
                'Importance': model_xgb.feature_importances_,
                'optimal_decision_threshold': round(optymalny_threshold, 4),
                'budget': budget
            })
            importances_list.append(node_importance_df)
            
            least_important_node = node_importance_df.sort_values(by=['Importance', 'Nodes'], ascending=[True, True]).iloc[0]['Nodes']
            unimportant_nodes_dropped.append(least_important_node)

            decision_thresholds = [round(x * 0.1, 1) for x in range(1, 10)]

            for decision_threshold in decision_thresholds:
                metadata_test['Final_Prediction'] = (metadata_test['Leak_Probability'] >= decision_threshold).astype(int)
                metadata_test['Decision_Threshold'] = decision_threshold

                tn, fp, fn, tp = confusion_matrix(
                    metadata_test['True_Is_Leak'], metadata_test['Final_Prediction']
                ).ravel()

                results_list.append({
                    'leak_diameter_parameter': leak_diameter_parameter,
                    'decision_threshold': decision_threshold,
                    'budget': budget,
                    'TP': int(tp),
                    'FP': int(fp),
                    'TN': int(tn),
                    'FN': int(fn)
                })

    pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_best_nodes_df_xgb.pkl'
    csv_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'csv' / 'confusion_matrix_best_nodes_df.csv'

    results_df = pd.DataFrame(results_list)
    with open(pickle_output_confusion_matrix_df, 'wb') as file:
        pickle.dump(results_df, file)
    # results_df.to_csv(csv_output_confusion_matrix_df, index=False)

    if importances_list:
        importances_df = pd.concat(importances_list, ignore_index=True)
        pickle_output_nodes = SIMULATION_CONFIG.output_folder / 'pickle' / 'best_nodes_xgb.pkl'
        csv_output_nodes = SIMULATION_CONFIG.output_folder / 'csv' / 'top_nodes_xgb.csv'
        
        with open(pickle_output_nodes, 'wb') as file:
            pickle.dump(importances_df, file)
        # importances_df.to_csv(csv_output_nodes, index=False)

# XGBoost_analysis_all_nodes()
# XGBoost_analysis_best_nodes()
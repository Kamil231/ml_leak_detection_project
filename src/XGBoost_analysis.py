import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np
from src.config import SIMULATION_CONFIG
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import optuna
from optuna.integration import XGBoostPruningCallback
import time

def objective_xgb(trial, dtrain, dval):
    param = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr", 
        "verbosity": 0,
        "tree_method": "hist", 
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    pruning_callback = XGBoostPruningCallback(trial, "validation-aucpr")
    
    evals_result = {}
    
    gbm = xgb.train(
        param, 
        dtrain, 
        num_boost_round=100,
        evals=[(dval, "validation")], 
        callbacks=[pruning_callback],
        evals_result=evals_result,
        verbose_eval=False
    )

    return evals_result["validation"]["aucpr"][-1]

def get_tuned_params_xgb(fun_name):
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')
    y = df_signals['Is_Leak']

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    
    X = df_signals.drop(columns=metadata_cols, errors='ignore')

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )   

    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    print("Rozpoczynam tuning hiperparametrów XGBoost za pomocą Optunay")
    
    study = optuna.create_study(direction="maximize") 
    study.optimize(lambda trial: objective_xgb(trial, dtrain, dval), n_trials=50)

    best_params = study.best_params

    file_path = SIMULATION_CONFIG.output_folder / 'parametrem_tuning.txt'
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n\n--- XGBoost Tuning: {fun_name} ---\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "aucpr"
    best_params["tree_method"] = "hist"
    best_params["verbosity"] = 0

    return best_params

def XGBoost_analysis_all_nodes():

    best_params = get_tuned_params_xgb('XGBoost_analysis_all_nodes')
    print(f"Użyte parametry do analizy wszystkich węzłów: {best_params}")

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')
    y = df_signals['Is_Leak']

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    
    unique_leaks = [x for x in df_signals['leak_diameter_parameter'].unique() if pd.notna(x)]
    leak_diameter_parameters = unique_leaks + ['All']

    results_list = []

    for leak_diameter_parameter in tqdm(leak_diameter_parameters, desc="XGB: Analiza wszystkich węzłów"):
        if leak_diameter_parameter == 'All':
            mask = pd.Series(True, index=df_signals.index) 
        else:
            mask = (df_signals['leak_diameter_parameter'] == leak_diameter_parameter) | (df_signals['leak_diameter_parameter'].isna())

        X_filtered = df_signals.loc[mask].copy().drop(columns=metadata_cols, errors='ignore')
        y_filtered = y.loc[mask].copy()

        unique_scenarios = df_signals.loc[mask, 'Scenario_Name'].unique()

        train_scenarios, test_scenarios = train_test_split(
            unique_scenarios, test_size=0.3, random_state=42
        )

        train_keys = df_signals.loc[mask, 'Scenario_Name'].isin(train_scenarios)
        test_keys = df_signals.loc[mask, 'Scenario_Name'].isin(test_scenarios)

        X_train, y_train = X_filtered.loc[train_keys], y_filtered.loc[train_keys]
        X_test, y_test = X_filtered.loc[test_keys], y_filtered.loc[test_keys]
        
        noise_number = np.sum(y_train == 0)
        leak_number = np.sum(y_train == 1)
        class_weight = noise_number / leak_number if leak_number > 0 else 1

        best_params['scale_pos_weight'] = class_weight

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        model_xgb = xgb.train(
            best_params,
            dtrain,
            num_boost_round=100
        )

        probs = model_xgb.predict(dtest)

        for decision_threshold in [round(x * 0.1, 1) for x in range(1, 10)]:

            preds = (probs >= decision_threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

            results_list.append({
                'leak_diameter_parameter': leak_diameter_parameter,
                'decision_threshold': decision_threshold,
                'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)
            })

    pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_df_xgb.pkl'
    results_df = pd.DataFrame(results_list)
    with open(pickle_output_confusion_matrix_df, 'wb') as file:
        pickle.dump(results_df, file)

def XGBoost_analysis_best_nodes():

    best_params = get_tuned_params_xgb('XGBoost_analysis_best_nodes')
    print(f"Użyte parametry do RFE: {best_params}")

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')
    y = df_signals['Is_Leak']

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    
    unique_leaks = [x for x in df_signals['leak_diameter_parameter'].unique() if pd.notna(x)]
    leak_diameter_parameters = unique_leaks + ['All']

    initial_nodes = [c for c in df_signals.columns if c not in metadata_cols]
    nodes_number = len(initial_nodes) 

    results_list = []
    importances_list = []

    for leak_diameter in tqdm(leak_diameter_parameters, desc="XGB RFE: Średnice"):
        
        if leak_diameter == 'All':
            mask = pd.Series(True, index=df_signals.index)
        else:
            mask = (df_signals['leak_diameter_parameter'] == leak_diameter) | (df_signals['leak_diameter_parameter'].isna())

        X_base = df_signals.loc[mask].copy().drop(columns=metadata_cols, errors='ignore')
        y_base = y.loc[mask].copy()

        train_scenarios, test_scenarios = train_test_split(
            df_signals.loc[mask, 'Scenario_Name'].unique(), test_size=0.3, random_state=42
        )

        train_keys = df_signals.loc[mask, 'Scenario_Name'].isin(train_scenarios)
        test_keys = df_signals.loc[mask, 'Scenario_Name'].isin(test_scenarios)

        X_train_base, y_train = X_base.loc[train_keys], y_base.loc[train_keys]
        X_test_base, y_test = X_base.loc[test_keys], y_base.loc[test_keys]

        noise_number = np.sum(y_train == 0)
        leak_number = np.sum(y_train == 1)
        class_weight = noise_number / leak_number if leak_number > 0 else 1
        
        best_params['scale_pos_weight'] = class_weight

        unimportant_nodes_dropped = []

        for budget in tqdm(range(nodes_number, 0, -1), desc=f"Budgeting {leak_diameter}", leave=False):

            X_train = X_train_base.drop(columns=unimportant_nodes_dropped, errors='ignore')
            X_test = X_test_base.drop(columns=unimportant_nodes_dropped, errors='ignore')

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test)

            model_xgb = xgb.train(
                best_params,
                dtrain,
                num_boost_round=100
            )

            probs = model_xgb.predict(dtest)

            fpr, tpr, thresholds_roc = roc_curve(y_test, probs)
            opt_thresh = float(thresholds_roc[(tpr - fpr).argmax()]) if len(fpr) > 0 else 0.5

            scores_dict = model_xgb.get_score(importance_type='gain')
            importances_array = [scores_dict.get(col, 0.0) for col in X_train.columns]

            node_importance_df = pd.DataFrame({
                'leak_diameter_parameter': leak_diameter,
                'Nodes': X_train.columns,
                'Importance': importances_array,
                'optimal_decision_threshold': round(opt_thresh, 4),
                'budget': budget
            })
            importances_list.append(node_importance_df)
            
            least_node = node_importance_df.sort_values(by=['Importance', 'Nodes'], ascending=[True, True]).iloc[0]['Nodes']
            unimportant_nodes_dropped.append(least_node)

            for decision_threshold in [round(x * 0.1, 1) for x in range(1, 10)]:
                preds = (probs >= decision_threshold).astype(int)

                tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

                results_list.append({
                    'leak_diameter_parameter': leak_diameter,
                    'decision_threshold': decision_threshold,
                    'budget': budget,
                    'TP': int(tp),
                    'FP': int(fp),
                    'TN': int(tn),
                    'FN': int(fn)
                })

    pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_best_nodes_df_xgb.pkl'
    results_df = pd.DataFrame(results_list)
    with open(pickle_output_confusion_matrix_df, 'wb') as file:
        pickle.dump(results_df, file)

    if importances_list:
        importances_df = pd.concat(importances_list, ignore_index=True)
        pickle_output_nodes = SIMULATION_CONFIG.output_folder / 'pickle' / 'best_nodes_xgb.pkl'
        
        with open(pickle_output_nodes, 'wb') as file:
            pickle.dump(importances_df, file)

def XGBoost_analysis_global():

    best_params = get_tuned_params_xgb('XGBoost_analysis_global')

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')
    y = df_signals['Is_Leak']

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]
    
    unique_leaks = [x for x in df_signals['leak_diameter_parameter'].unique() if pd.notna(x)]
    eval_leak_diameters = ['All'] + unique_leaks

    X_base = df_signals.drop(columns=metadata_cols, errors='ignore')

    unique_scenarios = metadata['Scenario_Name'].unique()
    train_scenarios, test_scenarios = train_test_split(
        unique_scenarios, test_size=0.3, random_state=42
    )

    train_keys = metadata['Scenario_Name'].isin(train_scenarios)
    test_keys = metadata['Scenario_Name'].isin(test_scenarios)

    X_train_global, y_train_global = X_base.loc[train_keys], y.loc[train_keys]
    X_test_global, y_test_global = X_base.loc[test_keys], y.loc[test_keys]
    metadata_test_global = metadata.loc[test_keys].copy()

    noise_number = np.sum(y_train_global == 0)
    leak_number = np.sum(y_train_global == 1)
    class_weight = noise_number / leak_number if leak_number > 0 else 1

    best_params['scale_pos_weight'] = class_weight

    initial_nodes = X_train_global.columns.tolist()
    nodes_number = len(initial_nodes)

    results_list = []
    importances_list = []
    unimportant_nodes_dropped = []

    for budget in tqdm(range(nodes_number, 0, -1), desc="Iterowanie po budżetach (Model Globalny XGB)"):

        X_train = X_train_global.drop(columns=unimportant_nodes_dropped, errors='ignore')
        X_test = X_test_global.drop(columns=unimportant_nodes_dropped, errors='ignore')

        dtrain = xgb.DMatrix(X_train, label=y_train_global)
        dtest = xgb.DMatrix(X_test)

        model_xgb = xgb.train(
            best_params,
            dtrain,
            num_boost_round=100
        )

        probabilities_global = model_xgb.predict(dtest)
        
        metadata_test_global['Leak_Probability'] = probabilities_global
        metadata_test_global['True_Is_Leak'] = y_test_global

        fpr, tpr, thresholds_roc = roc_curve(
            metadata_test_global['True_Is_Leak'], metadata_test_global['Leak_Probability']
        )
        idx_optymalne = (tpr - fpr).argmax()
        optymalny_threshold = float(thresholds_roc[idx_optymalne])

        scores_dict = model_xgb.get_score(importance_type='gain')
        importances_array = [scores_dict.get(col, 0.0) for col in X_train.columns]

        node_importance_df = pd.DataFrame({
            'leak_diameter_parameter': 'Global', 
            'Nodes': X_train.columns,
            'Importance': importances_array,
            'optimal_decision_threshold': round(optymalny_threshold, 4),
            'budget': budget
        })
        importances_list.append(node_importance_df)
        
        least_important_node = node_importance_df.sort_values(by=['Importance', 'Nodes'], ascending=[True, True]).iloc[0]['Nodes']
        unimportant_nodes_dropped.append(least_important_node)

        decision_thresholds = [round(x * 0.1, 1) for x in range(1, 10)]

        for eval_leak in eval_leak_diameters:
            
            if eval_leak == 'All':
                mask_test = pd.Series(True, index=metadata_test_global.index)
            else:
                mask_test = (metadata_test_global['leak_diameter_parameter'] == eval_leak) | (metadata_test_global['leak_diameter_parameter'].isna())

            meta_subset = metadata_test_global.loc[mask_test].copy()

            for decision_threshold in decision_thresholds:
                meta_subset['Final_Prediction'] = (meta_subset['Leak_Probability'] >= decision_threshold).astype(int)

                tn, fp, fn, tp = confusion_matrix(
                    meta_subset['True_Is_Leak'], meta_subset['Final_Prediction']
                ).ravel()

                results_list.append({
                    'leak_diameter_parameter': eval_leak, 
                    'decision_threshold': decision_threshold,
                    'budget': budget,
                    'TP': int(tp),
                    'FP': int(fp),
                    'TN': int(tn),
                    'FN': int(fn)
                })

    pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_global_xgb.pkl'
    results_df = pd.DataFrame(results_list)
    with open(pickle_output_confusion_matrix_df, 'wb') as file:
        pickle.dump(results_df, file)

    if importances_list:
        importances_df = pd.concat(importances_list, ignore_index=True)
        pickle_output_nodes = SIMULATION_CONFIG.output_folder / 'pickle' / 'top_nodes_global_xgb.pkl'
        
        with open(pickle_output_nodes, 'wb') as file:
            pickle.dump(importances_df, file)

import time

start_time = time.time()
XGBoost_analysis_global()
t_global = time.time()
XGBoost_analysis_all_nodes()
t_all_nodes = time.time()
XGBoost_analysis_best_nodes()
t_best_nodes = time.time()

print('t_global: ', t_global - start_time)
print('t_all_nodes: ', t_all_nodes - t_global)
print('t_best_nodes: ', t_best_nodes - t_all_nodes)
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np
from src.config import SIMULATION_CONFIG
from tqdm import tqdm
import pickle
import lightgbm as lgb
from optuna.integration import LightGBMPruningCallback
import optuna

def objective_lgb(trial, train_data, val_data):
    param = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
    }

    pruning_callback = LightGBMPruningCallback(trial, "average_precision")
    
    gbm = lgb.train(
        param, 
        train_data, 
        valid_sets=[val_data], 
        callbacks=[pruning_callback],
        num_boost_round=100
    )

    return gbm.best_score['valid_0']['average_precision']

def get_tuned_model(fun_name='no_fun_name'):
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')

    y = df_signals['Is_Leak']

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]
    
    unique_leaks = [x for x in df_signals['leak_diameter_parameter'].unique() if pd.notna(x)]
    eval_leak_diameters = ['All'] + unique_leaks

    X = df_signals.drop(columns=metadata_cols, errors='ignore')

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )   

    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    study = optuna.create_study(direction="maximize") 

    study.optimize(lambda trial: objective_lgb(trial, train_data, val_data), n_trials=50)

    best_params = study.best_params
    
    file_path = SIMULATION_CONFIG.output_folder / 'parametrem_tuning.txt'
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n\n--- LightGBM Tuning: {fun_name} ---\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    
    best_params["objective"] = "binary"
    best_params["metric"] = "average_precision"
    best_params["boosting_type"] = "gbdt"
    best_params["verbosity"] = -1

    print("Trenowanie docel modelu")
    
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    full_train_data = lgb.Dataset(X_train_full, label=y_train_full)
    
    model_lgb = lgb.train(
        best_params, 
        full_train_data, 
        num_boost_round=100
    )

    return model_lgb

def LightGBM_analysis_all_nodes():

    global_model = get_tuned_model('LightGBM_analysis_all_nodes')
    best_params = global_model.params.copy()
    best_params.pop('num_iterations', None)
    print(f"Użyte parametry do analizy wszystkich węzłów: {best_params}")

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')
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

        unique_scenarios = df_signals.loc[mask, 'Scenario_Name'].unique()
        train_scenarios, test_scenarios = train_test_split(unique_scenarios, test_size=0.3, random_state=42)
        
        train_keys = df_signals.loc[mask, 'Scenario_Name'].isin(train_scenarios)
        test_keys = df_signals.loc[mask, 'Scenario_Name'].isin(test_scenarios)

        X_train, y_train = X_filtered.loc[train_keys], y_filtered.loc[train_keys]
        X_test, y_test = X_filtered.loc[test_keys], y_filtered.loc[test_keys]
        
        class_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1
        
        best_params['scale_pos_weight'] = class_weight

        current_train_data = lgb.Dataset(X_train, label=y_train)
        
        model_lgb = lgb.train(
            best_params,
            current_train_data,
            num_boost_round=100
        )

        probs = model_lgb.predict(X_test)

        for threshold in [round(x * 0.1, 1) for x in range(1, 10)]:
            preds = (probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            results_list.append({
                'leak_diameter_parameter': leak_diameter_parameter,
                'decision_threshold': threshold,
                'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)
            })

    results_df = pd.DataFrame(results_list)
    results_df.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_df_lgb.pkl')

def LightGBM_analysis_best_nodes():

    model_lgb = get_tuned_model('LightGBM_analysis_best_nodes')
    best_params = model_lgb.params.copy()
    best_params.pop('num_iterations', None) 
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
        best_params['scale_pos_weight'] = class_weight

        dropped_nodes = []

        for budget in tqdm(range(nodes_number, 0, -1), desc=f"Budgeting {leak_diameter}", leave=False):
            
            X_tr = X_train_base.drop(columns=dropped_nodes, errors='ignore')
            X_ts = X_test_base.drop(columns=dropped_nodes, errors='ignore')

            # model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, num_leaves=31, scale_pos_weight=class_weight, random_state=42, n_jobs=1, verbosity=-1)
            # model.fit(X_tr, y_train)

            current_train_data = lgb.Dataset(X_tr, label=y_train)
            model = lgb.train(
                best_params,
                current_train_data,
                num_boost_round=100
            )
            
            #probs = model.predict_proba(X_ts)[:, 1]
            probs = model.predict(X_ts)

            fpr, tpr, thresholds_roc = roc_curve(y_test, probs) 
            opt_thresh = float(thresholds_roc[(tpr - fpr).argmax()]) if len(fpr) > 0 else 0.5

            importances_list.append(pd.DataFrame({
                'leak_diameter_parameter': leak_diameter, 'Nodes': X_tr.columns, 
                'Importance': model.feature_importance(), 'optimal_decision_threshold': round(opt_thresh, 4), 'budget': budget
            }))

            least_node = X_tr.columns[np.argmin(model.feature_importance())]
            dropped_nodes.append(least_node)

            for th in [round(x * 0.1, 1) for x in range(1, 10)]:
                tn, fp, fn, tp = confusion_matrix(y_test, (probs >= th).astype(int)).ravel()
                results_list.append({'leak_diameter_parameter': leak_diameter, 'decision_threshold': th, 'budget': budget, 'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)})

    pd.DataFrame(results_list).to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_best_nodes_df_lgb.pkl')
    pd.concat(importances_list).to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'best_nodes_lgb.pkl')

def LightGBM_analysis_global():

    model_lgb = get_tuned_model('LightGBM_analysis_global')

    # df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')

    # y = df_signals['Is_Leak']

    # metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    # metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]
    
    # unique_leaks = [x for x in df_signals['leak_diameter_parameter'].unique() if pd.notna(x)]
    # eval_leak_diameters = ['All'] + unique_leaks

    # X = df_signals.drop(columns=metadata_cols, errors='ignore')

    unique_scenarios = metadata['Scenario_Name'].unique()
    train_scenarios, test_scenarios = train_test_split(
        unique_scenarios, test_size=0.3, random_state=42
    )

    train_keys = metadata['Scenario_Name'].isin(train_scenarios)
    test_keys = metadata['Scenario_Name'].isin(test_scenarios)

    X_train_global, y_train_global = X.loc[train_keys], y.loc[train_keys]
    X_test_global, y_test_global = X.loc[test_keys], y.loc[test_keys]
    metadata_test_global = metadata.loc[test_keys].copy()

    noise_number = np.sum(y_train_global == 0)
    leak_number = np.sum(y_train_global == 1)
    class_weight = noise_number / leak_number if leak_number > 0 else 1

    initial_nodes = X_train_global.columns.tolist()
    nodes_number = len(initial_nodes)

    results_list = []
    importances_list = []
    unimportant_nodes_dropped = []

    for budget in tqdm(range(nodes_number, 0, -1), desc="Iterowanie po budżetach (Model Globalny LGB)"):

        X_train = X_train_global.drop(columns=unimportant_nodes_dropped, errors='ignore')
        X_test = X_test_global.drop(columns=unimportant_nodes_dropped, errors='ignore')

        # model_lgb = lgb.LGBMClassifier(
        #     n_estimators=100,         
        #     max_depth=5,              
        #     learning_rate=0.1,        
        #     class_weight={0: 1, 1: class_weight},
        #     random_state=42,
        #     n_jobs=1,
        #     verbose=-1 
        # )

        # model_lgb.fit(X_train, y_train_global)


        current_train_data = lgb.Dataset(X_train, label=y_train_global)

        model_lgb = lgb.train(
            best_params,
            current_train_data,
            num_boost_round=100
        )

        # probabilities_global = model_lgb.predict_proba(X_test)[:, 1]
        probabilities_global = model_lgb.predict(X_test)
        metadata_test_global['Leak_Probability'] = probabilities_global
        metadata_test_global['True_Is_Leak'] = y_test_global

        fpr, tpr, thresholds_roc = roc_curve(
            metadata_test_global['True_Is_Leak'], metadata_test_global['Leak_Probability']
        )
        idx_optymalne = (tpr - fpr).argmax()
        optymalny_threshold = float(thresholds_roc[idx_optymalne])

        node_importance_df = pd.DataFrame({
            'leak_diameter_parameter': 'Global',
            'Nodes': X_train.columns,
            # 'Importance': model_lgb.feature_importances_,
            'Importance': model_lgb.feature_importance(),
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

    pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_global_lgb.pkl'
    results_df = pd.DataFrame(results_list)
    with open(pickle_output_confusion_matrix_df, 'wb') as file:
        pickle.dump(results_df, file)

    if importances_list:
        importances_df = pd.concat(importances_list, ignore_index=True)
        pickle_output_nodes = SIMULATION_CONFIG.output_folder / 'pickle' / 'top_nodes_global_lgb.pkl'
        
        with open(pickle_output_nodes, 'wb') as file:
            pickle.dump(importances_df, file)
           

import time

start_time = time.time()
LightGBM_analysis_global()
t_global = time.time()
LightGBM_analysis_all_nodes()
t_all_nodes = time.time()
LightGBM_analysis_best_nodes()
t_best_nodes = time.time()

print('t_global: ', t_global - start_time)
print('t_all_nodes: ', t_all_nodes - start_time)
print('t_best_nodes: ', t_best_nodes - start_time)


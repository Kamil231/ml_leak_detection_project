import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.config import SIMULATION_CONFIG
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_curve, confusion_matrix, average_precision_score, roc_auc_score
from joblib import Parallel, delayed
import pickle
from sklearn.inspection import permutation_importance
import warnings
import optuna
import time
import numpy as np

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

class LeakDetectionMLP(nn.Module):
    def __init__(self, input_size, hidden_1=64, hidden_2=32):
        super(LeakDetectionMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),            
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1)          
        )

    def forward(self, x):
        return self.network(x)

def objective_nn(trial, X_train_t, y_train_t, X_val_t, y_val_t, pos_weight):
    
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    hidden_1 = trial.suggest_int("hidden_1", 32, 128, step=16)
    hidden_2 = trial.suggest_int("hidden_2", 16, 64, step=16)

    model = LeakDetectionMLP(X_train_t.shape[1], hidden_1, hidden_2)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(30): 
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).squeeze().numpy()
            val_aucpr = average_precision_score(y_val_t.numpy(), val_probs)

        trial.report(val_aucpr, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_aucpr

def get_tuned_params_nn(fun_name):
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

    noise_number = np.sum(y_train == 0)
    leak_number = np.sum(y_train == 1)
    pos_weight = noise_number / leak_number if leak_number > 0 else 1.0

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val.values).unsqueeze(1)

    print("Rozpoczynam tuning hiperparametrów PyTorch (NN) za pomocą Optunay")
    
    study = optuna.create_study(direction="maximize") 
    study.optimize(lambda trial: objective_nn(trial, X_train_t, y_train_t, X_val_t, y_val_t, pos_weight), n_trials=50)

    best_params = study.best_params
    
    file_path = SIMULATION_CONFIG.output_folder / 'parametrem_tuning.txt'
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n\n--- pytorch nn Tuning: {fun_name} ---\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    
    return best_params

def run_nn(X_train, y_train, best_params, parallel=False):

    if parallel:
        torch.set_num_threads(1)

    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    input_features = X_train_scaled.shape[1] 

    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1) 

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    hidden_1 = best_params.get('hidden_1', 64)
    hidden_2 = best_params.get('hidden_2', 32)
    lr = best_params.get('lr', 0.001)
    weight_decay = best_params.get('weight_decay', 0.0001)
    pos_weight = best_params.get('pos_weight', 1.0)

    model_pt = LeakDetectionMLP(input_size=input_features, hidden_1=hidden_1, hidden_2=hidden_2)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = optim.Adam(model_pt.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 100
    model_pt.train() 

    for epoch in tqdm(range(epochs), desc="Epochs", position=2, leave=False, disable=parallel):
        for batch_X, batch_y in train_loader: 
            optimizer.zero_grad()
            predictions = model_pt(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    model_pt.eval()

    return model_pt, scaler

class SklearnPyTorchWrapper:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.model.eval() 

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).squeeze().numpy()

        return np.vstack((1 - probs, probs)).T

def auc_scorer(estimator, X, y):
    probs = estimator.predict_proba(X)[:, 1] 
    return roc_auc_score(y, probs)

def process_single_leak_all_nodes(leak_diameter, df_signals, metadata_cols, y, best_params):
    
    if pd.isna(leak_diameter):
        return []
        
    if leak_diameter == 'All':
        mask = pd.Series(True, index=df_signals.index) 
    else:
        mask = (df_signals['leak_diameter_parameter'] == leak_diameter) | (df_signals['leak_diameter_parameter'].isna())

    X_filtered = df_signals.loc[mask].copy().drop(columns=metadata_cols, errors='ignore')
    y_filtered = y.loc[mask].copy()

    train_scenarios, test_scenarios = train_test_split(
        df_signals.loc[mask, 'Scenario_Name'].unique(), 
        test_size=0.3, 
        random_state=42
    )

    train_keys = df_signals.loc[mask, 'Scenario_Name'].isin(train_scenarios)
    test_keys = df_signals.loc[mask, 'Scenario_Name'].isin(test_scenarios)

    X_train, y_train = X_filtered.loc[train_keys], y_filtered.loc[train_keys]
    X_test, y_test = X_filtered.loc[test_keys], y_filtered.loc[test_keys]

    noise_number = np.sum(y_train == 0)
    leak_number = np.sum(y_train == 1)
    class_weight = noise_number / leak_number if leak_number > 0 else 1.0

    local_params = best_params.copy()
    local_params['pos_weight'] = class_weight

    model_pt, scaler = run_nn(X_train, y_train, local_params, parallel=True)

    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled)

    with torch.no_grad():
        test_logits = model_pt(X_test_tensor)
        test_probs = torch.sigmoid(test_logits)
        probs = test_probs.squeeze().numpy()

    local_results = []

    fpr, tpr, thresholds_roc = roc_curve(y_test, probs)

    valid_thresholds = thresholds_roc[np.isfinite(thresholds_roc)]
    if len(valid_thresholds) > 100:
        idx = np.linspace(0, len(valid_thresholds) - 1, 100, dtype=int)
        decision_thresholds = valid_thresholds[idx]
    else:
        decision_thresholds = valid_thresholds
        
    decision_thresholds = np.unique(np.append(decision_thresholds, [0.0, 1.0]))


    # for decision_threshold in [round(x * 0.1, 1) for x in range(1, 10)]:
    for decision_threshold in decision_thresholds:
        preds = (probs >= decision_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

        local_results.append({
            'leak_diameter_parameter': leak_diameter,
            'decision_threshold': decision_threshold,
            'TP': int(tp), 
            'FP': int(fp), 
            'TN': int(tn), 
            'FN': int(fn)
        })
        
    return local_results

def nn_analysis_all_nodes():
    
    best_params = get_tuned_params_nn('nn_analysis_all_nodes')

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')
    y = df_signals['Is_Leak']

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']

    unique_leaks = [x for x in df_signals['leak_diameter_parameter'].unique() if pd.notna(x)]
    leak_diameter_parameters = unique_leaks + ['All']

    parallel_output = Parallel(n_jobs=4)(
        delayed(process_single_leak_all_nodes)(leak_diameter, df_signals, metadata_cols, y, best_params)
        for leak_diameter in tqdm(leak_diameter_parameters, desc="NN: Analiza wszystkich węzłów (Równolegle)")
    )

    results_list = []
    for local_results in parallel_output:
        if local_results:
            results_list.extend(local_results)

    pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_df_nn.pkl'
    results_df = pd.DataFrame(results_list)
    with open(pickle_output_confusion_matrix_df, 'wb') as file:
        pickle.dump(results_df, file)

def nn_analysis_best_nodes(): 
    
    best_params = get_tuned_params_nn('nn_analysis_best_nodes')

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')
    y = df_signals['Is_Leak']
    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']

    unique_leaks = [x for x in df_signals['leak_diameter_parameter'].unique() if pd.notna(x)]
    leak_diameter_parameters = unique_leaks + ['All']

    initial_nodes = [c for c in df_signals.columns if c not in metadata_cols]
    nodes_number = len(initial_nodes) 

    results_list = []
    importances_list = []

    for leak_diameter in tqdm(leak_diameter_parameters, desc="NN RFE: Średnice"):

        if pd.isna(leak_diameter):
            continue
            
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
        class_weight = noise_number / leak_number if leak_number > 0 else 1.0

        local_params = best_params.copy()
        local_params['pos_weight'] = class_weight

        unimportant_nodes_dropped = []

        for budget in tqdm(range(nodes_number, 0, -1), desc=f"Budgeting {leak_diameter}", leave=False):

            X_train = X_train_base.drop(columns=unimportant_nodes_dropped, errors='ignore')
            X_test = X_test_base.drop(columns=unimportant_nodes_dropped, errors='ignore')

            model_pt, scaler = run_nn(X_train, y_train, local_params, parallel=True)

            X_test_scaled = scaler.transform(X_test)
            X_test_tensor = torch.FloatTensor(X_test_scaled)

            with torch.no_grad():
                test_logits = model_pt(X_test_tensor)
                test_probs = torch.sigmoid(test_logits)
                probs = test_probs.squeeze().numpy()

            fpr, tpr, thresholds_roc = roc_curve(y_test, probs)
            opt_thresh = float(thresholds_roc[(tpr - fpr).argmax()]) if len(fpr) > 0 else 0.5

            wrapped_model = SklearnPyTorchWrapper(model_pt, scaler)
            perm_results = permutation_importance(
                estimator=wrapped_model,
                X=X_test, 
                y=y_test,
                scoring=auc_scorer,
                n_repeats=2,
                random_state=42,
                n_jobs=-1          
            )

            node_importance_df = pd.DataFrame({
                'leak_diameter_parameter': leak_diameter,
                'Nodes': X_train.columns,
                'Importance': perm_results.importances_mean,
                'optimal_decision_threshold': round(opt_thresh, 4),
                'budget': budget
            })
            importances_list.append(node_importance_df)
            
            least_node = node_importance_df.sort_values(by=['Importance', 'Nodes'], ascending=[True, True]).iloc[0]['Nodes']
            unimportant_nodes_dropped.append(least_node)

            valid_thresholds = thresholds_roc[np.isfinite(thresholds_roc)]
            if len(valid_thresholds) > 100:
                idx = np.linspace(0, len(valid_thresholds) - 1, 100, dtype=int)
                decision_thresholds = valid_thresholds[idx]
            else:
                decision_thresholds = valid_thresholds
                
            decision_thresholds = np.unique(np.append(decision_thresholds, [0.0, 1.0]))

            # for decision_threshold in [round(x * 0.1, 1) for x in range(1, 10)]:
            for decision_threshold in decision_thresholds:
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

    pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_best_nodes_df_nn.pkl'
    results_df = pd.DataFrame(results_list)
    with open(pickle_output_confusion_matrix_df, 'wb') as file:
        pickle.dump(results_df, file)

    if importances_list:
        importances_df = pd.concat(importances_list, ignore_index=True)
        pickle_output_nodes = SIMULATION_CONFIG.output_folder / 'pickle' / 'best_nodes_nn.pkl'
        with open(pickle_output_nodes, 'wb') as file:
            pickle.dump(importances_df, file)

def nn_analysis_global():
    
    best_params = get_tuned_params_nn('nn_analysis_global')

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
    class_weight = noise_number / leak_number if leak_number > 0 else 1.0
    
    local_params = best_params.copy()
    local_params['pos_weight'] = class_weight

    initial_nodes = X_train_global.columns.tolist()
    nodes_number = len(initial_nodes)

    results_list = []
    importances_list = []
    unimportant_nodes_dropped = []

    for budget in tqdm(range(nodes_number, 0, -1), desc="Iterowanie po budżetach (RFE Globalny NN)"):

        X_train = X_train_global.drop(columns=unimportant_nodes_dropped, errors='ignore')
        X_test = X_test_global.drop(columns=unimportant_nodes_dropped, errors='ignore')

        model_pt, scaler = run_nn(X_train, y_train_global, local_params, parallel=True)

        X_test_scaled = scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled)

        with torch.no_grad():
            test_logits = model_pt(X_test_tensor)
            test_probs = torch.sigmoid(test_logits)
            probabilities_global = test_probs.squeeze().numpy()

        metadata_test_global['Leak_Probability'] = probabilities_global
        metadata_test_global['True_Is_Leak'] = y_test_global

        fpr, tpr, thresholds_roc = roc_curve(
            metadata_test_global['True_Is_Leak'], metadata_test_global['Leak_Probability']
        )
        idx_optymalne = (tpr - fpr).argmax()
        optymalny_threshold = float(thresholds_roc[idx_optymalne])

        wrapped_model = SklearnPyTorchWrapper(model_pt, scaler)
        perm_results = permutation_importance(
            estimator=wrapped_model,
            X=X_test, 
            y=y_test_global,
            scoring=auc_scorer,
            n_repeats=2, 
            random_state=42,
            n_jobs=-1          
        )

        node_importance_df = pd.DataFrame({
            'leak_diameter_parameter': 'Global',
            'Nodes': X_train.columns,
            'Importance': perm_results.importances_mean,
            'optimal_decision_threshold': round(optymalny_threshold, 4),
            'budget': budget
        })
        importances_list.append(node_importance_df)
        
        least_important_node = node_importance_df.sort_values(by=['Importance', 'Nodes'], ascending=[True, True]).iloc[0]['Nodes']
        unimportant_nodes_dropped.append(least_important_node)

        # decision_thresholds = [round(x * 0.1, 1) for x in range(1, 10)]

        valid_thresholds = thresholds_roc[np.isfinite(thresholds_roc)]
        if len(valid_thresholds) > 100:
            idx = np.linspace(0, len(valid_thresholds) - 1, 100, dtype=int)
            decision_thresholds = valid_thresholds[idx]
        else:
            decision_thresholds = valid_thresholds
            
        decision_thresholds = np.unique(np.append(decision_thresholds, [0.0, 1.0]))

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

    pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_global_nn.pkl'
    results_df = pd.DataFrame(results_list)
    with open(pickle_output_confusion_matrix_df, 'wb') as file:
        pickle.dump(results_df, file)

    if importances_list:
        importances_df = pd.concat(importances_list, ignore_index=True)
        pickle_output_nodes = SIMULATION_CONFIG.output_folder / 'pickle' / 'top_nodes_global_nn.pkl'
        
        with open(pickle_output_nodes, 'wb') as file:
            pickle.dump(importances_df, file)


import time

# start_time = time.time()
# print('\n\nnn_analysis_parallel()\n\n')
# nn_analysis_all_nodes()
# t_all_nodes = time.time()
# print('\n\nNN_analysis_best_nodes()\n\n')
# nn_analysis_best_nodes()
# t_best_nodes = time.time()
# print('\n\nnn_analysis_global()\n\n')
# nn_analysis_global()          #11h
# t_global = time.time()

# print('t_all_nodes: ', t_all_nodes - t_global)
# print('t_best_nodes: ', t_best_nodes - t_all_nodes)
# print('t_global: ', t_global - start_time)
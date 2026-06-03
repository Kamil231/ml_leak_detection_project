import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.config import SIMULATION_CONFIG
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import pickle
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

class LeakDetectionMLP(nn.Module):
    def __init__(self, input_size):
        super(LeakDetectionMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)          
        )

    def forward(self, x):
        return self.network(x)

def run_nn(X_train, y_train, parallel = False):

	if parallel:
		torch.set_num_threads(1)

	scaler = StandardScaler()
	
	X_train_scaled = scaler.fit_transform(X_train)
	X_train_tensor = torch.FloatTensor(X_train_scaled)
	input_features = X_train_scaled.shape[1] 

	y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1) 

	train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
	train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

	model_pt = LeakDetectionMLP(input_size=input_features)

	criterion = nn.BCEWithLogitsLoss()

	optimizer = optim.Adam(model_pt.parameters(), lr=0.001, weight_decay=0.0001)

	epochs = 200
	model_pt.train() 

	for epoch in tqdm(range(epochs), desc="Epochs: ", position=2, leave=False, disable=parallel):
	    for batch_X, batch_y in tqdm(train_loader, desc="Batches", position=3, leave=False, disable=parallel):
	        
	        optimizer.zero_grad()
	        
	        predictions = model_pt(batch_X)
	        
	        loss = criterion(predictions, batch_y)
	        
	        loss.backward()
	        
	        optimizer.step()

	model_pt.eval()

	return model_pt, scaler

def nn_analysis():

	df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')

	metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
	metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]

	y = df_signals['Is_Leak']
	X = df_signals.drop(columns=metadata_cols, errors='ignore')

	leak_diameter_parameters = df_signals['leak_diameter_parameter'].unique().tolist()
	leak_diameter_parameters.append('All')

	results_list = []
	model_dict = {}

	for leak_diameter_parameter in tqdm(leak_diameter_parameters, desc="Iterating over leaks", position=1, leave=False):

	    if pd.isna(leak_diameter_parameter):
	        continue 
	    elif leak_diameter_parameter == 'All':
	        mask = pd.Series(True, index=df_signals.index) 
	    else:
	        mask = (df_signals['leak_diameter_parameter'] == leak_diameter_parameter) | (df_signals['leak_diameter_parameter'].isna())

	    X_filtered = X.loc[mask].copy()
	    y_filtered = y.loc[mask].copy()
	    metadata_filtered = metadata.loc[mask].copy()

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

	    nn_model, scaler = run_nn(X_train, y_train)
	    model_dict[leak_diameter_parameter] = {
	    	'model_state': nn_model, 
	    	'scaler': scaler
	    }
	    nn_model.eval()

	    #scaler = StandardScaler()

	    X_train_scaled = scaler.fit_transform(X_train)
	    input_features = X_train_scaled.shape[1] 

	    X_test_scaled = scaler.transform(X_test)
	    X_test_tensor = torch.FloatTensor(X_test_scaled)

	    with torch.no_grad():
	    	test_logits = nn_model(X_test_tensor)
	    	test_probs = torch.sigmoid(test_logits)
	    	probabilities = test_probs.squeeze().numpy()


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

	pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_df_nn.pkl'
	pth_nn_models_dict = SIMULATION_CONFIG.output_folder / 'pickle' / 'nn_models_dict.pth'
	csv_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'csv' / 'confusion_matrix_df_nn.csv'


	results_df = pd.DataFrame(results_list)

	with open(pickle_output_confusion_matrix_df, 'wb') as file:
		pickle.dump(results_df, file)

	torch.save(model_dict, pth_nn_models_dict)

	# results_df.to_csv(csv_output_confusion_matrix_df)

def process_single_leak(leak_diameter_parameter, df_signals, metadata, X, y):

    if pd.isna(leak_diameter_parameter):
        return [], None, None, None
    elif leak_diameter_parameter == 'All':
        mask = pd.Series(True, index=df_signals.index) 
    else:
        mask = (df_signals['leak_diameter_parameter'] == leak_diameter_parameter) | (df_signals['leak_diameter_parameter'].isna())

    X_filtered = X.loc[mask].copy()
    y_filtered = y.loc[mask].copy()
    metadata_filtered = metadata.loc[mask].copy()

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

    nn_model, fitted_scaler = run_nn(X_train, y_train, True)

    X_test_scaled = fitted_scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled)

    with torch.no_grad():
        test_logits = nn_model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits)
        probabilities = test_probs.squeeze().numpy()

    metadata_test['Leak_Probability'] = probabilities
    metadata_test['True_Is_Leak'] = y_test

    decision_thresholds = [round(x * 0.1, 1) for x in range(1, 10)]
    local_results = []

    for decision_threshold in decision_thresholds:
        metadata_test['Final_Prediction'] = (metadata_test['Leak_Probability'] >= decision_threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(
            metadata_test['True_Is_Leak'], 
            metadata_test['Final_Prediction']
        ).ravel()

        local_results.append({
            'leak_diameter_parameter': leak_diameter_parameter,
            'decision_threshold': decision_threshold,
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn)
        })

    return local_results, leak_diameter_parameter, nn_model.state_dict(), fitted_scaler

def nn_analysis_parallel():

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]

    y = df_signals['Is_Leak']
    X = df_signals.drop(columns=metadata_cols, errors='ignore')

    leak_diameter_parameters = df_signals['leak_diameter_parameter'].unique().tolist()
    leak_diameter_parameters.append('All')
    
    parallel_output = Parallel(n_jobs=4)(
        delayed(process_single_leak)(leak, df_signals, metadata, X, y)
        for leak in tqdm(leak_diameter_parameters, desc="Iterating over leaks")
    )

    results_list = []
    model_dict = {}

    for local_results, leak_param, model_state, scaler in parallel_output:
        if local_results: 
            results_list.extend(local_results)
            model_dict[leak_param] = {
                'model_state': model_state,
                'scaler': scaler
            }

    pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_df_nn.pkl'
    csv_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'csv' / 'confusion_matrix_df_nn.csv'
    pth_nn_models_dict = SIMULATION_CONFIG.output_folder / 'pickle' / 'nn_models_dict.pth'

    results_df = pd.DataFrame(results_list)

    with open(pickle_output_confusion_matrix_df, 'wb') as file:
        pickle.dump(results_df, file)

    # results_df.to_csv(csv_output_confusion_matrix_df, index=False)
    
    torch.save(model_dict, pth_nn_models_dict)

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

        import numpy as np
        return np.vstack((1 - probs, probs)).T

def auc_scorer(estimator, X, y):
    probs = estimator.predict_proba(X)[:, 1] 
    return roc_auc_score(y, probs)

def NN_pick_best_nodes():

	df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')
	pth_nn_models_dict = SIMULATION_CONFIG.output_folder / 'pickle' / 'nn_models_dict.pth'

	metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
	metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]

	y = df_signals['Is_Leak']
	X = df_signals.drop(columns=metadata_cols, errors='ignore')

	leak_diameter_parameters = df_signals['leak_diameter_parameter'].unique().tolist()
	leak_diameter_parameters.append('All')

	nn_model_dict = torch.load(pth_nn_models_dict, weights_only=False)

	results_list = []

	for leak_diameter_parameter in tqdm(leak_diameter_parameters, desc="Iterating over leaks", position=1, leave=False):

		if pd.isna(leak_diameter_parameter):
		    continue 
		elif leak_diameter_parameter == 'All':
		    mask = pd.Series(True, index=df_signals.index) 
		else:
		    mask = (df_signals['leak_diameter_parameter'] == leak_diameter_parameter) | (df_signals['leak_diameter_parameter'].isna())

		X_filtered = X.loc[mask].copy()
		y_filtered = y.loc[mask].copy()
		metadata_filtered = metadata.loc[mask].copy()

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

		model_state = nn_model_dict[leak_diameter_parameter]['model_state']
		scaler = nn_model_dict[leak_diameter_parameter]['scaler']

		input_size = len(X_test.columns)

		nn_model = LeakDetectionMLP(input_size=input_size)
		nn_model.load_state_dict(model_state)

		wrapped_model = SklearnPyTorchWrapper(nn_model, scaler)

		perm_results = permutation_importance(
		    estimator=wrapped_model,
		    X=X_test, 
		    y=y_test,
		    scoring=auc_scorer,
		    n_repeats=5,       
		    random_state=42,
		    n_jobs=-1          
		)

		local_df = pd.DataFrame({
			'leak_diameter_parameter': leak_diameter_parameter,
		    'Node': X_test.columns,
		    'Importance_Mean': perm_results.importances_mean,  
		    'Importance_Std': perm_results.importances_std     
		})
		
		results_list.append(local_df)

	importance_df = pd.concat(results_list, ignore_index=True)

	importance_df = importance_df.sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)

	pickle_output_nodes = SIMULATION_CONFIG.output_folder / 'pickle' / 'best_nodes_nn.pkl'
	csv_output_nodes = SIMULATION_CONFIG.output_folder / 'csv' / 'top_nodes_nn.csv'

	with open(pickle_output_nodes, 'wb') as file:
		pickle.dump(importance_df, file)

	# importance_df.to_csv(csv_output_nodes, index=False)

	return importance_df

def NN_analysis_best_nodes(): 

	importance_df = NN_pick_best_nodes()

	min_budget = 1
	max_budget = 20

	budget_list = list(range(min_budget, max_budget + 1))

	df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_ml_dataset.pkl')

	metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
	metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]

	leak_diameter_parameters = df_signals['leak_diameter_parameter'].unique().tolist()
	leak_diameter_parameters.append('All')

	results_list = []

	for budget in tqdm(budget_list, desc="Budgets: ", position=1, leave=False):

		top_nodes = importance_df['Node'].to_list()[:budget]

		y = df_signals['Is_Leak']
		X = df_signals.drop(columns=metadata_cols, errors='ignore')
		X = X[top_nodes]

		parallel_output = Parallel(n_jobs=4)(
		    delayed(process_single_leak)(leak, df_signals, metadata, X, y)
		    for leak in tqdm(leak_diameter_parameters, desc="Iterating over leaks")
		)

		for local_results, leak_param, model_state, scaler in parallel_output:
			if local_results: 
				for output_dict in local_results:
					output_dict['budget'] = budget
				results_list.extend(local_results)


	pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_best_nodes_df_nn.pkl'
	csv_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'csv' / 'confusion_matrix_df_nn_top_nodes.csv'

	results_df = pd.DataFrame(results_list)

	with open(pickle_output_confusion_matrix_df, 'wb') as file:
	    pickle.dump(results_df, file)

	# results_df.to_csv(csv_output_confusion_matrix_df, index=False)


# nn_analysis_parallel()
# NN_analysis_best_nodes()
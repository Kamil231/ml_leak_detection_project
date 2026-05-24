import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from src.config import SIMULATION_CONFIG
import wntr
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve
import pickle
from pprint import pprint

signal_leak_long = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals.pkl')
scenario_metadata = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'scenario_metadata.pkl')

def generate_bp_signals(seed_offset = 0):
    
    wn_base = SIMULATION_CONFIG.create_network_base()
    sim_base = wntr.sim.WNTRSimulator(wn_base)
    results_base = sim_base.run_sim()

    wn = SIMULATION_CONFIG.create_network_real(seed_offset)
    sim_real = wntr.sim.WNTRSimulator(wn)
    results_real = sim_real.run_sim()

    residuals_matrix = results_base.node['pressure'] - results_real.node['pressure']                    
    residuals_stacked = residuals_matrix.stack()
    scenario_name = f'blueprint_scenario_{seed_offset}'
    residuals_stacked.name = scenario_name

    signal_final = residuals_stacked.reset_index()
    signal_final.rename(columns={'level_0': 'T', 'level_1': 'Node'}, inplace=True) 

    signal_final = signal_final.pivot_table(
        index='T',
        columns='Node',
        values=scenario_name
    )

    signal_final.columns.name = None
    signal_final = signal_final.reset_index()
    signal_final['Scenario_Name'] = scenario_name

    old_cols = list(signal_final.columns)
    old_cols.remove('Scenario_Name')
    old_cols.remove('T')
    new_order = ['Scenario_Name', 'T'] + old_cols
    signal_final = signal_final[new_order]

    return signal_final

def get_signals_df():

    max_seed = 2000

    signal_leak_wide = signal_leak_long.melt(
        id_vars=['T', 'Node'], 
        var_name='Scenario_Name', 
        value_name='Signal_Value'
    )

    signal_leak_wide_with_meta = pd.merge(
        signal_leak_wide, 
        scenario_metadata, 
        on='Scenario_Name', 
        how='left'
    )

    signal_leak_wide_with_meta['leak_diameter_parameter'] = signal_leak_wide_with_meta['leak_diameter_parameter'].round(4)

    signal_leak_wide_with_meta = signal_leak_wide_with_meta[signal_leak_wide_with_meta['is_outlier'] == False]

    signal_leak_wide_final = signal_leak_wide_with_meta.pivot_table(
        index=[
            'Scenario_Name', 
            'leak_diameter_parameter', 
            'time_of_failure_h', 
            'leak_location', 
            'is_outlier',
            'T'
        ],
        columns='Node',
        values='Signal_Value'
    ).reset_index()

    signal_leak_wide_final.columns = [str(col) for col in signal_leak_wide_final.columns]

    signal_leak_wide_final['Is_Leak'] = (signal_leak_wide_final['T'] > (signal_leak_wide_final['time_of_failure_h'] * 3600)).astype(int)

    results_list = Parallel(n_jobs=-1)(
        delayed(generate_bp_signals)(seed) 
        for seed in tqdm(range(max_seed), desc="Parallel WNTR Base Simulations")
    )

    df_temp = pd.concat(results_list, axis=0, ignore_index=True)

    df_temp['Is_Leak'] = 0

    df_final = pd.concat([signal_leak_wide_final, df_temp], axis=0, ignore_index=True)

    df_final.to_csv(SIMULATION_CONFIG.output_folder / 'csv' / 'signals_dataset_XGB.csv')
    df_final.to_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_dataset_XGB.pkl')

    return df_final

def XGBoost_analysis():

    #get_signals_df()

    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_dataset_XGB.pkl')

    y = df_signals['Is_Leak']

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]
    print('df_signals[\'leak_diameter_parameter\'].unique(): ', df_signals['leak_diameter_parameter'].unique())

    leak_diameter_parameters = df_signals['leak_diameter_parameter'].unique().tolist()
    leak_diameter_parameters.append('All')

    results_list = []

    for leak_diameter_parameter in df_signals['leak_diameter_parameter'].unique():

        print('leak_diameter_parameter: ', leak_diameter_parameter, '\t', type(leak_diameter_parameter))

        # X = df_signals.loc[
        #     (df_signals['leak_diameter_parameter'] == leak_diameter_parameter) | 
        #     (df_signals['leak_diameter_parameter'].isna())
        # ]

        if pd.isna(leak_diameter_parameter):
            continue # Albo `mask = df_signals['leak_diameter_parameter'].isna()` jeśli chcesz modelować same zera/NaN
        elif leak_diameter_parameter == 'All':
            mask = pd.Series(True, index=df_signals.index) # Bierzemy wszystko
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
            eval_metric='logloss'
        )

        model_xgb.fit(X_train, y_train)

        probabilities = model_xgb.predict_proba(X_test)[:, 1]

        metadata_test['Leak_Probability'] = probabilities
        metadata_test['True_Is_Leak'] = y_test

        col_names = ["TP", "FP", "TN", "FN"]

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

            # waznosc = pd.DataFrame({
            #     'Node': X_train.columns,
            #     'Importance': model_xgb.feature_importances_
            # }).sort_values(by='Importance', ascending=False)

            # top_nodes = waznosc.head(len(SIMULATION_CONFIG.scenarios.sensor_budget))['Node'].tolist()

            # top_nodes_path = SIMULATION_CONFIG.output_folder / 'pickle' / 'top_nodes_xgb.pkl'

            # nodes_threshold_list = []

            # for node in top_nodes:
            #     fpr, tpr, thresholds = roc_curve(y_train, X_train[node])
                
            #     idx_optymalne = (tpr - fpr).argmax()
            #     optymalny_threshold = thresholds[idx_optymalne].item()

            #     nodes_threshold_list.append((node, optymalny_threshold))


    # pickle_output_path_report_thd_dict = SIMULATION_CONFIG.output_folder / 'pickle' / 'classification_report.pkl'
    pickle_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'pickle' / 'confusion_matrix_df.pkl'
    csv_output_confusion_matrix_df = SIMULATION_CONFIG.output_folder / 'csv' / 'confusion_matrix_df.csv'

    # with open(pickle_output_path_report_thd_dict, 'wb') as file:
    #     pickle.dump(report_thd_dict, file)

    results_df = pd.DataFrame(results_list)

    print(results_df.shape, '\n\n\n')
    print(results_df)

    with open(pickle_output_confusion_matrix_df, 'wb') as file:
        pickle.dump(results_df, file)

    results_df.to_csv(csv_output_confusion_matrix_df)

XGBoost_analysis()

# signals = get_signals_df()
# print(signals)

# bp_singals = generate_bp_signals()
# print(bp_singals)

# signals_df = get_signals_df()
# print(signals_df)

# df_final = pd.concat([signals_df, bp_singals], axis=0, ignore_index=True)

# df_final.to_csv(path_csv+'/signal_leak_wide_final_concate_bp.csv')

# print(df_final)
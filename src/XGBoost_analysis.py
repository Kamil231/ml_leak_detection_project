import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
import pandas as pd
import numpy as np
from src.config import SIMULATION_CONFIG
import wntr
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import roc_curve


path_str = r"/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/code/leak_simulation/output_folder/pickle"
path_csv = r"/Users/kamilzawitaj/Documents/Studia/PW OKNO - AiR/Praca Mgr/code/leak_simulation/output_folder/csv"

signal_leak_long = pd.read_pickle(path_str + '/signals.pkl')
scenario_metadata = pd.read_pickle(path_str + '/scenario_metadata.pkl')

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

    # df_temp = pd.DataFrame() #generate_bp_signals(0)

    # for seed in tqdm(range(max_seed), desc="WNTR Base Simulations"): 
    #     df_bp = generate_bp_signals(seed)
    #     df_temp = pd.concat([df_temp, df_bp], axis=0, ignore_index=True)

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

    #df_signals = get_signals_df()
    df_signals = pd.read_pickle(SIMULATION_CONFIG.output_folder / 'pickle' / 'signals_dataset_XGB.pkl')

    y = df_signals['Is_Leak']

    metadata_cols = ['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T', 'Is_Leak']
    X = df_signals.drop(columns=metadata_cols, errors='ignore')

    metadata = df_signals[['Scenario_Name', 'leak_diameter_parameter', 'time_of_failure_h', 'leak_location', 'is_outlier', 'T']]

    unique_scenarios = metadata['Scenario_Name'].unique()

    train_scenarios, test_scenarios = train_test_split(
        unique_scenarios, 
        test_size=0.3,         
        random_state=42        
    )

    train_keys = metadata['Scenario_Name'].isin(train_scenarios)
    test_keys = metadata['Scenario_Name'].isin(test_scenarios)

    X_train, y_train = X[train_keys], y[train_keys]
    X_test, y_test = X[test_keys], y[test_keys]
    metadata_test = metadata[test_keys].copy() 

    noise_number = np.sum(y_train == 0)
    leak_number = np.sum(y_train == 1)
    class_weight = noise_number / leak_number

    print('noise_number: ', noise_number)
    print('leak_number: ', leak_number)

    # Inicjalizacja klasyfikatora
    model_xgb = xgb.XGBClassifier(
        n_estimators=100,         # liczba drzew decyzyjnych
        max_depth=5,              # głębokość drzewa (zapobiega overfitingowi)
        learning_rate=0.1,        # szybkość uczenia
        scale_pos_weight=class_weight,# ratuje nas przed niezbalansowanym zbiorem danych
        random_state=42,
        eval_metric='logloss'
    )

    model_xgb.fit(X_train, y_train)

    # czy jest wyciek? - prawdopodob
    probabilities = model_xgb.predict_proba(X_test)[:, 1]

    metadata_test['Leak_Probability'] = probabilities
    metadata_test['True_Is_Leak'] = y_test

    prog_decyzyjny = 0.5
    metadata_test['Final_Prediction'] = (metadata_test['Leak_Probability'] >= prog_decyzyjny).astype(int)

    print("XGBoost wyniki:")
    print(classification_report(metadata_test['True_Is_Leak'], metadata_test['Final_Prediction']))

    waznosc = pd.DataFrame({
        'Node': X_train.columns,
        'Importance': model_xgb.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    top_nodes = waznosc.head(20)['Node'].tolist()
    print("Najważniejsze węzły wybrane przez XGBoost:", top_nodes)

    for node in top_nodes:
        fpr, tpr, thresholds = roc_curve(y_train, X_train[node])
        
        idx_optymalne = (tpr - fpr).argmax()
        optymalny_threshold = thresholds[idx_optymalne]
        
        print(f"Dla Węzła [{node}] optymalny próg detekcji wynosi: {optymalny_threshold:.4f}")

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
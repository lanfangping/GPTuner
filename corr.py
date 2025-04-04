from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
import shap
import pickle
import numpy as np

def test():
    file_path = "../DBtuningDataset/reflexion_memory/memory_tpcc_sf20_t10_newflow_newimp_SR10_M8_Binary_IS1_TP8_IN0__202412032307.json"
    # file_path = "optimization_results/postgres_sf20_t10/fine/100/runhistory.json"
    tuning_hisory = json.load(open(file_path, 'r'))

    performances = []
    config_dicts = []
    for _, data in tuning_hisory['recent_settings_history'].items():
        if data['settings'] == 'default settings':
            continue
        performances.append(data['performance'])
        config_dicts.append(data['settings'])

    # # Extract tuning history
    # performances =[ -d[4] for d in tuning_hisory['data']]  # Raw list of knob values + performance
    # config_dicts =[config for _, config in tuning_hisory['configs'].items()]  # Knob metadata

    # transfered_configs = []
    # for config in config_dicts:
    #     temp_config = {}
    #     for knob, value in config.items():
    #         # print('knob', knob)
    #         if knob.startswith('control_'):
    #             knob = knob[8:]
    #             # print(knob)
    #             if value == '0':
    #                 continue
    #             elif value == '1':
    #                 temp_config[knob] = config[f'special_{knob}']
    #         elif knob.startswith('special_'):
    #             continue
    #         else:
    #             temp_config[knob] = value
    #     transfered_configs.append(temp_config)

    # df = pd.DataFrame(transfered_configs)
    df = pd.DataFrame(config_dicts)
    df["performance"] = performances  # Add performance metric

    df.to_json("temp_files/tuning_history_our.jsonl", orient="records", lines=True)
    print(df.head())

    # exit()
    # Separate features (X) and target (y)
    X = df.drop(columns=["performance"])  # Knob configurations
    for col in X.columns:
        if X[col].dtype == "O":  # If column is categorical
            X[col] = LabelEncoder().fit_transform(X[col])

    print(X.head())
    X.to_json("temp_files/tuning_history_dummies_our.json")
    y = df["performance"]  # Performance metric

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    pickle.dump(model, open("temp_files/model_our.pkl", 'wb'))

    explainer = shap.Explainer(model, X)  # Initialize SHAP explainer
    shap_values = explainer(X)  # Compute SHAP values

    # shap.summary_plot(shap_values, X)
    # sample_ind = 0  # Choose the first tuning configuration
    # shap.plots.waterfall(shap_values[sample_ind])
    # shap.dependence_plot("shared_buffers", shap_values.values, X)
    # Compute mean absolute SHAP values for each feature
    importance = pd.DataFrame({
        "knob": X.columns,
        "importance": shap_values.abs.mean(0).values
    })
    importance = importance.sort_values(by="importance", ascending=False)

    print(importance)
    print(type(importance))
    importance.to_json("temp_files/importance_our.json")


# Initialize an empty model
model = RandomForestRegressor(n_estimators=10, warm_start=True, random_state=42)

# Store all tuning history
X_history = pd.DataFrame()  # DataFrame to store knob values
y_history = []
encoders = {}  # Store LabelEncoders for categorical columns

def iteratively_corr():

    def update_model(new_X_dict, new_y):
        """Update the model incrementally with new tuning data, handling new knobs dynamically."""
        global X_history, y_history, model, encoders
        
        # Convert new_X_dict (knob dictionary) to DataFrame
        new_X_df = pd.DataFrame([new_X_dict])  # Convert dict to DataFrame with one row

        if not X_history.empty:
            # Find missing columns in X_history
            missing_cols = set(new_X_df.columns) - set(X_history.columns)
            if missing_cols:
                missing_df = pd.DataFrame(np.nan, index=X_history.index, columns=list(missing_cols))
                X_history = pd.concat([X_history, missing_df], axis=1)  # Add missing columns in one operation

            # Find missing columns in new_X_df
            missing_cols_new = set(X_history.columns) - set(new_X_df.columns)
            if missing_cols_new:
                missing_df_new = pd.DataFrame(np.nan, index=new_X_df.index, columns=list(missing_cols_new))
                new_X_df = pd.concat([new_X_df, missing_df_new], axis=1)  # Add missing columns in one operation

        # Append new data to history
        X_history = pd.concat([X_history, new_X_df], ignore_index=True)

        # Identify categorical columns (non-numeric)
        categorical_cols = [col for col in X_history.columns if X_history[col].dtype == "O" or X_history[col].isna().any()]

        # Convert categorical columns to string type
        for col in categorical_cols:
            X_history[col] = X_history[col].astype(str)

        # Apply Label Encoding to categorical variables
        for col in categorical_cols:
            if col not in encoders:
                encoders[col] = LabelEncoder()
                X_history[col] = encoders[col].fit_transform(X_history[col])
            else:
                unseen_values = set(X_history[col].unique()) - set(encoders[col].classes_)
                if unseen_values:
                    encoders[col].classes_ = np.append(encoders[col].classes_, list(unseen_values))
                X_history[col] = encoders[col].transform(X_history[col])

        # Fill NaN values with 0
        X_history.fillna(0, inplace=True)

        # Append new performance value
        y_history.append(new_y)

        # Ensure model is trained with all past data
        X_train = X_history.to_numpy()
        y_train = np.array(y_history)

        # Incrementally retrain model
        model.n_estimators += 5  # Increment estimators for better learning
        model.fit(X_train, y_train)

        # Compute SHAP values
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        return shap_values
    
    file_path = "../DBtuningDataset/reflexion_memory/memory_tpcc_sf20_t10_newflow_newimp_SR10_M8_Binary_IS1_TP8_IN0__202412032307.json"
    tuning_hisory = json.load(open(file_path, 'r'))

    for _, data in tuning_hisory['recent_settings_history'].items():
        if data['settings'] == 'default settings':
            continue
        
        new_X = data['settings']
        new_Y = data['performance']

        shap_values = update_model(new_X, new_Y)

        importance = pd.DataFrame({
            "knob": X_history.columns,
            "importance": shap_values.abs.mean(0).values
        })
        importance = importance.sort_values(by="importance", ascending=False)

        print(importance.head(50))
        print("==========================\n")
        input()

if __name__ == '__main__':
    iteratively_corr()
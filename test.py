### Start user input ###
path_to_zips = [r'C:\Users\slcup\Documents\Aerospace Engineering\Minor\Capstone\Capstone data\Data']
### End user input ###

# Libary imports
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
device = 'cpu'
from sklearn.model_selection import train_test_split

# Function imports
from advanced_dataloading import process_folder
from advanced_preprocessing import frame_waves, valid_velo_data
from models import load_scalers, load_models, LSTMModel, GRUModel

# Load the models and scalers
gru1, gru2, lstm = load_models()
gru1.eval(), gru2.eval(), lstm.eval()
feature_scaler, target_scaler, feature_scaler2, target_scaler2 = load_scalers()

# Loop over all zips
for file_path in path_to_zips:
    # Load and preprocess the input
    df = process_folder(file_path, plot=True, labels=True)
    X_gru1 = frame_waves(df['VoltageOut'], length=150, jump=0)[0]
    X_gru1_scaled = torch.tensor(feature_scaler.transform(X_gru1)[...,np.newaxis], dtype=torch.float32)   
    X_gru2 = frame_waves(df['VoltageOut'], length=150, jump=0)[0]
    X_gru2_scaled = torch.tensor(feature_scaler.transform(X_gru2)[...,np.newaxis], dtype=torch.float32)
    X_lstm = frame_waves(df['VoltageOut'], length=150, jump=0)[0]
    X_lstm_scaled = torch.tensor(feature_scaler.transform(X_lstm)[...,np.newaxis], dtype=torch.float32)

    # Make predictions
    with torch.no_grad():  
        y_gru1_scaled = gru1(X_gru1_scaled)
        y_gru2_scaled = gru2(X_gru2_scaled)
        y_lstm_scaled = lstm(X_lstm_scaled)
    y_gru1 = target_scaler.inverse_transform(y_gru1_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_gru2 = target_scaler.inverse_transform(y_gru2_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()
    y_lstm = target_scaler.inverse_transform(y_lstm_scaled.detach().cpu().numpy().reshape(-1, 1)).flatten()

    y_pred = ((y_lstm+y_gru1+y_gru2)/3).flatten()
    outcome_df = pd.DataFrame({"predictions model 1": y_gru1, "predictions model 2": y_gru2, "predictions model 3": y_lstm, "model prediction": y_pred})
    outcome_df['Standard deviation'] = outcome_df[["predictions model 1", "predictions model 2", "predictions model 3"]].std(axis=1)
    outcome_df['Standard deviation %'] = outcome_df['Standard deviation'] / outcome_df['model prediction'] * 100
 
print(outcome_df.head(10))

# Evaluation metrics
#valid_bubbles = len(outcome_df[outcome_df['Standard deviation %'] < 10])/len(outcome_df) * 100
#print(f'Percentage of valid bubbles: {valid_bubbles}%')

def test_eval(X_test, y_test, model, feature_scaler, target_scaler, n_bins=10, plot_hist=True):
    
    model.eval()
    X_test = feature_scaler.transform(X_test)
    X_test = torch.tensor(X_test[...,np.newaxis], dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    model.to(device)

    with torch.no_grad():
        predictions = model(X_test)
    # Returning to original scale
    predictions = target_scaler.inverse_transform(predictions.detach().cpu().numpy().reshape(-1, 1)).flatten()
    # Getting y_test to the cpu
    y_test = y_test.detach().cpu().numpy()
    mae = torch.mean(torch.abs(torch.tensor(predictions, dtype=torch.float32) - y_test))
    print(f"Test Set Mean Absolute Error: {mae.item():.4f}")
    print(f"Test set Mean: {np.mean(y_test)}")
    print(f"Average deviation: {mae.item() / np.mean(y_test) * 100}%")

    if plot_hist:
        # Plotting a histogram
        min_value = min(predictions.min().item(), y_test.min().item())
        max_value = max(predictions.max().item(), y_test.max().item())
        bins = np.linspace(min_value, max_value, n_bins)
        plt.hist(predictions.flatten(), bins=bins, alpha=0.5, edgecolor="black", label="prediction")
        plt.hist(y_test, bins=bins, alpha=0.5, edgecolor="black", label="ground truth")
        plt.xlabel("velocity")
        plt.legend()
        plt.show()

    predictions = predictions.flatten()
    y_test = y_test.flatten()
    return pd.DataFrame({"predictions": predictions, 
                         "true value": y_test,
                        "deviation (%)": np.abs(y_test - predictions)/predictions * 100 })

def avg_results_test(model_1, model_2, model_3, X_test, y_test, feature_scaler, target_scaler, 
                plot_hist=False, n_bins=20):
    """Plots the final results as an average of three models.
    
    Args:
        model_1, model_2, model_3: three different and trained models
        X_test: The data you want the models to make predictions of
        y_test: The true values
        feature_scaler: Scaler used to scale X_train of the models
        target_scaler: Scaler used to scale y_train of the models
        plot_hist: Will plot a histogram if set to True
        n_bins: Amount of bins in the histogram

    Output:
        dataframe with predictions of 3 models, final prediction, true prediction and standard deviation
    """
    
    results_1 = test_eval(X_test, y_test, model_1, feature_scaler, target_scaler, plot_hist=False)
    results_2 = test_eval(X_test, y_test, model_2, feature_scaler, target_scaler, plot_hist=False)
    results_3 = test_eval(X_test, y_test, model_3, feature_scaler, target_scaler, plot_hist=False)

    results_1 = results_1.rename(columns={"predictions": "predictions model 1"})
    results_2 = results_2.rename(columns={"predictions": "predictions model 2"})
    results_3 = results_3.rename(columns={"predictions": "predictions model 3"})
    
    # Combining the dataframes into one big dataframe
    combined_df = results_1.merge(results_2, on="true value").merge(results_3, on="true value")
    
    # Final prediction is the mean of LSTM, GRU and TCN
    combined_df["final prediction"] = combined_df[["predictions model 1", 
                                                   "predictions model 2", 
                                                   "predictions model 3"]].mean(axis=1)
    
    # Calculate the standard deviation over predictions of LSTM, GRU, and TCN
    combined_df["standard deviation"] = combined_df[["predictions model 1", "predictions model 2", "predictions model 3"]].std(axis=1)
    
    # Calculate the final deviation (%)
    combined_df["final deviation (%)"] = (abs(combined_df["final prediction"] - combined_df["true value"]) / combined_df["true value"]) * 100
    
    # Select the desired columns
    final_df = combined_df[[
        "predictions model 1",
        "predictions model 2",
        "predictions model 3",
        "final prediction",
        "true value",
        "standard deviation",
        "final deviation (%)"
    ]]

    if plot_hist:
        # Plotting a histogram
        predictions = final_df["final prediction"].tolist()
        y_test = final_df["true value"].tolist()
        min_value = min(min(predictions), min(y_test))
        max_value = max(max(predictions), max(y_test))
        bins = np.linspace(min_value, max_value, n_bins)
        plt.hist(predictions, bins=bins, alpha=0.5, edgecolor="black", label="prediction")
        plt.hist(y_test, bins=bins, alpha=0.5, edgecolor="black", label="ground truth")
        plt.xlabel("velocity")
        plt.legend()
        plt.show()
    
    return final_df

X, y = valid_velo_data(df)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

X_train_val, y_train_val = frame_waves(X_train_val, length=150, n_crops=2, jump=0, labels=y_train_val)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.67, random_state=0)
X_test1, y_test1 = frame_waves(X_test, labels=y_test, n_crops=1, length=150, jump=0)
results = avg_results_test(gru1, lstm, gru2, 
                      X_test1, y_test1, feature_scaler, target_scaler, plot_hist=True)
valid_test_results = results[(results["standard deviation"]/results["final prediction"]) <= 0.1]
print("Percentage found bubbles (<10% deviation from truth): ", len(valid_test_results) / (len(results)) * 100, "%")

uncertainty_threshold = 10
certain_valid_results = valid_test_results[valid_test_results["final deviation (%)"] < uncertainty_threshold]
print(f"From those found bubbles, {len(certain_valid_results) / len(valid_test_results) * 100}% are valid (<{uncertainty_threshold}% standard deviation on final prediction)")


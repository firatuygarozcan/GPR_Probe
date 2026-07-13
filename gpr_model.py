import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pandapower.topology as top
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler


def prepare_topology_data(net, all_buses, measured_buses):
    """
    Extracts the grid topology using NetworkX, calculates the shortest path
    distance matrix, and prepares the scaled training/testing sets for GPR.
    """
    print("--> GPR: Extracting grid topology (Graph) from Pandapower...")

    # Extracting the grid topology (respecting open switches as dead ends)
    mg = top.create_nxgraph(net, respect_switches=True)

    # Sort all buses by their index
    all_buses_sorted = all_buses.sort_index()
    all_indices = all_buses_sorted.index.tolist()
    N = len(all_indices)

    print("--> GPR: Creating the physical distance matrix (Shortest Path)...")
    # What is the shortest path between bus x and bus j
    # Output: Distance matrix. Input for Kernel function
    dist_matrix = np.zeros((N, N))
    for i, source in enumerate(all_indices):
        for j, target in enumerate(all_indices):
            try:
                dist_matrix[i, j] = nx.shortest_path_length(mg, source=source, target=target)
            except nx.NetworkXNoPath:
                # High distance for islanded/disconnected grid parts
                dist_matrix[i, j] = 100

    print("--> GPR: Preparing input features (X) and scaling...")
    # AI sees the value of 30% of values
    train_idx_positions = [all_indices.index(idx) for idx in measured_buses.index]

    X_train_raw = dist_matrix[train_idx_positions]
    y_train_raw = measured_buses['vm_pu'].values.reshape(-1, 1)

    X_test_raw = dist_matrix
    y_true = all_buses_sorted['vm_pu'].values

    # Scaling the data (Mandatory for distance matrices)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train_raw)
    y_train = scaler_y.fit_transform(y_train_raw)
    X_test = scaler_X.transform(X_test_raw)

    return X_train, y_train, X_test, y_true, scaler_y, all_buses_sorted


def train_and_predict_gpr(X_train, y_train, X_test, scaler_y):
    """
    Initializes the GPR model with Matern Kernel, trains it, and returns
    the inverse-transformed real voltage predictions and uncertainties.
    """
    print("--> GPR: Initializing and training the model...")
    # Using Matern Kernel (nu=1.5) combined with WhiteKernel for noise handling
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-3)

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, random_state=42)
    gpr.fit(X_train, y_train)

    print("--> GPR: Predicting unmeasured nodes...")
    y_pred_scaled, sigma_scaled = gpr.predict(X_test, return_std=True)

    # Inverse transform to real p.u. values
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    sigma = sigma_scaled * scaler_y.scale_[0]

    print("--> GPR: Topology-Aware GPR Training Completed Successfully!")
    return y_pred, sigma


def evaluate_and_visualize(all_buses_sorted, measured_buses, y_true, y_pred, sigma, title='Topology-Aware Probabilistic State Estimation via GPR'):
    """
    Plots the predictions vs ground truth and prints the tabular results
    for the unmeasured buses to evaluate model performance.
    """
    # print("\n" + "=" * 60)
    # print(" COMPARISON: ACTUAL vs PREDICTED VOLTAGE (UNMEASURED BUSES)")
    # print("=" * 60)

    # 1. TABULAR RESULTS
    results_df = pd.DataFrame({
        'Bus_ID': all_buses_sorted.index,
        'Actual_V_pu': y_true,
        'Predicted_V_pu': y_pred,
        'Uncertainty_Std': sigma
    })

    unmeasured_mask = ~results_df['Bus_ID'].isin(measured_buses.index)
    unseen_results = results_df[unmeasured_mask].copy()

    unseen_results['Abs_Error'] = abs(unseen_results['Actual_V_pu'] - unseen_results['Predicted_V_pu'])
    unseen_results['Error_%'] = (unseen_results['Abs_Error'] / unseen_results['Actual_V_pu']) * 100

    formatted_results = unseen_results.copy()
    formatted_results['Actual_V_pu'] = formatted_results['Actual_V_pu'].apply(lambda x: f"{x:.4f}")
    formatted_results['Predicted_V_pu'] = formatted_results['Predicted_V_pu'].apply(lambda x: f"{x:.4f}")
    formatted_results['Uncertainty_Std'] = formatted_results['Uncertainty_Std'].apply(lambda x: f"±{x:.4f}")
    formatted_results['Abs_Error'] = formatted_results['Abs_Error'].apply(lambda x: f"{x:.4f}")
    formatted_results['Error_%'] = formatted_results['Error_%'].apply(lambda x: f"{x:.2f}%")

    # Print the tabular results to the console
    # print(formatted_results[
    #          ['Bus_ID', 'Actual_V_pu', 'Predicted_V_pu', 'Abs_Error', 'Error_%', 'Uncertainty_Std']].to_string(
    #    index=False))

    # Create a dynamic file name based on the plot title
    file_name = f"{title.replace(' ', '_').replace('/', '_')}_Results.xlsx"

    # Save the formatted tabular results to a CSV file in the project folder
    formatted_results[['Bus_ID', 'Actual_V_pu', 'Predicted_V_pu', 'Abs_Error', 'Error_%', 'Uncertainty_Std']].to_excel(file_name, index=False)

    print(f"--> SYSTEM INFO: Tabular results successfully saved to '{file_name}'!")
    # =============================================================

    # Calculate and print overall metrics
    mean_absolute_error = unseen_results['Abs_Error'].mean()
    rmse = np.sqrt((unseen_results['Abs_Error'] ** 2).mean())

    print("-" * 60)
    print(f"Overall MAE (Mean Absolute Error)  on Unseen Buses: {mean_absolute_error:.6f} p.u.")
    print(f"Overall RMSE (Root Mean Square)    on Unseen Buses: {rmse:.6f} p.u.")
    print("=" * 60)

    # 2. VISUALIZATION
    plt.figure(figsize=(12, 6))
    sequential_x = np.arange(len(all_buses_sorted))
    train_x_seq = [np.where(all_buses_sorted.index == bus_id)[0][0] for bus_id in measured_buses.index]
    y_train_plot = measured_buses['vm_pu'].values

    plt.plot(sequential_x, y_true, 'k.--', label='Actual Voltage (Pandapower Ground Truth)', linewidth=1.2,
             markersize=8)
    plt.plot(sequential_x, y_pred, 'b-', label='Topology-Aware GPR Prediction (Mean)', linewidth=2)
    plt.fill_between(sequential_x, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='blue',
                     label='95% Confidence Interval (Uncertainty)')
    plt.scatter(train_x_seq, y_train_plot, c='red', s=60, zorder=5, label='Measured Buses (30% Smart Meters)')

    plt.xticks(sequential_x, all_buses_sorted.index, rotation=90, fontsize=8)
    plt.title(title, fontsize=14)
    plt.xlabel('Real Bus Index (Node ID)', fontsize=12)
    plt.ylabel('Voltage Magnitude (p.u.)', fontsize=12)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
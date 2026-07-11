import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import simbench as sb
import pandapower as pp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import pandapower.topology as top
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import pandas as pd
import pandapower.plotting.plotly as pplotly


# ==========================================
# 1. DOWNLOAD THE NETWORK
# ==========================================

sb_code = "1-LV-semiurb4--0-sw"
print(f"Downloading network with code: '{sb_code}'...\n")
net = sb.get_simbench_net(sb_code)

print("Creating interactive grid map...")
pplotly.simple_plotly(net)

plt.show()


# ==========================================
# 2. POWER FLOW ANALYSIS (LV GRID MODE)
# ==========================================

print("Starting Power Flow calculation...")

# algorithm="nr": Standard algorithm
# check_connectivity=True: Isolates open switches and dead zones
# voltage_depend_loads=False: Prevents the NaN/Inf math crash
# calculate_voltage_angles=False: Prevents divergence in high-resistance cables
pp.runpp(net,
         algorithm="nr",
         check_connectivity=True,
         voltage_depend_loads=False,
         calculate_voltage_angles=False)

print("Calculation completed successfully!\n")


# ==========================================
# 3. DISPLAYING THE RESULTS
# ==========================================

print("--- BUS VOLTAGE RESULTS (First 5 Buses) ---")
print(net.res_bus[['vm_pu', 'va_degree']].head())
print("-" * 50)

print("\n--- LINE LOADING RESULTS (First 5 Lines) ---")
print(net.res_line[['loading_percent', 'i_ka']].head())
print("-" * 50)


# ==========================================
# 4. SUBSET PREPERATION FOR STATE ESTIMATION
# ==========================================

print("\n------- STATE ESTIMATION: BUS DATA -------")

# Results of all buses in a variable
all_buses = net.res_bus[['vm_pu', 'va_degree']]

# Choose the random 30% (frac=0,3) with the "sample" feature of pandas
# random_state=42 parameter makes sure to choose the same 30% whenever we run the code.
measured_buses = all_buses.sample(frac=0.30, random_state=42)

print(f"Number of total buses in the network: {len(all_buses)}")

print("--- ALL NODES IN NETWORK ---")
print(net.bus[['name', 'vn_kv']])

print(f"Number of measured Buses (%30): {len(measured_buses)}\n")

print("--- MEASURED BUSES AND VALUES ---")
print(measured_buses)
print("-" * 50)


# ==========================================
# 5. TOPOLOGY-AWARE GPR (GRAPH EMBEDDING)
# ==========================================

print("\n--- STARTING TOPOLOGY-AWARE GPR TRAINING (NETWORKX) ---")

# 1. EXTRACTING THE GRID TOPOLOGY (GRAPH) FROM PANDAPOWER
# The 'respect_switches=True' parameter is crucial! It ensures the algorithm
# recognizes open switches and disconnected lines, treating them as dead ends.
mg = top.create_nxgraph(net, respect_switches=True)

# We must sort all buses by their index before extracting the list
all_buses_sorted = all_buses.sort_index()

# Get the list of all actual bus IDs (indices) in the grid
all_indices = all_buses_sorted.index.tolist()
N = len(all_indices)

# 2. CREATING THE PHYSICAL DISTANCE MATRIX (SHORTEST PATH)
# This matrix will store how many "hops" (cables) it takes to travel from any bus to any other bus.
dist_matrix = np.zeros((N, N))

for i, source in enumerate(all_indices):
    for j, target in enumerate(all_indices):
        try:
            # Calculate the shortest physical path (number of edges) between two buses
            dist_matrix[i, j] = nx.shortest_path_length(mg, source=source, target=target)
        except nx.NetworkXNoPath:
            # If there is no physical connection (e.g., an islanded grid part), set a very high distance
            dist_matrix[i, j] = 100

# 3. PREPARING THE INPUT FEATURES (X) FOR GPR
# CRITICAL DIFFERENCE: X is no longer a single number (Bus ID).
# Every bus is now represented by an entire row from the distance matrix
# (its spatial relationship to all other buses in the grid).
train_idx_positions = [all_indices.index(idx) for idx in measured_buses.index]

X_train_raw = dist_matrix[train_idx_positions] # Physical distance vectors of measured buses only
y_train_raw = measured_buses['vm_pu'].values.reshape(-1, 1)

X_test_raw = dist_matrix # Physical distance vectors of all buses in the grid
y_true = all_buses_sorted['vm_pu'].values

# Scaling the data (Mandatory for GPR when dealing with distance matrices)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
y_train = scaler_y.fit_transform(y_train_raw)
X_test = scaler_X.transform(X_test_raw)

# 4. INITIALIZING AND TRAINING THE MODEL
# We use the Matern Kernel, which is excellent at processing these multi-dimensional topology vectors
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-3)

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, random_state=42)
gpr.fit(X_train, y_train)

# 5. PREDICTION (Topology-Aware Results)
y_pred_scaled, sigma_scaled = gpr.predict(X_test, return_std=True)

# Inverse transform the scaled results back to real voltage values (p.u.)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
sigma = sigma_scaled * scaler_y.scale_[0]

print("Topology-Aware GPR Training Completed Successfully!")


# ==========================================
# 6. VISUALIZATION OF THE TOPOLOGY-AWARE RESULTS
# ==========================================

plt.figure(figsize=(12, 6))

# We create a sequential array (0 to N) for a clean, gapless X-axis plot
sequential_x = np.arange(len(all_buses_sorted))

# Find the sequential positions of our training (measured) buses
train_x_seq = [np.where(all_buses_sorted.index == bus_id)[0][0] for bus_id in measured_buses.index]
y_train_plot = measured_buses['vm_pu'].values

# Plot the Ground Truth (Black Dashed Line with Dots)
plt.plot(sequential_x, y_true, 'k.--', label='Actual Voltage (Pandapower Ground Truth)', linewidth=1.2, markersize=8)

# Plot the GPR Prediction Mean (Blue Line)
plt.plot(sequential_x, y_pred, 'b-', label='Topology-Aware GPR Prediction (Mean)', linewidth=2)

# Plot the 95% Confidence Interval / Uncertainty (Blue Shaded Area)
plt.fill_between(sequential_x,
                 y_pred - 1.96 * sigma,
                 y_pred + 1.96 * sigma,
                 alpha=0.2, color='blue', label='95% Confidence Interval (Uncertainty)')

# Scatter plot for the actual Smart Meter measurements (Red Dots)
plt.scatter(train_x_seq, y_train_plot, c='red', s=60, zorder=5, label='Measured Buses (30% Smart Meters)')

# Replace the sequential X-axis ticks with the actual Bus IDs for accurate referencing
plt.xticks(sequential_x, all_buses_sorted.index, rotation=90, fontsize=8)

plt.title('Topology-Aware Probabilistic State Estimation via GPR', fontsize=14)
plt.xlabel('Real Bus Index (Node ID)', fontsize=12)
plt.ylabel('Voltage Magnitude (p.u.)', fontsize=12)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 7. TABULAR RESULTS FOR UNMEASURED BUSES
# ==========================================

print("\n" + "="*60)
print(" COMPARISON: ACTUAL vs PREDICTED VOLTAGE (UNMEASURED BUSES)")
print("="*60)

# Create a DataFrame to hold all results
results_df = pd.DataFrame({
    'Bus_ID': all_buses_sorted.index,
    'Actual_V_pu': y_true,
    'Predicted_V_pu': y_pred,
    'Uncertainty_Std': sigma
})

# Filter out the buses that the model ALREADY knows (the 30% smart meters)
# We only want to see how it performed on the "unseen" / "unmeasured" buses
unmeasured_mask = ~results_df['Bus_ID'].isin(measured_buses.index)
unseen_results = results_df[unmeasured_mask].copy()

# Calculate Absolute Error and Percentage Error
unseen_results['Abs_Error'] = abs(unseen_results['Actual_V_pu'] - unseen_results['Predicted_V_pu'])
unseen_results['Error_%'] = (unseen_results['Abs_Error'] / unseen_results['Actual_V_pu']) * 100

# Format the dataframe for a clean print output
formatted_results = unseen_results.copy()
formatted_results['Actual_V_pu'] = formatted_results['Actual_V_pu'].apply(lambda x: f"{x:.4f}")
formatted_results['Predicted_V_pu'] = formatted_results['Predicted_V_pu'].apply(lambda x: f"{x:.4f}")
formatted_results['Uncertainty_Std'] = formatted_results['Uncertainty_Std'].apply(lambda x: f"±{x:.4f}")
formatted_results['Abs_Error'] = formatted_results['Abs_Error'].apply(lambda x: f"{x:.4f}")
formatted_results['Error_%'] = formatted_results['Error_%'].apply(lambda x: f"{x:.2f}%")

# Print the top 15 unseen buses (You can remove .head(15) to see all of them)
print(formatted_results[['Bus_ID', 'Actual_V_pu', 'Predicted_V_pu', 'Abs_Error', 'Error_%', 'Uncertainty_Std']].to_string(index=False))

# Calculate and print the overall evaluation metrics (MAE and RMSE) for the Unseen Data
mean_absolute_error = unseen_results['Abs_Error'].mean()
rmse = np.sqrt((unseen_results['Abs_Error']**2).mean())

print("-" * 60)
print(f"Overall MAE (Mean Absolute Error)  on Unseen Buses: {mean_absolute_error:.6f} p.u.")
print(f"Overall RMSE (Root Mean Square)    on Unseen Buses: {rmse:.6f} p.u.")
print("="*60 + "\n")
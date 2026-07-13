import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import simbench as sb
import pandapower as pp
import pandapower.plotting.plotly as pplotly

# Import custom modules
from simbench_test import run_study_case, run_annual_time_series
from gpr_model import prepare_topology_data, train_and_predict_gpr, evaluate_and_visualize


def main():
    print("==========================================")
    print(" 1. DOWNLOAD THE NETWORK & PROFILES")
    print("==========================================\n")

    sb_code = "1-LV-semiurb4--0-sw"
    print(f"Downloading network with code: '{sb_code}'...")
    net = sb.get_simbench_net(sb_code)

    print("Fetching annual profiles (Absolute values in p.u.)...")
    profiles_pu = sb.get_absolute_values(net, profiles_instead_of_study_cases=False)

    print("\n==========================================")
    print(" 2. BASELINE POWER FLOW (NORMAL DAY)")
    print("==========================================\n")
    pp.runpp(net,
             algorithm="nr",
             check_connectivity=True,
             voltage_depend_loads=False,
             calculate_voltage_angles=False)
    print("Calculation completed successfully!\n")

    # ==========================================
    # 3. SENSOR PREPARATION (30% SMART METERS)
    # ==========================================
    all_buses = net.res_bus[['vm_pu', 'va_degree']].copy()

    measured_buses = all_buses.sample(frac=0.30, random_state=42)
    print(f"Total buses in the network: {len(all_buses)}")
    print(f"Number of measured buses (30% Smart Meters): {len(measured_buses)}\n")

    # ==========================================
    # 4. NORMAL DAY: GPR STATE ESTIMATION
    # ==========================================
    print("\n==========================================")
    print(" 4. AI TEST 1: NORMAL DAY STATE ESTIMATION")
    print("==========================================\n")

    X_train, y_train, X_test, y_true, scaler_y, all_buses_sorted = prepare_topology_data(net, all_buses, measured_buses)
    y_pred, sigma = train_and_predict_gpr(X_train, y_train, X_test, scaler_y)

    evaluate_and_visualize(all_buses_sorted, measured_buses, y_true, y_pred, sigma,
                           title="NORMAL DAY - Topology-Aware GPR")

    # ==========================================
    # 5. PHYSICAL SIMULATION: CRISIS (HIGH LOAD)
    # ==========================================
    print("\n==========================================")
    print(" 5. PHYSICAL SIMULATION: EXTREME WINTER NIGHT (HIGH LOAD)")
    print("==========================================\n")

    crisis_voltages = run_study_case(net, profiles_pu, study_case_name="hL")

    # ==========================================
    # 6. CRISIS DAY: GPR STATE ESTIMATION
    # ==========================================
    print("\n==========================================")
    print(" 6. AI TEST 2: CRISIS DAY STATE ESTIMATION")
    print("==========================================\n")

    all_buses_crisis = all_buses.copy()
    all_buses_crisis['vm_pu'] = crisis_voltages


    measured_buses_crisis = all_buses_crisis.loc[measured_buses.index]

    X_train_c, y_train_c, X_test_c, y_true_c, scaler_y_c, all_buses_sorted_c = prepare_topology_data(net,
                                                                                                     all_buses_crisis,
                                                                                                     measured_buses_crisis)
    y_pred_c, sigma_c = train_and_predict_gpr(X_train_c, y_train_c, X_test_c, scaler_y_c)

    evaluate_and_visualize(all_buses_sorted_c, measured_buses_crisis, y_true_c, y_pred_c, sigma_c,
                           title="CRISIS SCENARIO (High Load) - Topology-Aware GPR")


if __name__ == "__main__":
    main()
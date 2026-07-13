import pandapower as pp
import simbench as sb


def run_study_case(net, profiles, study_case_name="hL"):
    """
    Tests predefined Extreme study cases from Simbench (e.g., 'hL' for High Load, 'hPV' for High PV).
    """
    print(f"--> SIMBENCH: Running Study Case '{study_case_name}'...")

    for (element, column), df in profiles.items():
        if study_case_name in df.index:
            net[element][column] = df.loc[study_case_name].values

    pp.runpp(net,
             algorithm="nr",
             check_connectivity=True,
             voltage_depend_loads=False,
             calculate_voltage_angles=False)

    print(f"    Result: Power flow solved successfully for '{study_case_name}'.")
    return net.res_bus.vm_pu

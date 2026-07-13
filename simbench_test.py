import pandapower as pp
import simbench as sb
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl


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


def run_annual_time_series(net, profiles, time_steps=5):
    """
    Executes a time series power flow analysis.
    Adjust time_steps to 35000 for a full year analysis.
    """
    print(f"--> SIMBENCH: Starting Time Series analysis... (Steps: {time_steps})")

    # Convert profiles to DFData format required by Pandapower TimeSeries
    load_p = DFData(profiles[("load", "p_mw")])
    load_q = DFData(profiles[("load", "q_mvar")])
    sgen_p = DFData(profiles[("sgen", "p_mw")])

    # Attach controllers to update loads and generators dynamically
    ConstControl(net, element='load', variable='p_mw', element_index=net.load.index, data_source=load_p,
                 profile_name=load_p.df.columns)
    ConstControl(net, element='load', variable='q_mvar', element_index=net.load.index, data_source=load_q,
                 profile_name=load_q.df.columns)
    ConstControl(net, element='sgen', variable='p_mw', element_index=net.sgen.index, data_source=sgen_p,
                 profile_name=sgen_p.df.columns)

    # Setup OutputWriter to store results in memory
    ow = OutputWriter(net, time_steps=range(time_steps), output_path=None)
    ow.log_variable('res_bus', 'vm_pu')

    # Execute the time series simulation
    run_timeseries(net, time_steps=range(time_steps))
    print("    Result: Time series power flow completed successfully.")

    return ow.output["res_bus.vm_pu"]
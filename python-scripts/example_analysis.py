"""
Run script designed to take three defined classes (
two optimization and one formatting)
with provided input data and generate performance metrics.

Created for ME 401 @ Boise State Univerity
Author: Maxwell Hewes, Summer 2025
"""
import bess_optimizer as heo # hybrid energy optimizer
import data_formatter as idf # input data formatter
import supply_optimizer as opt # wind / solar capacity optimizer

# Original Datasets
# wind = 'python-scripts/data_sets/NOLA_wind.csv'
# solar = 'python-scripts/data_sets/NOLA_solar.csv'
# heat = 'python-scripts/data_sets/NOLA_heat.csv'
# port_location = 'New Orleans'

# Seattle datasets
wind = 'python-scripts/data_sets/Seattle/wind.csv'
solar = 'python-scripts/data_sets/Seattle/solar.csv'
heat = 'python-scripts/data_sets/Seattle/heat.csv'
port_location = 'Seattle'

# Data sourced from https://www.renewables.ninja
port_file = 'python-scripts/data_sets/port_demand_raw.csv'


data = idf.DataFormatter()
data.gather_port_demand(port_file,port_location)
data.gather_heat_demand(heat)
data.build_demand()
data.gather_solar(solar)
data.gather_wind(wind)
data.check_array_length()

optimizer = opt.SolarWindOptimizer()
optimizer.make_arrays(data.wind, data.solar, data.demand)
optimizer.optimize()

data.solar = data.solar * optimizer.opt_solar_
data.wind = data.wind * optimizer.opt_wind_

system = heo.HybridEnergyOptimizer()
system.time_series = data.produce_avg_data_frame()

results = system.simulate_bess_operation(10,3)
system.optimize_bess_capacity()
print(system.results['optimal_capacity'])

pMetrics = system.calculate_performance_indicators()
system.display_results()



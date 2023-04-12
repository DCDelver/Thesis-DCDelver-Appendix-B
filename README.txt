Dear reader, thank you for your interest and welcome to the data and files of my thesis.

RUNNING THE MODEL
To run the model that I created, use simulation_runner.py
The simulation_runner.py is used to declare inputs and run scenarios with the model.
This file uses simulation_setup_scenarios.py and simulation_functions.py to run the simulation framework.

In addition, this folder contains three scripts for processing results:
-plot_validation_graphs: plots the data for different scenario, based on the data that was saved in aggregated_data_validation.csv
-write_output_indicators: this will write outputs for scenarios to a csv file. It looks a little more
clumsy, the output_indicators csv looks a little bit clumsy but serves its purpose.
-aggregate_results: this file allows for aggregating the results of the files in a folder. When it is used
to aggregate the result folders, it would replicate the already existing aggregated_data files. Be aware that this may take a while.

To aggregate results, the aggregate_results file can be run. For each output csv that it finds in the designated folder this will create a new entry in an aggregated_data csv.

DATA FOR VALIDATION AND SCENARIO RUNS
The data that was obtained from the validation runs for my thesis are stored in the results_csv_validation folder.
The data that was obtained from the scenario runs for my thesis can be found in the results_csv_scenarios folder

The aggregated results for both folders are stored in aggregated_data_validation.csv and aggregated_data_scenarios.csv

For any further questions, please contact me at d.c.delver@student.tudelft.nl

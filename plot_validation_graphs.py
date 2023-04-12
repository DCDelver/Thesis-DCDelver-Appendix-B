# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 19:48:59 2023

@author: dcdel
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import matplotlib as mpl
import scipy.stats

# Import plotdata
file_name = 'simulation_rundata2023224.csv'
df_output = pd.read_csv(file_name,delimiter=',')

''' DISAGGREGATE PLOTS '''

'first pick a case'
df_plot = df_output.loc[df_output['Name'] == 'Iteration_2023-02-24_13:06:11']
#df_plot = df_output.index == 12


'plot how the convergence occurs'
demand = pd.eval(df_plot['demand'])[0]
served = pd.eval(df_plot['served'])[0]
runs = len(demand)
run_axis = range(1,runs+1)
sns.set_style("darkgrid")

#served demand plots
fig, ax = plt.subplots(figsize=(16,16))
ax.set_title('Service demand and served trips per run')
sns.lineplot(x=run_axis, y=demand, ax=ax,  linestyle = "-", color = 'blue', label='Demand')
sns.lineplot(x=run_axis, y=served, ax=ax,  linestyle = "-", color = 'red', label='Served')

ax.set(xlabel='simulation run', ylabel='nr of requests')
plt.show()


'first pick a case'
df_plot = df_output.loc[df_output['Name'] == 'Iteration_2023-02-24_12:53:25']
#df_plot = df_output.index == 12

'plot how the convergence occurs'
demand = pd.eval(df_plot['demand'])[0]
served = pd.eval(df_plot['served'])[0]
runs = len(demand)
run_axis = range(1,runs+1)

#served demand plots
fig, ax = plt.subplots(figsize=(16,16))
ax.set_title('Service demand and served trips per run')
sns.lineplot(x=run_axis, y=demand, ax=ax,  linestyle = "-", color = 'blue', label='Demand')
sns.lineplot(x=run_axis, y=served, ax=ax,  linestyle = "-", color = 'red', label='Served')

ax.set(xlabel='simulation run', ylabel='nr of requests')
plt.show()

'first pick a case'
df_plot = df_output.loc[df_output['Name'] == 'Iteration_2023-02-24_12:34:21']
#df_plot = df_output.index == 12

'plot how the convergence occurs'
demand = pd.eval(df_plot['demand'])[0]
served = pd.eval(df_plot['served'])[0]
runs = len(demand)
run_axis = range(1,runs+1)

#served demand plots
fig, ax = plt.subplots(figsize=(16,16))
ax.set_title('Service demand and served trips per run')
sns.lineplot(x=run_axis, y=demand, ax=ax,  linestyle = "-", color = 'blue', label='Demand')
sns.lineplot(x=run_axis, y=served, ax=ax,  linestyle = "-", color = 'red', label='Served')

ax.set(xlabel='simulation run', ylabel='nr of requests')
plt.show()


'Also collect: service ratio over iterations, walktime, waittime, devFact'

''' AGGREGATED PLOTS '''

sns.set_style("dark")
sns.set(font_scale = 2)

file_name = 'aggregated_data_validation.csv'
df_output = pd.read_csv(file_name,delimiter=',')
sns.set(font_scale = 2.5)
sns.set_style("darkgrid") # "white", "dark", "ticks"

def plot_variability_data(df_sorted_data):  

    nr_mode0 = df_sorted_data['nr_mode0']
    nr_mode1 = df_sorted_data['nr_mode1']
    nr_mode2 = df_sorted_data['nr_mode2']
    nr_mode3 = df_sorted_data['nr_mode3']
    nr_mode4 = df_sorted_data['nr_mode4']
    
    nr_mode0_means = np.zeros(len(df_sorted_data))
    nr_mode1_means = np.zeros(len(df_sorted_data))
    nr_mode2_means = np.zeros(len(df_sorted_data))
    nr_mode3_means = np.zeros(len(df_sorted_data))
    nr_mode4_means = np.zeros(len(df_sorted_data))
    
    MS_served = df_sorted_data['MS_served']
    MS_served_pk = df_sorted_data['MS_served_pk']
    MS_served_offpk = df_sorted_data['MS_served_offpk']

    demand_means = np.zeros(len(df_sorted_data))
    served_means = np.zeros(len(df_sorted_data))
    demand_std_error_sample_means = np.zeros(len(df_sorted_data))
    served_std_error_sample_means = np.zeros(len(df_sorted_data))
    
    perc_diff_demand_sample_means = np.zeros(len(df_sorted_data))
    perc_diff_served_sample_means = np.zeros(len(df_sorted_data))
    demand_std_deviation_sample_means = np.zeros(len(df_sorted_data))
    served_std_deviation_sample_means = np.zeros(len(df_sorted_data))
    perc_diff_demand_sample_std_error = np.zeros(len(df_sorted_data))
    perc_diff_served_sample_std_error = np.zeros(len(df_sorted_data))
    
    for i in range(1,len(df_sorted_data)+1):
        
        demand_new_mean = df_sorted_data['demand'][0:i].mean()
        served_new_mean = df_sorted_data['served'][0:i].mean()
        nr_mode0_new_mean = df_sorted_data['nr_mode0'][0:i].mean()
        nr_mode1_new_mean = df_sorted_data['nr_mode1'][0:i].mean()
        nr_mode2_new_mean = df_sorted_data['nr_mode2'][0:i].mean()
        nr_mode3_new_mean = df_sorted_data['nr_mode3'][0:i].mean()
        nr_mode4_new_mean = df_sorted_data['nr_mode4'][0:i].mean()

        demand_means[i-1] = demand_new_mean
        served_means[i-1] = served_new_mean
        
        nr_mode0_means[i-1] = nr_mode0_new_mean
        nr_mode1_means[i-1] = nr_mode1_new_mean
        nr_mode2_means[i-1] = nr_mode2_new_mean
        nr_mode3_means[i-1] = nr_mode3_new_mean
        nr_mode4_means[i-1] = nr_mode4_new_mean
        
        demand_std_deviation_sample_means[i-1] = np.sqrt(np.sum(((df_sorted_data['demand'][0:i]-demand_means[0:i])**2))/i)         
        served_std_deviation_sample_means[i-1] = np.sqrt(np.sum(((df_sorted_data['served'][0:i]-served_means[0:i])**2))/i)
        
        # CHECK IF THIS INDICES CORRECTLY, IT GIVES STANDARD ERROR OF MEANS
        demand_std_error_sample_means[i-1] =\
            np.sqrt(np.sum(((df_sorted_data['demand'][0:i]-demand_means[0:i])**2))/i)/\
                np.sqrt(i)
        served_std_error_sample_means[i-1] =\
            np.sqrt(np.sum(((df_sorted_data['served'][0:i]-served_means[0:i])**2))/i)/\
                np.sqrt(i)
    
    perc_diff_demand_sample_means = (demand_means-demand_means[-1])/demand_means[-1]*100
    perc_diff_served_sample_means = (served_means-served_means[-1])/served_means[-1]*100    

    perc_diff_demand_sample_std_error = np.sqrt((demand_std_error_sample_means-demand_std_error_sample_means[-1])**2)/demand_std_error_sample_means[-1]*100
    perc_diff_served_sample_std_error = np.sqrt((served_std_error_sample_means-served_std_error_sample_means[-1])**2)/served_std_error_sample_means[-1]*100
    
    perc_diff_mode0 = (nr_mode0_means-nr_mode0_means[-1])/nr_mode0_means[-1]*100
    perc_diff_mode1 = (nr_mode1_means-nr_mode1_means[-1])/nr_mode1_means[-1]*100
    perc_diff_mode2 = (nr_mode2_means-nr_mode2_means[-1])/nr_mode2_means[-1]*100
    perc_diff_mode3 = (nr_mode3_means-nr_mode3_means[-1])/nr_mode3_means[-1]*100
    perc_diff_mode4 = (nr_mode4_means-nr_mode4_means[-1])/nr_mode4_means[-1]*100

    # [%] DIFF THE MEAN & [%] DIFF OF STD ERROR TO FINAL MEAN & STD ERROR
    sns.set_color_codes('bright')
    fig, axes = plt.subplots(1, 2, figsize=(48,24))
    #axes[0,0].set_title('Served', weight='bold')    
    axes[0].set_title('Difference of sample mean with final mean', weight='bold')    
    
    #sns.lineplot(x=df_sorted_data['run_seed'], y=served_means, ax=axes[0,0], marker='o',  linestyle = "-", color = 'g', label='All morning')
    sns.lineplot(x=df_sorted_data['run_seed'], y=perc_diff_demand_sample_means, ax=axes[0], marker='o',  linestyle = "-", color = 'b', label='Demand')
    sns.lineplot(x=df_sorted_data['run_seed'], y=perc_diff_served_sample_means, ax=axes[0], marker='o',  linestyle = "-", color = 'r', label='Served')
    #sns.lineplot(x=df_sorted_data['run_seed'], y=MS_served_offpk*100, ax=axes[0,0], marker='o',  linestyle = "-", color = 'b', label='Off-peak')
    axes[0].set(xlabel='Runs', ylabel='Difference of sample mean with final mean [%]')

    axes[1].set_title('Diff of std errors of sample means with std error of final sample mean, per run', weight='bold')    
    sns.lineplot(x=df_sorted_data['run_seed'], y=perc_diff_demand_sample_std_error, ax=axes[1], marker='o',  linestyle = "-", color = 'b', label='Demand')
    sns.lineplot(x=df_sorted_data['run_seed'], y=perc_diff_served_sample_std_error, ax=axes[1], marker='o',  linestyle = "-", color = 'r', label='Served')
    #sns.lineplot(x=df_sorted_data['run_seed'], y=MS_served_offpk*100, ax=axes[0,0], marker='o',  linestyle = "-", color = 'b', label='Off-peak')
    axes[1].set(xlabel='Runs', ylabel='Difference std error of sample mean and std error of final sample mean [%]')    
    
    plt.show()
    
    " THIS IS THE STANDARD ERROR OF THE MEAN FOR DEMAND"
    fig, ax = plt.subplots(figsize=(24,24))
    ax.set_title('Standard error of the sample mean of the demand and served demand up until each run')
    sns.lineplot(x=df_sorted_data['run_seed'][1:], y=demand_std_error_sample_means[1:], ax=ax, marker='o',  linestyle = "-", color = 'b', label='Demand')
    #sns.lineplot(x=df_sorted_data['run_seed'], y=served_std_error_sample_means, ax=ax, marker='o',  linestyle = "-", color = 'r', label='Served')
    ax.set(xlabel = 'Runs', ylabel = 'Standard error of the sample mean of the demand')
    plt.show()
    
    #
    fig, ax = plt.subplots(figsize=(24,24))
    ax.set_title('Sample mean of the demand up until each run')
    sns.lineplot(x=df_sorted_data['run_seed'][1:], y=demand_means[1:], ax=ax, marker='o',  linestyle = "-", color = 'b', label='Demand')
    #sns.lineplot(x=df_sorted_data['run_seed'], y=served_std_error_sample_means, ax=ax, marker='o',  linestyle = "-", color = 'r', label='Served')
    ax.set(xlabel = 'Runs', ylabel = 'Sample mean of the demand')
    plt.show()
    
    # STANDARD DEVIATION OF SAMPLE MEAN FOR DEMAND AND SERVED DEMAND
    fig, ax = plt.subplots(figsize=(24,24))
    ax.set_title('Standard deviation of the sample mean of the demand and served demand up until each run')
    sns.lineplot(x=df_sorted_data['run_seed'], y=demand_std_deviation_sample_means, ax=ax, marker='o',  linestyle = "-", color = 'b', label='Demand')
    sns.lineplot(x=df_sorted_data['run_seed'], y=served_std_deviation_sample_means, ax=ax, marker='o',  linestyle = "-", color = 'r', label='Served')
    ax.set(xlabel = 'Runs', ylabel = 'Standard deviation of the sample mean of the demand')
    plt.show()
    """
    """
    # FOR ALL MODAL SHARES, THESE FLUCTUATIONS WERE SIMILAR
    fig, ax = plt.subplots(figsize=(24,24))
    ax.set_title('Mode shares')
    sns.lineplot(x=df_sorted_data['run_seed'], y=nr_mode0_means/682*100, ax=ax, marker='o',  linestyle = "-", color = 'b', label='AV-taxi')
    sns.lineplot(x=df_sorted_data['run_seed'], y=nr_mode1_means/682*100, ax=ax, marker='o',  linestyle = "-", color = 'g', label='Shared bicycle')
    sns.lineplot(x=df_sorted_data['run_seed'], y=nr_mode2_means/682*100, ax=ax, marker='o',  linestyle = "-", color = 'r', label='Shared e-step')
    sns.lineplot(x=df_sorted_data['run_seed'], y=nr_mode3_means/682*100, ax=ax, marker='o',  linestyle = "-", color = 'k', label='Shared e-scooter')
    sns.lineplot(x=df_sorted_data['run_seed'], y=nr_mode4_means/682*100, ax=ax, marker='o',  linestyle = "-", color = 'y', label='Walking')    
    ax.set(xlabel = 'Runs', ylabel = ['Modal shares [%]'])
    plt.show()
    
    fig, ax = plt.subplots(figsize=(24,24))
    "DIFFERENCE OF MEANS OF MODE SHARES OF SAMPLE SIZE UP UNTIL EACH RUN TO THE MEAN MODE SHARES AFTER 100 RUNS"
    ax.set_title('Difference of means of mode shares')
    sns.lineplot(x=df_sorted_data['run_seed'], y=perc_diff_mode0, ax=ax, marker='o',  linestyle = "-", color = 'b', label='AV-taxi')
    sns.lineplot(x=df_sorted_data['run_seed'], y=perc_diff_mode1, ax=ax, marker='o',  linestyle = "-", color = 'g', label='Shared bicycle')
    sns.lineplot(x=df_sorted_data['run_seed'], y=perc_diff_mode2, ax=ax, marker='o',  linestyle = "-", color = 'r', label='Shared e-step')
    sns.lineplot(x=df_sorted_data['run_seed'], y=perc_diff_mode3, ax=ax, marker='o',  linestyle = "-", color = 'k', label='Shared e-scooter')
    sns.lineplot(x=df_sorted_data['run_seed'], y=perc_diff_mode4, ax=ax, marker='o',  linestyle = "-", color = 'y', label='Walking')    
    ax.set(xlabel = 'Runs', ylabel = 'Modal shares [% diff]')
    plt.show()
    
    return

'''
df_plot_data = df_output.loc[(df_output['fleetsize'] == 5) &\
                             (df_output['price1'] == 200) &\
                             (df_output['price2'] == 37.5) &\
                             (df_output['spacing'] == 0) &\
                             (df_output['optout'] == 700) &\
                             (df_output['demfact'] == 1)&\
                             (df_output['area_size'] == 1) &\
                             (df_output['arrival_frequency'] == 1) &\
                             (df_output['groups'] == 1)] # ADD max_dev_factor !!!!!!!!!!!!!!!
plot_variability_data(df_plot_data)    
'''    

sns.set(font_scale = 2.5)
sns.set_style("darkgrid") # "white", "dark", "ticks"


# This is the function runner for the pre-results 
def run_fleetsize_pre_results(df_output):
    df_plot_fleetsizes_base = df_output.loc[(df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 1) &\
                                 (df_output['area_size'] == 1) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 1)]
    fig_suptitle = 'Base scenario'
    fleetsize_vs_waiting_times(df_plot_fleetsizes_base,fig_suptitle)
    
    df_plot_fleetsizes_base2 = df_output.loc[(df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 2) &\
                                 (df_output['area_size'] == 1) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 1)]
    fig_suptitle = 'Base scenario, 2x demand'
    fleetsize_vs_waiting_times(df_plot_fleetsizes_base2,fig_suptitle)
    
    df_plot_fleetsizes_space = df_output.loc[(df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0.4) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 1) &\
                                 (df_output['area_size'] == 1) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 1)]
    fig_suptitle = 'Stops with a spacing of 400 [m]'
    fleetsize_vs_waiting_times(df_plot_fleetsizes_space,fig_suptitle)
    
    df_plot_fleetsizes_space2 = df_output.loc[(df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0.4) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 2) &\
                                 (df_output['area_size'] == 1) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 1)]
    fig_suptitle = 'Stops with a spacing of 400 [m], 2x demand'
    fleetsize_vs_waiting_times(df_plot_fleetsizes_space2,fig_suptitle)
    
    
    df_plot_fleetsizes_group = df_output.loc[(df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 1) &\
                                 (df_output['area_size'] == 1) &\
                                 (df_output['groups'] == 2) &\
                                 (df_output['arrival_frequency'] == 1)]
    fig_suptitle = 'Traveller groups'
    fleetsize_vs_waiting_times(df_plot_fleetsizes_group,fig_suptitle)
    
    df_plot_fleetsizes_group2 = df_output.loc[(df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 2) &\
                                 (df_output['area_size'] == 1) &\
                                 (df_output['groups'] == 2) &\
                                 (df_output['arrival_frequency'] == 1)]
    fig_suptitle = 'Traveller groups, 2x demand'
    fleetsize_vs_waiting_times(df_plot_fleetsizes_group2,fig_suptitle)
    
    df_plot_fleetsizes_freq = df_output.loc[(df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 1) &\
                                 (df_output['area_size'] == 1) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 2)]
    fig_suptitle = 'Decreased headway between arrivals'
    fleetsize_vs_waiting_times(df_plot_fleetsizes_freq,fig_suptitle)
    
    df_plot_fleetsizes_freq2 = df_output.loc[(df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 2) &\
                                 (df_output['area_size'] == 1) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 2)]
    fig_suptitle = 'Decreased headway between arrivals, 2x demand'
    fleetsize_vs_waiting_times(df_plot_fleetsizes_freq2,fig_suptitle)
    
    
    df_plot_fleetsizes_area = df_output.loc[(df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 1) &\
                                 (df_output['area_size'] == 2) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 1)]
    fig_suptitle = 'Large service area'
    fleetsize_vs_waiting_times(df_plot_fleetsizes_area,fig_suptitle)
    
    df_plot_fleetsizes_area2 = df_output.loc[(df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 2) &\
                                 (df_output['area_size'] == 2) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 1)]      
    fig_suptitle = 'Large service area, 2x demand'
    fleetsize_vs_waiting_times(df_plot_fleetsizes_area2,fig_suptitle)
# This is the plot function for pre-results, called by function runner 
def fleetsize_vs_waiting_times(df_plot_data, fig_suptitle):
    
    x_axis_name = str(df_plot_data['fleetsize'].name)
    x_axis_data = df_plot_data[x_axis_name]
    
    wait_avg = df_plot_data['wait_avg']
    #wait_avg_pk = df_plot_data['wait_avg_pk']
    #wait_avg_offpk = df_plot_data['wait_avg_offpk']

    sns.set_palette('bright')    
    sns.set_color_codes('bright')
    fig, ax = plt.subplots(figsize=(18,12))
    fig.suptitle(fig_suptitle)
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1, axis='y')
    ax.grid(b=True, which='minor', color='w', linewidth=0.5, axis='y')
    ax.set_title('Average waiting time per request', weight='bold')
    sns.lineplot(x=x_axis_data, y=wait_avg, ax=ax, marker='o',  linestyle = "-", color = 'b', label='All morning')
    #sns.lineplot(x=x_axis_data, y=wait_avg_pk, ax=ax, marker='o',  linestyle = "-", color = 'r', label='Peak')
    #sns.lineplot(x=x_axis_data, y=wait_avg_offpk, ax=ax, marker='o',  linestyle = "-", color = 'g', label='Off-peak')
    ax.set(xlabel=x_axis_name)

#run_fleetsize_pre_results(df_output) # THINK THIS IS THE TRUE PRE-RESULTS

def run_validation_results(df_output):
    df_plot_fleetsizes = df_output.loc[
                                 (df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 1) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 1) &\
                                 (df_output['area_size'] == 1)]
    df_plot_data = df_plot_fleetsizes
    plot_over_data = df_plot_data['fleetsize']
    fig_suptitle1 = 'System KPIs with standard deviations plotted over fleetsize'
    plot_systemKPIs(df_plot_data, plot_over_data, fig_suptitle1)
    fig_suptitle2 = 'Demand KPIs with standard deviations plotted over fleetsize'
    plot_demandKPIs(df_plot_data, plot_over_data, fig_suptitle2)
    
    
    df_plot_price1 = df_output.loc[(df_output['fleetsize'] == 5) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 1) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 1) &\
                                 (df_output['area_size'] == 1)]
    df_plot_data = df_plot_price1
    plot_over_data = df_plot_data['price1']
    fig_suptitle1 = 'System with standard deviations KPIs plotted over starting fare'
    #plot_data_confidence_int(df_plot_data, plot_over_data, fig_suptitle)
    plot_systemKPIs(df_plot_data, plot_over_data, fig_suptitle1)
    fig_suptitle2 = 'Demand KPIs with standard deviations plotted over starting fare'
    plot_demandKPIs(df_plot_data, plot_over_data, fig_suptitle2)
    
    df_plot_price2 = df_output.loc[(df_output['fleetsize'] == 5) &\
                                 (df_output['price1'] == 200) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 1) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 1) &\
                                 (df_output['area_size'] == 1)]
    df_plot_data = df_plot_price2
    plot_over_data = df_plot_data['price2']
    fig_suptitle1 = 'System KPIs with standard deviations plotted over ride time fare'
    #plot_data_confidence_int(df_plot_data, plot_over_data, fig_suptitle)
    plot_systemKPIs(df_plot_data, plot_over_data, fig_suptitle1)
    fig_suptitle2 = 'Demand KPIs with standard deviations plotted over ride time fare'
    plot_demandKPIs(df_plot_data, plot_over_data, fig_suptitle2)
    
    df_plot_spacing = df_output.loc[(df_output['fleetsize'] == 5) &\
                                 (df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['demfact'] == 1) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 1) &\
                                 (df_output['area_size'] == 1)]
    df_plot_data = df_plot_spacing
    plot_over_data = df_plot_data['spacing']
    fig_suptitle1 = 'System KPIs with standard deviations plotted over spacing distance [m]'
    plot_systemKPIs(df_plot_data, plot_over_data, fig_suptitle1)
    fig_suptitle2 = 'Demand KPIs with standard deviations plotted over spacing distance [m]'
    plot_demandKPIs(df_plot_data, plot_over_data, fig_suptitle2)
    
    df_plot_demfact = df_output.loc[(df_output['fleetsize'] == 5) &\
                                 (df_output['price1'] == 200) &\
                                 (df_output['price2'] == 37.5) &\
                                 (df_output['spacing'] == 0) &\
                                 (df_output['optout'] == 700) &\
                                 (df_output['groups'] == 1) &\
                                 (df_output['arrival_frequency'] == 1) &\
                                 (df_output['area_size'] == 1)] # ADD max_dev_factor !!!!!!!!!!!!!!!
    df_plot_data = df_plot_demfact
    plot_over_data = df_plot_data['demfact']
    fig_suptitle1 = 'SystemKPIs with standard deviations plotted over demand factor'
    #plot_data_confidence_int(df_plot_data, plot_over_data, fig_suptitle)
    plot_systemKPIs(df_plot_data, plot_over_data, fig_suptitle1)
    fig_suptitle2 = 'Demand KPIs with standard deviations plotted over demand factor'
    plot_demandKPIs(df_plot_data, plot_over_data, fig_suptitle2)

    return

def plot_systemKPIs(df_plot_data, plot_over_data, fig_suptitle):
    sns.set(font_scale = 2)
    # prepare name string for selection of x-axis range
    x_axis_name = str(plot_over_data.name)
    x_axis_data = df_plot_data[x_axis_name]
    #print(x_axis_data)
    # prepare mean values
    
#    for i in 
    nr_mode0 = df_plot_data['nr_mode0']
    nr_mode1 = df_plot_data['nr_mode1']
    nr_mode2 = df_plot_data['nr_mode2']
    nr_mode3 = df_plot_data['nr_mode3']
    nr_mode4 = df_plot_data['nr_mode4']
    
    MS_served = df_plot_data['MS_served']
    MS_served_pk = df_plot_data['MS_served_pk']
    MS_served_offpk = df_plot_data['MS_served_offpk']
    
    serviceratio = df_plot_data['served']/df_plot_data['demand']
    serviceratio_pk = df_plot_data['served_pk']/df_plot_data['demand_pk']
    serviceratio_offpk = df_plot_data['served_offpk']/df_plot_data['demand_offpk']
    
    punctuality = df_plot_data['punctuality']
    punctuality_pk = df_plot_data['punctuality_pk']
    punctuality_offpk = df_plot_data['punctuality_offpk']
    
    wait_avg = df_plot_data['wait_avg']
    wait_avg_pk = df_plot_data['wait_avg_pk']
    wait_avg_offpk = df_plot_data['wait_avg_offpk']
    wait_avg_served = df_plot_data['wait_served_avg']
    wait_avg_served_pk = df_plot_data['wait_served_avg_pk']
    wait_avg_served_offpk = df_plot_data['wait_served_avg_offpk']
    
    dev_avg = df_plot_data['dev_avg']
    dev_avg_pk = df_plot_data['dev_avg_pk']
    dev_avg_offpk = df_plot_data['dev_avg_offpk']
    
    load_factor = df_plot_data['load_factor']
    zero_time = df_plot_data['zero_time']
    use_time_ratio = df_plot_data['use_time_ratio']
    avg_trip_distance = df_plot_data['avg_trip_distance']
    avg_direct_trip_distance = df_plot_data['avg_direct_distance']
    avg_trip_duration = df_plot_data['avg_trip_duration']
    avg_direct_trip_duration = df_plot_data['avg_direct_ride_time']
    #fraction_deadheading
    #fraction_consolidated_dropoffs = df_plot_data.loc[df_plot_data['fraction_consolidated_dropoffs']]
    served = df_plot_data['served']
    served_pk = df_plot_data['served_pk']
    served_offpk = df_plot_data['served_offpk']
    
    tot_distance = df_plot_data['tot_distance']
    effective_transport_ratio = df_plot_data['effective_transport_ratio']
    effective_transport_dist = df_plot_data['effective_transport_dist']
    
    if plot_over_data.name == 'price1' or plot_over_data.name == 'price2' or plot_over_data.name == 'optout':
        plot_over_data = plot_over_data/100

    sns.set_palette('bright')    
    sns.set_color_codes('bright')
    fig, axes = plt.subplots(2, 3, figsize=(27,18))
    fig.suptitle(fig_suptitle, fontsize = 'x-large')
    axes[0,0].set_title('Passengers', weight='bold')
    sns.lineplot(x=plot_over_data, y=served, ax=axes[0,0], marker='o',  linestyle = "-", color = 'b', ci='sd')
    sns.lineplot(x=plot_over_data, y=served_pk, ax=axes[0,0], marker='o',  linestyle = "-", color = 'r', ci='sd')
    sns.lineplot(x=plot_over_data, y=served_offpk, ax=axes[0,0], marker='o',  linestyle = "-", color = 'g', ci='sd')
    axes[0,0].set(ylabel = 'Passengers')
    if plot_over_data.name == 'price1':
        axes[0,0].set(xlabel='Starting fare [euro]')
    elif plot_over_data.name == 'price2':
        axes[0,0].set(xlabel='Fare/min of direct ride time [euro/min]')
    elif plot_over_data.name == 'spacing':
        axes[0,0].set(xlabel='Stop spacing distance [km]')
    elif plot_over_data.name == 'demfact':
        axes[0,0].set(xlabel='Demand multiplicaton factor')
    else:
        axes[0,0].set(xlabel=x_axis_name)
    
    axes[0,1].set_title('Total distance', weight='bold')
    sns.lineplot(x=plot_over_data, y=tot_distance, ax=axes[0,1], marker='o',  linestyle = "-", color = 'b', ci='sd')
    axes[0,1].set(ylabel = 'Total distance [m]')
    if plot_over_data.name == 'price1':
        axes[0,1].set(xlabel='Starting fare [euro]')
    elif plot_over_data.name == 'price2':
        axes[0,1].set(xlabel='Fare/min of direct ride time [euro/min]')
    elif plot_over_data.name == 'spacing':
        axes[0,1].set(xlabel='Stop spacing distance [km]')
    elif plot_over_data.name == 'demfact':
        axes[0,1].set(xlabel='Demand multiplicaton factor')
    else:
        axes[0,1].set(xlabel=x_axis_name)
    
    #fig, ax4 = plt.subplots(figsize=(8,8))
    axes[1,0].set_title('Vehicle occupation factor', weight='bold')
    sns.lineplot(x=plot_over_data, y=load_factor, ax=axes[1,0], marker='o',  linestyle = "-", color = 'b', ci='sd')
    axes[1,0].set(ylabel = 'Vehicle occupation factor')
    if plot_over_data.name == 'price1':
        axes[1,0].set(xlabel='Starting fare [euro]')
    elif plot_over_data.name == 'price2':
        axes[1,0].set(xlabel='Fare/min of direct ride time [euro/min]')
    elif plot_over_data.name == 'spacing':
        axes[1,0].set(xlabel='Stop spacing distance [km]')
    elif plot_over_data.name == 'demfact':
        axes[1,0].set(xlabel='Demand multiplicaton factor')
    else:
        axes[1,0].set(xlabel=x_axis_name)
    
    #fig, ax5 = plt.subplots(figsize=(8,8))
    # WHY DOES VEHICLE UTIZILATION RATIO EXCEED 1 BY FAR?
    # BECAUSE:
    # The vehicle utilization ratio can exceed 1 by far if the last travellers
    # are willing to wait a long time!!!
    axes[1,1].set_title('Veh. use time to serive period ratio', weight='bold')
    sns.lineplot(x=plot_over_data, y=use_time_ratio, ax=axes[1,1], marker='o',  linestyle = "-", color = 'b', ci='sd')
    if plot_over_data.name == 'price1':
        axes[1,1].set(xlabel='Starting fare [euro]', ylabel='Vehicle utilization ratio')
    elif plot_over_data.name == 'price2':
        axes[1,1].set(xlabel='Fare/min of direct ride time [euro/min]', ylabel='Vehicle utilization ratio')
    elif plot_over_data.name == 'spacing':
        axes[1,1].set(xlabel='Stop spacing distance [km]', ylabel='Vehicle utilization ratio')
    elif plot_over_data.name == 'demfact':
        axes[1,1].set(xlabel='Demand multiplication factor', ylabel='Vehicle utilization ratio')
    else:
        axes[1,1].set(xlabel=x_axis_name, ylabel='Vehicle utilization ratio')
    
    axes[0,2].set_title('Effective transportation distance', weight='bold')
    sns.lineplot(x=plot_over_data, y=effective_transport_dist, ax=axes[0,2], marker='o',  linestyle = "-", color = 'b', label='All morning', ci='sd')
    axes[0,2].set(ylabel='Effective transportation distance [pkm]')
    if plot_over_data.name == 'price1':
        axes[0,2].set(xlabel='Starting fare [euro]')
    elif plot_over_data.name == 'price2':
        axes[0,2].set(xlabel='Fare/min of direct ride time [euro/min]')
    elif plot_over_data.name == 'spacing':
        axes[0,2].set(xlabel='Stop spacing distance [km]')
    elif plot_over_data.name == 'demfact':
        axes[0,2].set(xlabel='Demand multiplicaton factor')
    else:
        axes[0,2].set(xlabel=x_axis_name)
    
    axes[1,2].set_title('Effective transportation distance ratio', weight='bold')
    sns.lineplot(x=plot_over_data, y=effective_transport_ratio, ax=axes[1,2], marker='o',  linestyle = "-", color = 'b', label='All morning', ci='sd')
    axes[1,2].set(ylabel='Effective work ratio [pkm/vkm]')
    if plot_over_data.name == 'price1':
        axes[1,2].set(xlabel='Starting fare [euro]')
    elif plot_over_data.name == 'price2':
        axes[1,2].set(xlabel='Fare/min of direct ride time [euro/min]')
    elif plot_over_data.name == 'spacing':
        axes[1,2].set(xlabel='Stop spacing distance [km]')
    elif plot_over_data.name == 'demfact':
        axes[1,2].set(xlabel='Demand multiplicaton factor')
    else:
        axes[1,2].set(xlabel=x_axis_name)
    
    plt.show()
    
    '''
    sns.set(font_scale = 1.5)
    fig2, ax = plt.subplots(figsize=(16,8))
    ax.set_title('Effective transportation distance ratio', weight='bold')
    sns.lineplot(x=plot_over_data, y=effective_transport_ratio, ax=ax, marker='o',  linestyle = "-", color = 'b', label='All morning')
    ax.set(ylabel='Effective work ratio [pkm/vkm]')
    if plot_over_data.name == 'price1':
        ax.set(xlabel='Starting fare [euro]')
    elif plot_over_data.name == 'price2':
        ax.set(xlabel='Fare/min of direct ride time [euro/min]')
    elif plot_over_data.name == 'spacing':
        ax.set(xlabel='Stop spacing distance [km]')
    elif plot_over_data.name == 'demfact':
        ax.set(xlabel='Demand multiplicaton factor')
    else:
        ax.set(xlabel=x_axis_name)
    plt.show()
    '''
    
    avg_trip_distance = df_plot_data['avg_trip_distance'] 
    avg_direct_distance = df_plot_data['avg_direct_distance']
    tot_distance = df_plot_data['tot_distance']
    tot_deadheading_distance = df_plot_data['tot_deadheading_distance']
    deadheading_ratio = tot_deadheading_distance/tot_distance

def plot_demandKPIs(df_plot_data, plot_over_data, fig_suptitle):
    sns.set(font_scale = 2)
    # prepare name string for selection of x-axis range
    x_axis_name = str(plot_over_data.name)
    x_axis_data = df_plot_data[x_axis_name]
    # print(x_axis_data)
    # prepare mean values
    
    # for i in 
    nr_mode0 = df_plot_data['nr_mode0']
    nr_mode1 = df_plot_data['nr_mode1']
    nr_mode2 = df_plot_data['nr_mode2']
    nr_mode3 = df_plot_data['nr_mode3']
    nr_mode4 = df_plot_data['nr_mode4']
    
    MS_served = df_plot_data['MS_served']
    MS_served_pk = df_plot_data['MS_served_pk']
    MS_served_offpk = df_plot_data['MS_served_offpk']
    
    serviceratio = df_plot_data['served']/df_plot_data['demand']
    serviceratio_pk = df_plot_data['served_pk']/df_plot_data['demand_pk']
    serviceratio_offpk = df_plot_data['served_offpk']/df_plot_data['demand_offpk']
    
    punctuality = df_plot_data['punctuality']
    punctuality_pk = df_plot_data['punctuality_pk']
    punctuality_offpk = df_plot_data['punctuality_offpk']
    
    wait_avg = df_plot_data['wait_avg']
    wait_avg_pk = df_plot_data['wait_avg_pk']
    wait_avg_offpk = df_plot_data['wait_avg_offpk']
    wait_avg_served = df_plot_data['wait_served_avg']
    wait_avg_served_pk = df_plot_data['wait_served_avg_pk']
    wait_avg_served_offpk = df_plot_data['wait_served_avg_offpk']
    
    dev_avg = df_plot_data['dev_avg']
    dev_avg_pk = df_plot_data['dev_avg_pk']
    dev_avg_offpk = df_plot_data['dev_avg_offpk']
    
    load_factor = df_plot_data['load_factor']
    zero_time = df_plot_data['zero_time']
    use_time_ratio = df_plot_data['use_time_ratio']
    avg_trip_distance = df_plot_data['avg_trip_distance']
    avg_direct_trip_distance = df_plot_data['avg_direct_distance']
    avg_trip_duration = df_plot_data['avg_trip_duration']
    avg_direct_trip_duration = df_plot_data['avg_direct_ride_time']
    #fraction_consolidated_dropoffs = df_plot_data.loc[df_plot_data['fraction_consolidated_dropoffs']]
    effective_transport_ratio = df_plot_data['effective_transport_ratio']
    
    if plot_over_data.name == 'price1' or plot_over_data.name == 'price2' or plot_over_data.name == 'optout':
        plot_over_data = plot_over_data/100

    sns.set_palette('bright')    
    sns.set_color_codes('bright')
    fig, axes = plt.subplots(2, 3, figsize=(27,18))
    fig.suptitle(fig_suptitle, fontsize = 'x-large')
    axes[0,0].set_title('Modal shares', weight='bold')    
    sns.lineplot(x=plot_over_data, y=MS_served*100, ax=axes[0,0], marker='o',  linestyle = "-", color = 'b', label='All morning', ci='sd')
    sns.lineplot(x=plot_over_data, y=MS_served_pk*100, ax=axes[0,0], marker='o',  linestyle = "-", color = 'r', label='Peak', ci='sd')
    sns.lineplot(x=plot_over_data, y=MS_served_offpk*100, ax=axes[0,0], marker='o',  linestyle = "-", color = 'g', label='Off-peak', ci='sd')
    if plot_over_data.name == 'price1':
        axes[0,0].set(xlabel='Starting fare [euro]', ylabel='Modal shares [%]')
    elif plot_over_data.name == 'price2':
        axes[0,0].set(xlabel='Fare/min of direct ride time [euro/min]', ylabel='Modal shares [%]')
    elif plot_over_data.name == 'spacing':
        axes[0,0].set(xlabel='Stop spacing distance [km]', ylabel='Modal shares [%]')
    elif plot_over_data.name == 'demfact':
        axes[0,0].set(xlabel='Demand multiplicaton factor', ylabel='Modal shares [%]')
    else:
        axes[0,0].set(xlabel=x_axis_name, ylabel='Modal shares [%]')
    #plt.show()
   
    #fig, (ax2,ax3) = plt.subplots(2, 1, figsize=(12,24))
    axes[0,2].set_title('Rejection rate', weight='bold')
    sns.lineplot(x=plot_over_data, y=1-serviceratio, ax=axes[0,2], marker='o',  linestyle = "-", color = 'b', label='All morning', ci='sd')
    sns.lineplot(x=plot_over_data, y=1-serviceratio_pk, ax=axes[0,2], marker='o',  linestyle = "-", color = 'r', label='Peak', ci='sd')
    sns.lineplot(x=plot_over_data, y=1-serviceratio_offpk, ax=axes[0,2], marker='o',  linestyle = "-", color = 'g', label='Off-peak', ci='sd')
    if plot_over_data.name == 'price1':
        axes[0,2].set(xlabel='Starting fare [euro]', ylabel='Rejection rate')
    elif plot_over_data.name == 'price2':
        axes[0,2].set(xlabel='Fare/min of direct ride time [euro/min]', ylabel='Rejection rate')
    elif plot_over_data.name == 'spacing':
        axes[0,2].set(xlabel='Stop spacing distance [km]', ylabel='Rejection rate')
    elif plot_over_data.name == 'demfact':
        axes[0,2].set(xlabel='Demand multiplicaton factor', ylabel='Rejection rate')
    else:
        axes[0,2].set(xlabel=x_axis_name, ylabel='Rejection rate')

    punctuality_served = df_plot_data['punctuality_served']
    puntuality_served_pk = df_plot_data['punctuality_served_pk']
    puntuality_served_offpk = df_plot_data['punctuality_served_offpk']

    #fig, ax6 = plt.subplots(figsize=(8,8))
    axes[1,0].set_title('Travel time deviation factor (TTD)', weight='bold')
    sns.lineplot(x=plot_over_data, y=dev_avg, ax=axes[1,0], marker='o',  linestyle = "-", color = 'b', label='All morning', ci='sd')
    sns.lineplot(x=plot_over_data, y=dev_avg_pk, ax=axes[1,0], marker='o',  linestyle = "-", color = 'r', label='Peak', ci='sd')
    sns.lineplot(x=plot_over_data, y=dev_avg_offpk, ax=axes[1,0], marker='o',  linestyle = "-", color = 'g', label='Off-peak', ci='sd')
    axes[1,0].set(ylabel ='TTD')
    if plot_over_data.name == 'price1':
        axes[1,0].set(xlabel='Starting fare [euro]')
    elif plot_over_data.name == 'price2':
        axes[1,0].set(xlabel='Fare/min of direct ride time [euro/min]')
    elif plot_over_data.name == 'spacing':
        axes[1,0].set(xlabel='Stop spacing distance [km]')
    elif plot_over_data.name == 'demfact':
        axes[1,0].set(xlabel='Demand multiplicaton factor')
    else:
        axes[1,0].set(xlabel=x_axis_name)

    axes[0,1].set_title('Average waiting time per request', weight='bold')
    sns.lineplot(x=plot_over_data, y=wait_avg, ax=axes[0,1], marker='o',  linestyle = "-", color = 'b', label='All morning', ci='sd')
    sns.lineplot(x=plot_over_data, y=wait_avg_pk, ax=axes[0,1], marker='o',  linestyle = "-", color = 'r', label='Peak', ci='sd')
    sns.lineplot(x=plot_over_data, y=wait_avg_offpk, ax=axes[0,1], marker='o',  linestyle = "-", color = 'g', label='Off-peak', ci='sd')
    axes[0,1].get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    axes[0,1].get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    axes[0,1].grid(b=True, which='major', color='w', linewidth=1, axis='y')
    axes[0,1].grid(b=True, which='minor', color='w', linewidth=0.5, axis='y')
    axes[0,1].set(ylabel='Average waiting time')
    if plot_over_data.name == 'price1':
        axes[0,1].set(xlabel='Starting fare [euro]')
    elif plot_over_data.name == 'price2':
        axes[0,1].set(xlabel='Fare/min of direct ride time [euro/min]')
    elif plot_over_data.name == 'spacing':
        axes[0,1].set(xlabel='Stop spacing distance [km]')
    elif plot_over_data.name == 'demfact':
        axes[0,1].set(xlabel='Demand multiplicaton factor')
    else:
        axes[0,1].set(xlabel=x_axis_name)
    
    axes[1,1].set_title('Avg direct and actual trip times', weight='bold')
    sns.lineplot(x=plot_over_data, y=avg_trip_duration, ax=axes[1,1], marker='o',  linestyle = "-", label='avg trip', ci='sd')
    sns.lineplot(x=plot_over_data, y=avg_direct_trip_duration, ax=axes[1,1], marker='o',  linestyle = "-", label='avg direct trip', ci='sd')
    axes[1,1].set(ylabel ='Avg trip times [min]')
    if plot_over_data.name == 'price1':
        axes[1,1].set(xlabel='Starting fare [euro]')
    elif plot_over_data.name == 'price2':
        axes[1,1].set(xlabel='Fare/min of direct ride time [euro/min]')
    elif plot_over_data.name == 'spacing':
        axes[1,1].set(xlabel='Stop spacing distance [km]')
    elif plot_over_data.name == 'demfact':
        axes[1,1].set(xlabel='Demand multiplicaton factor')
    else:
        axes[1,1].set(xlabel=x_axis_name)
        
    axes[1,2].set_title('Avg direct and actual trip distances', weight='bold')
    sns.lineplot(x=plot_over_data, y=avg_trip_distance, ax=axes[1,2], marker='o',  linestyle = "-", label='avg trip', ci='sd')
    sns.lineplot(x=plot_over_data, y=avg_direct_trip_distance, ax=axes[1,2], marker='o',  linestyle = "-", label='avg direct trip', ci='sd')
    axes[1,2].set(ylabel ='Avg trip distances [km]')
    if plot_over_data.name == 'price1':
        axes[1,2].set(xlabel='Starting fare [euro]')
    elif plot_over_data.name == 'price2':
        axes[1,2].set(xlabel='Fare/min of direct ride time [euro/min]')
    elif plot_over_data.name == 'spacing':
        axes[1,2].set(xlabel='Stop spacing distance [km]')
    elif plot_over_data.name == 'demfact':
        axes[1,2].set(xlabel='Demand multiplicaton factor')
    else:
        axes[1,2].set(xlabel=x_axis_name)

    plt.show()
    
    #avg_trip_distance = df_plot_data['avg_trip_distance'] 
    #avg_direct_distance = df_plot_data['avg_direct_distance']
    #tot_distance = df_plot_data['tot_distance']
    #tot_deadheading_distance = df_plot_data['tot_deadheading_distance']
    #deadheading_ratio = tot_deadheading_distance/tot_distance

run_validation_results(df_output)

def calc_mean_ci(data):
    confidence=0.95
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    
    return m, h


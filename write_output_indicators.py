# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 20:44:30 2023

@author: dcdel
"""

import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl

file_name = 'aggregated_data_scenarios.csv'
df_output = pd.read_csv(file_name,delimiter=',')
sns.set(font_scale = 1.5)
sns.set_style('darkgrid') # "white", "dark", "ticks"

df_base_scenario = df_output.loc[(df_output['fleetsize'] == 5) &\
                            (df_output['price1'] == 200) &\
                            (df_output['price2'] == 37.5) &\
                            (df_output['spacing'] == 0) &\
                            (df_output['optout'] == 700) &\
                            (df_output['demfact'] == 1) &\
                            (df_output['area_size'] == 1) &\
                            (df_output['groups'] == 1) &\
                            (df_output['arrival_frequency'] == 1)]
df_base_scenario2 = df_output.loc[(df_output['fleetsize'] == 8) &\
                            (df_output['price1'] == 200) &\
                            (df_output['price2'] == 37.5) &\
                            (df_output['spacing'] == 0) &\
                            (df_output['optout'] == 700) &\
                            (df_output['demfact'] == 2) &\
                            (df_output['area_size'] == 1) &\
                            (df_output['groups'] == 1) &\
                            (df_output['arrival_frequency'] == 1)]
df_stops_scenario = df_output.loc[(df_output['fleetsize'] == 5) &\
                            (df_output['price1'] == 200) &\
                            (df_output['price2'] == 37.5) &\
                            (df_output['spacing'] == 0.4) &\
                            (df_output['optout'] == 700) &\
                            (df_output['demfact'] == 1) &\
                            (df_output['area_size'] == 1) &\
                            (df_output['groups'] == 1) &\
                            (df_output['arrival_frequency'] == 1)]
df_stops_scenario2 = df_output.loc[(df_output['fleetsize'] == 7) &\
                            (df_output['price1'] == 200) &\
                            (df_output['price2'] == 37.5) &\
                            (df_output['spacing'] == 0.4) &\
                            (df_output['optout'] == 700) &\
                            (df_output['demfact'] == 2) &\
                            (df_output['area_size'] == 1) &\
                            (df_output['groups'] == 1) &\
                            (df_output['arrival_frequency'] == 1)]    
df_groups_scenario = df_output.loc[(df_output['fleetsize'] == 4) &\
                            (df_output['price1'] == 200) &\
                            (df_output['price2'] == 37.5) &\
                            (df_output['spacing'] == 0) &\
                            (df_output['optout'] == 700) &\
                            (df_output['demfact'] == 1) &\
                            (df_output['area_size'] == 1) &\
                            (df_output['groups'] == 2) &\
                            (df_output['arrival_frequency'] == 1)]
df_groups_scenario2 = df_output.loc[(df_output['fleetsize'] == 7) &\
                            (df_output['price1'] == 200) &\
                            (df_output['price2'] == 37.5) &\
                            (df_output['spacing'] == 0) &\
                            (df_output['optout'] == 700) &\
                            (df_output['demfact'] == 2) &\
                            (df_output['area_size'] == 1) &\
                            (df_output['groups'] == 2) &\
                            (df_output['arrival_frequency'] == 1)]
df_freq_scenario = df_output.loc[(df_output['fleetsize'] == 5) &\
                            (df_output['price1'] == 200) &\
                            (df_output['price2'] == 37.5) &\
                            (df_output['spacing'] == 0) &\
                            (df_output['optout'] == 700) &\
                            (df_output['demfact'] == 1) &\
                            (df_output['area_size'] == 1) &\
                            (df_output['groups'] == 1) &\
                            (df_output['arrival_frequency'] == 2)]
df_freq_scenario2 = df_output.loc[(df_output['fleetsize'] == 8) &\
                            (df_output['price1'] == 200) &\
                            (df_output['price2'] == 37.5) &\
                            (df_output['spacing'] == 0) &\
                            (df_output['optout'] == 700) &\
                            (df_output['demfact'] == 2) &\
                            (df_output['area_size'] == 1) &\
                            (df_output['groups'] == 1) &\
                            (df_output['arrival_frequency'] == 2)]
df_area_scenario = df_output.loc[(df_output['fleetsize'] == 32) &\
                            (df_output['price1'] == 200) &\
                            (df_output['price2'] == 37.5) &\
                            (df_output['spacing'] == 0) &\
                            (df_output['optout'] == 700) &\
                            (df_output['demfact'] == 1) &\
                            (df_output['area_size'] == 2) &\
                            (df_output['groups'] == 1) &\
                            (df_output['arrival_frequency'] == 1)]
df_area_scenario2 = df_output.loc[(df_output['fleetsize'] == 60) &\
                            (df_output['price1'] == 200) &\
                            (df_output['price2'] == 37.5) &\
                            (df_output['spacing'] == 0) &\
                            (df_output['optout'] == 700) &\
                            (df_output['demfact'] == 2) &\
                            (df_output['area_size'] == 2) &\
                            (df_output['groups'] == 1) &\
                            (df_output['arrival_frequency'] == 1)]
    
def create_df_barplot(df_scenario):
    df_barplot = pd.DataFrame(columns=['Mode', 'Segment', 'Count']) 
    if len(df_scenario[(df_scenario['area_size'] == 1)] == len(df_scenario)):
        
        modal_shares0_bins = df_scenario.loc[:, 'bin0_0_1000':'bin0_2500_plus']
        modal_shares1_bins = df_scenario.loc[:, 'bin1_0_1000':'bin1_2500_plus']
        modal_shares2_bins = df_scenario.loc[:, 'bin2_0_1000':'bin2_2500_plus']
        modal_shares3_bins = df_scenario.loc[:, 'bin3_0_1000':'bin3_2500_plus']
        #modal_shares4_bins = df_scenario.loc[:, 'bin4_0_1000':'bin4_2500_plus']
        
        # If it looks like I would want to have it, then:
        for colname in modal_shares0_bins.columns:
            mode=np.empty(len(modal_shares0_bins), dtype=np.int8); mode.fill(0)
            segment = [colname[5:]] * len(modal_shares0_bins)
            counts = modal_shares0_bins[colname].to_numpy()
            data = {'Mode': mode,
                    'Segment': segment,
                    'Count' : counts}
            df_add_data = pd.DataFrame(data)
            df_barplot = pd.concat([df_barplot, df_add_data])
        for colname in modal_shares1_bins.columns:
            mode=np.empty(len(modal_shares1_bins), dtype=np.int8); mode.fill(1)
            segment = [colname[5:]] * len(modal_shares1_bins)
            counts = modal_shares1_bins[colname].to_numpy()
            data = {'Mode': mode,
                    'Segment': segment,
                    'Count' : counts}
            df_add_data = pd.DataFrame(data)
            df_barplot = pd.concat([df_barplot, df_add_data])
        for colname in modal_shares2_bins.columns:
            mode=np.empty(len(modal_shares2_bins), dtype=np.int8); mode.fill(2)
            segment = [colname[5:]] * len(modal_shares2_bins)
            counts = modal_shares2_bins[colname].to_numpy()
            data = {'Mode': mode,
                    'Segment': segment,
                    'Count' : counts}
            df_add_data = pd.DataFrame(data)
            df_barplot = pd.concat([df_barplot, df_add_data])
        for colname in modal_shares3_bins.columns:
            mode=np.empty(len(modal_shares3_bins), dtype=np.int8); mode.fill(3)
            segment = [colname[5:]] * len(modal_shares3_bins)
            counts = modal_shares3_bins[colname].to_numpy()
            data = {'Mode': mode,
                    'Segment': segment,
                    'Count' : counts}
            df_add_data = pd.DataFrame(data)
            df_barplot = pd.concat([df_barplot, df_add_data])
        '''
        for colname in modal_shares4_bins.columns:
            mode=np.empty(len(modal_shares4_bins), dtype=np.int8); mode.fill(4)
            segment = [colname[5:]] * len(modal_shares4_bins)
            counts = modal_shares4_bins[colname].to_numpy()
            data = {'Mode': mode,
                    'Segment': segment,
                    'Count' : counts}
            df_add_data = pd.DataFrame(data)
            df_barplot = pd.concat([df_barplot, df_add_data])
        '''
    else:
        modal_shares0_bins = df_scenario.loc[:, 'bin0_0_1000':'bin0_2000_2500']
        modal_shares1_bins = df_scenario.loc[:, 'bin1_0_1000':'bin1_2000_2500']
        modal_shares2_bins = df_scenario.loc[:, 'bin2_0_1000':'bin2_2000_2500']
        modal_shares3_bins = df_scenario.loc[:, 'bin3_0_1000':'bin3_2000_2500']
        #modal_shares4_bins = df_scenario.loc[:, 'bin4_0_1000':'bin4_6000_7100']
        modal_shares0_bins = pd.concat([modal_shares0_bins,df_scenario.loc[:, 'bin0_2500_3000':'bin0_6000_7100']])
        modal_shares1_bins = pd.concat([modal_shares1_bins,df_scenario.loc[:, 'bin1_2500_3000':'bin1_6000_7100']])
        modal_shares2_bins = pd.concat([modal_shares2_bins,df_scenario.loc[:, 'bin2_2500_3000':'bin2_6000_7100']])
        modal_shares3_bins = pd.concat([modal_shares3_bins,df_scenario.loc[:, 'bin3_2500_3000':'bin3_6000_7100']])
        
        
        for colname in modal_shares0_bins.columns:
            if colname != 'bin0_2500_plus':
                # If it looks like I would want to have it, then:
                for colname in modal_shares0_bins.columns:
                    mode=np.empty(len(modal_shares0_bins), dtype=np.int8); mode.fill(0)
                    segment = [colname[5:]] * len(modal_shares0_bins)
                    counts = modal_shares0_bins[colname].to_numpy()
                    data = {'Mode': mode,
                            'Segment': segment,
                            'Count' : counts}
                    df_add_data = pd.DataFrame(data)
                    df_barplot = pd.concat([df_barplot, df_add_data])
            if colname != 'bin1_2500_plus':   
                for colname in modal_shares1_bins.columns:
                    mode=np.empty(len(modal_shares1_bins), dtype=np.int8); mode.fill(1)
                    segment = [colname[5:]] * len(modal_shares1_bins)
                    counts = modal_shares1_bins[colname].to_numpy()
                    data = {'Mode': mode,
                            'Segment': segment,
                            'Count' : counts}
                    df_add_data = pd.DataFrame(data)
                    df_barplot = pd.concat([df_barplot, df_add_data])
            if colname != 'bin2_2500_plus': 
                for colname in modal_shares2_bins.columns:
                    mode=np.empty(len(modal_shares2_bins), dtype=np.int8); mode.fill(2)
                    segment = [colname[5:]] * len(modal_shares2_bins)
                    counts = modal_shares2_bins[colname].to_numpy()
                    data = {'Mode': mode,
                            'Segment': segment,
                            'Count' : counts}
                    df_add_data = pd.DataFrame(data)
                    df_barplot = pd.concat([df_barplot, df_add_data])
            if colname != 'bin3_2500_plus':
                for colname in modal_shares3_bins.columns:
                    mode=np.empty(len(modal_shares3_bins), dtype=np.int8); mode.fill(3)
                    segment = [colname[5:]] * len(modal_shares3_bins)
                    counts = modal_shares3_bins[colname].to_numpy()
                    data = {'Mode': mode,
                            'Segment': segment,
                            'Count' : counts}
                    df_add_data = pd.DataFrame(data)
                    df_barplot = pd.concat([df_barplot, df_add_data])
            '''
            if colname != 'bin4_2500_plus':
                for colname in modal_shares4_bins.columns:
                    mode=np.empty(len(modal_shares4_bins), dtype=np.int8); mode.fill(3)
                    segment = [colname[5:]] * len(modal_shares4_bins)
                    counts = modal_shares4_bins[colname].to_numpy()
                    data = {'Mode': mode,
                            'Segment': segment,
                            'Count' : counts}
                    df_add_data = pd.DataFrame(data)
                    df_barplot = pd.concat([df_barplot, df_add_data])
            '''
    return (df_barplot)

def plot_barplot(df_barplot, df_name, font_size):
    sns.set(font_scale = font_size)

    x, y, hue = 'Segment', 'Count', 'Mode'
    data = df_barplot
    hue_order = [0, 1, 2, 3]   
    ax = sns.barplot(data = data, x=x, y=y,\
                hue=hue, capsize=.1)
    #ax = sns.barplot(data=data, estimator=lambda x: sum(x==0)*100.0/len(hue), x=x, y=y,\
    #            hue=hue, capsize=.1)
    if font_size == 1.1: 
        ax.set_title(df_name, fontsize = 'large', weight = 'bold')
    else:
        ax.set_title(df_name, weight = 'bold')
    
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1, axis='y')
    ax.grid(b=True, which='minor', color='w', linewidth=0.5, axis='y')
    
    numX=len([x for x in data['Segment'].unique() if x==x])
    # 2. The bars are created in hue order, organize them
    bars = ax.patches
    ## 2a. For each X variable
    for ind in range(numX):
        ## 2b. Get every hue bar
        ##     ex. 8 X categories, 4 hues =>
        ##    [0, 8, 16, 24] are hue bars for 1st X category
        hueBars=bars[ind:][::numX]
        ## 2c. Get the total height (for percentages)
        total = sum([x.get_height() for x in hueBars])

        # 3. Print the percentage on the bars
        for bar in hueBars:
            ax.text(bar.get_x() + bar.get_width()/2.,
                    bar.get_height()+5,
                    f'{bar.get_height()/total:.0%}',
                    ha="center",va="bottom")
    
    sns.set(rc={"figure.figsize":(16, 8)})
    plt.show()

font_size = 1.5
df_barplot = create_df_barplot(df_base_scenario)
df_name = 'Mode choice by distance for base scenario, normal volume'
plot_barplot(df_barplot, df_name, font_size)
df_barplot = create_df_barplot(df_base_scenario2)
df_name = 'Mode choice by distance for base scenario, high volume'
plot_barplot(df_barplot, df_name, font_size)
df_barplot = create_df_barplot(df_groups_scenario)
df_name = 'Mode choice by distance for business travelers scenario, normal volume'
plot_barplot(df_barplot, df_name, font_size)
df_barplot = create_df_barplot(df_groups_scenario2)
df_name = 'Mode choice by distance for business travelers scenario, high volume'
plot_barplot(df_barplot, df_name, font_size)
df_barplot = create_df_barplot(df_stops_scenario)
df_name = 'Mode choice by distance for drop-off stops scenario, normal volume'
plot_barplot(df_barplot, df_name, font_size)
df_barplot = create_df_barplot(df_stops_scenario2)
df_name = 'Mode choice by distance for drop-off stops scenario, high volume'
plot_barplot(df_barplot, df_name, font_size)
df_barplot = create_df_barplot(df_freq_scenario)
df_name = 'Mode choice by distance for decreased headways scenario, normal volume'
plot_barplot(df_barplot, df_name, font_size)
df_barplot = create_df_barplot(df_freq_scenario2)
df_name = 'Mode choice by distance for decreased headways scenario, high volume'
plot_barplot(df_barplot, df_name, font_size)
font_size = 1.1
df_barplot = create_df_barplot(df_area_scenario)
df_name = 'Mode choice by distance for increased area size scenario, normal volume'
plot_barplot(df_barplot, df_name, font_size)
df_barplot = create_df_barplot(df_area_scenario2)
df_name = 'Mode choice by distance for increased area size scenario, high volume'
plot_barplot(df_barplot, df_name, font_size)


def calc_mean_ci(data):
    confidence=0.95
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    #m=round(m,3)
    #h=round(h,3)
    return m, h

def write_output_indicators(df_scenario, scenario_name):
    output = pd.DataFrame(columns = ['scenario_name',
                                    'travelers',
                                    'served',
                                    'market_share',
                                    'tot_dist',
                                    'effective_distance',
                                    'effective_distance_ratio',
                                    'occupation_rate',
                                    'use_time_ratio',
                                    'reject_rate',
                                    'avg_waiting_time',
                                    'TTD',
                                    'avg_dist',
                                    'avg_dir_dist',
                                    'avg_trip_time',
                                    'avg_dir_trip_time',
                                    'avg_walk_time',
                                    'mean_served',
                                    'mean_market_share',
                                    'mean_tot_dist',
                                    'mean_effective_dist',
                                    'mean_effect_dist_ratio',
                                    'mean_occ_r',
                                    'mean_use_time_r',
                                    'mean_reject_r',
                                    'mean_wait_time',
                                    'mean_ttd',
                                    'mean_avg_dist',
                                    'mean_dir_dist',
                                    'mean_avg_trip_time',
                                    'mean_avg_dir_time',
                                    'mean_avg_walk_time'
                                     ])
    
    travelers = int(df_scenario['nr_travellers'].mean())
    mean_served, std_var_served = calc_mean_ci(df_scenario['served'])
    served = '{0}, [{1}, {2}]'.format(round(mean_served,1), round(mean_served-std_var_served,1), round(mean_served+std_var_served,1))
    mean_market_share, std_var_market_share = calc_mean_ci(df_scenario['MS_served'])
    market_share = '{0}, [{1}, {2}]'.format(round(mean_market_share,3), round(mean_market_share-std_var_market_share,3), round(mean_market_share+std_var_market_share,3))
    mean_tot_dist, std_var_tot_dist = calc_mean_ci(df_scenario['tot_distance'])
    tot_dist = '{0}, [{1}, {2}]'.format(round(mean_tot_dist,1), round(mean_tot_dist-std_var_tot_dist,1), round(mean_tot_dist+std_var_tot_dist,1))
    mean_effective_dist, std_var_effective_dist = calc_mean_ci(df_scenario['effective_transport_dist'])
    effective_distance = '{0}, [{1}, {2}]'.format(round(mean_effective_dist,1), round(mean_effective_dist-std_var_effective_dist,1), round(mean_effective_dist+std_var_effective_dist,1))
    mean_effect_dist_ratio, std_var_effect_dist_ratio = calc_mean_ci(df_scenario['effective_transport_ratio'])
    effective_distance_ratio = '{0}, [{1}, {2}]'.format(round(mean_effect_dist_ratio,3), round(mean_effect_dist_ratio-std_var_effect_dist_ratio,3), round(mean_effect_dist_ratio+std_var_effect_dist_ratio,3))
    mean_occ_r, std_variance_occ_r = calc_mean_ci(df_scenario['load_factor'])
    occupation_rate = '{0}, [{1}, {2}]'.format(round(mean_occ_r,2), round(mean_occ_r-std_variance_occ_r,2), round(mean_occ_r+std_variance_occ_r,2))
    mean_use_time_r, std_var_use_time_r = calc_mean_ci(df_scenario['use_time_ratio'])
    use_time_ratio = '{0}, [{1}, {2}]'.format(round(mean_use_time_r,3),round(mean_use_time_r-std_var_use_time_r,3),round(mean_use_time_r+std_var_use_time_r,3))    
    mean_reject_r, std_var_reject_r = calc_mean_ci(1-df_scenario['service_ratio'])
    reject_rate = '{0}, [{1}, {2}]'.format(round(mean_reject_r,3),round(mean_reject_r-std_var_reject_r,3),round(mean_reject_r+std_var_reject_r,3))
    mean_wait_time,std_var_wait_time = calc_mean_ci(df_scenario['wait_avg'])
    avg_waiting_time = '{0}, [{1}, {2}]'.format(round(mean_wait_time,2),round(mean_wait_time-std_var_wait_time,2),round(mean_wait_time+std_var_wait_time,2))
    mean_ttd, std_var_ttd = calc_mean_ci(df_scenario['dev_avg'])
    TTD = '{0}, [{1}, {2}]'.format(round(mean_ttd,2),round(mean_ttd-std_var_ttd,2),round(mean_ttd+std_var_ttd,2))
    mean_avg_dist, std_var_avg_dist = calc_mean_ci(df_scenario['avg_trip_distance'])
    avg_dist = '{0}, [{1}, {2}]'.format(round(mean_avg_dist,2),round(mean_avg_dist-std_var_avg_dist,2),round(mean_avg_dist+std_var_avg_dist,2))    
    mean_dir_dist, std_var_dir_dist = calc_mean_ci(df_scenario['avg_direct_distance'])
    avg_dir_dist = '{0}, [{1}, {2}]'.format(round(mean_dir_dist,2),round(mean_dir_dist-std_var_dir_dist,2),round(mean_dir_dist+std_var_dir_dist,2))
    mean_avg_trip_time, std_var_avg_trip_time = calc_mean_ci(df_scenario['avg_trip_duration'])
    avg_trip_time = '{0}, [{1}, {2}]'.format(round(mean_avg_trip_time,2),round(mean_avg_trip_time-std_var_avg_trip_time,2),round(mean_avg_trip_time+std_var_avg_trip_time,2))
    mean_avg_dir_time, std_var_avg_dir_time = calc_mean_ci(df_scenario['avg_direct_ride_time'])
    avg_dir_trip_time = '{0}, [{1}, {2}]'.format(round(mean_avg_dir_time,2),round(mean_avg_dir_time-std_var_avg_dir_time,2),round(mean_avg_dir_time+std_var_avg_dir_time,2))
    mean_avg_walk_time, std_var_avg_walk_time = calc_mean_ci(df_scenario['walk_avg'])
    avg_walk_time = '{0}, [{1}, {2}]'.format(round(mean_avg_walk_time,2),round(mean_avg_walk_time-std_var_avg_walk_time,2),round(mean_avg_walk_time+std_var_avg_walk_time,2))
    
    output_indicators = [scenario_name,
                        travelers,
                        served,
                        market_share,
                        tot_dist,
                        effective_distance,
                        effective_distance_ratio,
                        occupation_rate,
                        use_time_ratio,
                        reject_rate,
                        avg_waiting_time,
                        TTD,
                        avg_dist,
                        avg_dir_dist,
                        avg_trip_time,
                        avg_dir_trip_time,
                        avg_walk_time,
                        mean_served,
                        mean_market_share,
                        mean_tot_dist,
                        mean_effective_dist,
                        mean_effect_dist_ratio,
                        mean_occ_r,
                        mean_use_time_r,
                        mean_reject_r,
                        mean_wait_time,
                        mean_ttd,
                        mean_avg_dist,
                        mean_dir_dist,
                        mean_avg_trip_time,
                        mean_avg_dir_time,
                        mean_avg_walk_time
                        ]
    
    file_name =  'output_indicators.csv'   
    try:
        output.loc[len(output)] = output_indicators
        output.to_csv(file_name, index=False, sep=',', header=True, mode='x')
    except FileExistsError:
        output = pd.read_csv(file_name)
        output.loc[len(output)] = output_indicators
        output.to_csv(file_name, index=False, sep=',', header=True, mode = 'w+')
    
    print('done with writing indicators for ' + str(scenario_name) + ' to ' + str(file_name))

'''
write_output_indicators(df_base_scenario, 'base ')
write_output_indicators(df_base_scenario2, 'base 2x demand')
write_output_indicators(df_stops_scenario, 'stops')
write_output_indicators(df_stops_scenario2, 'stops 2x demand')
write_output_indicators(df_groups_scenario, 'groups scenario')
write_output_indicators(df_groups_scenario2, 'groups scenario2')
write_output_indicators(df_freq_scenario, 'decreased headway')
write_output_indicators(df_freq_scenario2, 'decreased headway 2x demand')
write_output_indicators(df_area_scenario, 'big area size')
write_output_indicators(df_area_scenario2, 'big area size 2 x demand')
'''

#sns.barplot(x=modal_shares0_brackets.columns, y=modal_shares0_brackets,\
#            hue=modal_shares0_brackets.columns)

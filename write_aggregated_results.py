# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:30:31 2023

@author: dcdel
"""

import os
import pandas as pd
import re
import numpy as np

data_dir = 'test_results_csv'
all_files = os.listdir(data_dir)

for file_name in all_files:
    df_output = pd.read_csv(str(data_dir)+'/'+str(file_name),delimiter=',',skip_blank_lines=True)

    print('start with ' + str(file_name))

    '''
    DEMAND KPI'S
    
    Market share of SAV (%) – distance bins (also by peak)
    Impact on other modal shares (%/mode) – distance bins <- scatter plot with different colors (also by peak)
    Rejector rate (%) – distance bins
    Rejector rate (%) – by peak
    Punctuality (% served at the requested time) – by peak
    Waiting time (min) – bins (also by peak)
    Walking time (min) – bins (also by peak)
    Walking times accepted vs. walking time offered (min) – distance bins (also by peak)
    Waiting times accepted vs. Average waiting times offered (min) – distance bins (also by peak)
    Avg deviation factor (ratio) (by peak)
    Arrivals (train bins)
    Demand served per train (bins) – train bins
    '''
    
    # Aggregate the outputs as:
    output = pd.DataFrame(columns = ['file_name','fleetsize','price1',\
                     'price2','spacing','optout','demfact','devfact',\
                     'area_size','arrival_frequency','groups','run_seed',\
                     'nr_travellers','nr_mode0','nr_mode1',\
                     'nr_mode2','nr_mode3','nr_mode4','demand', 'demand_pk',\
                     'demand_offpk','served','served_pk','served_offpk',\
                     'service_ratio','service_ratio_pk','service_ratio_offpk',\
                     'MS_demand','MS_demand_pk','MS_demand_offpk',\
                     'MS_served','MS_served_pk','MS_served_offpk',\
                     'punctuality','punctuality_pk','punctuality_offpk',\
                     'punctuality_served','punctuality_served_pk',
                     'punctuality_served_offpk','wait_avg','wait_avg_pk',\
                     'wait_avg_offpk','wait_served_avg','wait_served_avg_pk',\
                     'wait_served_avg_offpk','walk_avg','walk_avg_pk',\
                     'walk_avg_offpk','walk_served_avg','walk_served_avg_pk',\
                     'walk_served_avg_offpk','dev_avg','dev_avg_pk',\
                     'dev_avg_offpk','nr_of_routes','zero_time',\
                     'tot_use_time','use_time_ratio','avg_veh_use_time',\
                     'avg_idle_time','fraction_idle_time','tot_distance',\
                     'tot_deadheading_distance','tot_shared_distance',\
                     'tot_non_shared_dist','avg_deadheading_distance',\
                     'avg_direct_distance','avg_trip_distance',\
                     'avg_route_distance','avg_direct_ride_time',\
                     'avg_trip_duration','load_factor','consolidated_dropoffs',\
                     'fraction_consolidated_dropoffs','transportation_dist',\
                     'effective_transport_dist','transportation_ratio',\
                     'effective_transport_ratio','bin0_0_1000',\
                     'bin0_1000_2000','bin0_2000_2500','bin0_2500_plus',\
                     'bin0_2500_3000','bin0_3000_4000','bin0_4000_6000',\
                     'bin0_6000_7100','bin1_0_1000','bin1_1000_2000',\
                     'bin1_2000_2500','bin1_2500_plus',\
                     'bin1_2500_3000','bin1_3000_4000','bin1_4000_6000',\
                     'bin1_6000_7100','bin2_0_1000','bin2_1000_2000',\
                     'bin2_2000_2500','bin2_2500_plus',\
                     'bin2_2500_3000','bin2_3000_4000','bin2_4000_6000',\
                     'bin2_6000_7100','bin3_0_1000','bin3_1000_2000',\
                     'bin3_2000_2500','bin3_2500_plus',\
                     'bin3_2500_3000','bin3_3000_4000','bin3_4000_6000',\
                     'bin3_6000_7100','bin4_0_1000',\
                     'bin4_1000_2000','bin4_2000_2500','bin4_2500_plus',\
                     'bin4_2500_3000','bin4_3000_4000','bin4_4000_6000',\
                     'bin4_6000_7100'])
        
    # EXTRACT THE SIMULATION INPUT VARIABLES FROM THE FILENAMES    
    # Fleetsize
    fleetsize_pattern = '(?<=fl)\d+(?=_)'
    fleetsize = re.search(fleetsize_pattern, file_name).group()
    
    # Price
    price1_pattern = '(?<=pr)\d+(?=_)'
    price1 = re.search(price1_pattern, file_name)
    if price1 != None:
        price1 = price1.group()
    else:
        price1_pattern = '(?<=pr)\d+.\d+(?=_)'
        price1 = re.search(price1_pattern, file_name).group()
    
    price2_pattern = '(?<=_)\d+(?=_space)'
    price2 = re.search(price2_pattern, file_name)
    if price2 != None:
        price2 = price2.group()
    else:
        price2_pattern = '(?<=_)\d+.\d+(?=_space)'
        price2 = re.search(price2_pattern, file_name).group()
    
    spacing_pattern = '(?<=space)\d+(?=_)'
    spacing = re.search(spacing_pattern, file_name)
    if spacing != None:
        spacing = spacing.group()
    elif spacing == None: # This should be the one that is executed
        spacing_pattern = '(?<=space)\d+.\d+(?=_)'
        spacing = re.search(spacing_pattern, file_name).group()
    else:
        spacing_pattern = '(?<=space).\d+(?=_)'
        spacing = re.search(spacing_pattern, file_name).group()
    optout_pattern = '(?<=opt)\d+.\d+(?=_)'
    optout = re.search(optout_pattern, file_name)
    if optout != None:
        optout = optout.group()
    else: 
        optout_pattern = '(?<=opt)\d+(?=_)'
        optout = re.search(optout_pattern, file_name).group()
    
    demfact_pattern = '(?<=dem)\d+.\d+(?=_)'
    demfact = re.search(demfact_pattern, file_name)
    if demfact != None:
        demfact = demfact.group()
    else:
        try:
            demfact_pattern = '(?<=dem)\d+(?=_)'
            demfact = re.search(demfact_pattern, file_name).group()
        except:
            print('"demfact..._" lijkt niet in de naam voor te komen')
            demfact = False
            
    devfact_pattern = '(?<=dev)\d+.\d+(?=_)'
    devfact = re.search(devfact_pattern, file_name)
    if devfact != None:
        devfact = devfact.group()
    else:
        try:
            devfact_pattern = '(?<=dev)\d+(?=_)'
            devfact = re.search(devfact_pattern, file_name).group()
        except:
            print('"devfact..._" lijkt niet in de naam voor te komen')
            devfact = False
    
    area_pattern = '(?<=area)\d+(?=_)'
    area_size = re.search(area_pattern, file_name).group()
    
    frequency_pattern = '(?<=freq)\d+(?=_)'
    arrival_frequency = re.search(frequency_pattern, file_name).group()
    
    groups_pattern = '(?<=groups)\d+(?=_)'
    traveler_groups = re.search(groups_pattern, file_name).group()
    
    run_seed_pattern = '(?<=RS)\d+(?=.)'
    run_seed = re.search(run_seed_pattern, file_name).group()
    
    '''
    if run_seed != None:
        run_seed = run_seed.group()
    else:
        try:
            devfact_pattern = '(?<=RS)\d+(?=.)'
            devfact = re.search(demfact_pattern, file_name).group()
        except:
            print('"devfact..._" lijkt niet in de naam voor te komen')
            devfact = False
    '''
    
    #re.search(fleet\d{1,2}, 
    #is_year = re.search('(?:%s)' %year, file_name)    
        
    # MARKET SHARES
    nr_travellers = len(df_output.loc[df_output['mode choice'].notna()])
    nr_mode0 = len(df_output.loc[(df_output['mode choice'] == 0)].index)
    nr_mode1 = len(df_output.loc[(df_output['mode choice'] == 1)].index)
    nr_mode2 = len(df_output.loc[(df_output['mode choice'] == 2)].index)
    nr_mode3 = len(df_output.loc[(df_output['mode choice'] == 3)].index)
    nr_mode4 = len(df_output.loc[(df_output['mode choice'] == 4)].index)


    def bin_it(mode):
        bin_0_1000 = len(df_output.loc[(df_output['mode choice'] == mode) &\
                                     (df_output['distance'] < 1)])
        bin_1000_2000 = len(df_output.loc[(df_output['mode choice'] == mode) &\
                                     (df_output['distance'] >= 1) &\
                                     (df_output['distance'] < 2)])    
        #bin_1500_2000 = len(df_output.loc[(df_output['mode choice'] == mode) &\
        #                             (df_output['distance'] >= 1.5) &\
        #                             (df_output['distance'] < 2)])    
        bin_2000_2500 = len(df_output.loc[(df_output['mode choice'] == mode) &\
                                     (df_output['distance'] >= 2) &\
                                     (df_output['distance'] < 2.5)])
        bin_2500_plus = len(df_output.loc[(df_output['mode choice'] == mode) &\
                                     (df_output['distance'] >= 2.5)])
        bin_2500_3000 = len(df_output.loc[(df_output['mode choice'] == mode) &\
                                     (df_output['distance'] >= 2.5) &\
                                     (df_output['distance'] < 3)])
        bin_3000_4000 = len(df_output.loc[(df_output['mode choice'] == mode) &\
                                     (df_output['distance'] >= 3) &\
                                     (df_output['distance'] < 4)])
        bin_4000_6000 = len(df_output.loc[(df_output['mode choice'] == mode) &\
                                     (df_output['distance'] >= 4) &\
                                     (df_output['distance'] < 6)])    
        bin_6000_7100 = len(df_output.loc[(df_output['mode choice'] == mode) &\
                                     (df_output['distance'] >= 6) &\
                                     (df_output['distance'] < 8)])
            
        return bin_0_1000, bin_1000_2000, bin_2000_2500,\
               bin_2500_plus, bin_2500_3000, bin_3000_4000, bin_4000_6000,\
               bin_6000_7100 
    
    #bin4_0_1000 = len(df_output.loc[(df_output['mode choice'] == 4)])
    bin0_0_1000, bin0_1000_2000, bin0_2000_2500,\
           bin0_2500_plus, bin0_2500_3000, bin0_3000_4000, bin0_4000_6000,\
           bin0_6000_7100 = bin_it(0)       
    bin1_0_1000, bin1_1000_2000, bin1_2000_2500,\
           bin1_2500_plus, bin1_2500_3000, bin1_3000_4000, bin1_4000_6000,\
           bin1_6000_7100 = bin_it(1)
    bin2_0_1000, bin2_1000_2000, bin2_2000_2500,\
           bin2_2500_plus, bin2_2500_3000, bin2_3000_4000, bin2_4000_6000,\
           bin2_6000_7100 = bin_it(2)
    bin3_0_1000, bin3_1000_2000, bin3_2000_2500,\
           bin3_2500_plus, bin3_2500_3000, bin3_3000_4000, bin3_4000_6000,\
           bin3_6000_7100 = bin_it(3)
    bin4_0_1000, bin4_1000_2000, bin4_2000_2500,\
           bin4_2500_plus, bin4_2500_3000, bin4_3000_4000, bin4_4000_6000,\
           bin4_6000_7100 = bin_it(4) 
       
    # ACCEPTANCE RATES (total, peak and off-peak)
    df_interest = df_output.loc[df_output['interest'] == 0]
    demand = len(df_interest.index)
    servedDemand = len(df_interest.loc[df_interest['mode choice'] == 0].index)
    try:
        serviceRatio = servedDemand/demand
    except:
        serviceRatio = 0
    
    df_interest_pk = df_interest.loc[(df_interest['requested time'] >= 90) & (df_interest['requested time'] < 210)]
    demand_pk = len(df_interest_pk.index)
    servedDemand_pk = len(df_interest_pk.loc[df_interest_pk['mode choice'] == 0].index)
    try:
        serviceRatio_pk = servedDemand_pk/demand_pk
    except:
        serviceRatio_pk = 0
    
    df_interest_offpk = df_interest.loc[(df_interest['requested time'] < 90) | (df_interest['requested time'] >= 210)]
    demand_offpk = len(df_interest_offpk.index)
    servedDemand_offpk = len(df_interest_offpk.loc[df_interest_offpk['mode choice'] == 0].index)
    try:
        serviceRatio_offpk = servedDemand_offpk/demand_offpk
    except:
        servedDemand_offpk = 0
    
    #print('Travellers by total, peak, and off-peak')
    df_market = df_output.loc[df_output['mode choice'].notna()]
    market = nr_travellers
    #print(market)
    MS_demand = demand/market
    MS_served = servedDemand/market
    market_pk = len(df_market.loc[(df_market['requested time'] >= 90) & (df_market['requested time'] < 210)])
    MS_demand_pk = demand_pk/market_pk
    MS_served_pk = servedDemand_pk/market_pk
    market_offpk = len(df_market.loc[(df_market['requested time'] < 90) | (df_market['requested time'] >= 210)])
    MS_demand_offpk = demand_offpk/market_offpk
    MS_served_offpk = servedDemand_offpk/market_offpk
    #print(market_pk,market_offpk)

    #print(str(df_market.loc[(df_market['requested time'] >= 90) & (df_market['requested time'] < 210)]['traveler ID'].to_string()) + ' & ' + str(df_market.loc[(df_market['requested time'] >= 90) & (df_market['requested time'] < 210)]['requested time'].to_string()))
    #print(df_market.loc[(df_market['requested time'] < 90) | (df_market['requested time'] >= 210)]['traveler ID'].head(80).to_string())
    
    # PUNCTUALITY means nr. of requests that have no waiting time
    punctual = len(df_interest[(df_interest['waiting time'] == 0)].index)
    punctualServed = len(df_interest.loc[df_interest['mode choice'] == 0 | (df_interest['waiting time'] == 0)].index)
    
    try:
        punctuality = punctual/demand
    except:
        punctuality = 0
    try:
        punctuality_served = punctualServed/servedDemand
    except:
        punctuality_served = 0
    
    punctuality_served = punctualServed/servedDemand
    
    punctual_pk = len(df_interest_pk[(df_interest_pk['waiting time'] == 0)].index)
    punctualServed_pk = len(df_interest_pk.loc[df_interest_pk['mode choice'] == 0 | (df_interest_pk['waiting time'] == 0)].index)
    
    try:
        punctuality_pk = punctual_pk/demand_pk
    except:
        punctuality_pk = 0
    try:
        punctuality_served_pk = punctualServed_pk/servedDemand_pk
    except:
        punctuality_served_pk = 0
    
    punctual_offpk = len(df_interest_offpk[(df_interest_offpk['waiting time'] == 0)].index)
    punctualServed_offpk = len(df_interest_offpk.loc[df_interest_offpk['mode choice'] == 0 | (df_interest_offpk['waiting time'] == 0)].index)
    try:
        punctuality_offpk = punctual_offpk/demand_offpk
    except:
        punctuality_offpk = 0
    try:
        punctuality_served_offpk= punctualServed_offpk/servedDemand_offpk
    except:
        punctuality_served_offpk = 0    
    
    # AVERAGE WAITING TIMES OF OFFERS (total, peak, off-peak)
    wait_avg_offpk = df_interest_offpk['waiting time'].mean()
    wait_avg_pk = df_interest_pk['waiting time'].mean()
    wait_avg = df_interest['waiting time'].mean()
    
    # Create a dataframe to determine kpi's of served trips
    df_servedDemand = df_interest.loc[df_interest['mode choice'] == 0]
    df_servedDemand_pk = df_servedDemand.loc[(df_servedDemand['requested time'] >= 90) & (df_servedDemand['requested time'] < 210)]
    df_servedDemand_offpk = df_servedDemand.loc[(df_servedDemand['requested time'] < 90) | (df_servedDemand['requested time'] >= 210)]
    # AVERAGE WAITING TIMES ACCEPTED (total, peak, off-peak)    
    wait_served_avg = df_servedDemand['waiting time'].mean()
    wait_served_avg_pk = df_servedDemand_pk['waiting time'].mean()
    wait_served_avg_offpk = df_servedDemand_offpk['waiting time'].mean()
    
    # AVERAGE WALK TIMES OF OFFERS (total, peak, off-peak)
    walk_avg = df_interest['walking time'].mean()
    walk_avg_pk = df_interest_pk['walking time'].mean()
    walk_avg_offpk = df_interest_offpk['walking time'].mean()
    
    # AVERAGE WAITING TIMES ACCEPTED (total, peak, off-peak)
    walk_served_avg_pk = df_servedDemand_pk['walking time'].mean()
    walk_served_avg_offpk = df_servedDemand_offpk['walking time'].mean()
    walk_served_avg = df_servedDemand['walking time'].mean()
    
    # First the deviation factor (from the routes)
    df_devFacts = df_output.loc[(df_output['devFact'].notna() == True) & df_output['devFact'] != '']
    dev_avg = df_devFacts['devFact'].mean()                
    dev_avg_pk = df_output.loc[(df_output['requested time'] >= 90) & (df_output['requested time'] < 210)]['devFact'].mean()
    dev_avg_offpk = df_output.loc[(df_output['requested time'] < 90) | (df_output['requested time'] >= 210)]['devFact'].mean()
    # Then, the waiting time and walking time from the offers
        
    '''
    SUPPLY RELATED KPI'S

    Ride occupation [sort by unique nr. of cells with a factors and the requested time combinations, then divide total by total rides]
    Ridesharing rate (%) [sort by different]
    Shared distance (%)
    Time that x vehicles are in use
    Zero vehicle time
    Idle time %/Vehicle utilization%
    Avg direct ride distance
    Avg direct ride time
    Avg ride distance 
    Avg ride time
    Deadheading distance per trip
    Effective vehicle transportation distance ratio
    Consolidated drop-offs %
    '''
    
    rides = df_output.groupby(['vehicle', 'requested time', 'mode choice']).size()
    rides_nr = len(rides)
    served_trips = rides.sum()
    non_shared_trips = rides.value_counts()[1]
    ride_sharing_rate = served_trips/rides_nr
    
    'Vehicle IDLE TIME and USE TIME in [min] and as fractions'
    
    # Determine total use time
    df_routetimes = df_output.loc[df_output['route time'].notna()]
    tot_use_time = 0
    departure_times = []
    return_times = []
    fleetsize
    for vehicle_nr in range(int(fleetsize)):
        vehicle = 'SAV (sav.' + str(int(vehicle_nr)) + ')'
        vehicle_route_times = df_routetimes.loc[(df_routetimes['vehicle'] == vehicle)]['route time']
        vehicle_return_times = df_routetimes.loc[(df_routetimes['vehicle'] == vehicle)]['return time']
        for i in vehicle_route_times.index:
            trip_times = pd.eval(vehicle_route_times[i])
            tot_use_time += trip_times[-1]
            departure_time = vehicle_return_times[i]-trip_times[-1]
            departure_times.append(departure_time)
            return_times.append(vehicle_return_times[i])
    
    departure_times.sort()
    return_times.sort()
    
    'ZERO VEHICLE TIMES'
    # To determine 
    # Loop through departure times in ascending order
    # Compare nr. of vehicles that have left before each departure time
    # If the difference between the index (+1) of the departure time and the
    # number of returns before that time are equal to the number of vehicles,
    # zero vehicles are at the station. Then add the time until the next
    # vehicle returns
    
    nr_vehicles = int(fleetsize)
    #print('nr of vehicles ' + str(nr_vehicles))
    zero_time = 0
    #print('departures ' + str(len(departure_times)))
    for i in range(len(departure_times)):
        departure = departure_times[i]
        # to prevent rounding mistakes, add an 11th decimal
        y = sum(x <= (departure+0.000000000001) for x in return_times[0:i+1])
        if (i+1-y) == nr_vehicles:
            zero_time += return_times[i+1-nr_vehicles]-departure_times[i]   
    zero_time
    #print('________________')        
    #print('zero time = ' + str(zero_time))
    #print('total use time = ' + str(tot_use_time))        
    use_time_ratio = tot_use_time/(60*6*nr_vehicles)
    avg_veh_use_time = tot_use_time/nr_vehicles
    avg_idle_time = (60*6)-avg_veh_use_time # the service period has 6 hours 
    fraction_idle_time = avg_idle_time/(60*6) 
        
    '''NEXT THE SHARED DISTANCES, DEADHEADING, AVERAGE RIDE AND TOTAL 
    DISTANCES OF TRIPS'''
    # ADD ROUTE DISTANCES TO THE OUTPUT, THEN USE THE INDICES OF THE ROUTE DISTANCES TO DETERMINE
    # HOW MUCH IS SHARED.
    tot_deadheading_distance = 0
    tot_distance = 0
    tot_trip_distance = 0 # cumulative distances that each traveler sits in
    # SAV, which counts shared distances double
    tot_shared_distance = 0
    tot_trip_duration = 0
    trip_distance = []
    nr_of_routes = len(df_routetimes)
    route_distances = df_routetimes['route distance']
    route_times = df_routetimes['route time']
    consolidation_counter = 0

    for i in route_distances.index:
        route_dist = pd.eval(route_distances[i])
        route_time = pd.eval(route_times[i])
        tot_deadheading_distance += route_dist[-1] - route_dist[-2]
        tot_distance += route_dist[-1]
        if len(route_dist) != 2:
            tot_shared_distance += route_dist[-3]
        for j in range(len(route_dist)-1):
            trip_distance.append(route_dist[j])
            tot_trip_distance += route_dist[j]
            if route_dist[j] == route_dist[j+1]:
                consolidation_counter += 1
        for k in range(len(route_time)-2):
            tot_trip_duration += route_time[k+1]
            
    tot_non_shared_dist = tot_distance - tot_shared_distance - tot_deadheading_distance
    'Avg ride distance' 
    avg_trip_distance = tot_trip_distance/servedDemand
    'Avg direct ride distance per trip'
    # Just the average of the distance column
    distances = df_servedDemand['distance']
    avg_direct_distance = distances.mean()*np.sqrt(2)
    'Getting the effective transportationi distance + ratio (to total dist)'
    transport_dist = tot_distance-tot_deadheading_distance
    transport_ratio = transport_dist/tot_distance
    effective_transport_dist = distances.sum()*np.sqrt(2)
    effective_transport_ratio = effective_transport_dist/tot_distance
       
    'Avg deadheading distance per trip'
    avg_deadheading_distance = tot_deadheading_distance/servedDemand
    
    'Avg direct ride time (calculated with hardcoded service and boarding times)'
    # Just the average of in-vehicle times
    avg_direct_ride_time = avg_direct_distance/(27/60) + 2 + 1
    'Average ride time'
    avg_trip_duration = tot_trip_duration/servedDemand
    'Average of the route distances, get these from the route distances'
    avg_route_distance = tot_distance/nr_of_routes
    'Average nr of travelers per route'
    load_factor = servedDemand/nr_of_routes
    'Total nr of consolidated drop-offs'
    consolidated_dropoffs = consolidation_counter
    'Fraction of consolidated drop-offs'
    fraction_consolidated_dropoffs = consolidated_dropoffs/servedDemand
    

    
    'SHARED DISTANCES %'
    # Check definitions. Is this including deadheading or not? Is it compared
    # to direct trip or just of the total route?
    
    'Effective vehicle transportation distance ratio'
    # Double check what this one means
    
    aggregated_output = [file_name, fleetsize, price1, price2, spacing,\
                 optout,demfact,devfact,area_size,arrival_frequency,\
                 traveler_groups,run_seed,nr_travellers,nr_mode0,\
                 nr_mode1,nr_mode2,nr_mode3,nr_mode4,demand,demand_pk,\
                 demand_offpk,servedDemand,servedDemand_pk,servedDemand_offpk,\
                 serviceRatio,serviceRatio_pk,serviceRatio_offpk,\
                 MS_demand, MS_demand_pk, MS_demand_offpk,\
                 MS_served, MS_served_pk, MS_served_offpk,\
                 punctuality, punctuality_pk, punctuality_offpk,\
                 punctuality_served, punctuality_served_pk,
                 punctuality_served_offpk, wait_avg, wait_avg_pk,\
                 wait_avg_offpk, wait_served_avg, wait_served_avg_pk,\
                 wait_served_avg_offpk, walk_avg, walk_avg_pk,\
                 walk_avg_offpk,walk_served_avg, walk_served_avg_pk,\
                 walk_served_avg_offpk, dev_avg, dev_avg_pk,\
                 dev_avg_offpk,nr_of_routes,zero_time,tot_use_time,\
                 use_time_ratio,avg_veh_use_time,avg_idle_time,\
                 fraction_idle_time,tot_distance,tot_deadheading_distance,\
                 tot_shared_distance,tot_non_shared_dist,avg_deadheading_distance,\
                 avg_direct_distance,avg_trip_distance,avg_route_distance,\
                 avg_direct_ride_time,avg_trip_duration,load_factor,\
                 consolidated_dropoffs,fraction_consolidated_dropoffs,\
                 transport_dist,effective_transport_dist,transport_ratio,\
                 effective_transport_ratio,bin0_0_1000,\
                 bin0_1000_2000,bin0_2000_2500,bin0_2500_plus,\
                 bin0_2500_3000,bin0_3000_4000,bin0_4000_6000,\
                 bin0_6000_7100,bin1_0_1000,bin1_1000_2000,\
                 bin1_2000_2500,bin1_2500_plus,\
                 bin1_2500_3000,bin1_3000_4000,bin1_4000_6000,\
                 bin1_6000_7100,bin2_0_1000,bin2_1000_2000,\
                 bin2_2000_2500,bin2_2500_plus,\
                 bin2_2500_3000,bin2_3000_4000,bin2_4000_6000,\
                 bin2_6000_7100,bin3_0_1000,bin3_1000_2000,\
                 bin3_2000_2500,bin3_2500_plus,\
                 bin3_2500_3000,bin3_3000_4000,bin3_4000_6000,\
                 bin3_6000_7100,bin4_0_1000, bin4_1000_2000, bin4_2000_2500,\
                 bin4_2500_plus, bin4_2500_3000, bin4_3000_4000, bin4_4000_6000,\
                 bin4_6000_7100]           
        
    #''' TURN THIS BACK ON AGAIN TO WRITE AGGREGATION FILE
    aggregated_data_file =  'aggregated_data_test.csv'   
    try:
        output.loc[len(output)] = aggregated_output
        output.to_csv(aggregated_data_file, index=False, sep=',', header=True, mode='x')
    except FileExistsError:
        output = pd.read_csv(aggregated_data_file)
        output.loc[len(output)] = aggregated_output
        output.to_csv(aggregated_data_file, index=False, sep=',', header=True, mode = 'w+')
    
    print('done with ' + str(file_name))

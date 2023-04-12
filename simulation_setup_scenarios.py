# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 16:56:22 2022

@author: dcdel
"""
import salabim as sim
from simulation_functions import *
import numpy as np
import pandas as pd
import csv
from scipy.stats import gumbel_r
from datetime import datetime
import time

"""if the station service area is ever to be adjusted, also make sure that the
range of the coordinates function is adjusted so that the service area starts
at coordinate [0,0]"""

def simulation(waitExpectPeak, waitExpectOffPeak,\
               devExpectPeak, devExpectOffPeak,\
               walkExpectPeak, walkExpectOffPeak,\
               fleetsize, price1, price2, opt_out_distance,\
               stopSpacing, setup_dev_fact, demand_factor,\
               traveler_groups, arrival_frequency, area_size, run_seed):
    
    # Initiate the random seeds
    np.random.seed(run_seed*100000 + 7654321)
    env = sim.Environment(trace=False, random_seed = run_seed*100000 + 7654321)
    
    print('This iteration uses defFactOffPeak = ' + str(devExpectOffPeak) +\
          ', defFactPeak = ' + str(devExpectPeak) + ', waitTimeOffPeak = '+\
          str(waitExpectOffPeak) + ', waitExpectPeak = ' +\
          str(waitExpectPeak) + ', and walkTimeOffPeak = ' +\
          str(walkExpectOffPeak) + ', and walkTimePeak = ' + str(walkExpectPeak))
    
    sim_time = 6*60 # minutes from 06:00 to 12:00
    span_pre_peak = 1 # timespan of pre-peak period
    span_peak = 2 # timespan of peak period
    span_after_peak = 3 # timespan of after-peak period
    destination_ratio = 0.7 # share of traveler destinations within 'center' 

    # Distribute travelers over the periods over destinations in- and outside
    # of 'center'. Uses applicable ratios (demand_factor, destination_ratio,
    # and area_size) and hardcoded number of arrivals from the base case.
    # Variables ending with '1' indicate destinations outside of the center.
    # Variables ending with '2' indicate destinations inside of the center.
    # First check if it is the traveler_groups scenario. If 1, then base case.
    if traveler_groups == 1:
        travelers_pre_peak1 = round(78*demand_factor*(1-destination_ratio))*area_size**2
        travelers_peak1 = round(367*demand_factor*(1-destination_ratio))*area_size**2
        travelers_after_peak1 = round(237*demand_factor*(1-destination_ratio))*area_size**2
        travelers_pre_peak2 = round(78*demand_factor*destination_ratio)*area_size**2
        travelers_peak2 = round(367*demand_factor*destination_ratio)*area_size**2
        travelers_after_peak2 = round(237*demand_factor*destination_ratio)*area_size**2
        
        # Distribute the travellers over the trains that arrive during period
        # Use arrival_frequency to set number of arriving trains according
        # to the base case scenario or the high arrival frequency scenario
        bins_pre1 = arrival_distribution(travelers_pre_peak1, span_pre_peak, arrival_frequency)
        bins_peak1 = arrival_distribution(travelers_peak1, span_peak, arrival_frequency)
        bins_after1 = arrival_distribution(travelers_after_peak1, span_after_peak, arrival_frequency)
        bins_pre2 = arrival_distribution(travelers_pre_peak2, span_pre_peak, arrival_frequency)
        bins_peak2 = arrival_distribution(travelers_peak2, span_peak, arrival_frequency)
        bins_after2 = arrival_distribution(travelers_after_peak2, span_after_peak, arrival_frequency)
        
        # Combine so they can be used by the traveler generator
        bins1 = (bins_pre1 + bins_peak1 + bins_after1)
        bins2 = (bins_pre2 + bins_peak2 + bins_after2)
    
    # In case of traveler_groups scenario, else statements creates two
    # different traveler groups (3 and 4). These are used to generate 
    # business travelers.
    else:
        # determine how many travellers for each part of the day in- and
        # outside of service area, include business travelers (high tt value)
        travelers_pre_peak1 = round(78*demand_factor*0.5*(1-destination_ratio))*area_size**2
        travelers_peak1 = round(367*demand_factor*0.5*(1-destination_ratio))*area_size**2
        travelers_after_peak1 = round(237*demand_factor*0.5*(1-destination_ratio))*area_size**2
        travelers_pre_peak2 = round(78*demand_factor*0.5*destination_ratio)*area_size**2
        travelers_peak2 = round(367*demand_factor*0.5*destination_ratio)*area_size**2
        travelers_after_peak2 = round(237*demand_factor*0.5*destination_ratio)*area_size**2
        travelers_pre_peak3 = round(78*demand_factor*0.5*(1-destination_ratio))*area_size**2
        travelers_peak3 = round(367*demand_factor*0.5*(1-destination_ratio))*area_size**2
        travelers_after_peak3 = round(237*demand_factor*0.5*(1-destination_ratio))*area_size**2
        travelers_pre_peak4 = round(78*demand_factor*0.5*destination_ratio)*area_size**2
        travelers_peak4 = round(367*demand_factor*0.5*destination_ratio)*area_size**2
        travelers_after_peak4 = round(237*demand_factor*0.5*destination_ratio)*area_size**2   
        
        # Distribute the travellers over the trains that arrive during period.
        # Use arrival_frequency to set number of arriving trains according
        # to the base case scenario or the high arrival frequency scenario.
        bins_pre1 = arrival_distribution(travelers_pre_peak1, span_pre_peak, arrival_frequency)
        bins_peak1 = arrival_distribution(travelers_peak1, span_peak, arrival_frequency)
        bins_after1 = arrival_distribution(travelers_after_peak1, span_after_peak, arrival_frequency)
        bins_pre2 = arrival_distribution(travelers_pre_peak2, span_pre_peak, arrival_frequency)
        bins_peak2 = arrival_distribution(travelers_peak2, span_peak, arrival_frequency)
        bins_after2 = arrival_distribution(travelers_after_peak2, span_after_peak, arrival_frequency)
        bins_pre3 = arrival_distribution(travelers_pre_peak1, span_pre_peak, arrival_frequency)
        bins_peak3 = arrival_distribution(travelers_peak1, span_peak, arrival_frequency)
        bins_after3 = arrival_distribution(travelers_after_peak1, span_after_peak, arrival_frequency)
        bins_pre4 = arrival_distribution(travelers_pre_peak2, span_pre_peak, arrival_frequency)
        bins_peak4 = arrival_distribution(travelers_peak2, span_peak, arrival_frequency)
        bins_after4 = arrival_distribution(travelers_after_peak2, span_after_peak, arrival_frequency)

        # Combine the bins so they can be used by the traveler generator.
        bins1 = (bins_pre1 + bins_peak1 + bins_after1)
        bins2 = (bins_pre2 + bins_peak2 + bins_after2)
        bins3 = (bins_pre3 + bins_peak3 + bins_after3)
        bins4 = (bins_pre4 + bins_peak4 + bins_after4)
    
    # Set headway between arriving trains. 7.5 min is the default headway.
    inter_arrival_time = 7.5/arrival_frequency
    
    # Define station location at center of area of 5x5 and multiply with.
    # the area size factor.
    station = [2.5*area_size,2.5*area_size]

    # Check if stop spacing applies. If 0 then base case scenario. Otherwise
    # run scenario with stop spacing.
    if stopSpacing == 0:
       toWalkOrNotToWalk = False
    else:
       toWalkOrNotToWalk = True
    
    # Prepare dataframe for output. Set column names and name of output file.
    resultaten = pd.DataFrame(columns = ['traveler', 'traveler ID','requested time',\
                                         'distance','mode choice','waiting time',\
                                         'devFact_offer','x_coord_desti','y_coord_desti',\
                                         'vehicle','drop_in_sequence','inVehTime',\
                                         'devFact','walking time','interest',\
                                         'utility choice','utility interest',\
                                         'value tt','x_coord_stoploc','y_coord_stoploc',\
                                         'return time','route time','route distance',\
                                         'stoch parts'])
    now = datetime.now()
    output_csv = 'test_results_csv/results_fl'+str(fleetsize)+'_pr'+str(price1*100)+\
        '_'+str(price2*100)+'_space'+str(stopSpacing)+'_opt'+str(opt_out_distance*1000)+\
        '_dem'+str(demand_factor)+'_dev'+str(setup_dev_fact)+'_area'+str(area_size)+\
        '_freq'+str(arrival_frequency)+'_groups'+str(traveler_groups)+'_RS'+str(run_seed)+'.csv'
    
    '''Create traveler generator. The generator is set to release travelers
    from each bin into the system at the appropriate time. Depending on the
    bin, destination coordinates in- or outside of the center will be
    generated.'''
    class travelerGenerator(sim.Component):
        def process(self):
            
            train=0
            while train <= len(bins1)-1:
                
                arrivals1 = bins1[train]
                arrivals2 = bins2[train]
                for arrival in range(arrivals1):
                    traveler(destination_indicator='outside')
                for arrival in range(arrivals2):
                    traveler(destination_indicator='inside')
                
                if traveler_groups == 2:
                    arrivals3 = bins3[train]
                    arrivals4 = bins4[train]
                    for arrival in range(arrivals3):
                        traveler(destination_indicator='outside_high_tt')
                    for arrival in range(arrivals4):
                        traveler(destination_indicator='inside_high_tt')
                
                yield self.hold(inter_arrival_time) # fixed 7.5 time between each train
                train += 1    
    
    '''Traveler component class is used for the creation of travelers.
    
    Define setup to gives traveler the following data:
    -stoch_parts sets the alternative specific constants of last-mile mode
    -arrTime sets arrival time of traveler at the station, based on train
    -destination uses a custom function to generate coordinates in- or outside
    the center
    -distance sets direct travel distance from station to traveler destination
    -choice is used later to store mode choice of traveler, initiate with
    a non-sensible outcome (7 is not associated with an actual option).
    -travelerRoute is used to set the state of the traveler. Set to request
    so the traveler enters the flow of the model
    -nextResource sets the next state of the traveler. Iniate with 0
    -set the max_deviation_factor for the model
    -set walkTime to 0, used to store walking time if applicable
    
    Define process so traveler enters next applicable state when one state ends
    '''
    class traveler(sim.Component):
        def setup(self, destination_indicator):
            self.stoch_parts = [gumbel_r.rvs(scale=(1/0.55)),gumbel_r.rvs(),gumbel_r.rvs(),gumbel_r.rvs()]
            self.arrTime = env.now() + 30
            self.destination = coords(station, destination_indicator)
            if destination_indicator == 'outside_high_tt' or destination_indicator == 'inside_high_tt':
                self.value_tt = 'high'
            else:
                self.value_tt = 'normal'
            self.distance = distance_func(station,self.destination)
            self.choice = 7
            self.travelerRoute = []
            self.travelerRoute.append(request_service)
            self.nextResource = 0
            self.max_dev_fact = setup_dev_fact
            self.walkTime = 0
            
        def process(self):
            while True:
                if self.nextResource <= len(self.travelerRoute) - 1:
                    self.enter(self.travelerRoute[self.nextResource])
                    self.nextResource += 1
                    yield self.passivate()
                else:
                    yield self.passivate()
    
    '''serviceScheduler class creates a process to handle the flow of the ABM.
    Based on the system time in the simulation, the serviceSchedule.
    1) First, it check the period to determine if the MSA of the peak- or off-peak
    period apply. Then, it evaluates mode choice 1 with the actual_demand()
    function.
    2) If the traveler has demand, interest is set to 0 (0 represents the
    service) and the routeDetermination() function is run.
    routeDetermination() runs the dispatching algorithm and evaluates the mode
    choice by the traveler. Upon completion of those functions, the
    state update of traveler is returned, along with the offer, mode_choice,
    and utility of the offer (for review purposes). Results of the traveler are
    logged accordingly. Routes that have become fixed are logged as well.
    '''
    class serviceScheduler(sim.Component):
        def process(self):
            while True:
                while len(request_service) == 0:
                    yield self.standby()              
                self.traveler = request_service.pop()
                
                """Check if interested in the service and if so, determine
                offer and mode choice"""
                if env.now() < 90 or env.now() >= 210:
                    interest, utility_expected = actual_demand(station,self.traveler.destination, waitExpectOffPeak, devExpectOffPeak, walkExpectOffPeak, price1, price2, opt_out_distance, self.traveler.stoch_parts, self.traveler.value_tt)
                    self.traveler.devExpect = devExpectOffPeak # to log results according to arrival time
                else:
                    interest, utility_expected = actual_demand(station,self.traveler.destination, waitExpectPeak, devExpectPeak, walkExpectPeak, price1, price2, opt_out_distance, self.traveler.stoch_parts, self.traveler.value_tt)
                    self.traveler.devExpect = devExpectPeak # to log results according to arrival time
                
                if interest == 0:               
                    self.traveler.travelerRoute, self.traveler.offer, self.traveler.choice, utility_offer = routeDetermination(self.traveler.stoch_parts, self.traveler.destination, self.traveler.arrTime, sav, self.traveler)
                    walkTimePassenger = self.traveler.walkTime
                    waitTimePassenger = self.traveler.offer[2] - self.traveler.arrTime
                    
                    "Update the route of the vehicle in the offer (at index 0) with the generated route"
                    if self.traveler.choice == 0: # If traveler accepts the offer
                        # First check if the route is new or not. If new, the final
                        # result of the previous route are logged to results
                        # for all boarded passengers before storing new route
                        if self.traveler.offer[6] == True and self.traveler.offer[0].sequence != []:
                            
                            for passenger in range(len(self.traveler.offer[0].sequence)): # Then log every traveler in the old route
                                b=str(self.traveler.offer[0].sequence[passenger])
                                tav,nr = b.split('.')
                                if self.traveler.offer[0].routeTime == []:
                                    print(self.traveler.offer[0].sequence)
                                    print(self.traveler.offer[0].routeTime)
                                    print(self.traveler)
                                    print('wait here')
                                    print('at index ' + str(passenger))
                                    print(self.traveler.offer)
                                # if the passener is the last one in the schedule, save the whole route time and returntime
                                if passenger == range(len(self.traveler.offer[0].sequence))[-1]:
                                    routeTime = self.traveler.offer[0].routeTime
                                    returnTime = self.traveler.offer[0].returnTime
                                    sequence = self.traveler.offer[0].sequence
                                    routeDist = route_distance_func(sequence, station)
                                    if isinstance(routeTime, np.ndarray):
                                        routeTime = list(routeTime)
                                else:
                                    routeTime = ''
                                    returnTime = ''
                                    routeDist = ''
                                
                                inVehTimePassenger = self.traveler.offer[0].routeTime[passenger+1] #- self.traveler.offer[0].depTime
                                devFactPassenger = self.traveler.offer[0].routeScore[passenger]
                                walkTimePassenger = self.traveler.offer[0].sequence[passenger].walkTime
                                distancePassenger = self.traveler.offer[0].sequence[passenger].distance/np.sqrt(2) # distance otherwise given as penalized distance
                                
                                # Write the results of the ride
                                resultaten.loc[len(resultaten)] = [self.traveler.offer[0].sequence[passenger]] + [nr[:-1]]\
                                    + [self.traveler.offer[0].sequence[passenger].arrTime] + [distancePassenger]\
                                    + [''] + ['']  + [''] + [self.traveler.offer[0].sequence[passenger].destination[0]] + [self.traveler.offer[0].sequence[passenger].destination[1]] + [self.traveler.offer[0]] + [passenger+1]  + [inVehTimePassenger] + [devFactPassenger]\
                                        + [walkTimePassenger] + [''] + [''] + [''] + [''] + [self.traveler.offer[0].sequence[passenger].offer[10][0]] + [self.traveler.offer[0].sequence[passenger].offer[10][1]] + [returnTime] + [routeTime] + [routeDist] + ['']
                                        
                        self.traveler.offer[0].routeScore = self.traveler.offer[5]
                        self.traveler.offer[0].depTime = self.traveler.offer[2]
                        self.traveler.offer[0].returnTime = self.traveler.offer[3]
                        self.traveler.offer[0].route = self.traveler.offer[4]
                        self.traveler.offer[0].passengers = len(self.traveler.offer[0].route)-2
                        self.traveler.offer[0].sequence = self.traveler.offer[7]
                        self.traveler.offer[0].routeTime = self.traveler.offer[8] 
                        if self.traveler.offer[8] == []:
                            print('Check. Hoe kan de routetime hier leeg zijn?')
                            print(self.traveler.offer)
                        walkTimePassenger = self.traveler.walkTime
                        distancePassenger = self.traveler.distance/np.sqrt(2) # distance otherwise given as penalized distance
                        
                        b=str(self.traveler)
                        tav,nr = b.split('.')
                        
                        resultaten.loc[len(resultaten)] = [self.traveler] + [nr[:-1]] + [self.traveler.arrTime] + [distancePassenger]\
                            + [self.traveler.choice] + [self.traveler.offer[2] - self.traveler.arrTime]  + [self.traveler.devExpect] + [self.traveler.destination[0]]+ [self.traveler.destination[1]]\
                                + [self.traveler.offer[0]] + [''] + [''] + [''] + [walkTimePassenger] + [interest] + [utility_offer] + [utility_expected] + [self.traveler.value_tt] \
                                    + [''] + [''] + [''] + [''] + [''] + [self.traveler.stoch_parts]#
                    
                    else: # If traveler did not choose the service
                        b=str(self.traveler)
                        tav,nr = b.split('.')
                        distancePassenger = self.traveler.distance/np.sqrt(2) # distance otherwise given as penalized distance
                        resultaten.loc[len(resultaten)] = [self.traveler] + [nr[:-1]] + [self.traveler.arrTime] + [distancePassenger]\
                            + [self.traveler.choice] + [self.traveler.offer[2] - self.traveler.arrTime]  + [self.traveler.devExpect] + [self.traveler.destination[0]]+ [self.traveler.destination[1]]\
                                + ['']+['']+[''] + [''] + [walkTimePassenger] + [interest] + [utility_offer] + [utility_expected] + [self.traveler.value_tt] \
                                    + [''] + [''] + [''] + [''] + [''] + [self.traveler.stoch_parts]
                    self.traveler.activate(delay=30)
                    
                else: # interest != 0
                    self.traveler.choice = interest
                    distancePassenger = self.traveler.distance/np.sqrt(2) # distance otherwise given as penalized distance
                    b=str(self.traveler)
                    tav,nr = b.split('.')
                    resultaten.loc[len(resultaten)] = [self.traveler] + [nr[:-1]] + [self.traveler.arrTime] + [distancePassenger]\
                        + [self.traveler.choice] + ['']  + [''] + [self.traveler.destination[0]]+ [self.traveler.destination[1]]\
                            + ['']+['']+[''] + [''] + [''] + [interest] + [''] + [''] + [''] + [''] + [''] + [''] + [''] + [''] + [self.traveler.stoch_parts]
       
    '''Use SAV class is used to create the SAVs. The setup is used to take in
    specification data from the input. Next, this data is used in the
    dispatching function and to determine the mode choice.'''
    class SAV(sim.Component):  
        def setup(self):
            self.capacity = 4 # capacity
            self.speed = 27
            self.passengers = 0 # initiate with no passengers on board
            self.depTime = 0 # initiate departure time variable
            self.returnTime = 0 # initiate return time variable
            self.routeTime = []
            self.route = [station,station]
            self.sequence = []
            self.devFact = []
            self.routeScore = []
            self.stopLocations = toWalkOrNotToWalk
            self.stopSpacing = stopSpacing
            self.price1 = price1 # starting fee
            self.price2 = price2 # charge/minute
            self.station = station
            
        def process(self):
            while True:
                # Haal SAV_queue index op uit string "<bound method Component.name of SAV (sav.3)>"
                while len(sav_queues['SAV_queue' + str(self.name)[-3]]) == 0:
                    yield self.standby()
                self.traveler = sav_queues['SAV_queue' + str(self.name)[-3]].pop()
                s = str(self)
                b = str(self.traveler)
                component, nummer = s.split('.')
                tav,nr = b.split('.')
                
    def routeDetermination(stoch_parts, destination, reqTime, sav, travelerID):
        offer = schedulingFunction(destination, travelerID, reqTime, sav)
        station = offer[0].station
        price1 = offer[0].price1
        price2 = offer[0].price2
        travelerID.walkTime = offer[9] # Append walktime
        if env.now() < 90 or env.now() >= 210:         
            travelerChoice, utility_offer = modeChoice(station, destination, travelerID, offer, reqTime, devExpectOffPeak, price1, price2)
        else:
            travelerChoice, utility_offer = modeChoice(station, destination, travelerID, offer, reqTime, devExpectPeak, price1, price2)
        if travelerChoice == 0:
            travelerID.travelerRoute.append(sav_queues['SAV_queue' + str(offer[0])[-2]])
            
        return travelerID.travelerRoute, offer, travelerChoice, utility_offer
    
    def saveLastRoutes():
        for SAV in sav:                    
            for passenger in range(len(SAV.sequence)):
                b=str(SAV.sequence[passenger])
                tav,nr = b.split('.')               
                inVehTimePassenger = SAV.routeTime[passenger+1] #- self.traveler.offer[0].depTime
                devFactPassenger = SAV.routeScore[passenger]
                distancePassenger = SAV.sequence[passenger].distance/np.sqrt(2) # distance otherwise given as penalized distance
                
                if passenger == range(len(SAV.sequence))[-1]:
                    routeTime = SAV.routeTime
                    returnTime = SAV.returnTime
                    sequence = SAV.sequence
                    routeDist = route_distance_func(sequence, station)
                    if isinstance(routeTime, np.ndarray):
                        routeTime = list(routeTime)
                else:
                    routeTime = ''
                    returnTime = ''
                    routeDist = ''
                    
                resultaten.loc[len(resultaten)] = [SAV.sequence[passenger]] + [nr[:-1]] + [SAV.sequence[passenger].arrTime] + [distancePassenger]\
                    + [''] + ['']  + [''] + [SAV.sequence[passenger].destination[0]]+ [SAV.sequence[passenger].destination[0]] + [SAV] + [passenger+1]  + [inVehTimePassenger] + [devFactPassenger] + [SAV.sequence[passenger].walkTime] +\
                        [''] + [''] + [''] + ['']  + [SAV.sequence[passenger].offer[10][0]] + [SAV.sequence[passenger].offer[10][1]] + [returnTime] + [routeTime] + [routeDist] + ['']
    
    # Now initiate the travelerGenerator, appropriate number of SAVs,
    # and request_service queue and run the environment
    travelerGenerator()
    sav = []
    sav_queues = {}
    for i in range(fleetsize): # initiate vehicles as active components
        sav.append(SAV())
        sav_queues[('SAV_queue' + str(i))] = sim.Queue('SAV_queue' + str(i))
    serviceScheduler()
    request_service = sim.Queue('request_service')
    
    env.run(till=sim_time+30)
    
    # Log the last completed routes
    saveLastRoutes()
    
    finish = datetime.now()
    print('started at ' + str(now) + ' and finished at ' + str(finish) +\
          ' with duration ' + str(finish-now))
    
    # Save the output in CSV. If a file for the run already exists from
    # earlier iterations, it is overwritten with the latest results.
    outfile = open(output_csv, 'w')
    resultaten.to_csv(output_csv, index=False, sep=',', header=True)
    outfile.close()
    env.reset_now()
    return output_csv

"""
Runs the simulation framework. The framework applies the MSA for convergence
based on the demand. MSA inputs consider the waiting, walking, and travel time
deviation factor. The function also ensures that the appropriate number of runs
is completed and saves the logs the development of the MSA values and demand
behavior.
"""
def run_simulation(fleetsize,spacing,price1,price2,max_dev_fact,demand_factor,opt_out_distance,traveler_groups,high_frequency,area_size,runs):        
    start_instance = datetime.now()
  
    for run in range(1,runs+1):
        start_run = datetime.now()

        # Create empty arrays to keep track the development of MSA values,
        # demand, served demand, and service ratio subsequent iterations
        # within the run.
        serviceMSA = []
        devFactP_MSA = []
        devFactO_MSA = []
        walkTP_MSA = []
        walkTO_MSA = []
        waitTP_MSA = []
        waitTO_MSA = []
        demand_tracker = []
        served_demand_tracker = []
        service_ratio_tracker = []
        serviceRatio = 1
        
        # Initiate MSA values that are later used to hold the latest MSA value.
        # Need to be non-empty.
        devAvgOffPeakMSA = devAvgPeakMSA = 1
        walkTime = 0.5*spacing*np.sqrt(2)/(4.6/60)
        walkAvgOffPeakMSA = walkAvgPeakMSA = 0.5*walkTime
        waitAvgOffPeakMSA = waitAvgPeakMSA = 0
		
        # Initiate the MSA holding values. Overwritten later,
        waitAvgPeak = waitAvgOffPeak = 0
        walkAvgPeak = walkAvgOffPeak = 0.5*walkTime
        devAvgPeak = devAvgOffPeak = 1
        
        # Initiate demand values. These are overwritte later to chech the
        # output for any alternating outcomes. Does not seem to occur anymore
        # after random seed was fixed.
        demand=demandOld=demandOld2=demandOld3=demandOld4=demandOld5=\
            demandOld6=demandOld7=demandOld8=demandOld9=demandOld10=\
            demandOld11=1
        alternate = False
        
        iteration = 0
        converge = False
        
        # Show number of current run and its input.
        print('')
        print('Simulation run ' + str(run) + ' for fleetsize = ' + str(fleetsize)\
			  + ', spacing = ' + str(spacing)\
			  + ', price1 = '+ str(price1)\
			  + ', price2 = '+ str(price2)\
			  + ', max_dev = ' + str(max_dev_fact)\
			  + ', opt_out_dist = ' + str(opt_out_distance)\
			  + ', demand_factor = ' + str(demand_factor)\
              + ', groups = ' + str(traveler_groups)\
              + ', frequency = ' + str(high_frequency)\
              + ', area_size = ' + str(area_size))
			
        while converge == False:
            iteration +=1
            print('')
            print('__________________________')
            print('iteration ' + str(iteration))

            demandOld11 = demandOld10
            demandOld10 = demandOld9
            demandOld9 = demandOld8
            demandOld8 = demandOld7
            demandOld7 = demandOld6        
            demandOld6 = demandOld5
            demandOld5 = demandOld4        
            demandOld4 = demandOld3
            demandOld3 = demandOld2
            demandOld2 = demandOld
            demandOld = demand
            
            # Update the MSA value with each run.
            devAvgOffPeakMSA = (devAvgOffPeakMSA*(iteration-1) + devAvgOffPeak)/iteration
            devAvgPeakMSA = (devAvgPeakMSA*(iteration-1) + devAvgPeak)/iteration
            walkAvgOffPeakMSA = (walkAvgOffPeakMSA*(iteration-1) + walkAvgOffPeak)/iteration
            walkAvgPeakMSA = (walkAvgPeakMSA*(iteration-1) + walkAvgPeak)/iteration
            waitAvgOffPeakMSA = (waitAvgOffPeakMSA*(iteration-1) + waitAvgOffPeak)/iteration
            waitAvgPeakMSA = (waitAvgPeakMSA*(iteration-1) + waitAvgPeak)/iteration
            
            # Log the MSA values for the iteration
            serviceMSA.append(serviceRatio)
            devFactP_MSA.append(devAvgPeakMSA)
            walkTP_MSA.append(walkAvgPeakMSA)
            waitTP_MSA.append(waitAvgPeakMSA)
            devFactO_MSA.append(devAvgOffPeakMSA)
            walkTO_MSA.append(walkAvgOffPeakMSA)
            waitTO_MSA.append(waitAvgOffPeakMSA)
			
            # Run the model and keep the name of the output file to open it
            # and determine the MSA values.
            csv_output = simulation(waitAvgPeakMSA, waitAvgOffPeakMSA,\
									devAvgPeakMSA, devAvgOffPeakMSA,\
									walkAvgPeakMSA, walkAvgOffPeakMSA,\
									fleetsize, price1, price2, opt_out_distance,\
                                    spacing,max_dev_fact,demand_factor,\
                                    traveler_groups, high_frequency, area_size,\
                                    run)
            outfile = open(csv_output, 'r')
            df_output = pd.read_csv(outfile, delimiter=',')
            outfile.close()

            '''Out of the three indicators, the deviation factor is the
			only one that can only be estimated afterward. For the other
			two (waiting time and walking time), the times are fixed when
			an offer is made'''
            devAvgOffPeak = df_output.loc[(df_output['requested time'] < 90) | (df_output['requested time'] >= 210)]['devFact'].mean()
            devAvgPeak = df_output.loc[(df_output['requested time'] >= 90) & (df_output['requested time'] < 210)]['devFact'].mean()
            df_interest = df_output.loc[df_output['interest'] == 0]
            waitAvgOffPeak = df_interest.loc[(df_interest['requested time'] < 90) | (df_interest['requested time'] >= 210)]['waiting time'].mean()
            waitAvgPeak = df_interest.loc[(df_interest['requested time'] >= 90) & (df_interest['requested time'] < 210)]['waiting time'].mean()
            walkAvgOffPeak = df_interest.loc[(df_interest['requested time'] < 90) | (df_interest['requested time'] >= 210)]['walking time'].mean()
            walkAvgPeak = df_interest.loc[(df_interest['requested time'] >= 90) & (df_interest['requested time'] < 210)]['walking time'].mean()    
            walkAvg = df_interest['walking time'].mean()

            # Determine the demand, served demand, and service ratio
            demand = len(df_interest.index)
            servedDemand = len(df_interest.loc[df_interest['mode choice'] == 0].index)
            try:
                serviceRatio = servedDemand/demand
            except:
                serviceRatio = 0
			
            print(demand,demandOld,demandOld2)
            
			# Determine convergence and document results accordingly
            if (demand/demandOld <= 0.99 or demand/demandOld >= 1.01) or (demand/demandOld2 <= 0.99 or demand/demandOld2 >= 1.01):
                print('demand = ' + str(demand) + ' and served = ' + str(servedDemand))
                demand_tracker.append(demand)
                served_demand_tracker.append(servedDemand)
                service_ratio_tracker.append(serviceRatio)
				
                # Add an additional check to make sure no alternating patterns
                # are occuring. Check alternating pattern for 6 consecutive
                # iterations
                if demand==demandOld6 and demandOld==demandOld7 and\
                    demandOld2==demandOld8 and demandOld3==demandOld9 and\
                    demandOld4==demandOld10 and demandOld5==demandOld11:
                    if not ((demand-demandOld) < -1 or (demand-demandOld) > 1): 
                        alternate = True
                        patternlength = 6
                            
                if alternate == True:
                    print('Iteration stopped after demand pattern started alternating')
                    print('PATTERNLENGTH = ' + str(patternlength))
                    print('______Final_outcome_______')
                    print('demand = ' + str(demand) + ', served = ' +\
                        str(servedDemand) + 'service ratio = ' + str(serviceRatio))
                    print('per iteration the demand, served demand, and service ratio developed as follows:')
                    print('Demand for each iteration')
                    print(demand_tracker)
                    print('Served demand for each iteration')
                    print(served_demand_tracker)
                    print('Service ratio for each iteration')
                    print(service_ratio_tracker)
                    print("MSA's:")
                    print('MSA waiting time Off Peak = ' + str(waitTO_MSA))
                    print('MSA waiting time Peak = ' + str(waitTP_MSA))
                    print('MSA walking time Off Peak = ' + str(walkTO_MSA))
                    print('MSA walking time Peak = ' + str(walkTP_MSA))
                    print('MSA dev fact Off Peak = ' + str(devFactO_MSA))
                    print('MSA dev fact Peak = ' + str(devFactP_MSA))
					
                    converge = True
                    
                if not (((demand-demandOld) < -1 or (demand-demandOld) > 1) or ((demand-demandOld2) < -1 or (demand-demandOld2) > 1)):
                    print('Convergence based on absolute outcome')
                    print('______Final_outcome_______')
                    print('demand = ' + str(demand) + ', served = ' +\
                        str(servedDemand) + 'service ratio = ' + str(serviceRatio))
                    print('per iteration the demand, served demand, and service ratio developed as follows:')
                    print('Demand for each iteration')
                    print(demand_tracker)
                    print('Served demand for each iteration')
                    print(served_demand_tracker)
                    print('Service ratio for each iteration')
                    print(service_ratio_tracker)

                    converge = True
                    
            else:
                print('______Final_outcome_______')
                print('per iteration the demand, served demand, and service ratio developed as follows:')
                print('Demand for each iteration')
                print(demand_tracker)
                print('Served demand for each iteration')
                print(served_demand_tracker)
                print('Service ratio for each iteration')
                print(service_ratio_tracker)
				
                converge = True 
		
        # Track the development of the MSA values and the demand behavior 
        demand_tracker.append(demand)
        served_demand_tracker.append(servedDemand)
        service_ratio_tracker.append(serviceRatio)
        serviceMSA.append(serviceRatio)
        devFactP_MSA.append(devAvgPeakMSA)
        walkTP_MSA.append(walkAvgPeakMSA)
        waitTP_MSA.append(waitAvgPeakMSA)
        devFactO_MSA.append(devAvgOffPeakMSA)
        walkTO_MSA.append(walkAvgOffPeakMSA)
        waitTO_MSA.append(waitAvgOffPeakMSA)
        
        # Log the iterative development of the MSA values and the demand
        # behavior together with the details of the run.
        finish_run = datetime.now()
        duration = finish_run-start_run
        finish_str = finish_run.strftime("%H:%M:%S")
        duration_str = str(duration.seconds) + '.' +str(duration.microseconds)
        rundata_filename = 'simulation_rundata' + str(datetime.now().year) + str(datetime.now().month) + str(datetime.now().day) + '.csv'
        name = 'Iteration_' + str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        try:
            rundata = pd.DataFrame(columns = ['Name','Finish','Duration', 'run','iterations', 'Fleetsize', 'Price1','Price2', 'Spacing','Demand Factor Adjust','Max Deviation Factor','Opt out distance','demand','served','serviceratio', 'alternate','serviceMSA', 'devFactP_MSA', 'walkTP_MSA', 'waitTP_MSA', 'devFactO_MSA', 'walkTO_MSA', 'waitTO_MSA.append'])
            rundata.loc[len(rundata)] = [name, finish_str, duration_str, run, iteration, fleetsize, price1, price2, spacing, demand_factor, max_dev_fact, opt_out_distance, demand_tracker, served_demand_tracker, service_ratio_tracker, alternate, serviceMSA, devFactP_MSA, walkTP_MSA, waitTP_MSA, devFactO_MSA, walkTO_MSA, waitTO_MSA]
            outfile = open(rundata_filename, 'x')
            rundata.to_csv(rundata_filename, index=False, sep=',', header=True)
            outfile.close()
        except FileExistsError:
            rundata = pd.read_csv(rundata_filename)
            rundata.loc[len(rundata)] = [name, finish_str, duration_str, run, iteration, fleetsize, price1, price2, spacing, demand_factor, max_dev_fact, opt_out_distance, demand_tracker, served_demand_tracker, service_ratio_tracker, alternate, serviceMSA, devFactP_MSA, walkTP_MSA, waitTP_MSA, devFactO_MSA, walkTO_MSA, waitTO_MSA]
            outfile = open(rundata_filename, 'w+')
            rundata.to_csv(rundata_filename, index=False, sep=',', header=True, mode='w+')
            outfile.close()

        rundata = False
        print('Done with run ' + str(run))
        print('Duration of run : ' + str(duration))
        
    print('Finished runs at ' + str(finish_run))
    print('Duration of instance : ' + str(finish_run - start_instance))
    
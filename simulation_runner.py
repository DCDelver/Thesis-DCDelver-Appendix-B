# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:50:16 2022

@author: dcdel

"""

import re
from simulation_setup_scenarios import *

'''Declare input parameters
-fleetsizes sets the number of vehicles that the service will be using
-spacing sets the distance between stops, setting 0 means the service will be
a station-to-door service
-price1 sets the starting fare
-price2 sets the ride fare
-max_dev_fact sets the maximum travel time deviation factor, and is kept at 2
-demand_factor sets a multiplier to vary the number of arriving travelers
compared to the base case.
-opt_out_distance sets the radius where of the area (in km) within which
travelers are assumed to walk to their destination. It is kept at 0.7 km.
-traveler_groups sets the simulation setting for the traveler groups scenario.
1 = base case setting, 2 = traveler groups scenario 
-sets the simulation setting for the traveler groups scenario.
1 = base case setting, 2 = double arrival frequency of trains
-area_size sets the simulation setting for the big area size scenario.
1 = base case setting, 2 = big area size scenario
-runs sets the number of model runs that wil be completed with different random
seeds.
'''

fleetsize = 5
spacing = 0.4 # [km]
price1 = 2
price2 = 0.375
max_dev_fact = 2
demand_factor = 1
opt_out_distance=0.7
traveler_groups = 1
high_frequency = 1
area_size = 1
runs = 1

run_simulation(fleetsize,spacing,price1,price2,max_dev_fact,demand_factor,\
    opt_out_distance,traveler_groups,high_frequency,area_size,runs)
 
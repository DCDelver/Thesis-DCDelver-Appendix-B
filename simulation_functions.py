# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:35:31 2022

@author: dcdel
"""

from random import randrange # for distributing travelers over trains
import numpy as np
import salabim as sim
import csv

'''Defines the dispatching function. It uses the following steps.
First, comparisson values and logging values are created for the route that
will be selected.
Second, the algorithm attempts to select a vehicle that already has a route
to perform ridesharing. It then checks if it can make a ridesharing route.
Next, check idle vehicles. If there are idle vehicles, there can not have been
future departures, hence this order makes sense.
Next, check all future departures.
If previous options do not give a feasible route, find the first vehicle to
become idle.
Return the best result.
'''

def schedulingFunction(destination, travelerID, reqTime, vehicles):
    # Algorithm to determine best vehicle and route
    bestScore = 999
    bestRoute = False
    selectVehicle = False
    selectRoute = False
    selectRouteScore = False
    selectStop = False    
    selectDepTime = False
    selectReturnTime = False
    selectInVehTime = False
    newSequence = False
    newRoute = False
    routeTime = []
    walkTime = False
    
    '''
    OPTION 1 is to try and add it to an existing vehicle route.
    Note: since all requests are made 30 min in advance, any vehicles
    scheduled earlier have already left. Vehicles departing later have not
    yet been scheduled. Hence, only look at existing schedules at the
    requested time. 
    '''
    selected = select(reqTime, vehicles)
    if selected != False:
        for vehicle in selected:
            routeOption, inVehTime, routeScore, score, routeTime, sequence, walkTime, stop = requestHandlerInsert(vehicle, travelerID, destination, reqTime)
            # If insertions were feasible, the best route is saved
            if bestScore > score and score != False:
                bestScore = score
                selectRouteScore = routeScore
                selectRoute = routeOption
                selectStop = [stop[0],stop[1]]
                selectVehicle = vehicle
                selectDepTime = vehicle.depTime
                selectReturnTime = selectDepTime + routeTime[-1]
                selectInVehTime = inVehTime
                selectTime = routeTime
                selectWalkTime = walkTime
                newSequence = sequence
              
        # Check if insertions were succesful insertions were compared.
        # If this is the case return the offer.
        if bestScore < 999 and selectVehicle != False:
            return selectVehicle, selectInVehTime, selectDepTime, selectReturnTime, selectRoute, selectRouteScore, newRoute, newSequence, selectTime, selectWalkTime, selectStop
    
    '''
    OPTION 2 is to select an available idle vehicle at the requested time.
    Ofcourse, if ridesharing is possible for a future departure, no vehicles
    can even be found here.If there is no such vehicle for a feasible
    insertion, go to option 3 (select vehicle with departure time in future). 
    '''
    selected = select_idle(reqTime, vehicles)
    if selected != False:
        # Only one idle vehicle is enough to make a new route
        newRoute = True
        vehicle = selected
        routeOption, inVehTime, routeScore, score, routeTime, sequence, walkTime, stop = requestHandlerIdle(vehicle, travelerID, destination, reqTime)
        # Now append the new route
        bestScore = score
        selectRouteScore = routeScore
        selectRoute = routeOption
        selectStop = stop
        selectVehicle = vehicle
        selectDepTime = reqTime
        selectReturnTime = selectDepTime + routeTime[-1]
        selectInVehTime = inVehTime
        selectRouteTime = routeTime
        selectWalkTime = walkTime
        newSequence = sequence
        
        return selectVehicle, selectInVehTime, selectDepTime, selectReturnTime, selectRoute, selectRouteScore, newRoute, newSequence, selectRouteTime, selectWalkTime, selectStop 

    '''OPTION 3 is to try inserting into all scheduled future departures'''
    selected = select_future_departures(reqTime, vehicles)
    if selected != False:
        for vehicle in selected:
            routeOption, inVehTime, routeScore, score, routeTime, sequence, walkTime, stop = requestHandlerFutureInsert(vehicle, travelerID, destination, reqTime)
            # This part only gets done when insertions were feasible
            if bestScore > score and score != False:
                bestScore = score
                selectRouteScore = routeScore
                selectRoute = routeOption
                selectStop = [stop[0],stop[1]]
                selectVehicle = vehicle
                selectDepTime = vehicle.depTime
                selectReturnTime = selectDepTime + routeTime[-1]
                selectInVehTime = inVehTime
                selectTime = routeTime
                selectWalkTime = walkTime
                newSequence = sequence
                
        # Check if insertions were succesful insertions were compared.
        # If this is the case return the offer.
        if bestScore < 999 and selectVehicle != False:
            if selectTime == []:
                print('How did best score get lower than 999, if no feasible route was found')
                print(travelerID)
                print(vehicle)
                print(selectRoute, selectInVehTime, selectRouteScore, bestScore, selectTime, sequence, walkTime, stop)
            return selectVehicle, selectInVehTime, selectDepTime, selectReturnTime, selectRoute, selectRouteScore, newRoute, newSequence, selectTime, selectWalkTime, selectStop

    '''OPTION 4 is to select the FIRST vehicle to become idle. This must also be
    triggered if select vehicle fails first..
    NOTE: since it does not matter which vehicle is idle, any vehicle will
    do. Hence, return the first "first idle" vehicle to be found.'''
    selected = select_first_available(reqTime, vehicles)              
    if selectInVehTime == False:
        #print('There was no schedule available to combine the request and no idle vehicle. Select first vehicle to return (become idle): ' + str(vehicle))
        newRoute = True
        earliestReturn = 10000
        for try_vehicle in vehicles:
            if try_vehicle.returnTime < earliestReturn:
                earliestReturn = try_vehicle.returnTime
                vehicle = try_vehicle

        routeOption, inVehTime, routeScore, score, routeTime, sequence, walkTime, stop = requestHandlerDesperate(vehicle, travelerID, destination, reqTime)
        selectRouteScore = routeScore
        selectRoute = routeOption
        selectStop = stop
        selectVehicle = vehicle
        selectInVehTime = inVehTime
        newSequence = sequence
        selectDepTime = vehicle.returnTime
        selectReturnTime = selectDepTime + routeTime[-1]
        selectTime = routeTime
        selectWalkTime = walkTime
        
        return selectVehicle, selectInVehTime, selectDepTime, selectReturnTime, selectRoute, selectRouteScore, newRoute, newSequence, routeTime, walkTime, selectStop
        
    # If all else fails, throw a message
    elif (selectInVehTime != False) and routeTime == []:
        print('Als we hier komen worden er reizigers niet goed voorzien van een aanbod... ')
        print('empty routeTime returned for' + str(travelerID) + 'with offer')
        print(selectVehicle, selectInVehTime, selectDepTime, selectReturnTime, selectRoute, selectRouteScore, newRoute, newSequence, routeTime, walkTime, selectStop)
    
    return selectVehicle, selectInVehTime, selectDepTime, selectReturnTime, selectRoute, selectRouteScore, newRoute, newSequence, selectTime, selectWalkTime, selectStop

def requestHandlerInsert(vehicle, travelerID, destination, reqTime):
    'Initiate insertion parameters'
    speed = vehicle.speed # [km/h]
    boardingTime = 2 # [min]
    serviceTime = 1 # [min]
    maxDev = travelerID.max_dev_fact # maximum allowed deviation factor
    station = vehicle.station    
    walkTime = False
    'Set values to try inserting at differ'
    vehicle_route = vehicle.route    
    originalScore = vehicle.routeScore
    bestScore = 1000
    bestRoute = np.zeros((len(vehicle_route)+1,2))
    bestStop = False
    
    # Check if the dispatching algorithm needs to select the nearest stops
    # to the destination
    if vehicle.stopLocations == True:
        spacing = vehicle.stopSpacing # assume at first this is 5 min walk max
        stops = selectStops(station,destination,spacing) 
        # first outer loop to permute drop-off location of possible insertion
        for stop in stops:
            for i in range(1,len(vehicle_route)):
                # add dropoff 'destination' at index i in newRoute
                feasibility = True
                # construct the sequence from the vehicle object
                testSequence = []
                
                for drop in vehicle.sequence:
                    testSequence.append(drop)
                # Create empty arrays with the length of the existing route -1
                # to calculate the deviation factors for the passengers
                directDist = np.zeros(len(vehicle_route)-1)
                directTime = np.zeros(len(vehicle_route)-1) # calculate direct time as speed/euclidean distance + boarding and service times
                devFact = np.zeros(len(vehicle_route)-1)  # calculate deviation factor based on totaltime/direct time
                directDevFact = np.zeros(len(vehicle_route)-1)
                # Create empty arrays with the length of the existing route +1
                # To keep track of the total route length and duration
                totalDist = np.zeros(len(vehicle_route)+1) # At each iteration, calculate added distance as euclidean distance between last stop and the new stop.        
                totalTime = np.zeros(len(vehicle_route)+1) # At each iteration, calculate the added time as 1 minute service time + dist to next stop divide by: (assumed speed [km/h]/min in an hr) 
                totalTime[0] = boardingTime    # Add one minute for boarding at the start      
                # Variables to iterate the route with
                route = np.zeros((len(vehicle_route)+1,2))
                route[0] = station # initiate the route
                lastStop = station # initiate the last stop variable
                
                # Route before insertion index
                for j in range(1,i):
                    newStop = vehicle_route[j]
                    route[j] = newStop
                    # The route associated parameters (vector length equals nr. passengers + 2 (start & stop))
                    totalDist[j] = totalDist[j-1] + distance_func(lastStop,newStop)
                    if lastStop[0] == newStop[0] and lastStop[1] == newStop[1]:
                        totalTime[j] = totalTime[j-1]+ distance_func(lastStop,newStop)/(speed/60)
                    else:     
                        totalTime[j] = totalTime[j-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
                    # The passenger associated parameters (vector length equals nr. of passengers)
                    directDist[j-1] = distance_func(vehicle_route[0],newStop)
                    directTime[j-1] = directDist[j-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed + boarding + parking
                    devFact[j-1] = (totalTime[j]+vehicle.sequence[j-1].walkTime)/directTime[j-1]
                    directDevFact[j-1] = totalTime[j]/directTime[j-1]
                    lastStop = newStop #update the last stop variable
                
                # Insertion at index i
                newStop = stop
                route[i] = newStop
                t_walk = dist_walk(stop,destination)/(5/60)            
                
                # The route associated parameters (vector length equals nr. passengers + 2 (start & stop))
                totalDist[i] = totalDist[i-1] + distance_func(lastStop,newStop)
                if lastStop[0] == newStop[0] and lastStop[1] == newStop[1]:
                    totalTime[i] = totalTime[i-1]+ distance_func(lastStop,newStop)/(speed/60)
                else:     
                    totalTime[i] = totalTime[i-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
                
                # The passenger associated parameters (vector length equals nr. of passengers)
                # calculate the deviation factor
                directDist[i-1] = distance_func(vehicle_route[0],newStop)
                directTime[i-1] = directDist[i-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed
                deviationFactor = (totalTime[i]+t_walk)/directTime[i-1]
                directDevFactor = totalTime[i]/directTime[i-1]
                if deviationFactor <= maxDev:
                    devFact[i-1] = deviationFactor
                    directDevFact[i-1] = directDevFactor
                else:
                    feasibility = False  
                
                lastStop = newStop #update the last stop variable
            
                for j in range(i+1,len(vehicle_route)):
                    newStop = vehicle_route[j-1] # Correct index for insertion
                    route[j] = newStop
                    # The route associated parameters (vector length equals nr. passengers + 2 (start & stop))
                    totalDist[j] = totalDist[j-1] + distance_func(lastStop,newStop)
                    if lastStop[0] == newStop[0] and lastStop[1] == newStop[1]:
                        totalTime[j] = totalTime[j-1]+ distance_func(lastStop,newStop)/(speed/60)
                    else:     
                        totalTime[j] = totalTime[j-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
                    # The passenger associated parameters (vector length equals nr. of passengers)
                    directDist[j-1] = distance_func(vehicle_route[0],newStop)
                    directTime[j-1] = directDist[j-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed
                    deviationFactor = (totalTime[j]+vehicle.sequence[j-2].walkTime)/directTime[j-1]
                    directDevFactor = totalTime[j]/directTime[j-1]
                    if deviationFactor <= maxDev:
                        devFact[j-1] = deviationFactor
                        directDevFact[j-1] = directDevFactor
                    else:
                        feasibility = False
                    lastStop = newStop # update the last stop variable
                
                route[-1] = vehicle_route[-1]
                newStop = route[-1]
                totalDist[-1] = totalDist[-2] + distance_func(lastStop,newStop)
                totalTime[-1] = totalTime[-2] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
                
                if feasibility is True:
                    totalDev = 0
                    for factor in devFact:
                        totalDev += factor
                    if totalDev <= bestScore:
                        bestScore = totalDev
                        bestScores = directDevFact
                        bestRoute = route
                        bestStop = stop
                        inVehTime = totalTime[i]
                        bestTime = totalTime
                        walkTime = t_walk
                        bestSequence = testSequence # initiate the sequence of the try
                        bestSequence.insert(i-1,travelerID)       
    
    # If no stops need to be selected, proceed here as station-to-door
    else:
        'first outer loop to permute the drop-off location of possible insertion' 
        for i in range(1,len(vehicle_route)):
            # add dropoff 'destination' at index i in newRoute
            feasibility = True
            # construct the sequence from the vehicle object elements (carefully)
            testSequence = []
            for drop in vehicle.sequence:
                testSequence.append(drop)
            
            # To calculate the deviation factors for the passengers
            directDist = np.zeros(len(vehicle_route)-1)
            directTime = np.zeros(len(vehicle_route)-1) # calculate direct time as speed/euclidean distance + boarding and service times
            devFact = np.zeros(len(vehicle_route)-1)  # calculate deviation factor based on totaltime/direct time
            # To keep track of the total route length and duration
            totalDist = np.zeros(len(vehicle_route)+1) # At each iteration, calculate added distance as euclidean distance between last stop and the new stop.        
            totalTime = np.zeros(len(vehicle_route)+1) # At each iteration, calculate the added time as 1 minute service time + dist to next stop divide by: (assumed speed [km/h]/min in an hr) 
            totalTime[0] = boardingTime    # Add one minute for boarding at the start      
            # Variables to iterate the route with
            route = np.zeros((len(vehicle_route)+1,2))
            route[0] = station # initiate the route
            lastStop = station # initiate the last stop variable

            # Route before insertion
            for j in range(1,i):
                newStop = vehicle_route[j]
                route[j] = newStop
                totalDist[j] = totalDist[j-1] + distance_func(lastStop,newStop)
                totalTime[j] = totalTime[j-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
                directDist[j-1] = distance_func(vehicle_route[0],newStop)
                directTime[j-1] = directDist[j-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed + boarding + parking
                devFact[j-1] = totalTime[j]/directTime[j-1]
                lastStop = newStop #update the last stop variable
 
            # Insertion at index i
            newStop = destination
            route[i] = newStop
            totalDist[i] = totalDist[i-1] + distance_func(lastStop,newStop)
            totalTime[i] = totalTime[i-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
            
            # calculate the deviation factor
            directDist[i-1] = distance_func(vehicle_route[0],newStop)
            directTime[i-1] = directDist[i-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed
            deviationFactor = totalTime[i]/directTime[i-1]
            if deviationFactor <= maxDev:
                devFact[i-1] = deviationFactor
            else:
                feasibility = False
            lastStop = newStop #update the last stop variable
        
            for j in range(i+1,len(vehicle_route)):
                newStop = vehicle_route[j-1] # Correct index for insertion
                route[j] = newStop

                totalDist[j] = totalDist[j-1] + distance_func(lastStop,newStop)
                totalTime[j] = totalTime[j-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
                directDist[j-1] = distance_func(vehicle_route[0],newStop)
                directTime[j-1] = directDist[j-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed
                deviationFactor = totalTime[j]/directTime[j-1]
                if deviationFactor <= maxDev:
                    devFact[j-1] = deviationFactor
                else:
                    feasibility = False
                lastStop = newStop #update the last stop variable
            
            route[-1] = vehicle_route[-1]
            newStop = route[-1]
            totalDist[-1] = totalDist[-2] + distance_func(lastStop,newStop)
            totalTime[-1] = totalTime[-2] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
            
            if feasibility is True:
                totalDev = 0
                for factor in devFact:
                    totalDev += factor
                if totalDev <= bestScore:
                    bestScore = totalDev
                    bestScores = devFact
                    bestRoute = route
                    bestStop = route[i]
                    inVehTime = totalTime[i]
                    bestTime = totalTime
                    bestSequence = testSequence # initiate the sequence of the try
                    bestSequence.insert(i-1,travelerID)

    if bestScore == 1000: # When no feasible insertion could be made
         bestScores = False # to hold the devFactors
         inVehTime = False
         bestRoute = False
         bestSequence = False
         walkTime = False
         bestTime = []
         bestStop = False

    return bestRoute, inVehTime, bestScores, bestScore, bestTime, bestSequence, walkTime, bestStop
    
def requestHandlerIdle(vehicle, travelerID, destination, reqTime):
    'Initiate insertion parameters'
    speed = vehicle.speed # [km/h]
    boardingTime = 2 # [min]
    serviceTime = 1 # [min]
    maxDev = travelerID.max_dev_fact # maximum allowed deviation factor
    station = vehicle.station    
    walkTime = False

    'Define new route for the vehicle that is first to return'
    # Check if the dispatching algorithm needs to select the nearest stops
    # to the destination
    if vehicle.stopLocations == True:
        # if spacing is max of 5 min walk, 5/60 = 0.08333 km max walking dist
        # spacing = 0.083333/(0.5*np.sqrt(2))
        spacing = vehicle.stopSpacing # assume at first this is 5 min walk max
        stops = selectStops(station,destination,spacing) 
        t_best = 100
        for stop in stops:
            t_walk = dist_walk(stop,destination)/(5/60)
            t_drive = boardingTime+serviceTime+distance_func(station,stop)/(speed/60)
            if t_walk+t_drive <= t_best:
                t_best = t_walk+t_drive
                walkTime = t_walk
                inVehTime = t_drive
                bestStop = stop  
         
        directTime = distance_func(station,destination)/(speed/60) + boardingTime + serviceTime
        bestScores = [1]
        bestScore = 1
        totalTime = [boardingTime, boardingTime + serviceTime + distance_func(station,stop)/(speed/60),\
                 boardingTime + serviceTime + 2*distance_func(station,stop)/(speed/60)]
        bestRoute = [station, stop, station]
        
    # If no stops need to be selected, proceed here
    else:
        bestRoute = [station, destination, station]    
        inVehTime = boardingTime+ serviceTime + distance_func(station,destination)/(speed/60)
        bestScores = [1] # to hold the devFactors
        bestScore = 1
        totalTime = [boardingTime, boardingTime + serviceTime + distance_func(station,destination)/(speed/60),\
                 boardingTime + serviceTime + 2*distance_func(station,destination)/(speed/60)]
        bestStop = destination
    sequence = [travelerID]
    sequence2 = sequence

    return bestRoute, inVehTime, bestScores, bestScore, totalTime, sequence, walkTime, bestStop

def requestHandlerFutureInsert(vehicle, travelerID, destination, reqTime):
    'Initiate insertion parameters'
    speed = vehicle.speed # [km/h]
    boardingTime = 2 # [min]
    serviceTime = 1 # [min]
    maxDev = travelerID.max_dev_fact # maximum allowed deviation factor
    station = vehicle.station    
    walkTime = False
    'Set values to try inserting at differ'
    vehicle_route = vehicle.route    
    originalScore = vehicle.routeScore
    bestScore = 1000
    devScore = 1000
    bestRoute = np.zeros((len(vehicle_route)+1,2))
    bestStop = False
    waitingTime = vehicle.depTime - travelerID.arrTime
    
    # Check if the dispatching algorithm needs to select the nearest stops
    # to the destination
    if vehicle.stopLocations == True:
        spacing = vehicle.stopSpacing # assume at first this is 5 min walk max
        stops = selectStops(station,destination,spacing) 
        # first outer loop to permute drop-off location of possible insertion
        for stop in stops:
            for i in range(1,len(vehicle_route)):
                # add dropoff 'destination' at index i in newRoute
                feasibility = True
                # construct the sequence from the vehicle object elements (carefully)
                testSequence = []
                
                for drop in vehicle.sequence:
                    testSequence.append(drop)
                
                # Create empties to calculate deviation factors for the passengers
                directDist = np.zeros(len(vehicle_route)-1)
                directTime = np.zeros(len(vehicle_route)-1) # calculate direct time as speed/euclidean distance + boarding and service times
                devFact = np.zeros(len(vehicle_route)-1)  # calculate deviation factor based on totaltime/direct time
                directDevFact = np.zeros(len(vehicle_route)-1)
                # Create empties to keep track of total route length and duration
                totalDist = np.zeros(len(vehicle_route)+1) # At each iteration, calculate added distance as euclidean distance between last stop and the new stop.        
                totalTime = np.zeros(len(vehicle_route)+1) # At each iteration, calculate the added time as 1 minute service time + dist to next stop divide by: (assumed speed [km/h]/min in an hr) 
                totalTime[0] = boardingTime    # Add one minute for boarding at the start      
                # Variables to iterate the route with
                route = np.zeros((len(vehicle_route)+1,2))
                route[0] = station # initiate the route
                lastStop = station # initiate the last stop variable
                
                # Route before insertion index
                for j in range(1,i):
                    newStop = vehicle_route[j]
                    route[j] = newStop
                    # The route associated parameters (vector length equals nr. passengers + 2 (start & stop))
                    totalDist[j] = totalDist[j-1] + distance_func(lastStop,newStop)
                    if lastStop[0] == newStop[0] and lastStop[1] == newStop[1]:
                        totalTime[j] = totalTime[j-1]+ distance_func(lastStop,newStop)/(speed/60)
                    else:     
                        totalTime[j] = totalTime[j-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
                    # The passenger associated parameters (vector length equals nr. of passengers)
                    directDist[j-1] = distance_func(vehicle_route[0],newStop)
                    directTime[j-1] = directDist[j-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed + boarding + parking
                    devFact[j-1] = (totalTime[j]+vehicle.sequence[j-1].walkTime)/directTime[j-1]
                    directDevFact[j-1] = totalTime[j]/directTime[j-1]
                    lastStop = newStop # update the last stop variable
                
                # Insertion at index i
                newStop = stop
                route[i] = newStop
                t_walk = dist_walk(stop,destination)/(5/60)            
                
                # The route associated parameters (vector length equals nr. passengers + 2 (start & stop))
                totalDist[i] = totalDist[i-1] + distance_func(lastStop,newStop)
                if lastStop[0] == newStop[0] and lastStop[1] == newStop[1]:
                    totalTime[i] = totalTime[i-1]+ distance_func(lastStop,newStop)/(speed/60)
                else:     
                    totalTime[i] = totalTime[i-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
                
                # The passenger associated parameters (vector length equals nr. of passengers)
                # calculate the deviation factor
                directDist[i-1] = distance_func(vehicle_route[0],newStop)
                directTime[i-1] = directDist[i-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed
                deviationFactor = (totalTime[i]+t_walk)/directTime[i-1]
                directDevFactor = totalTime[i]/directTime[i-1]
                if deviationFactor <= maxDev:
                    devFact[i-1] = deviationFactor 
                    #devFact[i-1] = (totalTime[i]+t_walk+waitingTime)/directTime[i-1] # Save the corrected deviation factor
                    directDevFact[i-1] = directDevFactor
                else:
                    feasibility = False  
                
                lastStop = newStop #update the last stop variable
            
                for j in range(i+1,len(vehicle_route)):
                    newStop = vehicle_route[j-1] # Correct index for insertion
                    route[j] = newStop
                    # The route associated parameters (vector length equals nr. passengers + 2 (start & stop))
                    totalDist[j] = totalDist[j-1] + distance_func(lastStop,newStop)
                    if lastStop[0] == newStop[0] and lastStop[1] == newStop[1]:
                        totalTime[j] = totalTime[j-1]+ distance_func(lastStop,newStop)/(speed/60)
                    else:     
                        totalTime[j] = totalTime[j-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
                    # The passenger associated parameters (vector length equals nr. of passengers)
                    directDist[j-1] = distance_func(vehicle_route[0],newStop)
                    directTime[j-1] = directDist[j-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed
                    deviationFactor = (totalTime[j]+vehicle.sequence[j-2].walkTime)/directTime[j-1]
                    directDevFactor = totalTime[j]/directTime[j-1]
                    if deviationFactor <= maxDev:
                        devFact[j-1] = deviationFactor
                        directDevFact[j-1] = directDevFactor
                    else:
                        feasibility = False
                    lastStop = newStop #update the last stop variable
                
                route[-1] = vehicle_route[-1]
                newStop = route[-1]
                totalDist[-1] = totalDist[-2] + distance_func(lastStop,newStop)
                totalTime[-1] = totalTime[-2] + serviceTime + distance_func(lastStop,newStop)/(speed/60)

                timeScore = totalTime[-1] - vehicle.routeTime[-1] + waitingTime

                if feasibility is True and timeScore <= bestScore: # To assure that 
                    totalDev = 0
                    devScore = 1000
                    for factor in devFact: # OR directDevFact, don't know which was the ORIGINAL
                        totalDev += factor
                    if totalDev <= devScore:    
                        bestScore = totalDev
                        bestScores = directDevFact
                        bestRoute = route
                        bestStop = stop
                        inVehTime = totalTime[i]
                        bestTime = totalTime
                        walkTime = t_walk
                        bestSequence = testSequence # initiate the sequence of the try
                        bestSequence.insert(i-1,travelerID)

    # If no stops are used, proceed here as station-to-door               
    else:
        'first outer loop to permute the drop-off location of possible insertion' 
        for i in range(1,len(vehicle_route)):
            # add dropoff 'destination' at index i in newRoute
            feasibility = True
            # construct the sequence from the vehicle object elements (carefully)
            testSequence = []
            for drop in vehicle.sequence:
                testSequence.append(drop)
            
            # To calculate the deviation factors for the passengers
            directDist = np.zeros(len(vehicle_route)-1)
            directTime = np.zeros(len(vehicle_route)-1) # calculate direct time as speed/euclidean distance + boarding and service times
            directDevFact = np.zeros(len(vehicle_route)-1)  # calculate deviation factor based on totaltime/direct time
            # To keep track of the total route length and duration
            totalDist = np.zeros(len(vehicle_route)+1) # At each iteration, calculate added distance as euclidean distance between last stop and the new stop.        
            totalTime = np.zeros(len(vehicle_route)+1) # At each iteration, calculate the added time as 1 minute service time + dist to next stop divide by: (assumed speed [km/h]/min in an hr) 
            totalTime[0] = boardingTime    # Add one minute for boarding at the start      
            # Variables to iterate the route with
            route = np.zeros((len(vehicle_route)+1,2))
            route[0] = station # initiate the route
            lastStop = station # initiate the last stop variable
            
            # Route before insertion
            for j in range(1,i):
                newStop = vehicle_route[j]
                route[j] = newStop

                totalDist[j] = totalDist[j-1] + distance_func(lastStop,newStop)
                totalTime[j] = totalTime[j-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
                directDist[j-1] = distance_func(vehicle_route[0],newStop)
                directTime[j-1] = directDist[j-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed + boarding + parking
                directDevFact[j-1] = totalTime[j]/directTime[j-1]
                lastStop = newStop #update the last stop variable
            
            # Insertion at index i
            newStop = destination
            route[i] = newStop

            totalDist[i] = totalDist[i-1] + distance_func(lastStop,newStop)
            totalTime[i] = totalTime[i-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
            # calculate the deviation factor
            directDist[i-1] = distance_func(vehicle_route[0],newStop)
            directTime[i-1] = directDist[i-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed
            deviationFactor = totalTime[i]/directTime[i-1]
            if deviationFactor <= maxDev:
                directDevFact[i-1] = deviationFactor
            else:
                feasibility = False
            lastStop = newStop #update the last stop variable
        
            for j in range(i+1,len(vehicle_route)):
                newStop = vehicle_route[j-1] # Correct index for insertion
                route[j] = newStop

                totalDist[j] = totalDist[j-1] + distance_func(lastStop,newStop)
                totalTime[j] = totalTime[j-1] + serviceTime + distance_func(lastStop,newStop)/(speed/60)
                directDist[j-1] = distance_func(vehicle_route[0],newStop)
                directTime[j-1] = directDist[j-1]/(speed/60) + boardingTime + serviceTime # time = distance/speed
                deviationFactor = totalTime[j]/directTime[j-1]
                if deviationFactor <= maxDev:
                    directDevFact[j-1] = deviationFactor
                else:
                    feasibility = False
                lastStop = newStop #update the last stop variable
            
            route[-1] = vehicle_route[-1]
            newStop = route[-1]
            totalDist[-1] = totalDist[-2] + distance_func(lastStop,newStop)
            totalTime[-1] = totalTime[-2] + serviceTime + distance_func(lastStop,newStop)/(speed/60)

            timeScore = totalTime[-1] - vehicle.routeTime[-1] + waitingTime + walkTime

            if feasibility is True and timeScore <= timeScore:
                totalDev = 0
                devScore = 1000
                for factor in directDevFact:
                    totalDev += factor
                if totalDev <= devScore:
                    bestScore = totalDev
                    bestScores = directDevFact
                    bestRoute = route
                    bestStop = route[i]
                    inVehTime = totalTime[i]
                    bestTime = totalTime
                    bestSequence = testSequence # set the sequence of the try
                    bestSequence.insert(i-1,travelerID)

    if bestScore == 1000: # When no feasible insertion could be made in the future
         bestScores = False # to hold the devFactors
         inVehTime = False
         bestRoute = False
         bestSequence = False
         walkTime = False
         bestTime = []
         bestStop = False
    
    return bestRoute, inVehTime, bestScores, bestScore, bestTime, bestSequence, walkTime, bestStop
      
def requestHandlerDesperate(vehicle, travelerID, destination, reqTime):
    'Initiate insertion parameters'
    speed = vehicle.speed # [KM/H]
    boardingTime = 1 # [min]
    serviceTime = 1 # [min]
    maxDev = travelerID.max_dev_fact # maximum allowed deviation factor
    station = vehicle.station 
    walkTime = False
    
    'Define new route for the vehicle that is first to return'
    # Check if the dispatching algorithm needs to select the nearest stops
    # to the destination
    if vehicle.stopLocations == True:
        spacing = vehicle.stopSpacing # assume at first this is 5 min walk max
        stops = selectStops(station,destination, spacing)
        t_best = 100
        for stop in stops:
            t_walk = dist_walk(stop,destination)/(5/60)
            t_drive = boardingTime+serviceTime+distance_func(station,stop)/(speed/60)
            if t_walk+t_drive <= t_best:
                t_best = t_walk+t_drive
                walkTime = t_walk
                inVehTime = t_drive
                bestStop = stop  
        
        directTime = distance_func(station,destination)/(speed/60) + boardingTime + serviceTime
        dev = t_best/directTime
        bestScores = [1]
        bestScore = 1
        totalTime = [boardingTime, boardingTime + serviceTime + distance_func(station,stop)/(speed/60),\
                 boardingTime + serviceTime + 2*distance_func(station,stop)/(speed/60)]
        bestRoute = [station, stop, station]
    # If no stops are used, proceed here as station-to-door
    else:
        bestRoute = [station, destination, station]    
        inVehTime = boardingTime+ serviceTime + distance_func(station,destination)/(speed/60)
        bestScores = [1] # to hold the devFactors
        bestScore = 1
        totalTime = [boardingTime, boardingTime + serviceTime + distance_func(station,destination)/(speed/60),\
                 boardingTime + serviceTime + 2*distance_func(station,destination)/(speed/60)]
        bestStop = destination
    sequence = [travelerID]
    return bestRoute, inVehTime, bestScores, bestScore, totalTime, sequence, walkTime, bestStop

# Determine the mode choice based on the offer made by the service.
# Use the optional strategy for the utility of alternative modes so this
# function will be more simple and efficient to run.   
def modeChoice(station, destination, travelerID, offer, reqTime, devExpect, price1, price2):    
    stoch_parts = travelerID.stoch_parts
    bestUtil = -1000 # initiate low value for overwriting with best option
    utility = np.zeros(len(stoch_parts))
    start = station # include start for distance_func
    distance = distance_func(start,destination)
    waitingTime = offer[2] - reqTime
    
    for x in range(len(stoch_parts)):        
        if x == 0:
            util_deterministic =  offer_utility(offer, distance, reqTime, devExpect, price1, price2, travelerID.value_tt)
            utility[0] = stoch_parts[0] + 0.55 * util_deterministic
        else:
            utility[x] = stoch_parts[x] + util_func(x, distance, travelerID.value_tt)
    for x in range(len(utility)):
        if utility[x] >= bestUtil:
            bestUtil = utility[x]
            best_choice = x
    return(best_choice, bestUtil) # 0 = SAV, 1 = ebicycle, 2 = estep, 3 = escooter

def util_func(mode, distance, value_tt):
    'Determine the deterministic part of util function for each mode'
    tPickup = 2 # add a pickup time or the shared mode
    if value_tt == 'high':
        utility_tt = 0.109 * 2
    else:
        utility_tt = 0.109
    
    if mode == 1: # bicycle
        speed = 15 # the average google maps result for two tests in rotterdam
        tt = distance/15*60 # t=s/v
        tc = 1.5              # cost = 1 euro
        av = 1            # availability of mode
        #print('Mode 1 tt = ' + str(tt))
        utility = -0.614*tc -utility_tt*(tt+tPickup) + 0.00537*av    
    elif mode == 2: # estep
        tt = distance/17*60 # speed assumed as lower then scooter
        #print('Mode 2 tt = ' + str(tt))
        tc = 0.75+0.21*tt           # unlock 0.75, fee/min = 0.21
        av = 1            # availability of mode 
        utility = -0.275 -0.614*tc -utility_tt*(tt+tPickup) + 0.00537*av
    else: # DCM == 3: # escooter
        tt = distance/20*60
        #print('Mode 3 tt = ' + str(tt))
        tc = 0.75 + tt*0.3       # unlock 0.75, fee/min = 0.3
        av = 1            # availability of mode 
        utility = -0.35 -0.614*tc -utility_tt*(tt+tPickup) + 0.00537*av
    return utility

def offer_utility(offer, distance, reqTime, devFact, price1, price2, value_tt):
    boarding_time = 2
    service_time = 1
    
    if value_tt == 'high':
        utility_tt = 0.109 * 2
    else:
        utility_tt = 0.109
    
    waitTime = offer[2] - reqTime # offer[2] = depTime
    if offer[9] != False:
        walkTime = offer[9] # the walktime of the offered service
    else:
        walkTime = 0
    speed = offer[0].speed # the assumed speed for a car in an urban newtork
    direct_tt = distance/speed*60
    tc = price1+price2*direct_tt
    in_vehicle_time = (direct_tt + boarding_time\
                       + service_time) * devFact
    if offer[0] == False:
        utility = -1000
    else:
        utility = 0.081 + 0.00537 -0.614*tc  -utility_tt*(in_vehicle_time +\
            1.6*walkTime + 2.2*waitTime) # includes availability (1 * 0.00537)    
    return utility

def actual_demand(station, destination, waitTimeExpect, devFact, walkTimeExpect, price1, price2, opt_out_distance, stoch_parts, value_tt):
    start = station # include start for distance_func
    distance = distance_func(start,destination)
    walking_distance = distance/(np.sqrt(2))
    utility_best = -30
    boarding_time = 2
    service_time = 1
    tPickup = 2
    
    if value_tt == 'high':
        utility_tt = 0.109 * 2
    else:
        utility_tt = 0.109
            
    if walking_distance < opt_out_distance:
        interest = 4
    else:
        for mode in range(len(stoch_parts)):
            if mode == 0:
                'hardcoded speed definitions here'
                speed = 27 # the assumed speed for a car in an urban network
                direct_tt = distance/speed*60
                tc = price1+price2*direct_tt
                in_vehicle_time = (direct_tt + boarding_time\
                                   + service_time) * devFact
                av = 1
                utility = 0.55*(0.081 + 0.00537 - 0.614*tc - utility_tt*(in_vehicle_time\
                    + 1.6*walkTimeExpect + 2.2*waitTimeExpect)) + stoch_parts[mode] # Adding waiting time
            elif mode == 1: # bicycle, reference mode so alternative specific
                # constant is determined by the random part (ASC[mode])
                speed = 15 # based on avg google maps for two tests in Rdam
                tt = distance/15*60  # t=s/v
                tc = 1.5              # cost = 1 euro
                av = 1            # availability of mode 
                utility = -0.614*tc -utility_tt*(tt+ tPickup) + 0.00537*av + stoch_parts[mode]
            elif mode == 2: # estep
                tt = distance/17*60
                tc = 0.75+0.21*tt           # unlock 0.75, fee/min = 0.21
                av = 1            # availability of mode 
                utility = -0.275 -0.614*tc -utility_tt*(tt+ tPickup) + 0.00537*av + stoch_parts[mode]
            else: # DCM == 3: # escooter
                tt = distance/20*60
                tc = 0.75 + tt*0.3       # unlock 0.75, fee/min = 0.3
                av = 1            # availability of mode 
                utility = -0.35 -0.614*tc -utility_tt*(tt+ tPickup) + 0.00537*av + stoch_parts[mode]
            if utility >= utility_best:
                utility_best = utility
                interest = mode

    return interest, utility_best

'Selects all vehicles that departe at the requested time for scheduling'    
def select(reqTime, vehicles):
    selected = []    
    for vehicle in vehicles:
        if vehicle.depTime == reqTime and len(vehicle.sequence) < vehicle.capacity:
            selected.append(vehicle)
    if selected == []:
        selected = False
    return selected

'Then see if there is an empty vehicle available'
def select_idle(reqTime, vehicles):
    select = False
    for vehicle in vehicles:
        if vehicle.returnTime <= reqTime:
            select = vehicle
    return select

def select_future_departures(reqTime, vehicles):
    preselection = []
    selected = []
    for vehicle in vehicles:
        if vehicle.depTime > reqTime and len(vehicle.sequence) < vehicle.capacity:
            preselection.append(vehicle)        
    selected = sorted(preselection, key = lambda select:select.depTime)  
    return selected

'''selects earliest vehicle to return'''
def select_first_available(reqTime, vehicles): 
    select = False
    firstAvailable = 100000
    for vehicle in vehicles:
        if firstAvailable > vehicle.returnTime:
                firstAvailable = vehicle.returnTime
                select = vehicle
    return(select)

'Create the arrivals by train using bins'
def arrival_distribution(total_travelers, span, arrival_frequency):
    
    trains_hour = 8*arrival_frequency # the number of trains arriving each hour
    nr_bins = int(trains_hour*span) # determine number of trains during period
    bins = [0]*nr_bins # make list containing each bin
    for i in range(total_travelers):
        bins[sim.IntUniform(0,len(bins)-1).sample()] += 1
    return bins # return trains with arriving train travelers

'used in traveler generation to create random coordinates'
def coords(station,destination_indicator):
    # Use station as center coordinate, meaning that total canvas ranges for
    # x and y are twice the coordinates of the station coordinates.
    area_size_fact=station[0]/2.5
    if destination_indicator == 'outside' or destination_indicator == 'outside_high_tt':
        x = sim.Uniform(0,2*station[0]).sample()
        y = sim.Uniform(0,2*station[1]).sample()
        while np.sqrt((x-station[0])**2 + (y-station[1])**2) < (1*area_size_fact):
            x = sim.Uniform(0,2*station[0]).sample()
            y = sim.Uniform(0,2*station[1]).sample()
    else: # Define center area as rectangle 1 km from center, multiply with area size
        x = sim.Uniform(0.6*station[0],1.4*station[0]).sample()
        y = sim.Uniform(0.6*station[1],1.4*station[1]).sample()
        while np.sqrt((x-station[0])**2 + (y-station[1])**2) >= 1*(area_size_fact):
            x = sim.Uniform(0.6*station[0],1.4*station[0]).sample()
            y = sim.Uniform(0.6*station[0],1.4*station[0]).sample()
        
    return [x,y] # return coordinates

'to select the nearest stop locations to the passenger destination'
def selectStops(station,destination, spacing):
    
    for i in range(len(destination)):    
    	goal = (destination[i]-station[i])/spacing
    	if round(goal) <= goal:
    		low = (round(goal))*spacing # x_low coordinate of stoplocation
    		up = (round(goal)+1)*spacing        # x_up coordinate of stoplocation
    	if round(goal) >= goal:
    		up = (round(goal))*spacing  # x_low coordinate of stoplocation
    		low = (round(goal)-1)*spacing       # x_up coordinate of stoplocation
    	if i == 0:
    		x_stop=[low+station[i],up+station[i]]
    	elif i == 1:
    		y_stop=[low+station[i],up+station[i]]  
    
    stop1 = [x_stop[0],y_stop[1]]
    stop2 = [x_stop[1],y_stop[1]]
    stop3 = [x_stop[0],y_stop[0]]
    stop4 = [x_stop[1],y_stop[0]]
    stop_options = [stop1,stop2,stop3,stop4]
    
    stops = []
    for stop in stop_options:
        if stop != station:
            stops.append(stop)
    
    return(stops)

'determine euclidean distances'
def distance_func(start, stop):
    # Determine euclidean distance between two coordinates
    # Use the root of 2 as an assumed factor to determine route distance
    dist = np.sqrt(2) * np.sqrt((stop[0]-start[0])**2 + (stop[1]-start[1])**2)
    return dist

def route_distance_func(sequence, station):
    routeDist = []
    for passenger in range(len(sequence)): # get the route distances old route
        if (passenger == 0):
            start_coords = station
            stop_coords = sequence[passenger].offer[10]
            dist = distance_func(start_coords,stop_coords)
            routeDist.append(dist)
            if (passenger == range(len(sequence))[-1]):
                start_coords = sequence[passenger].offer[10]
                stop_coords = station
                dist = distance_func(start_coords,stop_coords)
                dist += routeDist[-1]
                routeDist.append(dist)
        
        # If the passenger is the last passenger on the route, also add the
        # route distance to get back to the station
        elif (passenger == range(len(sequence))[-1]):
            start_coords = sequence[passenger-1].offer[10]
            stop_coords = sequence[passenger].offer[10]
            dist = distance_func(start_coords,stop_coords)
            dist += routeDist[-1]
            routeDist.append(dist)
            
            start_coords = sequence[passenger].offer[10]
            stop_coords = station
            dist = distance_func(start_coords,stop_coords)
            dist += routeDist[-1]
            routeDist.append(dist)    
        else:
            start_coords = sequence[passenger-1].offer[10]
            stop_coords = sequence[passenger].offer[10]
            dist = distance_func(start_coords,stop_coords)
            dist += routeDist[-1]
            routeDist.append(dist)
    return routeDist

def dist_walk(start, stop):
    dist = np.sqrt((stop[0]-start[0])**2 + (stop[1]-start[1])**2)
    return dist
    
'Use this just one time, or multiple times depending on how want to do this'
def write_to_csvfile(input_data, fileName):
    wrt = csv.writer(open(fileName + '.csv', 'w'), delimiter=',', lineterminator='\n')
    for row in input_data:
        wrt.writerow([row])
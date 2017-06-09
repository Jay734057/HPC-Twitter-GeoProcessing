# -*- coding: utf-8 -*-
import json
import os
from mpi4py import MPI
import numpy
import re   
import time 

# Dictionary for rows: define the row that the grid box belongs to.
Rows = {'A1' : 'A-Row','A2' :'A-Row','A3' : 'A-Row','A4' : 'A-Row',
'B1' : 'B-Row','B2' : 'B-Row','B3' : 'B-Row','B4' : 'B-Row',
'C1' : 'C-Row','C2' : 'C-Row','C3' : 'C-Row','C4' : 'C-Row','C5' : 'C-Row',
'D3' : 'D-Row','D4' : 'D-Row','D5' : 'D-Row'}

# Dictionary for columns: define the column that the grid box belongs to.
Columns = {'A1' : 'Column 1','A2' :'Column 2','A3' : 'Column 3',
'A4' : 'Column 4', 'B1' : 'Column 1','B2' : 'Column 2','B3' : 'Column 3',
'B4' : 'Column 4', 'C1' : 'Column 1','C2' : 'Column 2','C3' : 'Column 3',
'C4' : 'Column 4','C5' : 'Column 5','D3' : 'Column 3','D4' : 'Column 4',
'D5' : 'Column 5'
}

# Function parse_coordinates: parse the coordinate information from one line. 
# If the line does not contain coordinate information, return None. 
def parse_coordinates(data):
    match = re.search('{\"type\":\"Point\",\"coordinates\":\[[-+]?[0-9]*\.?[0-9]*,[-+]?[0-9]*\.?[0-9]*\]}',data)
    if match:
        # Find the match in the line, parse the coordinate information.
        coordinates = json.loads(match.group(0))
        return coordinates
    else:
        return None

# Function is_in_grid: determine if the coordinate is in the grid box.
# If it is in the grid box, return True; Otherwise, return False.
def is_in_grid(coordinates,grid):
    coordinates_x = coordinates[0]
    coordinates_y = coordinates[1]
    if coordinates_x >= grid[0] and coordinates_x < grid[1] \
            and coordinates_y >= grid[2] and coordinates_y < grid[3]:
        return True
    else:
        return False

# Function print_result: output the results of searching including the sorted
# grid boxes, rows, columns by number of tweets, and the execution time.       
def print_result(ids_of_grids, results):
    # Sort the grid boxes by number of tweets
    sorted_ids = [x for (y,x) in sorted(zip(results,ids_of_grids), \
                    reverse=True)]
    sorted_results = sorted(results,reverse=True)
    
    # Initial the dictionaries of {Row: number of tweet} and 
    # {Column: number of tweet}
    dict_for_rows = {'A-Row':0,'B-Row':0,'C-Row':0,'D-Row':0}
    dict_for_columns = {'Column 1':0,'Column 2':0,'Column 3':0, \
                        'Column 4':0,'Column 5':0}
    
    # Output the sorted list of grid boxes
    print("\nRank by grid boxes:")
    for index in range(0,len(ids_of_grids)):
        print("%s : %d tweets" % (sorted_ids[index],sorted_results[index]))
        # Calculate the number of tweets in rows or columns
        dict_for_rows[Rows[sorted_ids[index]]] += sorted_results[index]
        dict_for_columns[Columns[sorted_ids[index]]] += sorted_results[index]
        
    # Output the sorted list of Rows
    print("\nRank by rows:")   
    sorted_by_rows = sorted(dict_for_rows.items(),key=lambda x: x[1], \
                            reverse=True)
    for tuple in sorted_by_rows:
        print("%s : %d tweets" % (tuple[0],tuple[1]))
    
    # Output the sorted list of Columns 
    print("\nRank by columns:")  
    sorted_by_columns = sorted(dict_for_columns.items(),key=lambda x: x[1], \
                            reverse=True)
    for tuple in sorted_by_columns:
        print("%s : %d tweets" % (tuple[0],tuple[1]))
    
    # Output the execution time
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

# Get the start time of the searching
start_time = time.time()

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

comm.Barrier()

# Core with rank 0 retrieves the coordinate information of all grid boxes
if comm.rank == 0:
    # Get the file path
    script_dir = os.path.dirname(__file__)
    file_path_for_grid_data = os.path.join(script_dir, 'melbGrid.json')
    # Read the melbGrid file
    with open(file_path_for_grid_data) as melb_grid_file:    
        melb_grid_json = json.load(melb_grid_file)
        melb_grid_data = melb_grid_json["features"]
        num_of_grids = len(melb_grid_data)
        ids_of_grids = []
        coordinates_of_grids = []
        # Get detailed information
        for grid in melb_grid_data:
            grid_id = grid["properties"]["id"]
            xmin = grid["properties"]["xmin"]
            xmax = grid["properties"]["xmax"]
            ymin = grid["properties"]["ymin"]
            ymax = grid["properties"]["ymax"]
            ids_of_grids.append(grid_id)
            coordinates_of_grids.append([xmin,xmax,ymin,ymax]) 
    results = numpy.zeros(num_of_grids)    
    grid_data = {'num_of_grids' : num_of_grids,
            'coordinates_of_grids' : coordinates_of_grids,}
else:
    # Other cores do not need to read the file
    grid_data = None


comm.Barrier() 

# Determine the number of resources
if comm_size > 1:
    # Broadcast the coordinate information of all grid boxes
    grid_data = comm.bcast(grid_data, root = 0)
    
    # Core with rank 0 reads the file line by line and send the lines to other
    # nodes
    if comm.rank == 0:
        # Get the file path for bigTwitter.json
        file_path_for_twitter = os.path.join(script_dir, 'bigTwitter.json')
        twitter_file = open(file_path_for_twitter)
        f_results = numpy.zeros(num_of_grids)
        destinations = range(1,comm_size)
        index = 0
        # Read the file and send lines to other cores one by one
        line = twitter_file.readline()
        while line: 
            comm.send(line,dest=destinations[index])
            index = (index + 1) % len(destinations)
            line = twitter_file.readline()
        twitter_file.close() 
        # Reach the end of the file, send None to all other cores
        for i in destinations: 
            comm.send(None,i)
            
    # Cores with ranks higher than 0 process the lines they receive from 
    # core with rank 0
    else:
        coordinates_of_grids = grid_data['coordinates_of_grids']
        f_results = numpy.zeros(grid_data['num_of_grids'])
        results = numpy.zeros(grid_data['num_of_grids'])
        # Keep processing the received data if it is not None
        while True:
            line = comm.recv(source=0)
            if line == None:
                break
            else:
                # Get the coordinates from the line
                coordinates = parse_coordinates(line)
                if coordinates:
                    # Determine the grid box that the coordinates belongs to
                    for index in range(0, len(coordinates_of_grids)):
                        if is_in_grid(coordinates["coordinates"], \
                                        coordinates_of_grids[index]):
                            results[index] += 1
                            break
    comm.Barrier()
    
    # Core with rank 0 gathers the searching results from all other nodes
    comm.Reduce(results,f_results,root=0)
    
    # Core with rank 0 outputs the final searching result
    if comm.rank == 0:
        print_result(ids_of_grids, f_results)

# The script runs on a single core
else:
    # Open the file
    file_path_for_twitter = os.path.join(script_dir, 'bigTwitter.json')
    twitter_file = open(file_path_for_twitter)
    line = twitter_file.readline()
    # Process the file line by line
    while line: 
        # Get the coordinates from the line 
        coordinates = parse_coordinates(line)
        if coordinates:
            # Determine the grid box that the coordinates belongs to
            for index in range(0, len(coordinates_of_grids)):
                if is_in_grid(coordinates["coordinates"], \
                                coordinates_of_grids[index]):
                    results[index] += 1
                    break
        line = twitter_file.readline()
    twitter_file.close()
    # Outputs the searching result
    print_result(ids_of_grids, results)
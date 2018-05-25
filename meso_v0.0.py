'''
    Script to perform simple simulation of spheres on a plane
    Assumption:
    Once spheres come within a certain distance of one another they move 'as one'
    
    Authors: M CHavent, J Helie, A Duncan
    
    13 Jan 2017:
    added better image output (circles and wedges)
    cushion factor as an argument
    no PBC functionality
    
    8 March:
    added functionality to do restarts
    
    7 Apr:
    make angles list mod 2p*i
    
    Sept 2017:
    add options for starting struct ie. grid in space
    changed parser options so that default values are shown with help
    changed default angles for sticky patches (to amnuscript version), 
        output_frequency (from 10 to 100) and 
        cushion_cutoff (0.9 to 0.98 - consistent w manuscript)
    
    Added more parser options
    cleaned up log file output
    changed options for restarting so that restarting shd happen once inside existing sim directory
    changed precision of angles to float32 (in line w x-,y-coords)
    
    Oct 10-11 2017:
    attempted to make 'check_for_clustering_stickypatches' more efficient:
    cut out extra PBC noPBC-dist search
    made all 'for' loops into array calculations
    
    Oct 13 2017:
    Added the writing of a 'restart.txt.' file
    which includes commands necessary to restart sim painlessly
    
    Oct 17 2017
    Tried to get rid of for loops in movement and clashing clusts step
    This is mostly done except for the rotation step
    Also added functionality to read in previous frame cluster list when restarting
    
    Oct 26 2017
    Tried to optimise number of pairwise dists calculated when looking at clashing clusts
    '''
import numpy as np
import random
import math
import MDAnalysis
import MDAnalysis.analysis.distances
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import PatchCollection
from scipy.stats import norm

import os
import sys
import ast

import argparse
import time

import re

np.set_printoptions(precision=2) # sets number of decimal points to show when printin arrays - makes output a bit easier to read

#################################################################
# function to initiate system
#################################################################
# create the starting positions (square, trinagle or hex grid)
def initiate_system(n_particles, init_dist, grid_name, initial_boxsize=0, max_box=0):  
    # set up variables
    box_dimensions = np.array([0,0],dtype=np.float32)
    if(grid_name == 'square' or grid_name == 'square_in_space'):
        #check if n_particles is a square number otherwise take the nearest number:
        sqrt_n_particles = int(round(math.sqrt(n_particles)))

        # set up an array of (sqrt_n_particles)^2 particles, initially on a square grid. grid size is init_dist
        x_coords = np.array([[i*init_dist]*sqrt_n_particles for i in range(sqrt_n_particles)],dtype=np.float32).flatten() + 0.5*init_dist
        y_coords = np.array([[i*init_dist for i in range(sqrt_n_particles)]*sqrt_n_particles],dtype=np.float32).flatten() + 0.5*init_dist
        
		# calculate the dimension of the periodic box:
        if(grid_name == 'square'):
            box_dimensions = np.array([init_dist*(sqrt_n_particles),init_dist*(sqrt_n_particles)],dtype=np.float32)
        elif(grid_name == 'square_in_space'):
            if initial_boxsize < init_dist*(sqrt_n_particles):
                box_dimensions = np.array([init_dist*(sqrt_n_particles),init_dist*(sqrt_n_particles)],dtype=np.float32)
            else:
                box_dimensions = np.array([initial_boxsize,initial_boxsize],dtype=np.float32)
            print('initial_box_size = ',box_dimensions)
        if max_box:
            x_coords += (max_box - box_dimensions[0])/2.
            y_coords += (max_box - box_dimensions[1])/2.
            box_dimensions = np.array([max_box, max_box],dtype=np.float32)
        actual_n_particles = sqrt_n_particles**2
        angles = np.array([random.uniform(0,1)*2*np.pi for i in range(actual_n_particles)],dtype=np.float32)
        cluster_list = np.arange(actual_n_particles)
#        cluster_list = np.array([[i] for i in range(sqrt_n_particles**2)])

    if(grid_name == 'hex' or grid_name == 'triangle'):
		# define Height and width:
        H = math.sqrt(3)*init_dist

        counter = 0
        break_flag = 0
        x_list = []
        y_list = []
        size_x = 0
        size_y = 0

        sqrt_n_particles = math.sqrt(n_particles)
        i_grid_size = int(round(sqrt_n_particles*1.7))
        j_grid_size = int(round(sqrt_n_particles))
    
        for i in range(i_grid_size):
            for j in range(j_grid_size):
                if (counter==n_particles):
                    break_flag = 1
                    break;
                else:
                    if (j%2==1):
                        if(grid_name == 'hex'):
                            if (i%6 == 1 or i%6 == 3):
                                x_list.append(i*0.5*init_dist)
                                y_list.append(j*H*0.5)
                                size_x = i*0.5*init_dist
                                size_y = j*H*0.5
                                counter+=1
                    
                        if(grid_name == 'triangle'):
                            if (i%2 == 1):
                                x_list.append(i*0.5*init_dist)
                                y_list.append(j*H*0.5)
                                size_x = i*0.5*init_dist
                                size_y = j*H*0.5
                                counter+=1

                    if (j%2==0):
                        if(grid_name == 'hex'):
                            if (i%6 == 0 or i%6 == 4):
                                x_list.append(i*0.5*init_dist)
                                y_list.append(j*H*0.5)
                                size_x = i*0.5*init_dist
                                size_y = j*H*0.5
                                counter+=1

                        if(grid_name == 'triangle'):
                            if (i%2 == 0):
                                x_list.append(i*0.5*init_dist)
                                y_list.append(j*H*0.5)
                                size_x = i*0.5*init_dist
                                size_y = j*H*0.5
                                counter+=1
        
            if (break_flag==1):
                break;

        dim_x = max(x_list)-min(x_list)
        dim_y = max(y_list)-min(y_list)
        move_x = np.array([dim_x/2.0 for i in range(len(x_list))])
        move_y = np.array([dim_y/2.0 for i in range(len(y_list))])
        x_coords = np.array(x_list, dtype=np.float32) - np.array(move_x, dtype=np.float32)
        y_coords = np.array(y_list, dtype=np.float32) - np.array(move_y, dtype=np.float32)
        angles = np.array([random.uniform(0,1)*2*np.pi for i in range(n_particles)], dtype=np.float32)

        box_dimensions = np.array([dim_x+init_dist,dim_y+H],dtype=np.float32)
        print(box_dimensions)
        cluster_list = np.arange(len(x_coords))
#        cluster_list = np.array([[i] for i in range(len(x_coords))])
    
    return x_coords, y_coords, angles, cluster_list, box_dimensions

#################################################################
# functions for use with PBCs
#################################################################

# for when PBCs are used: puts coords back in a box
def coords_in_box(x_coords, y_coords, box_dim):
    '''this function centers coords around center and apply pbc along x and y
    convention: coords between 0 and +box_dim along x and y'''
    coords_loc_x = np.copy(x_coords)
    coords_loc_y = np.copy(y_coords)
    #pbc applied along x and y directions using modular function
    coords_loc_x = np.mod(coords_loc_x, box_dim[0])
    coords_loc_y = np.mod(coords_loc_y, box_dim[1])
    return coords_loc_x, coords_loc_y

def calculate_cog_2D(tmp_coords, box_dim):
    # from Jean membrane_prop scripts, altered to make two dimensional
    #this method allows to take pbc into account when calculcating the center of geometry 
    #see: http://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
    cog_coord = np.zeros(2) # changed from 3
    tmp_nb_atoms = np.shape(tmp_coords)[0]
    for n in range(0,2): # changed from (0,3) to (0,2)
        tet = tmp_coords[:,n] * 2 * math.pi / float(box_dim[n])
        xsi = np.cos(tet)
        zet = np.sin(tet)
        tet_avg = math.atan2(-np.average(zet),-np.average(xsi)) + math.pi
        cog_coord[n] = tet_avg * box_dim[n] / float(2*math.pi)
    return cog_coord 

#################################################################
# functions used to check clustering 
#################################################################

# this checks clustering in the case that sticky patches are NOT present (ie. when there are no specific interaction interfaces)
def check_for_clustering(dist, x_coords, y_coords, angles, n_particles, cutoff):
    # detect which proteins are clustering now
    proteins_array = np.array(zip(x_coords, y_coords, np.zeros(n_particles,dtype=np.float32)))
    connected = dist < float(cutoff)
    network = nx.Graph(connected)
    cluster_list = np.zeros(n_particles,dtype=np.int64)
    for i,component in enumerate(list(nx.connected_components(network))):
        cluster_list[component] = i
    return cluster_list

# this function is used in the check_for_clustering_stickypatches function
def values_in_ranges(value,list_of_endpoint_tuples):
    for pair in list_of_endpoint_tuples:
        if pair[0] <= value <= pair[1]:
            return True
    return False

# used to check clustering when sticky patches are present
def check_for_clustering_stickypatches(dist, x_coords, y_coords, angles, box_dimensions, half_box, n_particles, cutoff, sticky_patches_range, cluster_list, verbose, PBC, report_timings=False):
    # detect which proteins are clustered via their 'sticky patches'
    if report_timings:
        StartTime = time.time()
    connected = dist < float(cutoff)
    if len(connected) == 0:
#        print 'no connections to check'
        return cluster_list
    if verbose:
        pass
#        print 'initial_cluster_list: ', initial_cluster_list
    #print 'initial cluster list: ', initial_cluster_list
    if report_timings:
        currentTime1 = time.time()
#        print 'finding connected prots took: {} seconds'.format(currentTime1 - StartTime)
    
    angle_check_indices = np.nonzero(connected)
    if report_timings:
        currentTimeNonzero = time.time()
#        print 'generating indices of connected pairs took: {} seconds'.format(currentTimeNonzero - currentTime1)
    connected_pairs_indices_1 = angle_check_indices[0][angle_check_indices[0]> angle_check_indices[1]]
    connected_pairs_indices_2 = angle_check_indices[1][angle_check_indices[0]> angle_check_indices[1]]
    if report_timings:
        currentTimePairs = time.time()
#        print 'generating index pairs took: {} seconds'.format(currentTimePairs - currentTimeNonzero)
    #if pairs are already in the same cluster, no need to check them
    # so choose just those pairs that are not in the same cluster:
    tocheck = ( cluster_list[connected_pairs_indices_1] != cluster_list[connected_pairs_indices_2] )
    connected_pairs_tocheck_indices_1 = connected_pairs_indices_1[tocheck]
    connected_pairs_tocheck_indices_2 = connected_pairs_indices_2[tocheck]
    if report_timings:
        currentTimePairsFilter = time.time()
#        print 'filtering index pairs took: {} seconds'.format(currentTimePairsFilter - currentTimePairs)

    if len(connected_pairs_tocheck_indices_1) == 0:
        if report_timings:
            pass
            #print 'no connected pairs to check'
        return cluster_list

    if report_timings:
        currentTime2 = time.time()
#        print 'checking connected pairs to check took: {} seconds'.format(currentTime2 - currentTimeNonzero)

    delta_x = x_coords[connected_pairs_tocheck_indices_2] - x_coords[connected_pairs_tocheck_indices_1]
    delta_y = y_coords[connected_pairs_tocheck_indices_2] - y_coords[connected_pairs_tocheck_indices_1]
    if PBC:
        delta_x[delta_x > half_box[0]] -= box_dimensions[0]
        delta_x[delta_x < -half_box[0]] += box_dimensions[0]
        delta_y[delta_y > half_box[1]] -= box_dimensions[1]
        delta_y[delta_y < -half_box[1]] += box_dimensions[1]

    if report_timings:
        currentTime3 = time.time()
#        print 'getting delta_x and delta_y took: {} seconds'.format(currentTime3 - currentTime2)

    angle_between_proteins = np.arctan2(np.array(delta_y), np.array(delta_x))  
    # angle is given in radians, which is the units of 'angles' also.  According to documentation of arctan2, y-coord comes first.
    angle1 = ( angle_between_proteins - angles[connected_pairs_tocheck_indices_1] ) % (2*np.pi)
    angle2 = ( (angle_between_proteins+np.pi) - angles[connected_pairs_tocheck_indices_2] ) % (2*np.pi)
    # the two lines above take into account the individual protein orientation 

    if report_timings:
        currentTime4 = time.time()
#        print 'getting angles took: {} seconds'.format(currentTime4 - currentTime3)

    if verbose:
        pass
#        print 'len(connected_pairs), connected_pairs, len(angle_between_proteins), angle_between_proteins, angles[[0]], angles[[1]]: '
#        print len(connected_pairs_tocheck), connected_pairs_tocheck, len(angle_between_proteins), angle_between_proteins, angles[[index_pair[0] for index_pair in connected_pairs_tocheck]], angles[[index_pair[1] for index_pair in connected_pairs_tocheck]]

    # check whether the line that would run between connected proteins also runs through a 'sticky patch'...
    # ( the initial setup assumes that no proteins interact via sticky patches and this is recursively updated)
    angle1_in_sticky_patches = np.zeros(len(connected_pairs_tocheck_indices_1),dtype=bool)
    angle2_in_sticky_patches = np.zeros(len(connected_pairs_tocheck_indices_2),dtype=bool)
    for sticky_patch in sticky_patches_range:
        angle1_in_sticky_patch = np.logical_and(angle1 >= sticky_patch[0], angle1 <= sticky_patch[1])
        angle2_in_sticky_patch = np.logical_and(angle2 >= sticky_patch[0], angle2 <= sticky_patch[1])
        angle1_in_sticky_patches = np.logical_or(angle1_in_sticky_patches, angle1_in_sticky_patch)
        angle2_in_sticky_patches = np.logical_or(angle2_in_sticky_patches, angle2_in_sticky_patch)
    # ...on BOTH proteins...
    actually_connected_bool = np.logical_and(angle1_in_sticky_patches, angle2_in_sticky_patches)
    # ... and then marks as 'not connected' those protein pairs that do NOT touch via sticky patches
    actually_NOT_connected_1 = np.array(connected_pairs_tocheck_indices_1)[np.logical_not(actually_connected_bool)]
    actually_NOT_connected_2 = np.array(connected_pairs_tocheck_indices_2)[np.logical_not(actually_connected_bool)]
    connected[actually_NOT_connected_1, actually_NOT_connected_2] = 0
    connected[actually_NOT_connected_2, actually_NOT_connected_1] = 0

    if report_timings:
        currentTime5 = time.time()
#        print 'check angles in sticky patches took: {} seconds'.format(currentTime5 - currentTime4)

    # finally, remake the cluster list:
    network = nx.Graph(connected)
    if report_timings:
        currentTime6 = time.time()
#        print 'network -> graph took: {} seconds'.format(currentTime6 - currentTime5)
    cluster_list = np.zeros(n_particles,dtype=np.int64)
    for i,component in enumerate(list(nx.connected_components(network))):
        cluster_list[component] = i
#    cluster_list = sorted(nx.connected_components(network))
    if verbose:
        pass
#        print 'cluster_list: ', cluster_list
    if report_timings:
        currentTime7 = time.time()
#        print 'making modified cluster list took: {} seconds'.format(currentTime7 - currentTime6)

    return cluster_list

#################################################################
# function used to read in gro file from a previous frame - for restarting
#################################################################
def parse_gro_file(start_struct, cutoff, cluster_text, sticky_patches, sticky_patches_range, verbose, PBC):
    print 'Reading in gro file provided: '+start_struct
    f = open(start_struct)
    x_coords = []
    y_coords = []
    angles = []
    natoms = 2
    for i,line in enumerate(f):
        if i == 1:
            natoms = int(line.strip())
            print 'natoms = {}'.format(natoms)
        elif i == natoms + 2:
            m = re.match(r'\s*(\d+\.*\d*)\s+(\d+\.*\d*)\s+(\d+\.*\d*)',line)
            box_dimensions = np.array([float(m.group(1))*10, float(m.group(2))*10],dtype=np.float32)
            print 'box_dimensions = {}'.format(box_dimensions)
            print 'Finished reading file'
        elif i > 1:
            m = re.match(r'\s*(\d+\.\d+)\s+(\d+\.\d+)\s+\d+\.\d+\s+(\d+\.\d+)', line[20:])
            # should save gro with -ndec at least 8, so can't assume, eg. x-coords are always at posn [20:24] in the gro file
            # the re.match ensures that the coords are read in regardless of the number of decimal places present in the gro file
            x_coords.append(float(m.group(1))*10)
            y_coords.append(float(m.group(2))*10)
            angles.append(float(m.group(3))*10)
    f.close()
    x_coords = np.array(x_coords,dtype=np.float32)
    y_coords = np.array(y_coords,dtype=np.float32)
    angles = np.array(angles,dtype=np.float32)
    half_box = 0.5*box_dimensions
    print 'Now assessing clustering..'
    proteins_array = np.array(zip(x_coords, y_coords, np.zeros(natoms,dtype=np.float32)))
    if PBC:
        dist = MDAnalysis.analysis.distances.distance_array(proteins_array, proteins_array, np.array([box_dimensions[0], box_dimensions[1], 1.],dtype=np.float32))
    else:
        dist = MDAnalysis.analysis.distances.distance_array(proteins_array, proteins_array)
    cluster_list = np.arange(natoms)
#    cluster_list = np.array([[i] for i in range(natoms)])
    if cluster_text:
        g = open(cluster_text)
        last_line = g.readlines()[-1]
        cluster_list = np.array(ast.literal_eval(last_line))
        g.close()
    elif sticky_patches:
        cluster_list = check_for_clustering_stickypatches(dist, x_coords, y_coords, angles, box_dimensions, half_box, natoms, cutoff, sticky_patches_range, cluster_list, verbose, PBC)
        #print cluster_list
    else:
        cluster_list = check_for_clustering(dist, x_coords, y_coords, angles, natoms, cutoff)  ## PBC functionality not needed as 'dist' has already been calculated taking existence (or not) of PBCs into account
    return x_coords, y_coords, angles, cluster_list, box_dimensions

#################################################################
# this is the update step function
#################################################################
def update_algorithm(x_coords, y_coords, angles, cluster_list, box_dimensions, choose_velocities, choose_rotations, cutoff, sticky_patches_range, sticky_patches, verbose, very_verbose, png_out = False, report_timings=False, clustersize_sd_angle_array=None, clustersize_sd_array=None, cushion_ratio=0.8, PBC=True, max_box=0):
    if report_timings:
        startTime = time.time()
    # 0a. set up some variables
    n_clusters = np.amax(cluster_list) + 1
    n_particles = len(x_coords)
    half_box = 0.5*box_dimensions
    len_clusters = np.bincount(cluster_list)
    # 0b. set up containers for new coordinates
    new_x_coords = x_coords.copy()
    new_y_coords = y_coords.copy()
    new_angles = angles.copy()
    old_x_coords = x_coords.copy()
    old_y_coords = y_coords.copy()
    old_angles = angles.copy()

    # 1a. choose n_c rotations
    if choose_rotations == 'random':
        sigma_rotation = (5./180)*np.pi
        timestep = 1
        rotations = [np.random.normal(0,(sigma_rotation)*timestep)/len_clusters] # these values should be in radians
    elif choose_rotations == 'random_clustersize':
        timestep = 1
        rotations = np.random.normal(0,(clustersize_sd_angle_array[len_clusters-1])*timestep) # the minus 1 is because clustersize_sd_angle_array starts with a cluster size of 1, not 0.
        if n_clusters == 1:
            rotations = np.array([rotations])
    elif choose_rotations == 'random_clustersize_powerlaw':
        rotations = np.random.normal(0,clustersize_sd_angle_array[len_clusters-1]) # the minus 1 is because clustersize_sd_angle_array starts with a cluster size of 1, not 0.
        if n_clusters == 1:
            rotations = np.array([rotations])
        # we need rotations to be an array so that elements can be referred to by index when applying rotations
    # 1b. rotate particles around cluster CoM
    ## TO DO: generate a 'rotatable clusters' list
    ## clusters which form a loop that spans the whole plane cannot be rotated
    ## ie. with periodic boundary conditions the cluster has infinite size
    ## these clusters can be searched using networkX:
    ## first find loops in clusters, then look for edge length of loops
    ## if there are an odd number of long edges ( ie. edges > half_box ) then the loop covers the infinite plane
    if report_timings:
        currentTime1 = time.time()
#        print 'setup took: {} seconds'.format(currentTime1 - startTime)
    # 1b. rotate clusters
    # need tp rotate a cluster w.r.t. its centre of mass
    # this requires the particles in the cluster to be from the periodic image which is closest to the centre of mass
    #  rotating raw coordinates about the centre of mass will cause clusters split across the boundary to go totally wrong (with the exception of rotation by angles 0, 90,180,270)
    cluster_CoM = np.zeros((n_clusters,2))
    for i in range(n_clusters):
        if PBC:
            cluster_coords = np.array(zip(old_x_coords[cluster_list == i], old_y_coords[cluster_list == i]))  # NOTE: use the copies of the ORIGINAL coords            
            cluster_CoM[i] = np.array(calculate_cog_2D(cluster_coords, box_dimensions))
        else:
            cluster_CoM[i] = np.array([np.mean(old_x_coords[cluster_list == i]), np.mean(old_y_coords[cluster_list == i])])
        if very_verbose:
            pass
#        print 'cluster_CoM (i): ',cluster_CoM, i

    if report_timings:
        currentTime2 = time.time()
#        print 'finding cluster CoG took: {} seconds'.format(currentTime2 - currentTime1)

    if PBC:
        closest_coords = np.array(zip(old_x_coords, old_y_coords))
        dist_to_cluster_CoM = closest_coords - cluster_CoM[cluster_list]
        closest_coords[:,0][dist_to_cluster_CoM[:,0] > half_box[0]] -= box_dimensions[0]
        closest_coords[:,0][dist_to_cluster_CoM[:,0] < -half_box[0]] += box_dimensions[0]
        closest_coords[:,1][dist_to_cluster_CoM[:,1] > half_box[1]] -= box_dimensions[1]
        closest_coords[:,1][dist_to_cluster_CoM[:,1] < -half_box[1]] += box_dimensions[1]
    else:
        closest_coords = np.array(zip(old_x_coords, old_y_coords))
    if report_timings:
        currentTime3 = time.time()
#        print 'finding closest coords to cluster CoG took: {} seconds'.format(currentTime3 - currentTime2)

    cos_theta = np.cos(rotations)
    sin_theta = np.sin(rotations)
    for i in range(n_clusters):
        rotation_matrix = np.array([[cos_theta[i], -sin_theta[i]], [sin_theta[i], cos_theta[i]]])
        rotated_coords = np.dot(closest_coords[cluster_list == i] - cluster_CoM[i], rotation_matrix) + cluster_CoM[i]
#        rotated_coords = np.dot(closest_coords - cluster_CoM, rotation_matrix) + cluster_CoM
        new_x_coords[cluster_list == i], new_y_coords[cluster_list == i] = rotated_coords[:,0], rotated_coords[:,1]
    # update angle of each individual protein
    new_angles = (old_angles - rotations[cluster_list]) % (2*np.pi) # NOTE: subtract from copies of the ORIGINAL angles and calulate mod(2*pi)
    if report_timings:
        currentTime4 = time.time()
#        print 'rotating coords took: {} seconds'.format(currentTime4 - currentTime3)

    # 2a. choose n_c translations
    if choose_velocities == 'random':
        max_dist = 0.5
        velocities_x = [random.uniform(-1,1)*max_dist for i in range(n_clusters)]
        velocities_y = [random.uniform(-1,1)*max_dist for i in range(n_clusters)]
    elif choose_velocities == 'brownian':
        sigma = 1.
        timestep = 1
        velocities_x = np.random.normal(np.zeros(n_clusters),(sigma)*timestep)
        velocities_y = np.random.normal(np.zeros(n_clusters),(sigma)*timestep)
    elif choose_velocities == 'brownian_clustersize':
        timestep = 1
        # choose velocities from gaussian distributions with mean, 0, s.d. defined as in clustersize_sd_array
        velocities_x = np.random.normal(0,(clustersize_sd_array[len_clusters-1])*timestep)
        velocities_y = np.random.normal(0,(clustersize_sd_array[len_clusters-1])*timestep)
    elif choose_velocities == 'brownian_clustersize_powerlaw':
        # choose velocities from gaussian distributions with mean, 0; s.d. as defined by power law - 
        velocities_x = np.random.normal(0,clustersize_sd_array[len_clusters-1])
        velocities_y = np.random.normal(0,clustersize_sd_array[len_clusters-1])
    elif choose_velocities == 'brownian_clustersize_exp':
        # y = Aexp(-bx)
        # choose velocities from gaussian distributions with mean, 0; s.d. as defined by exponential decay law - 
        velocities_x = np.random.normal(0,clustersize_sd_array[len_clusters-1])
        velocities_y = np.random.normal(0,clustersize_sd_array[len_clusters-1])
    if n_clusters == 1:
        velocities_x = np.array([velocities_x])
        velocities_y = np.array([velocities_y])
    # 2b. translate particles
    new_x_coords += velocities_x[cluster_list]
    new_y_coords += velocities_y[cluster_list]
    
    #2c. If max_box defined, make sure all proteins are within the confines of the box
    if max_box:
        outside_max_box_x = np.logical_or(new_x_coords > max_box, new_x_coords < 0)
        outside_max_box_y = np.logical_or(new_y_coords > max_box, new_y_coords < 0)
        outside_max_box = np.logical_or(outside_max_box_x, outside_max_box_y)
        outside_box_clusters = np.unique( cluster_list[outside_max_box] )
        outside_box_list = np.in1d(cluster_list, outside_box_clusters)
        new_x_coords[outside_box_list] = old_x_coords[outside_box_list]
        new_y_coords[outside_box_list] = old_y_coords[outside_box_list]

    # 3. check if particles clash and find out which clusters are involved
    # if they do clash, move them back to original positions, and if this means that new clashes appear, then the while loop should ensure that these are eventually eliminated
    if report_timings:
        currentTime5 = time.time()
#        print 'translating particles took:  {} seconds'.format(currentTime5 - currentTime4)
    # set up initial cluster lists required
    n_clashing_clusters = n_clusters
    clashing_clusters = range(n_clusters)
    clashing_cluster_list = np.in1d(cluster_list, clashing_clusters)
    iteration = 0
    dist = np.zeros([n_particles, n_particles])
#    if report_timings:
#        print 'setting up clashing cluster took:  {} seconds'.format(currentTime5 - time.time())
    while n_clashing_clusters > 0:
        
#        if report_timings:
#            startlooptime = time.time()
#            print 'iteration:  {} '.format(iteration)
        proteins_array = np.array(zip(new_x_coords, new_y_coords, np.zeros(n_particles,dtype=np.float32)))
        if iteration > 0:
            clashing_proteins_array = np.array(zip(new_x_coords[clashing_cluster_list], new_y_coords[clashing_cluster_list], np.zeros(len(clashing_cluster_list),dtype=np.float32)))
            actual_indices_of_clashing = clashing_cluster_list.nonzero()[0]
#        if report_timings:
#            arraysetuptime = time.time()
#            print 'setting up clashing cluster arrays took:  {} seconds'.format(arraysetuptime - startlooptime)
        if PBC:
            if iteration == 0:
                dist = MDAnalysis.analysis.distances.distance_array(proteins_array, proteins_array, np.array([box_dimensions[0], box_dimensions[1], 1.],dtype=np.float32))
            else:
                clashing_dists = MDAnalysis.analysis.distances.distance_array(proteins_array, clashing_proteins_array, np.array([box_dimensions[0], box_dimensions[1], 1.],dtype=np.float32))
                dist[:,actual_indices_of_clashing] = clashing_dists
                dist[actual_indices_of_clashing] = clashing_dists.T
        else:
            if iteration == 0:
                dist = MDAnalysis.analysis.distances.distance_array(proteins_array, proteins_array)
            else:
                clashing_dists = MDAnalysis.analysis.distances.distance_array(proteins_array, clashing_proteins_array)
                dist[:,actual_indices_of_clashing] = clashing_dists
                dist[actual_indices_of_clashing] = clashing_dists.T
#        if report_timings:
#            diststime = time.time()
#            print 'measuring dists took:  {} seconds'.format(diststime - arraysetuptime)
        if iteration == 0:
            clashing = dist < float(cushion_ratio*cutoff)
            clashing_pair_indices = np.nonzero(clashing)
            clashing_pair_check_indices = clashing_pair_indices[0][clashing_pair_indices[0] != clashing_pair_indices[1]]
        else:
            clashing = clashing_dists < float(cushion_ratio*cutoff)
            clashing_pair_indices = np.nonzero(clashing)
            clashing_pair_check_indices = clashing_pair_indices[0][clashing_pair_indices[0] != actual_indices_of_clashing[clashing_pair_indices[1]]]
#        if report_timings:
#            checkdiststime = time.time()
#            print 'checking dists took:  {} seconds'.format(checkdiststime - diststime)
        clashing_clusters = np.unique( cluster_list[clashing_pair_check_indices] )
        n_clashing_clusters = len(clashing_clusters)
        clashing_cluster_list = np.in1d(cluster_list, clashing_clusters)
#        if report_timings:
#            indextime = time.time()
#            print 'finding correct indices took:  {} seconds'.format(indextime - checkdiststime)
        new_x_coords[clashing_cluster_list], new_y_coords[clashing_cluster_list] = old_x_coords[clashing_cluster_list], old_y_coords[clashing_cluster_list]
        new_angles[clashing_cluster_list] = old_angles[clashing_cluster_list]
        if very_verbose and iteration >2:
            print 'Clashing clusters after translation and rotation - more than two iterations: ',clashing_clusters,'\r',
        iteration += 1
        if iteration >10000:
            print 'Clashing clusters after translation and rotation - >1000 iterations: ',clashing_clusters,'\r',
            print (new_x_coords[clashing_cluster_list], new_y_coords[clashing_cluster_list])
#        if report_timings:
#            reassigntime = time.time()
#            print 'reassigning coords and angles took:  {} seconds'.format(reassigntime - indextime)


    # 4. commit new cluster positions
    x_coords, y_coords, angles = new_x_coords, new_y_coords, new_angles
    
    #### verbose mode is for debugging
    if very_verbose:
#        print 'x, y, angles after translation: '
        #print 'x_coords, y_coords: ', zip(x_coords, y_coords)
        #print 'angles: ', angles
        #print 'rotations, velocities_x, velocities_y: ', zip(rotations, velocities_x, velocities_y)
        pass
    if verbose or very_verbose:
        all_dists_btwn_prots = []
        all_sizes = []
        for i in range(n_clusters):
            dist_btwn_prots = half_box[0] + 1
            ic = 0
            while dist_btwn_prots > half_box[0] and ic < len_clusters[i]:
                dist_btwn_prots = np.sqrt((new_x_coords[cluster_list == ic] - old_x_coords[cluster_list == ic])**2 + (new_y_coords[cluster[ic]] - old_y_coords[cluster[ic]])**2)
                ic += 1
            if dist_btwn_prots < half_box[0]:
                all_dists_btwn_prots.append(dist_btwn_prots)
                all_sizes.append(len_clusters[i])
#        print zip(all_dists_btwn_prots, all_sizes)
        param = norm.fit(sorted(all_dists_btwn_prots))
        mean_norm = norm.mean(loc=param[0],scale=param[1]) 
        std_norm = norm.std(loc=param[0],scale=param[1])
        print 'mean and std dev of distance moved in 1 step: ', mean_norm, std_norm

    if report_timings:
        currentTime6 = time.time()
        print 'sorting out clashes ({} iterations)'.format(iteration)+' took:  {} seconds\r'.format(currentTime6 - currentTime5),

    # 5. check for clustering
    if sticky_patches:
        cluster_list = check_for_clustering_stickypatches(dist, x_coords, y_coords, angles, box_dimensions, half_box, n_particles, cutoff, sticky_patches_range, cluster_list, very_verbose, PBC, report_timings)
        #print cluster_list
    else:
        cluster_list = check_for_clustering(dist, x_coords, y_coords, angles, n_particles, cutoff)  ## PBC functionality not needed as 'dist' has already been calculated taking existence (or not) of PBCs into account
    
    if report_timings:
        currentTime7 = time.time()
#        print 'assessing clusters: {} seconds'.format(currentTime7 - currentTime6)

    # 6a. apply PBCs (if specified) - do this only now, since sticky patches clustering function fails for pairs split across boundaries
    if PBC:
        x_coords, y_coords = coords_in_box(x_coords, y_coords, box_dimensions)
    
    # 6b. If not using PBCs, redefine the box dimensions (this is only really necessary for the image creation)
    if not PBC:
        min_coord = min(min(x_coords),min(y_coords))
        max_coord = max(max(x_coords),max(y_coords))
        round_min = (np.fix(min_coord/100.) - 1)*100
        round_max = (np.fix(max_coord/100.) + 1)*100
        newdim = round_min - round_max
        if newdim < box_dimensions[0]:
            dim = box_dimensions[0]
        else:
            dim = newdim
        box_dimensions = np.array([dim,dim]) # box_dimesnion
        
    if report_timings:
        currentTime8 = time.time()
#        print 'applying PBCs: {} seconds'.format(currentTime8 - currentTime7)

    
    return x_coords, y_coords, angles, cluster_list, box_dimensions

#################################################################
# functions used to create output files
#################################################################

# create list of proteins with colours according to their cluster size
def cluster_colors(cluster_list):
    cluster_colors = np.zeros(len(cluster_list))
    len_clusters = np.bincount(cluster_list)
    n_clusters = np.amax(cluster_list) + 1 # need the +1 so that the max cluster size is also included in the range
    for i in range(n_clusters): 
        cluster_colors[cluster_list == i] = [len_clusters[i]]*len_clusters[i]
    return cluster_colors

# settings for output png file, for when output is required as an image (write='image')
def create_output_image(png_out, box_dimensions, colour, x_coords, y_coords, angles, cluster_list, cutoff, sticky_patches_range, PBC, max_box, max_colour=500):
            #sticky_patches_range, offset_angles=[], png_out=None, box_dimensions=[10,10],x_coords=[5], y_coords=[5], cluster_list=[1],  colour='green', cutoff=5):
    figure = plt.figure()
    ax = figure.add_subplot(1,1,1)
    ax.set_aspect('equal')
    if max_box:
        # putting 'if max_box' before PBC means that max_box overrides PBCs
        plt.xlim((0,max_box))
        plt.ylim((0,max_box))
    elif PBC:
        plt.xlim((0,box_dimensions[0]))
        plt.ylim((0,box_dimensions[1]))
    else:
        min_coord = min(min(x_coords),min(y_coords))
        max_coord = max(max(x_coords),max(y_coords))
        round_min = (np.fix(min_coord/100.) - 1)*100
        round_max = (np.fix(max_coord/100.) + 1)*100
        plt.xlim(round_min,round_max)
        plt.ylim(round_min,round_max)
#    if offset_angles == []:
#        offset_angles = [(sticky_patches_range_radians[0][1] - sticky_patches_range_radians[0][0])/2]
#        offset_angles_degrees = [(sticky_patches_range[1] - sticky_patches_range[0])/2]
#    else:
#        offset_angles_degrees = offset_angles*180./np.pi
    #point_area = np.pi*(cutoff/2)**2 / (box_dimensions[0]*box_dimensions[1])
    if colour == 'clustersize':
        pcolors = cluster_colors(cluster_list)
#        pcolors = cluster_colors(x_coords,cluster_list)
        cmjet = plt.get_cmap("jet")
        cnorm=colors.Normalize(vmin=1, vmax=max_colour)  ## added max_colour ALD, 31 Jan 2018
        circles = []
        for x,y in zip(x_coords, y_coords):
            circle = Circle((x, y), cutoff/2)
            circles.append(circle)
        c = PatchCollection(circles, cmap=cmjet, norm=cnorm, zorder=1, alpha=0.7, edgecolors='none')
        c.set_array(np.array(pcolors))
        ax.add_collection(c)
        #plt.scatter(x_coords, y_coords, s=point_area, c=pcolors,cmap=cmjet, norm=cnorm, edgecolors='none')
    else:
        for x, y in zip(x_coords, y_coords):
            c1 = plt.Circle((x,y), cutoff/2, color=colour, zorder=1, alpha=0.8)
            ax.add_artist(c1)
        #plt.scatter(x_coords, y_coords, s=point_area, c=colour, edgecolors='none')
    angles_degrees = angles*180./np.pi
    sticky_patches_range_degrees = np.array(sticky_patches_range)*180./np.pi
    patch_colours = ['blue','red','orange', 'yellow']
    patches = []
    #this took 2 HOURS
#    for i,theta in enumerate(offset_angles):
#        for j,patch in enumerate(sticky_patches_range_radians):
            #ax.annotate('',xytext=(x_coords[i], y_coords[i]), xy=(x_coords[i]+(cutoff/2)*np.cos(theta+sticky_patches_range_radians[j][0]), y_coords[i]+(cutoff/2)*np.sin(theta+sticky_patches_range_radians[j][0])), arrowprops=dict(fc=patch_colours[j]))
            #ax.annotate('',xytext=(x_coords[i], y_coords[i]), xy=(x_coords[i]+(cutoff/2)*np.cos(theta+sticky_patches_range_radians[j][1]), y_coords[i]+(cutoff/2)*np.sin(theta+sticky_patches_range_radians[j][1])), arrowprops=dict(fc=patch_colours[j]))
    for i,theta_degrees in enumerate(angles_degrees):
        for j in range(len(sticky_patches_range_degrees)):
            patches += [Wedge((x_coords[i], y_coords[i]), (cutoff/2), (theta_degrees+sticky_patches_range_degrees[j][0]),(theta_degrees+sticky_patches_range_degrees[j][1]),  facecolor=patch_colours[j], zorder=2, alpha=1., edgecolor= 'none')]
    for p in patches:
        ax.add_patch(p)

    if png_out:
        figure.savefig(png_out,dpi=300)
    plt.close()

# settings for output gro file, for when output is required as gro (write='trr')
def create_output_gro_file(gro_out, n_particles, x_coords, y_coords, box_dimensions):
    f = open(gro_out,'w')
    f.write('Meso model, one particle per protein, frame 0\n') # title on first line of gro file
    f.write('{}\n'.format(n_particles)) # n particles on 2nd line of gro file
    for i,coords in enumerate(zip(x_coords, y_coords)):
        f.write('{0:5d} PROT PROT{0:5d}{1:8.3f}{2:8.3f}   0.500\n'.format(i,coords[0]/10.,coords[1]/10.)) # gro file with the format '    1 PROT PROT    1  20.000  20.000 20.000' - coordinates are in nm - this ensures they match up to xtc file
    f.write(' {:10.3f} {:10.3f}      1.000\n'.format(box_dimensions[0]/10.,box_dimensions[1]/10.)) # box dimensions on last line, with z-dim set to 1 nm
    f.close()

#################################################################
# main function
#################################################################
def main(choose_velocities='brownian_clustersize_powerlaw', choose_rotations='random_clustersize_powerlaw', cutoff=5., cushion_ratio = 0.9, nsteps=50000000, write=['image', 'trr', 'cluster'], start_struct=None, start_time=0, start_clust=None, colour='clustersize',n_particles=144, init_dist=10.,  grid_name='square', initial_boxsize=0, sticky_patches_range=[(2*np.pi*0/360,2*np.pi*110/360),(2*np.pi*180/360,2*np.pi*240/360)], gro_out='brownian_frame0.gro', trr_out='brownian.trr', cluster_out='brownian_clusters.txt', output_frequency=10, sticky_patches=True, verbose=False, very_verbose=False, rot_png_out=False, parameter_dt='1ns', PBC=True, max_box=None):
    startTime = time.time()
    # initialise system
    if start_struct:
        x_coords, y_coords, angles, cluster_list, box_dimensions = parse_gro_file(start_struct, cutoff, start_clust, sticky_patches, sticky_patches_range, verbose, PBC)
    else:
        x_coords, y_coords, angles, cluster_list, box_dimensions = initiate_system(n_particles, init_dist, grid_name, initial_boxsize, max_box) #######
        initial_box_dimensions = box_dimensions
    if max_box:
#        # max_box sets the simulation to be in within a finite square so want to make sure that PBC is definitely not turned on
        PBC = False

    # write initial positions to preferred output file type - png and/or gro
    if 'image' in write:
        # if not PBC, box_dimnsions will change over the course of the sim - could modify this to start of with a system that is twice the len scale of intial - and expand as necessary
        # tried this initially but it didn't work that well - might be better to generate image of whole sys from final trr file and keep snapshots at system size
        png_out = 'brownian_{:08d}.png'.format(start_time)
        create_output_image(png_out, box_dimensions, colour, x_coords, y_coords, angles, cluster_list, cutoff, sticky_patches_range, PBC, max_box)
    if 'trr' in write :
        # TO DO: where to put the check for gro_out and xtc_file names?
        # write a gro file
        if start_time == 0:
            create_output_gro_file(gro_out, n_particles, x_coords, y_coords, box_dimensions)
        # create universe object and selection from the new gro file, and set up the filewriter for writing an xtc file
        u_temp = MDAnalysis.Universe(gro_out)
        AllProts = u_temp.selectAtoms("all")
        FileWriter = MDAnalysis.coordinates.core.writer(filename='temp_'+trr_out,numatoms=n_particles, multiframe=True, start=0, step=1)
        for t in range(nsteps/output_frequency):
            FileWriter.write(AllProts)
        FileWriter.close()
        u = MDAnalysis.Universe(gro_out, 'temp_'+trr_out)
        AllProts = u.selectAtoms("all")
        FileWriter = MDAnalysis.coordinates.core.writer(filename=trr_out,numatoms=n_particles, multiframe=True, start=0, step=1)
    if 'cluster' in write:
#        cluster_array = np.zeros((n_particles))
        f_cluster = open(cluster_out,'w')
    currentTime = time.time()
    print 'setup took: {} seconds'.format(currentTime-startTime)
    # initialise  rotation and translation s.d. dictionaries (where relevant)
    if choose_rotations == 'random_clustersize':
        # dummy data used in the early days
        clustersize_sd_angle_array = [np.pi*4./180,np.pi*3./180,np.pi*2.5/180,np.pi*2./180,np.pi*1.8/180]+[np.pi*1.6/180]*(n_particles-5)
#        clustersize_sd_angle_dict = dict(zip(range(1,n_particles+1), [np.pi*4./180,np.pi*3./180,np.pi*2.5/180,np.pi*2./180,np.pi*1.8/180]+[np.pi*1.6/180]*(n_particles-5)))
        if very_verbose:
            print 'clustersize_sd_angle_array: ',clustersize_sd_angle_array
#            print 'clustersize_sd_angle_dict: ',clustersize_sd_angle_dict
    elif choose_rotations == 'random_clustersize_powerlaw':
        if parameter_dt == '1ns':
            # using btub_313 parameters for dt  = 1 ns
            # file: /sansom/n22/bioc1280/uCG/params_btub313/results_dt1/transXandY_params_dt1.txt
            # b = 0.659709134934 A = 2.62069497775, y = Ax^-b
            A = 2.62069497775
            b = 0.659709134934
        elif parameter_dt == '10ns' or parameter_dt == '10ns_mixed' or parameter_dt == '10ns_mixed_slower':
            # /sansom/n22/bioc1280/uCG/params_btub313/results_dt10/powerlaw_fit_params_BtuB
            #A = 6.11377445251
            #b = 0.77320718339
            # new dt 10 ns sampling 1 ns
            #A = 6.10593412149
            #b = 0.766986886591
            # from /sansom/n08/chavent/Big_Omps_processed/rot_trans_vs_time/BtuB_313/results_dt1/results_dt10_sample-dt1/
            A = 6.10496233345
            b = 0.766939219951
        cluster_sizes = np.arange(1,n_particles+1)
        clustersize_sd_angle_array = (A*(cluster_sizes**(-b)))*np.pi/180.
#        clustersize_sd_angle_dict = dict(zip(cluster_sizes, (A*(cluster_sizes**(-b)))*np.pi/180.))
        if very_verbose:
            print 'clustersize_sd_angle_dict: ',clustersize_sd_angle_dict
    else:
        clustersize_sd_angle_array = None
#        clustersize_sd_angle_dict = None
    if choose_velocities == 'brownian_clustersize':
        # dummy data used in the early days
        clustersize_sd_array = [0.55,0.52,0.5,0.48,0.46,0.42,0.4,0.395,0.39,0.385,0.38,0.375,0.37,0.365,0.36,0.355,0.35,0.345,0.34,0.335, 0.33,0.325,0.32,0.315,0.31,0.305,0.3,0.295,0.29,0.285]+[0.28]*(n_particles-20)
#        clustersize_sd_dict = dict(zip(range(1,n_particles+1), [0.55,0.52,0.5,0.48,0.46,0.42,0.4,0.395,0.39,0.385,0.38,0.375,0.37,0.365,0.36,0.355,0.35,0.345,0.34,0.335, 0.33,0.325,0.32,0.315,0.31,0.305,0.3,0.295,0.29,0.285]+[0.28]*(n_particles-20)))
        # choose n velocities from gaussian distributions with mean, 0, s.d. defined as in clustersize_sd_dict
    elif choose_velocities == 'brownian_clustersize_powerlaw':
        # choose velocities from gaussian distributions with mean, 0; s.d. as defined by power law - 
        # s.d. = A(cluster_size)^(-b)
        if parameter_dt == '1ns':
            # parameters from file: /sansom/n22/bioc1280/uCG/params_btub313/results_dt1/transXandY_params_dt1.txt
            # b and A: 0.213981497243 0.166756024007 (nm)
            A = 0.166756024007 
            b = 0.213981497243
        elif parameter_dt == '10ns':
            # /sansom/n22/bioc1280/uCG/params_btub313/results_dt10/powerlaw_fit_params_BtuB - units converted from Angstroms (in file) to nm
            #A = 0.390962755661
            #b = 0.207262599967
            # new dt10ns sampling 1 ns
            #A = 0.386777052451
            #b = 0.195930222882
            # params from: /sansom/n08/chavent/Big_Omps_processed/rot_trans_vs_time/BtuB_313/results_dt1/results_dt10_sample-dt1/
            A = 0.386777043044
            b = 0.195930198766
        elif parameter_dt == '10ns_mixed':
            # mixed btub 313 alone
            A = 0.33
            b = 0.29
        elif parameter_dt == '10ns_mixed_slower':
            # mixed btub 313 alone
            A = 0.25
            b = 0.35
        cluster_sizes = np.arange(1,n_particles+1)
        clustersize_sd_array = A*cluster_sizes**(-b)
#        clustersize_sd_dict = dict(zip(cluster_sizes, A*cluster_sizes**(-b)))
        if very_verbose:
            print 'clustersize_sd_array: ',clustersize_sd_array
#            print 'clustersize_sd_dict: ',clustersize_sd_dict
    elif choose_velocities == 'brownian_clustersize_exp':
        # choose velocities from gaussian distributions with mean, 0; s.d. as defined by power law - 
        # s.d. = A*exp(-b*cluster_size)
        if parameter_dt == '1ns':
            # parameters from data in file: /sansom/n22/bioc1280/uCG/params_btub313/results_dt1/transXandY_params_dt1.txt
            # b and A: 0.0558551305612 0.162623021828 (nm)
            A = 0.162623021828 
            b = 0.0558551305612
        elif parameter_dt == '10ns':
            # /sansom/n22/bioc1280/uCG/params_btub313/results_dt10/exponential_fit_params_BtuB - units converted from Angstroms (in file) to nm
            #A = 0.381601764963
            #b = 0.0541161648434
            # new dt10ns sampling 1ns
            #A = 0.375938491676
            #b = 0.0500546420297
            # from /sansom/n08/chavent/Big_Omps_processed/rot_trans_vs_time/BtuB_313/results_dt1/results_dt10_sample-dt1/
            A = 0.375938478662
            b = 0.050054630549
        cluster_sizes = np.arange(1,n_particles+1)
        clustersize_sd_array = A*np.exp(cluster_sizes*(-b))
#        clustersize_sd_dict = dict(zip(cluster_sizes, A*np.exp(cluster_sizes*(-b))))
        if very_verbose:
            print 'clustersize_sd_array: ',clustersize_sd_array
#            print 'clustersize_sd_dict: ',clustersize_sd_dict
    else:
        clustersize_sd_array = None
#        clustersize_sd_dict = None
    # create trajectory
    for t in range(start_time,nsteps):
        # update step
        if verbose or very_verbose:
            print 'Step: ',t,
        if rot_png_out and t % output_frequency == 1:
            rot_png = 'brownian_{:06d}_afterrot.png'.format(t+1)
        else:
            rot_png = False
        if t % 10 == 0:
            report_timings = True
        else:
            report_timings = False
        x_coords, y_coords, angles, cluster_list, box_dimensions = update_algorithm(x_coords, y_coords, angles, cluster_list, box_dimensions, choose_velocities, choose_rotations, cutoff, sticky_patches_range, sticky_patches, verbose, very_verbose,rot_png, report_timings, clustersize_sd_angle_array, clustersize_sd_array, cushion_ratio, PBC, max_box)
        # write new positions with the specified output frequency and file format
        if t % output_frequency == 0:
            if 'image' in write:
                print 't = {}'.format(t),
                png_out = 'brownian_{:08d}.png'.format(t+1) # changed from 6 to 8 places - ALD 27 Feb 2017
                # if not PBC, box_dimnsions will change over the course of the sim - start of with a system that is twice the len scale of intial - and expand as necessary
                create_output_image(png_out, box_dimensions, colour, x_coords, y_coords, angles, cluster_list, cutoff, sticky_patches_range, PBC, max_box)
            if  'trr' in write:
                # write a trr file of the simulation
                timestep = t/output_frequency
                for ts in u.trajectory[timestep:timestep+1]:
                    AllProts.set_positions(zip(x_coords, y_coords, [0.5 for i in range(n_particles)]))
                    AllProts.set_velocities(zip(angles, [0.0 for i in range(n_particles)], [0.0 for i in range(n_particles)]))
                    FileWriter.write(AllProts) # AllProts is a selection of the universe object containing all proteins
            if 'cluster' in write:
                f_cluster.write('{}\n'.format(list(cluster_list)))
            newCurrentTime = time.time()
            print '{} steps took: {} seconds\r'.format(output_frequency, newCurrentTime-currentTime),
            currentTime = newCurrentTime
    if 'trr' in write:
        os.remove('temp_'+trr_out)
        FileWriter.close()
    if 'cluster' in write:
        f_cluster.close()
    print 'Finished'

#################################################################
# parser
#################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_directory", type=str, help='directory name for output files')
    parser.add_argument("-n", dest="n_particles",type=int, help='Number of particles in system', default=144)
    parser.add_argument("-a", dest="sticky_patches_range_degrees", type=float, nargs='+', help='space separated list of angle pairs, which define the "sticky patches" on the protein surface (angles given in degrees)', default=[15, 80, 180, 230]) # initial default = [60,120,240,300], default now based on manuscript
    parser.add_argument("-d", dest="init_dist",type=float, help='Initial distance between particles in system (nm)', default=10.)
    parser.add_argument("-c", dest="cutoff",type=float, help='Cutoff distance between particles (nm)', default=5.)
    parser.add_argument("-cc", dest="cushion_cutoff", type = float, help="number between 0 and 1 that gives the proportion of the interaction cutoff length which will give the 'hard core' of the protein", default=0.98) # default consistent w manuscript
    parser.add_argument("-T", dest="nsteps", type=int, help='Number of steps for simulation - given than one step is the length specified in --paramdt', default=2000)
    parser.add_argument("-v", dest="verbose", action='store_true', help='turns on the verbose mode')
    parser.add_argument("--vv", dest="very_verbose", action='store_true', help='turns on the very verbose mode')
    parser.add_argument("--paramdt", dest="parameter_dt", type=str, help='Timestep that parameters were collected at.  Options: 1ns, 10ns', default='10ns')
    parser.add_argument("--of", dest="output_frequency", type=int, help='frequency to with which to write output (in steps', default=100)
    parser.add_argument("--noPBC", dest="noPBC", action='store_true', help='turns off periodic boundary conditions (in nm)')
    parser.add_argument("--maxbox", dest="max_box", type=int, help='turns on use of a finite box to run simulation in - and if so selects size of box',default = 0)
    parser.add_argument("--trans_fit", dest="choose_velocities", choices=['brownian_clustersize_powerlaw','brownian_clustersize_exp'], help='specify the type of fitting of CG parameters in the graph of clustersize vs. S.D. 1-D translation - to determine how translations are extrapolated for larger clustersizes', default='brownian_clustersize_powerlaw')
    parser.add_argument("--rot_fit", dest="choose_rotations", choices=['random_clustersize_powerlaw'], help='specify the type of fitting of CG parameters in the graph of clustersize vs. S.D. rotation - to determine how rotations are extrapolated for larger clustersizes', default='random_clustersize_powerlaw')
    parser.add_argument("--start", dest="start_struct", default=None)
    parser.add_argument("--start_time", dest="start_time", type=int, help="start_time (in same units as steps)", default=0)
    parser.add_argument("--start_clust", dest="start_clust", help="cluster.txt file with clustering info for use when restarting", default=None)
    parser.add_argument("--grid", dest="grid_name", type=str, help="if no starting structure provided, this species the type of grid structure to start with", default="square", choices=["square","square_in_space"])
    parser.add_argument("--initial_boxsize", dest="initial_boxsize", type=float, help="if -grid square_in_space specified, this is the starting box size to be used (in nm)", default=0.)
    parser.add_argument("--noStickyPatch", dest="sticky_patches", action='store_false', help="turns off sticky patches")
    parser.add_argument("--write", dest="write", nargs='+', type=str, help='Choose which kind of files to write', default='trr image cluster')
    parser.add_argument("--trr", dest="trr_out", type=str, help='Name of output trr file', default='brownian.trr')
    parser.add_argument("--clustxt", dest="cluster_out", type=str, help='Name of output txt file containing cluster info', default='brownian_clusters.txt')
    parser.add_argument("--gro", dest="gro_out", type=str, help='Name of output gro - intial frame', default='brownian_frame0.gro')
    parser.add_argument("--colour", dest="colour", type=str, help="Specify colour of proteins in output image files.  Can choose a matplotlib colour name, or specify 'clustersize' for proteins to be coloured according to the size of clusters that they are part of.", default='clustersize')
    options = parser.parse_args()
    # set up directoris and make log file (of inputs used)
    if options.start_struct:
        # ie. if this is a restart:
        # assume that we are IN the simulation directory
        restart_recurrences = [0]
        for file in os.listdir('.'):
            m = re.match(r'settings_restart(\d+).log', file)
            if m:
                restart_recurrences.append(int(m.group(1)))
        logfile = 'settings_restart{:03d}.log'.format(max(restart_recurrences)+1)
        options.cluster_out = options.cluster_out[:-4]+'_restart{:03d}.txt'.format(max(restart_recurrences)+1)
        options.trr_out = options.trr_out[:-4]+'_restart{:03d}.trr'.format(max(restart_recurrences)+1)
    else:
        os.mkdir(options.output_directory)
        os.chdir(options.output_directory)
        logfile = 'settings.log'
    f = open(logfile,'w')
    f.write('Command used:\npython '+(' ').join(sys.argv)+'\n\n'+'All input values, including defaults: \n output_directory={}\n n_particles={}\n sticky_patches_range_degrees={}\n init_dist={}\n cutoff={}\n cushion_cutoff={}\n nsteps={}\n verbose={}\n very_verbose={}\n parameter_dt={}\n output_frequency={}\n noPBC={}\n maxbox={}\n choose_velocities={}\n choose_rotations={}\n start_struct={}\n start_time={}\n start_clust={}\n grid_name={}\n initial_boxsize={}\n sticky_patches={}\n write={}\n trr_out={}\n cluster_out={}\n gro_out={}\n color={}\n'.format(options.output_directory, options.n_particles, str(options.sticky_patches_range_degrees), options.init_dist, options.cutoff, options.cushion_cutoff, options.nsteps, options.verbose, options.very_verbose, options.parameter_dt, options.output_frequency, options.noPBC, options.max_box, options.choose_velocities, options.choose_rotations, options.start_struct, options.start_time, options.start_clust, options.grid_name, options.initial_boxsize, options.sticky_patches, str(options.write), options.trr_out, options.cluster_out, options.gro_out, options.colour))
    f.close()
    print 'log file written: '+logfile
    if not options.start_struct:
        # if this is not a restart, write txt file that includes all commands necessary to restart the sim
        f = open('restart_commands.txt', 'w')
        f.write('#!/bin/bash\n')
        f.write('# commands to restart a simulation\n# should be run within the current directory\n')
        f.write('# if there are multiple restarts, then the trr and txt file names should be changed:\n')
        f.write('# from {} to brownian_restart00n.trr and {} to brownian_clusters_restart00n.txt\n'.format(options.trr_out,options.cluster_out))
        f.write('# AND need to add number of frames ALREADY simulated in other trr files to $FRAME in the trjconv and python line\n')
        f.write("rm temp_brownian*.trr\n")
        f.write("XTCFR=$(gmxcheck -f {} 2>&1 >/dev/null | grep 'Step' | sed 's/Step *\([0-9]*\) *1/\\1/')\n".format(options.trr_out))
        f.write("XTCFR=$(($XTCFR - 1))\n")
        f.write("CLUSTFR=$(cat {} | wc -l )\n".format(options.cluster_out))
        f.write("MIN=$(( $XTCFR < $CLUSTFR ? $XTCFR : $CLUSTFR ))\n")
        f.write("FRAME=$MIN\n")
        f.write("echo 0 | trjconv -f {} -s {} -o brownian_frame$FRAME.gro -ndec 9 -b $FRAME -e $FRAME\n".format(options.trr_out, options.gro_out))
        f.write('sed "${FRAME}q;d" '+'{} > brownian_clusters_frame$FRAME.txt\n'.format(options.cluster_out))
        f.write("python {} {} -T {} -n {} --start brownian_frame$FRAME.gro --start_time $FRAME'00' --start_clust brownian_clusters_frame$FRAME.txt\n".format(sys.argv[0], options.output_directory, options.nsteps, options.n_particles))
        f.close()
        print 'restart commands written: restart_commands.txt'
    #print options.sticky_patches_range_degrees
    startTime = time.time()
    if options.sticky_patches_range_degrees:
        n_sticky_patch_angles = len(options.sticky_patches_range_degrees)
        if n_sticky_patch_angles % 2 == 1:
            print 'There should be an even number of sticky_patch angles given after the flag -a'
            sys.exit
    sticky_patches_range_radians = []
    for i in range(len(options.sticky_patches_range_degrees)/2):
        sticky_patches_range_radians.append((options.sticky_patches_range_degrees[2*i]*2*np.pi/360, options.sticky_patches_range_degrees[2*i+1]*2*np.pi/360))
    main(n_particles=options.n_particles, sticky_patches_range=sticky_patches_range_radians, init_dist=options.init_dist, cutoff=options.cutoff, cushion_ratio=options.cushion_cutoff, choose_velocities=options.choose_velocities, nsteps=options.nsteps, verbose=options.verbose, very_verbose=options.very_verbose, parameter_dt=options.parameter_dt, output_frequency=options.output_frequency, PBC=not options.noPBC, max_box=options.max_box, start_struct=options.start_struct, start_time=options.start_time, start_clust=options.start_clust, grid_name=options.grid_name, initial_boxsize=options.initial_boxsize, sticky_patches=options.sticky_patches, write=options.write, trr_out=options.trr_out, cluster_out=options.cluster_out, gro_out=options.gro_out, colour=options.colour)
    endTime = time.time()
    print 'Program took {:7.1f} seconds to run'.format(endTime-startTime)


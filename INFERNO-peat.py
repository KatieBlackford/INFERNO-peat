# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:59:18 2023

@author: katie
"""

## INFERNO-peat 

## Required packages

import jules
import iris
import math
import numpy as np
import matplotlib.pyplot as plt
import iris.plot as iplt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

## jules analysis code file is required for running this code 

####

## Model peat fires burnt area and depth:
    
# base = base cube for regridding
# year = year of analysis in str format
# plot = true/false whether you want to produce a map of burnt area
# peat_data = either 'npd' or 'peat_ml' - npd (Northern peatland dataset recommended)
# ig_method = either 1, 2 or 3
#             1 = Fixed natural and human ignitions
#             2 = LIS-OTD lightning climatology (https://doi.org/10.1029/2002JD002347) 
#                 and HYDE population density driven ignitions
#             3 = WGLC lightning timeseries (https://doi.org/10.5281/zenodo.4882792) 
#                 and HYDE population density driven ignitions
# bd_method = 1 or 2
#             1 = Fixed burn depth at 12cm, but modified by height of water table 
#             2 = Modelled burn depth through calculating critical temperature
# max_burndepth = 40cm max burn depth is used - please set this variable as 4 
#(refers to the 4th soil layer, which is 40cm deep)
# avg_ba = average peat fire burned area. Please use 381.7 - other values have 
#          not been fully tested and evaluated
# save = true/false whether or not to save output files
# output_folder = file path to where you want files to be saved. 

def model_peat_fires(base, year, plot, peat_data, ig_method, bd_method, max_burndepth, avg_ba, save, output_folder):
    # 1. Calculate peat fire ignitions
    # Load initial required files (JULES PFT flammability and fractions)
    var_constraint1 = iris.Constraint(cube_func=lambda x: x.var_name == 'flammability')
    flam = jules.load_cube('path/to/file/filename'+year+'.nc', var_constraint1)

    var_constraint2 = iris.Constraint(cube_func=lambda x: x.var_name == 'frac')
    frac = jules.load_cube('path/to/file/filename'+year+'.nc', var_constraint2)

    # Base cube and apply landmask to accurately calculate areas
    # Landmask = CRU-NCEPv7.landfrac.nc
    var_constraint1 = iris.Constraint(cube_func=lambda x: x.var_name == 'burnt_area_gb')
    cube = jules.load_cube('/path/to/base_cube.nc', var_constraint1, missingdata=np.ma.masked)

    var_constraintx = iris.Constraint(cube_func=lambda x: x.var_name == 'field36')
    landmask = iris.load_cube('/path/to/landmask/CRU-NCEPv7.landfrac.nc', var_constraintx)
    landmask = landmask.regrid(cube, iris.analysis.Linear()) 
    cs_new = iris.coord_systems.GeogCS(6371229.)
    landmask.coord('latitude').coord_system = cs_new
    landmask.coord('longitude').coord_system = cs_new

    # Calculate gridcell areas
    area1 = iris.analysis.cartography.area_weights(base, normalize=False)/1e+6

    lat_coord = base.coord('latitude')
    lon_coord = base.coord('longitude')
    time_coord = base.coord('time')

    area_cube2 = iris.cube.Cube(area1,
                                attributes=None,
                                dim_coords_and_dims=[(time_coord, 0),
                                                     (lat_coord, 1),
                                                     (lon_coord, 2)])
    area_cube2.data = area_cube2.data * landmask.data

    ## load population density data
    popden_cube = iris.load_cube('/path/to/HYDE/PD_2000_n96.nc')   
    popden_cube = popden_cube.regrid(cube, iris.analysis.Linear())
    
    ## create ignitions cubes
    array = np.zeros((12,112,192))
    ignitions_cube2 = iris.cube.Cube(array,
                                     attributes=None,
                                     dim_coords_and_dims=[(time_coord, 0),
                                                          (lat_coord, 1),
                                                          (lon_coord, 2)])
    array2 = np.zeros((12,112,192))
    ignitions_cube3 = iris.cube.Cube(array2,
                                     attributes=None,
                                     dim_coords_and_dims=[(time_coord, 0),
                                                          (lat_coord, 1),
                                                          (lon_coord, 2)])

    array3 = np.zeros((12,112,192))
    sup_cube = iris.cube.Cube(array3,
                                     attributes=None,
                                     dim_coords_and_dims=[(time_coord, 0),
                                                          (lat_coord, 1),
                                                          (lon_coord, 2)])
    array4 = np.zeros((12,112,192))
    total_ignitions = iris.cube.Cube(array4,
                                     attributes=None,
                                     dim_coords_and_dims=[(time_coord, 0),
                                                          (lat_coord, 1),
                                                          (lon_coord, 2)])
    array5 = np.zeros((12,112,192))
    natural_ignitions = iris.cube.Cube(array5,
                                       attributes=None,
                                       dim_coords_and_dims=[(time_coord, 0),
                                                            (lat_coord, 1),
                                                            (lon_coord, 2)])
    
    # Calculate ignitions using 1 of 3 methods 
    # 1 = Fixed natural and human ignitions
    # 2 = LIS-OTD lightning climatology (https://doi.org/10.1029/2002JD002347) 
    #     and HYDE population density driven ignitions
    # 3 = WGLC lightning timeseries (https://doi.org/10.5281/zenodo.4882792) 
    #     and HYDE population density driven ignitions
    # See Mangeon et al 2016 (https://doi.org/10.5194/gmd-9-2685-2016) for 
    # details on how ignitions are calculated
    if ig_method == 1: #fixed ignitions
        # natural ignitions are 2.7/km2/yr
        nat_ign = 2.7/12
        # human ignitions 1.5km2/month
        hum_ign = 1.5
        # ignitions per km2
        ignitions = nat_ign + hum_ign
        
        array = np.zeros((12,112,192))
        total_ignitions = iris.cube.Cube(array,
                                         attributes=None,
                                         dim_coords_and_dims=[(time_coord, 0),
                                                              (lat_coord, 1),
                                                              (lon_coord, 2)])

        for month in range(len(total_ignitions.coord('time').points)):
            for lat in range(len(total_ignitions.coord('latitude').points)):
                for lon in range(len(total_ignitions.coord('longitude').points)):
                    total_ignitions.data[month][lat][lon] = area_cube2.data[month][lat][lon] * ignitions

    elif ig_method == 2: #lightning climatology and population density
        lightning = iris.load_cube('/path/to/lightning/Cloud_to_ground_2013_n96.nc')
        lightning.data = lightning.data*30
        lightning_cube = lightning.regrid(cube, iris.analysis.Linear())

        for month in range(len(natural_ignitions.coord('time').points)):
            for lat in range(len(natural_ignitions.coord('latitude').points)):
                for lon in range(len(natural_ignitions.coord('longitude').points)):
                    natural_ignitions.data[month][lat][lon] = area_cube2.data[month][lat][lon] * lightning_cube.data[month][lat][lon]

        for month in range(len(ignitions_cube2.coord('time').points)):
            for lat in range(len(ignitions_cube2.coord('latitude').points)):
                for lon in range(len(ignitions_cube2.coord('longitude').points)):
                    ignitions_cube2.data[month][lat][lon] = area_cube2.data[month][lat][lon] * popden_cube.data[lat][lon]
    
        for month in range(len(ignitions_cube2.coord('time').points)):
            for lat in range(len(ignitions_cube2.coord('latitude').points)):
                for lon in range(len(ignitions_cube2.coord('longitude').points)):
                    ignitions_cube3.data[month][lat][lon] = (6.8 * (ignitions_cube2.data[month][lat][lon]**-0.6)) * ignitions_cube2.data[month][lat][lon] * 0.03
                          
        for month in range(len(sup_cube.coord('time').points)):
            for lat in range(len(sup_cube.coord('latitude').points)):
                for lon in range(len(sup_cube.coord('longitude').points)):
                    ig_data = ignitions_cube3.data[month][lat][lon]
                    sup_cube.data[month][lat][lon] = 7.7 * (0.05 + 0.9 * math.exp(-0.05 * ig_data))

        for month in range(len(ignitions_cube2.coord('time').points)):
            for lat in range(len(ignitions_cube2.coord('latitude').points)):
                for lon in range(len(ignitions_cube2.coord('longitude').points)):
                    total_ignitions.data[month][lat][lon] = (ignitions_cube3.data[month][lat][lon] + natural_ignitions.data[month][lat][lon])*sup_cube.data[month][lat][lon]

    elif ig_method == 3: # lightning timeseries and population density
        lightning = iris.load_cube('/path/to/wglc_lightning/yearly/wglc_30m_'+year+'.nc')
        lightning.data = lightning.data*30
        lightning.coord('latitude').coord_system = cs_new
        lightning.coord('longitude').coord_system = cs_new
        lightning_cube = lightning.regrid(base, iris.analysis.Linear())
        
        for month in range(len(natural_ignitions.coord('time').points)):
            for lat in range(len(natural_ignitions.coord('latitude').points)):
                for lon in range(len(natural_ignitions.coord('longitude').points)):
                    natural_ignitions.data[month][lat][lon] = area_cube2.data[month][lat][lon] * lightning_cube.data[month][lat][lon]

        for month in range(len(ignitions_cube2.coord('time').points)):
            for lat in range(len(ignitions_cube2.coord('latitude').points)):
                for lon in range(len(ignitions_cube2.coord('longitude').points)):
                    ignitions_cube2.data[month][lat][lon] = area_cube2.data[month][lat][lon] * popden_cube.data[lat][lon]
    
        for month in range(len(ignitions_cube2.coord('time').points)):
            for lat in range(len(ignitions_cube2.coord('latitude').points)):
                for lon in range(len(ignitions_cube2.coord('longitude').points)):
                    ignitions_cube3.data[month][lat][lon] = (6.8 * (ignitions_cube2.data[month][lat][lon]**-0.6)) * ignitions_cube2.data[month][lat][lon] * 0.03
                          
        for month in range(len(sup_cube.coord('time').points)):
            for lat in range(len(sup_cube.coord('latitude').points)):
                for lon in range(len(sup_cube.coord('longitude').points)):
                    ig_data = ignitions_cube3.data[month][lat][lon]
                    sup_cube.data[month][lat][lon] = 7.7 * (0.05 + 0.9 * math.exp(-0.05 * ig_data))

        for month in range(len(ignitions_cube2.coord('time').points)):
            for lat in range(len(ignitions_cube2.coord('latitude').points)):
                for lon in range(len(ignitions_cube2.coord('longitude').points)):
                    total_ignitions.data[month][lat][lon] = (ignitions_cube3.data[month][lat][lon] + natural_ignitions.data[month][lat][lon])*sup_cube.data[month][lat][lon]

    ## Calculate pft number of fires cube
    # Calculated by multiplying the total number of ignitions by the 
    # flammability of each pft (i.e. how many of those ignitions turn into 
    # fires), and the fraction of that pft in the gridbox to estimate number 
    # of fires per pft.
    array = np.zeros((12,13,112,192))
    lat_coord = flam.coord('latitude')
    lon_coord = flam.coord('longitude')
    time_coord = flam.coord('time')
    pft_coord = flam.coord('pft')
    pft_ignitions_cube = iris.cube.Cube(array,
                                        attributes=None,
                                        dim_coords_and_dims=[(time_coord, 0),
                                                             (pft_coord, 1),
                                                             (lat_coord, 2),
                                                             (lon_coord, 3)])

    for time in range(len(pft_ignitions_cube.coord('time').points)):
        for pft in range(len(pft_ignitions_cube.coord('pft').points)):
            for lat in range(len(pft_ignitions_cube.coord('latitude').points)):
                for lon in range(len(pft_ignitions_cube.coord('longitude').points)):
                    pft_ignitions_cube.data[time][pft][lat][lon] = total_ignitions.data[time][lat][lon] * flam.data[time][pft][lat][lon] * frac.data[time][pft][lat][lon]

    sum_pft = pft_ignitions_cube.collapsed('pft', iris.analysis.SUM)

    ## 2. Calculate peat combustibility
    
    # Load peat fraction
    # Northern peatland dataset (https://doi.org/10.17043/hugelius-2020-peatland-2)
    # recommended
    # ! peat-ml currently experimental (https://doi.org/10.5281/zenodo.5794336) !
    if peat_data == "npd":
        peat = iris.load_cube('/path/to/npd/filename.nc')/100
        scheme = iris.analysis.AreaWeighted()
        peat_cube = peat.regrid(base, scheme)
    elif peat_data == "peat_ml":
        peat = iris.load_cube('/path/to/Peat-ML/filename.nc')/100
        scheme = iris.analysis.AreaWeighted()
        peat_cube = peat.regrid(base, scheme)

    array3 = np.zeros((12,112,192))
    lat_coord = base.coord('latitude')
    lon_coord = base.coord('longitude')
    time_coord = base.coord('time')
    peat_flam = iris.cube.Cube(array3,
                               attributes=None,
                               dim_coords_and_dims=[(time_coord, 0),
                                                    (lat_coord, 1),
                                                    (lon_coord, 2)])                

    ## Load Soil moisture ancillaries and calculate mc % 
    # !!! Ensure use of peat specific soil moisture !!!
    var_constraint4 = iris.Constraint(cube_func=lambda x: x.var_name == 'smcl', soil=0)
    sm_cube = jules.load_cube('/path/to/file/filename'+year+'.nc', var_constraint4)

    cs_new = iris.coord_systems.GeogCS(6371229.)
    sm_cube.coord('latitude').coord_system = cs_new
    sm_cube.coord('longitude').coord_system = cs_new
    sm_cube.coord('longitude').guess_bounds()
    sm_cube.coord('latitude').guess_bounds()

    sm_cube = sm_cube.regrid(base, iris.analysis.AreaWeighted())
    
    var_constraint = iris.Constraint(cube_func=lambda x: x.var_name == 'BD')
    bd_cube = iris.load_cube('/path/to/qrparm.soil.latlon_fixed.nc', var_constraint)
    cs_new = iris.coord_systems.GeogCS(6371229.)
    coord_lat = bd_cube.dim_coords[0]
    coord_lon = bd_cube.dim_coords[1]
    coord_lat.rename('latitude')
    coord_lon.rename('longitude')
    bd_cube.coord('latitude').coord_system = cs_new
    bd_cube.coord('longitude').coord_system = cs_new
    bd_cube.coord('longitude').guess_bounds()
    bd_cube.coord('latitude').guess_bounds()

    bd_cube = bd_cube.regrid(base, iris.analysis.AreaWeighted())
    
    # Convert soil moisture to % 
    # 0.05 = depth of 1st layer
    for time in range(len(sm_cube.coord('time').points)):
        for lat in range(len(sm_cube.coord('latitude').points)):
            for lon in range(len(sm_cube.coord('longitude').points)):
                sm_cube.data[time][lat][lon] = (sm_cube.data[time][lat][lon]/(bd_cube.data[lat][lon]*0.05))*100

        
    ## calculate combustibility
    # NOTE: fixed values currently being used for inorganic content and 
    #       bulk density
    for time in range(len(peat_flam.coord('time').points)):
        for lat in range(len(peat_flam.coord('latitude').points)):
            for lon in range(len(peat_flam.coord('longitude').points)):
                ioc = 9.4
                bd = 222
                if peat_cube.data[lat][lon] > 0.1:
                    MC = sm_cube.data[time][lat][lon]
                    peat_flam.data[time][lat][lon] = 1/(1 + math.exp(-(-19.8198 + -0.1169 * MC + 1.0414 * ioc + 0.0782 * bd)))

    ## 3. Calculate peat burnt area 
    
    array3 = np.zeros((12,112,192))
    lat_coord = base.coord('latitude')
    lon_coord = base.coord('longitude')
    time_coord = base.coord('time')
    burnt_area = iris.cube.Cube(array3,
                                attributes=None,
                                dim_coords_and_dims=[(time_coord, 0),
                                                     (lat_coord, 1),
                                                     (lon_coord, 2)])     

    # Burnt area calculated by multiplying the total number of ignitions 
    # (sum of pft ignitions i.e. a peat fire can only start from a pre existing 
    # flaming vegetation fire), by the peat combustibility, peat fraction and
    # a average burnt area value
    
    # !!! PLease use avg_ba = 381.7 !!!
    # Other values have not been tested and may lead to different results.
    
    #avg_ba = 381.7
    for time in range(len(peat_flam.coord('time').points)):
        for lat in range(len(peat_flam.coord('latitude').points)):
            for lon in range(len(peat_flam.coord('longitude').points)):
                if peat_cube.data[lat][lon] > 0.05:
                    burnt_area.data[time][lat][lon] = sum_pft.data[time][lat][lon] * peat_flam.data[time][lat][lon] * peat_cube.data[lat][lon] * avg_ba

    ## burnt area fraction 
    lat_coord = base.coord('latitude')
    lon_coord = base.coord('longitude')
    time_coord = base.coord('time')
    array = np.zeros((12,112,192))
    ba_frac_cube = iris.cube.Cube(array,
                                  attributes=None,
                                  dim_coords_and_dims=[(time_coord, 0),
                                                       (lat_coord, 1),
                                                       (lon_coord, 2)])

    for month in range(len(burnt_area.coord('time').points)):
        for lat in range(len(burnt_area.coord('latitude').points)):
            for lon in range(len(burnt_area.coord('longitude').points)):
                if burnt_area.data[month][lat][lon] > 0:
                    ba_frac_cube.data[month][lat][lon] = burnt_area.data[month][lat][lon]/area_cube2.data[month][lat][lon]
    
    ## 4. Option to plot a map of average burnt area for the year
    if plot == "true":
        nhls = iris.Constraint(latitude = lambda cell: 35.0 < cell < 85.0)
        ba_frac_cube_nhls = ba_frac_cube.extract(nhls)
        avg = ba_frac_cube_nhls.collapsed('time', iris.analysis.SUM)
        fig = plt.figure(figsize=[10,4.5])
        cmap = plt.get_cmap('gist_heat_r')
        levels = MaxNLocator(nbins=20).tick_values(0.0, 0.32)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        mesh = iplt.pcolormesh(avg, cmap=cmap, norm=norm)
        plt.gca().coastlines()
        colorbar = plt.colorbar(mesh, orientation='horizontal', pad=0.02, aspect=50)
        colorbar.set_label('Gridbox burnt area fraction'+year, fontsize=11)   
    
    ## 5. Calculate burn depth 
    
    lat_coord = base.coord('latitude')
    lon_coord = base.coord('longitude')
    time_coord = base.coord('time')
    array = np.empty((12,112,192))
    array[:] = np.nan
    burndepth_cube = iris.cube.Cube(array,
                                    attributes=None,
                                    dim_coords_and_dims=[(time_coord, 0),
                                                         (lat_coord, 1),
                                                         (lon_coord, 2)])
    
    # !!! Ensure all soil ancillaries are specifically for peat/organic soils !!!
    # Load water table depth
    var_constraint6 = iris.Constraint(cube_func=lambda x: x.var_name == 'zw')
    water_table_cube = jules.load_cube('/path/to/filename'+year+'.nc', var_constraint6)

    cs_new = iris.coord_systems.GeogCS(6371229.)
    water_table_cube.coord('latitude').coord_system = cs_new
    water_table_cube.coord('longitude').coord_system = cs_new
    water_table_cube.coord('longitude').guess_bounds()
    water_table_cube.coord('latitude').guess_bounds()

    water_table_cube = water_table_cube.regrid(base, iris.analysis.AreaWeighted())
    
    # Load soil temperature
    var_constraint8 = iris.Constraint(cube_func=lambda x: x.var_name == 't_soil')
    soil_temp_cube = jules.load_cube('/path/to/filename'+year+'.nc', var_constraint8)
    soil_temp_cube.data = soil_temp_cube.data - 273.15
    
    soil_temp_cube.coord('latitude').coord_system = cs_new
    soil_temp_cube.coord('longitude').coord_system = cs_new
    soil_temp_cube.coord('longitude').guess_bounds()
    soil_temp_cube.coord('latitude').guess_bounds()

    soil_temp_cube = soil_temp_cube.regrid(base, iris.analysis.AreaWeighted())
    
    array_tcrt = np.zeros((12,20,112,192))
    
    # Load soil moisture and convert to fraction 
    var_constraint4 = iris.Constraint(cube_func=lambda x: x.var_name == 'smcl', soil=0)
    sm_cube = jules.load_cube('/path/to/filename'+year+'.nc', var_constraint4)

    cs_new = iris.coord_systems.GeogCS(6371229.)
    sm_cube.coord('latitude').coord_system = cs_new
    sm_cube.coord('longitude').coord_system = cs_new
    sm_cube.coord('longitude').guess_bounds()
    sm_cube.coord('latitude').guess_bounds()

    sm_cube = sm_cube.regrid(base, iris.analysis.AreaWeighted())
    
    var_constraint = iris.Constraint(cube_func=lambda x: x.var_name == 'BD')
    bd_cube = iris.load_cube('/path/to/qrparm.soil.latlon_fixed.nc', var_constraint)
    cs_new = iris.coord_systems.GeogCS(6371229.)
    coord_lat = bd_cube.dim_coords[0]
    coord_lon = bd_cube.dim_coords[1]
    coord_lat.rename('latitude')
    coord_lon.rename('longitude')
    bd_cube.coord('latitude').coord_system = cs_new
    bd_cube.coord('longitude').coord_system = cs_new
    bd_cube.coord('longitude').guess_bounds()
    bd_cube.coord('latitude').guess_bounds()

    bd_cube = bd_cube.regrid(base, iris.analysis.AreaWeighted())
    
    for time in range(len(sm_cube.coord('time').points)):
        for lat in range(len(sm_cube.coord('latitude').points)):
            for lon in range(len(sm_cube.coord('longitude').points)):
                sm_cube.data[time][lat][lon] = (sm_cube.data[time][lat][lon]/(bd_cube.data[lat][lon]*0.05))

    
    # Create critical temperature cube
    lat_coord = flam.coord('latitude')
    lon_coord = flam.coord('longitude')
    time_coord = flam.coord('time')
    soil_coord = soil_temp_cube.coord('soil')
    tcrt_cube = iris.cube.Cube(array_tcrt,
                                    attributes=None,
                                    dim_coords_and_dims=[(time_coord, 0),
                                                         (soil_coord, 1),
                                                         (lat_coord, 2),
                                                         (lon_coord, 3)])
    
    # Calculate depth of burn using either:
    # 1: Fixed burn depth at 12cm, but modified by height of water table 
    #   (not recommended)
    # 2: Modelled burn depth through calculating critical temperature
    if bd_method == 1: # fixed
        for month in range(len(ba_frac_cube.coord('time').points)):
                for lat in range(len(ba_frac_cube.coord('latitude').points)):
                    for lon in range(len(ba_frac_cube.coord('longitude').points)):
                        if ba_frac_cube.data[month][lat][lon] > 0.0:
                            burndepth_cube.data[month][lat][lon] = 0.12
                        if burndepth_cube.data[month][lat][lon] > water_table_cube.data[month][lat][lon]:
                            burndepth_cube.data[month][lat][lon] = water_table_cube.data[month][lat][lon]
                        
    elif bd_method == 3: # method 2 using t_crt
        for month in range(len(tcrt_cube.coord('time').points)):
            for layer in range(len(tcrt_cube.coord('soil').points)):
                for lat in range(len(tcrt_cube.coord('latitude').points)):
                    for lon in range(len(tcrt_cube.coord('longitude').points)):
                        tcrt_cube.data[month][layer][lat][lon] = (42*sm_cube.data[month][lat][lon]) - 28
    
        soil_depths = [0.05, 0.134, 0.248, 0.389, 0.556,
               0.748, 0.963, 1.201, 1.461, 1.742,
               2.044, 2.366, 2.708, 3.07, 3.87,
               4.67, 5.47, 6.27, 7.07, 7.87]

        for month in range(len(tcrt_cube.coord('time').points)):
            for lat in range(len(tcrt_cube.coord('latitude').points)):
                for lon in range(len(tcrt_cube.coord('longitude').points)):
                    if ba_frac_cube.data[month][lat][lon] > 0:
                        burndepth_cube.data[month][lat][lon] = 0.01
                        for n in range(0,max_burndepth):
                            #print('testing layer: ',n)
                            if tcrt_cube.data[month][n][lat][lon] < soil_temp_cube.data[month][n][lat][lon]:
                                burndepth_cube.data[month][lat][lon] = soil_depths[n]
                                #print('fire in this layer')
                            else:
                                break
                    
        for month in range(len(burndepth_cube.coord('time').points)):
            for lat in range(len(burndepth_cube.coord('latitude').points)):
                for lon in range(len(burndepth_cube.coord('longitude').points)):                
                    if burndepth_cube.data[month][lat][lon] > water_table_cube.data[month][lat][lon]:
                        burndepth_cube.data[month][lat][lon] = water_table_cube.data[month][lat][lon]
    
    ## 6. Option to save output files
    # ignitions files and peat flammability are outputted as diagnostics - can be removed
    if save == "true":
        iris.save(ba_frac_cube, output_folder+'ba_frac/ba_frac'+year+'.nc')
        iris.save(burnt_area, output_folder+'ba/ba'+year+'.nc')
        iris.save(burndepth_cube, output_folder+'bd/bd'+year+'.nc')
        iris.save(total_ignitions, output_folder+'ignitions/ignitions'+year+'.nc')
        iris.save(peat_flam, output_folder+'peat_flam/peat_flam'+year+'.nc')
        iris.save(sum_pft, output_folder+'sum_pft_ignitions/sum_pft'+year+'.nc')
        iris.save(pft_ignitions_cube, output_folder+'pft_ignitions/pft_ignitions'+year+'.nc')
    
    return ba_frac_cube, burnt_area, burndepth_cube, peat_flam, sum_pft, pft_ignitions_cube


## Modelling carbon emissions

# Requires inputs from the first function. 
# base_cube = base cube for regridding
# ba_cube = burnt area cube (km2)
# bd_cube = burn depth cube (m)
# year = year you want to run (str format)
# combust_completeness = 0.8 - please use 0.8 as a fixed value for combustion completeness
# graph = true/false whether you want to plot a map of carbon emissions
# save = true/flase whether you want to save the outputted carbon emission cube
# output_folder = file path where you want output to be saved to

def calc_carbon_emissions(base_cube, ba_cube, bd_cube, year, combust_completeness, graph, save, output_folder):
    # Load and regrid peat carbon
    peat_c = iris.load_cube('/path/to/file.nc')
    scheme = iris.analysis.AreaWeighted()
    peat_c_cube = peat_c.regrid(base_cube, scheme)
    lat_coord = ba_cube.coord('latitude')
    lon_coord = ba_cube.coord('longitude')
    time_coord = ba_cube.coord('time')
    
    # Create emissions cube
    array = np.zeros((12,112,192))
    emitted_carbon_cube = iris.cube.Cube(array,
                                         attributes=None,
                                         dim_coords_and_dims=[(time_coord, 0),
                                                              (lat_coord, 1),
                                                              (lon_coord, 2)])
    
    # Calculate carbon emissions as carbon content of peat * depth of burn * combust_completeness * burnt area
    for month in range(len(ba_cube.coord('time').points)):
        for lat in range(len(ba_cube.coord('latitude').points)):
            for lon in range(len(ba_cube.coord('longitude').points)):
                carbon = peat_c_cube.data[lat][lon]
                area = ba_cube.data[month][lat][lon]*1e+6
                depth = bd_cube.data[month][lat][lon]
                depth_frac = depth/3
                emitted_carbon_cube.data[month][lat][lon] = (carbon * depth_frac * combust_completeness * area)/1000

    # Option to plot a map of carbon emissions
    if graph == "true":    
        carbon_tot = emitted_carbon_cube.collapsed('time', iris.analysis.SUM)
        fig = plt.figure(figsize=[10,4.5])
        cmap = plt.get_cmap('gist_heat_r')
        mesh = iplt.pcolormesh(carbon_tot, cmap=cmap)
        plt.gca().coastlines()
        colorbar = plt.colorbar(mesh, orientation='horizontal', pad=0.02, aspect=50)
        colorbar.set_label('Carbon emissions (Mg) '+ year, fontsize=11)
    
    # Option to save the output
    if save == "true":
        iris.save(emitted_carbon_cube, output_folder+'carbon/c_'+year+'.nc')
    
    return emitted_carbon_cube

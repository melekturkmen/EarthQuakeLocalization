# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:06:58 2024

@author: Erdem Akagündüz
"""

from __future__ import print_function
import os
import torch
import numpy as np
import scipy.io as sio
import scipy.signal
from math import radians, cos, sin, asin, sqrt
import scipy
from torch.utils.data import Dataset 
import random
import matplotlib.pyplot as plt
import geopandas as gpd


def datasetCreator(**kwargs):
    """Create a dataset for the AFAD dataset based on the specified kwargs
    
    Parameters:
    **kwargs: Keyword arguments specifying the dataset details
    
    Returns:
    list: List of attributes for the dataset
    """
    input_signal = []
    groundTruth = []
    gettingfiles = []
    magnitude_val = []
    stat2epi = []
    signal_list = []
    epicenter_depth = []
    lats = []
    longs = []
    stationco = []
    epicentraldist = []
    altitudes = []
    stat_id = []
    sP_3_channel = []
    tpga_ind = []  # 0 = left augmentation, 1 = middle augmentation, 2 = right augmentation
    stat_info = []
    alt = []
    Vs30 = []
    #skipped_file=[[] for ii in range(2)]
    skipped_file = {'afad': {'epi2loc is greater than the radius': [],
                                'station_km is greater than the stat_dist': [],
                                'eliminated epiDist values' : [],
                                'depth is greater than the epicenterDepth': [],
                                'duration is smaller than desired': [],
                                'Epicenter is empty': [],
                                'Depth is empty': [],
                                'Not skipped, Magnitude is empty': [],
                                'Not skipped, Altitude is empty': [],
                                'Vs30': [],
                                'Misc: out of the try block': []
                                }}
     
    fname = kwargs.get('AFAD_Path')
    for b in os.listdir(fname)[0:len(os.listdir(fname))]: 
        # print(b)
        
        dictname = 'afad'
        dataset=sio.loadmat('{}/{}'.format(fname, b))
        # Extract various data points from the loaded file, such as the latitude and longitude of the station, the number of data points,
        # the epicenter location, and the altitude.
        
        stationLat = dataset['EQ'][0][0]['anEQ']['statco'][0][0][0][0]
        stationLon = dataset['EQ'][0][0]['anEQ']['statco'][0][0][0][1]
        stationcoor = dataset['EQ'][0][0]['anEQ']['statco'][0][0][0]
                
        numofData = dataset['EQ'][0][0]['anEQ']['numofData'][0][0][0][0]
        
        SNR = dataset['EQ'][0][0]['anEQ']['numofData'][0][0][0][0]
        
        epicenterLat = dataset['EQ'][0][0]['anEQ']['epicenter'][0][0][0][0]
        epicenterLon = dataset['EQ'][0][0]['anEQ']['epicenter'][0][0][0][1]
        
        station_name = dataset['EQ'][0][0]['anEQ']['statID'][0][0][0][0]
        stat_id.append(station_name)
        
        epicenterDepth = float(dataset['EQ'][0][0]['anEQ']['depth'][0][0][0][0])/kwargs.get('km2meter')

        # Vs30 = dataset['EQ'][0][0]['anEQ']["Vs30"]  
        # Pass if Vs30 is required
        if 'Vs30' in kwargs['gt_select']:
            if isinstance(Vs30, np.uint16):
                Vs30 = Vs30 
            else:
                skipped_file[dictname]['Vs30'].append(b)
                stat_id.pop()
                continue
        # Check if station altitude information is gonna be fed to FC layers or not.
        if kwargs['add_station_altitude']:
            if 'alt' in dataset['EQ'][0][0]['anEQ'].dtype.fields:
                if len(dataset['EQ'][0][0]['anEQ']['alt'][0][0])!=0:                
                    alt = dataset['EQ'][0][0]['anEQ']['alt'][0][0][0][0]
                else:
                    alt = 0            
        
        # If available, extract the magnitude value of the earthquake from the file.
        if 'magnitudeval' in dataset['EQ'][0][0]['anEQ'].dtype.fields:
            if len(dataset['EQ'][0][0]['anEQ']['magnitudeval'][0][0])!=0:                
                mag = dataset['EQ'][0][0]['anEQ']['magnitudeval'][0][0][0][0]
            else:
                mag = 0    
                skipped_file[dictname]["Not skipped, Magnitude is empty"].append(b)
        
        # Extract the signal data from the file and find the time of the maximum acceleration.                        
        signal = dataset['EQ'][0][0]['anEQ']['Accel'][0][0] 
        gettingfiles, groundTruth, input_signal,  magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco , epicentraldist ,altitudes , stat_id , sP_3_channel, tpga_ind ,stat_info, skipped_file, Vs30, SNR = loader_helper(
            b, signal, alt, mag, numofData, epicenterLat,epicenterLon, epicenterDepth, 
            stationLat,stationLon, stationcoor, gettingfiles, groundTruth, input_signal, 
            magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco, 
            epicentraldist, altitudes, stat_id , sP_3_channel , tpga_ind ,stat_info, 
            skipped_file, Vs30, SNR, **kwargs)
        
         
    # Making them arrays
    groundTruth = np.asarray(groundTruth, dtype=np.float32)
    input_signal = np.asarray(input_signal, dtype=np.float32)
    stat_info = np.asarray(stat_info, dtype=np.float32)


    # SUbtract the station coordinates from epicenters' to provide a directional sense.
    for order,element in enumerate(kwargs.get("gt_select")):
        # To ensure that nothing is affected unless epiLAT or epiLON is present, trMeans=0 and trSTD=1 are set.
        if element == 'epiLAT':
            groundTruth[:, order] -= stat_info[:,0]
        elif element == 'epiLON':
            groundTruth[:, order] -= stat_info[:,1]

    groundTruth=torch.from_numpy(groundTruth)
    stat_info = torch.from_numpy(stat_info)
    
    
    #return all of them in a dictionary
    attributes = {'Matfiles': gettingfiles, 'Signal': signal_list, 'station_km':stat2epi, 'Latitudes': lats,'Longitudes': longs, 'Depths': epicenter_depth, 'Station_Coordinates': stationco, 'Epicentral_distance(epi2loc)': epicentraldist, 'Magnitudes': magnitude_val, 'Altitudes': altitudes, 'Station ID': stat_id, 'groundTruth': groundTruth, 'stat_info': stat_info, "tpga_ind": tpga_ind, 'skipped_file':skipped_file}                                
 
    return attributes  


class structureData(Dataset):
    def __init__(self,attributes, attr, phase, **kwargs):
        
        # Get the 
        self.attributes = attributes
        self.attr = attr
        self.phase = phase
        self.aug = kwargs.get('augmentation_flag')
        self.freq = kwargs.get('freq_flag')
        self.aug_param = kwargs.get('Augmentation parameter')
        self.signaltime =  kwargs.get('signaltime')
        self.fs = kwargs.get("fs")
        self.startP = attributes["tpga_ind"]
        self.gtnorm = kwargs.get("gtnorm")
        self.window_size = kwargs.get("window_size")
        self.signals = attributes["Signal"]
        self.groundTruth = attributes["groundTruth"]
        self.stat_info = attributes["stat_info"]
                
                
    def __getitem__(self, index):
        
        gt = self.groundTruth[index].clone()
        k = self.fs
        l = self.signaltime * self.fs
        gtnorm = self.gtnorm
        
        if self.freq:
            k = self.window_size*2 #overlap constant
            l = 2*self.signaltime - 1
                 
        self.startP = self.attributes["tpga_ind"][index]
        
        if self.phase=="training" and (self.aug):
            self.startP = random.randint(0,(2*self.aug))
              
        # augment ettiğimiz yeri çıkar-6000,1,3
        sig_to_return = torch.tensor(self.signals[index][self.startP * k : self.startP * k + l][:][:]).clone()# 19x51x3
        # sig_to_return = torch.from_numpy(self.signals[index][self.startP * k : self.startP * k + l,:,:]).clone() # 19x51x3
        
        # transpose-3,6000,1
        sig_to_return = np.transpose(sig_to_return,(2,0,1)) #(3, 19, 51)
        
        # normalize
        sig_to_return[0,:,:] = sig_to_return[0,:,:] - self.attr["trMeanR"]
        sig_to_return[1,:,:] = sig_to_return[1,:,:] - self.attr["trMeanG"]
        sig_to_return[2,:,:] = sig_to_return[2,:,:] - self.attr["trMeanB"]
        
        
        # Normalize station information, i.e., latitude (deg), longitude (deg), altitude (km) info of station.
        stat_info = self.stat_info[index].clone()
       #  stat_info[0] = stat_info[0] - self.attr["MeanStatLat"]
       #  stat_info[1] = stat_info[1] - self.attr["MeanStatLon"]
       # # stat_info[2] = stat_info[2] - self.attributes["MeanStatAlt"]
        
        if gtnorm:            
            for channel in range(len(gt)):          
                gt[channel] = (gt[channel] - self.attr["trMeans"][channel]) / self.attr["trStds"][channel]   
                
        return sig_to_return, stat_info, gt

        
    def __len__(self):

        return len(self.signals) ## len(self.images)


def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    Parameters:
    lon1 (float): Longitude of point 1
    lat1 (float): Latitude of point 1
    lon2 (float): Longitude of point 2
    lat2 (float): Latitude of point 2
    
    Returns:
    float: Distance between the two points in kilometers
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r

            
def loader_helper(b, signal, alt, mag, numofData, epicenterLat, epicenterLon, epicenterDepth, 
                  stationLat, stationLon, stationcoor, gettingfiles, groundTruth, input_signal, 
                  magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco, 
                  epicentraldist, altitudes, stat_id, sP_3_channel, tpga_ind, stat_info,
                  skipped_file, Vs30, SNR, **kwargs):

    try: 
        if len(b)>24: #Afad files have upto 24 chars and stead have at least 27
            dictname = 'stead'
        else:
            last_four = b[len(b)-8:len(b)-4]
            if last_four.isdigit():
                dictname = 'afad'
            else:
                dictname = 'kandilli'
        
            
        counter=0
        gettingfiles.append(b) 
        
        
        max_indices = np.argmax(signal,axis=0)
        median_index = np.median(max_indices)/kwargs.get('fs')
    
    
        
        # Select groundTruth according to gt_select
        new_column = np.array([])
        the_list = np.array([])
        for element in kwargs.get("gt_select"):
            
            if element == 'epiLAT':
                new_column = np.array(epicenterLat)
                the_list = np.hstack((the_list, new_column))
                #groundTruth.append(epicenterLat)
            if element == 'Depth':
                new_column = np.array(epicenterDepth)
                the_list = np.hstack((the_list, new_column))
                #groundTruth.append(epicenterDepth)
            if element == 'epiLON':
                new_column = np.array(epicenterLon)
                the_list = np.hstack((the_list, new_column))
                #groundTruth.append(epicenterLon)
            if element == 'Vs30':
                new_column = np.array(Vs30)
                the_list = np.hstack((the_list, new_column))
        groundTruth.append(the_list)  
        

        # Calculate the distance from the epicenter to the desired location and the distance from the station to the epicenter.
        epi2loc = haversine(float(kwargs.get('latitude')), float(kwargs.get('longitude')),epicenterLat,epicenterLon) #Epicenter to desired location distance(Km)
        station_km = haversine(stationLat,stationLon,epicenterLat,epicenterLon) #Station to epicenter distance(Km)
        
        # Check if epicenter to location distance is greater than radius
        if (epi2loc > float(kwargs.get('radius'))):
            # If yes, remove the file from the list and delete the dataset
            gettingfiles.remove(b)
            groundTruth.pop()
            stat_id.pop()
            skipped_file[dictname]['epi2loc is greater than the radius'].append(b)
            
          
        else:
            if (SNR < float(kwargs.get('SNR_val'))):
                # If yes, remove the file from the list and delete the dataset
                gettingfiles.remove(b)
                groundTruth.pop()
                stat_id.pop()
                skipped_file[dictname]['SNR is lower than the desired level'].append(b)
                
              
            else:
                # Check if station to epicenter distance is greater than stat_dist
                if (station_km > int(kwargs.get('stat_dist'))):
                    # If yes, remove the file from the list 
                    gettingfiles.remove(b)
                    groundTruth.pop()
                    stat_id.pop()
                    skipped_file[dictname]['station_km is greater than the stat_dist'].append(b)
                    skipped_file[dictname]['eliminated epiDist values'].append(station_km)
                    
                       
                else:
                    # Check if the number of data points in the signal is greater than desired signal duration * fs
                    if numofData > (int(kwargs.get('signaltime'))*kwargs.get('fs'))-1:
                        # Check if epicenter depth is less than desired depth value
                         if epicenterDepth < float(kwargs.get('depth')):
                             # Check if median index(tpga) is less than or equal to desired signal duration
                             if median_index <= int(kwargs.get('signaltime')):
                                 
                                 # Check which part of the signal the tpga value corresponds to (far left of the signal) 
                                 if median_index < float(int(kwargs.get('signaltime'))*kwargs.get('signal_aug_rate')):
                                     # Apply signal augmentation by flipping the first part of signal                                
                                     signal = signal[0:int(kwargs.get('signaltime'))* kwargs.get('fs'),:]                                 
                                     # denemesignal = signal                                 
                                     firstnoise = signal[0:int(kwargs.get('Augmentation parameter')) * kwargs.get('fs') * 2,:]                                 
                                     flipped = firstnoise[::-1]                                                                                              
                                     signal = np.concatenate((flipped,signal),axis=0)
                                     # Save tpga index to store information about what kind of augmentation applied
                                     tpga = 0
                                 # Check which part of the signal the tpga value corresponds to (far right of the signal)   
                                 elif median_index > float(int(kwargs.get('signaltime'))* (1- kwargs.get('signal_aug_rate'))):
                                     # Apply signal augmentation by flipping the first part of signal                                 
                                     signal = signal[0:int(kwargs.get('signaltime'))* kwargs.get('fs'),:]
                                     # denemesignal = signal
                                     lastnoise = signal[-(int(kwargs.get('Augmentation parameter')) * kwargs.get('fs') * 2):,:]
                                     flipped2 = lastnoise[::-1]                                                             
                                     signal = np.concatenate((signal,flipped2),axis=0)
                                     # Save tpga index to store information about what kind of augmentation applied
                                     tpga = 2
                                 # Check which part of the signal the tpga value corresponds to (in the middle of the signal)    
                                 else:
                                     # Apply signal augmentation by flipping the first part of signal                                  
                                     signal = signal[0:int(kwargs.get('signaltime')) * kwargs.get('fs'),:]
                                     
                                     # denemesignal = signal
                                     
                                     firstnoise = signal[0:int(kwargs.get('Augmentation parameter'))* kwargs.get('fs'),:]
                                     flipped = firstnoise[::-1]
                                     
                                     lastnoise = signal[-(int(kwargs.get('Augmentation parameter'))* kwargs.get('fs')):,:]
                                     flipped2 = lastnoise[::-1]
                                                                 
                                     signal = np.concatenate((flipped,signal),axis=0)
                                     signal = np.concatenate((signal,flipped2),axis=0)
                                     # Save tpga index to store information about what kind of augmentation applied
                                     tpga = 1
                             # Check if the distance from the end of the signal to the median index is less than median_index/2    
                             elif (numofData/kwargs.get('fs')) - median_index < float(median_index/2):    
                                 # Check which part of the signal the tpga value corresponds to (far left of the signal)
                                 if median_index < float(int(kwargs.get('signaltime')) * (kwargs.get('signal_aug_rate'))):
                                     # Apply signal augmentation by flipping the first part of signal                                  
                                     signal = signal[0:int(kwargs.get('signaltime'))* kwargs.get('fs'),:]
                                     
                                     # denemesignal = signal
                                     
                                     firstnoise = signal[0:int(kwargs.get('Augmentation parameter'))* kwargs.get('fs') * 2,:]
                                     flipped = firstnoise[::-1]
                                                                                                  
                                     signal = np.concatenate((flipped,signal),axis=0)
                                     # Save tpga index to store information about what kind of augmentation applied
                                     tpga = 0
                                 # Check which part of the signal the tpga value corresponds to (far right of the signal)    
                                 elif median_index > float(int(kwargs.get('signaltime')) * (1-kwargs.get('signal_aug_rate'))):                                 
                                     # Apply signal augmentation by flipping the first part of signal 
                                     signal = signal[0:int(kwargs.get('signaltime'))* kwargs.get('fs'),:]
                                     
                                     # denemesignal = signal
                                                                      
                                     lastnoise = signal[-(int(kwargs.get('Augmentation parameter'))* kwargs.get('fs') * 2):,:]
                                     flipped2 = lastnoise[::-1]
                                                                 
                                     signal = np.concatenate((signal,flipped2),axis=0)
                                     # Save tpga index to store information about what kind of augmentation applied
                                     tpga = 2
                                 # Check which part of the signal the tpga value corresponds to (in the middle of the signal)    
                                 else:                                 
                                     # Apply signal augmentation by flipping the first part of signal 
                                     signal = signal[0:int(kwargs.get('signaltime')) * kwargs.get('fs'),:]
                                     
                                     # denemesignal = signal
                                     
                                     firstnoise = signal[0:int(kwargs.get('Augmentation parameter')) * kwargs.get('fs'),:]
                                     flipped = firstnoise[::-1]
                                     
                                     lastnoise = signal[-(int(kwargs.get('Augmentation parameter'))*kwargs.get('fs')):,:]
                                     flipped2 = lastnoise[::-1]
                                                                 
                                     signal = np.concatenate((flipped,signal),axis=0)
                                     signal = np.concatenate((signal,flipped2),axis=0)
                                     # Save tpga index to store information about what kind of augmentation applied
                                     tpga = 1
                             #Tpga is in the middle of the signal    
                             else:
                                 # Cut the desired length from the signal
                                 start = round((median_index - float(int(kwargs.get('signaltime'))/2))*kwargs.get('fs'))
                                 stop = round((median_index + float(int(kwargs.get('signaltime'))/2))*kwargs.get('fs'))
                                 signal = signal[start:stop,:]
                                 
                                 # denemesignal = signal
                                 # Apply signal augmentation by flipping the first part of signal 
                                 firstnoise = signal[0:int(kwargs.get('Augmentation parameter'))*kwargs.get('fs'),:]
                                 flipped = firstnoise[::-1]
                                 
                                 lastnoise = signal[-(int(kwargs.get('Augmentation parameter'))*kwargs.get('fs')):,:]
                                 flipped2 = lastnoise[::-1]
                                                             
                                 signal = np.concatenate((flipped,signal),axis=0)
                                 signal = np.concatenate((signal,flipped2),axis=0)
                                 # Save tpga index to store information about what kind of augmentation applied
                                 tpga = 1
                                 
                             
                             # Reshape the signal array to the specified shape       
                             signal = signal.reshape((int(kwargs.get('signaltime')) + (2*int(kwargs.get('Augmentation parameter'))))*kwargs.get('fs'),1,kwargs.get('channel_depth'))    
                             
                             # Convert the signal array to a list
                             a = signal
                             a = a.tolist()
                             
                             # Create an array to store the spectrogram data
                             sP_3_channel = np.zeros((int((kwargs.get('fs')/2)+1),2*(int(kwargs.get('signaltime'))+(2*int(kwargs.get('Augmentation parameter'))))-1,kwargs.get('channel_depth')))
                             
                             # If the frequency flag is set, compute the spectrogram for each channel
                             if kwargs.get('freq_flag'):
                                 for vay in range(0,kwargs.get('channel_depth')):
                                     f, t, sP = scipy.signal.spectrogram(signal[:,0,vay], window=scipy.signal.windows.hann(kwargs.get('fs')*kwargs.get('window_size')), fs=kwargs.get('fs'), nperseg=kwargs.get('fs')*kwargs.get('window_size'), noverlap=kwargs.get('fs')*kwargs.get('window_size')/2, mode='magnitude')
                                     sP_3_channel[:,:,vay] = sP                             
                                 signal = (np.moveaxis(sP_3_channel,0,1))      
                                 input_signal.append(np.moveaxis(sP_3_channel,0,1))
                             # If the frequency flag is not set, use input_signal as time signal  
                             else:
                                 input_signal.append(a)
                             counter += 1
                             
                             # Append the epicenter location data to the epicenter list
                             EpicenterDuo = [epicenterLat, epicenterLon]
                             EpicenterDuolist = list(EpicenterDuo)
                             EpicenterDepthlist = epicenterDepth                         
                             EpicenterDuolist.append(EpicenterDepthlist)
                             counter += 1
                             # Append the station latitude, longitude, and altitude to the station information list
                             stat_inf = [stationLat, stationLon] 
                             stat_info.append(stat_inf)
                             counter += 1
                             altitudes.append(alt)
                             counter += 1 #7
                             # Append the epicenter depth, magnitude, latitude, and longitude to their respective lists                
                             epicenter_depth.append(epicenterDepth)
                             counter += 1
                             # magnitude_val.append(mag)
                             # counter += 1
                             lats.append(epicenterLat)
                             counter += 1
                             longs.append(epicenterLon)
                             counter += 1
                             # Append the distance between the station and the epicenter to the station to epicenter distance list
                             stat2epi.append(station_km)
                             counter += 1
                             # Append the TPGA index to the TPGA index list
                             tpga_ind.append(tpga)
                             counter += 1
                             # Append the signal, station coordinates, and epicenter distance to their respective lists
                             signal_list.append(signal)
                             counter += 1
                             stationco.append(stationcoor)
                             counter += 1
                             epicentraldist.append(epi2loc)
                             counter += 1
                             
                         else:
                            gettingfiles.remove(b)
                            groundTruth.pop()
                            stat_id.pop()
                            skipped_file[dictname]['depth is greater than the epicenterDepth'].append(b)
                            
                    else:
                        gettingfiles.remove(b)
                        groundTruth.pop()
                        stat_id.pop()
                        skipped_file[dictname]['duration is smaller than desired'].append(b) 
                        
            
            # print("Gettingfiles, Signal list and filename--> ", len(gettingfiles),len(signal_list),b) 
            
            # if len(gettingfiles) != len(signal_list):
            #     print("ERR")
            
    except:
        # print(f"Error: Counter: {counter}")
        lists_to_pop = ['input_signal', 'EpicenterDuolist', 'stat_info', 'altitudes', 'epicenter_depth', 'lats', 'longs', 'stat2epi', 'tpga_ind', 'signal_list', 'stationco', 'epicentraldist']
        
        for lst_name in lists_to_pop[0:counter]:
            lst = locals()[lst_name]
            # print(lst_name,len(lst))
            if lst and lst != []:
                lst.pop()
            # print(lst_name,len(lst))
        skipped_file['except_block'].append(b)
    
    return gettingfiles, groundTruth, input_signal,  magnitude_val, stat2epi, signal_list, epicenter_depth, lats, longs, stationco, epicentraldist, altitudes, stat_id, sP_3_channel, tpga_ind, stat_info, skipped_file, Vs30, SNR


def getArguments(testDataFolder):
    ############################################################################## 
    # 1. ExpName & Parameters
    hyperparameters = {"fno" : 1,          
                    "fsiz" : 4,
                    "dropout_rate" : 0.2,
                    "batchsize" : 64,
                    "n_epochs": 1,
                    "lr":0.001, 
                    "step_size": 20,
                    "gamma": 0.9,
                    'training_loss_fcn': None,
                    }
    
    parameters = {"Transfer_model": False,
                  "Transfer_encoder":False,
                      "transfer_path":'',
                      "add_stat_info": True,
                      "add_station_altitude": True,
                      "gtnorm": False,
                      "test": True,
                      "gt_select": ["Depth", "epiLAT", "epiLON"],  #epiLAT,epiLON,Depth,Distance
                      "model_select": 'ResNet',           #"ResNet","TCN"
                      "FC_size" : 256,
                      "SP": True,
                      "SNR_val": 25,                         # Remove if it is lower than this value                
                      "statID" : None,                  # Station ID
                      "radius" : 3000000,               # The radius at which the experiment will be carried out.
                      "latitude" : '0',                # In which Latitude the experiment will be performed.
                      "longitude" : '0',             # In which Longitude the experiment will be performed.
                      "signaltime" : 60,              # Time length of EQ signals
                      "magnitude": 3.5,               # The magnitude of the EQ signals (If it is less than this value, wont take it.)
                      "depth":10000,                     # Depth of EQ signal (If it is greater than this value, will not use )(km)
                      "stat_dist": 120,               #(unit?) Distance between the station recording the EQ event and the epicenter. (If it is greater than this value, will not use )
                      "freq_flag": True,            # Will I use a frequency signal? Or is it a time signal? (T for Freq, F for Time)
                      "augmentation_flag": True,      # Will I augment the signal?
                      "Train percentage": 80,         # Train-val-test percentage. (80 means %80 = train + val)
                      "Augmentation parameter": 2,     # How much will I augment the signal (1 means 2 pieces of 1 seconds, so at total 2 seconds.)
                      "Crossvalidation_type": "Chronological",         # Crossvalidation_type (Chronological, Station-based)
                      "dataset": "AFAD"                # Dataset to use (AFAD, STEAD, KANDILLI_AFAD)
                    }
    
    constants = {"km2meter" : 1000,                 
                    "fs" : 100,                     
                    "signal_aug_rate" : 0.3,        # I augment the signal differently when the tpga is below this percentage value. (or over 1-signal_aug_rate)
                    "window_size" : 1,              
                    "channel_depth" : 3,
                    "AFAD_Path": testDataFolder,
                    "working_directory": os.getcwd()                                 
                    }    
    kwargs = {**parameters,**constants,**hyperparameters}    
    return kwargs



def plot_(station_lat, station_lon, predicted_lat,predicted_lon, true_lat,true_lon, event_name, path):
    
   # Create a Point object from your station coordinates
   fig, ax = plt.subplots(figsize=(9, 6))
   
   # Define colors for markers
   true_color = "green" # "#1f77b4" 
   pred_color = "blue"  #"#ff7f0e"
   station_color = "red" #"#2ca02c"
   
   # Determine which country each point is located in
   countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
   
   station_country=countries[countries['name'] == 'Turkey']
   station_country.plot(color="lightgrey", edgecolor="white", ax=ax)
   
   # Set axis limits
   lon_min = 26
   lon_max = 45
   lat_min = 36
   lat_max = 42
   ax.set_xlim(lon_min, lon_max)
   ax.set_ylim(lat_min, lat_max) 
   
   # Set axis labels and plot title
   ax.set_xlabel('latitude (deg)')
   ax.set_ylabel('longitude (deg)') #, fontsize=20
   title = f"{event_name}: {int(np.round(haversine(true_lon, true_lat, predicted_lon, predicted_lat)))}km error"
   ax.set_title(title)
   
   # add station location label to legend
   ax.scatter(station_lon,station_lat, marker='p',color=station_color)
   
   true_lat = true_lat
   true_lon = true_lon
   
   # Plot points
   plt.plot(true_lon, true_lat, marker='o', markersize=7, markerfacecolor="None", markeredgecolor=true_color, linestyle="None")
   plt.plot(predicted_lon, predicted_lat, marker='o', markersize=7, markerfacecolor="None", markeredgecolor=pred_color, linestyle="None")
      
   # Plot lines
   plt.plot([predicted_lon, true_lon], [predicted_lat, true_lat], linestyle='-',alpha=0.8, linewidth=0.5)
   
   # Add legend
   plt.scatter([], [], marker='o', facecolor='none', edgecolors=true_color, label='True Epicenter')
   plt.scatter([], [], marker='o', facecolor='none', edgecolors=pred_color, label='Predicted Epicenter')
   plt.scatter([], [], marker='p', facecolor=station_color, edgecolors=station_color, label='Station Location')
   plt.legend()
    
   base_name = os.path.splitext(event_name)[0]
   output_file = os.path.join(r"figs", f"{base_name}.png")
   plt.savefig(output_file)
   plt.close()
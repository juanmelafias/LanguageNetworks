# Importing relevant libraries: 
#import pandas as pd
#import numpy as np; 
#import matplotlib.pyplot as plt; 
#import matplotlib as mplt; 
import networkx as nx; 
#from copy import copy; 
# For 3D scatter: 
#from mpl_toolkits.mplot3d import Axes3D; 

# Importing libraries for I/O and system communication: 
import os, sys; 
#import pickle as pkl; 
#import scipy.io as sio; # To read .mat files! and .mnx files! 

# Importing functions for clustering: 
#from sklearn.cluster import KMeans;  
#import scipy.cluster.hierarchy as spc; 

# Importing homebrew libraries: 
import helper as h; 
#import loadHelper as lh; 

#importing functions

from utils import csv2df,json2dict,dict2json,load_network
filelist = os.listdir('./dictionaries/')
languagelist = [file.split('.')[0] for file in filelist]
CreateProperties = True
for netName in languagelist:
    netPath='networkslemma/'
    if CreateProperties:
        picsPath = 'pics/'
        langframe = csv2df(f'dataframeslemma/{netName}.csv')
        mostfreq =langframe.unique_id.to_list()
        jsonfile = f'dictionarieslemma/{netName}.json'
        thisNetwork = load_network(jsonfile)
        thisNetwork=thisNetwork.subgraph(mostfreq)
        
        # Creating Giant connected component
        Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
        thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 
        nNodes = len(thisNetwork.nodes()); 
        nEdges = thisNetwork.number_of_edges(); 
        # Computing and saving properties
        print(f'Computing properties for {netName}')
        fNeighborMean = True; 
        fNeighborStd = True; 
        (nodeList, propertiesDict, includedProperties, excludedProperties) = h.computeNodesProperties(thisNetwork, 
                                                                                            fNeighborMean, fNeighborStd); 
        h.writeNetworkProperties(netName, netPath, nodeList, propertiesDict); 
    #Readin properties back again
    (nodeList, propertiesDict) = h.readNetworkProperties(netName, netPath, fNeighborMean, fNeighborStd); 
    (includedProperties, excludedProperties) = h.findPathologicalProperties(propertiesDict); 
    #Creating mean properties
    meanpropertiesDict = {key:propertiesDict[key].mean() for key in propertiesDict.keys()}
    jsonname = f'avgpropertieslemma/{netName}.json'
    dict2json(meanpropertiesDict,jsonname)
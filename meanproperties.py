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
import common.helper as h; 
#import loadHelper as lh; 

#importing functions

"""
This scripts computes mean properties for all nodes in the network for every desired language
"""

from common.utils import csv2df,json2dict,dict2json,load_network
filelist = os.listdir('./files/inflected/dictionaries/')
languagelist = [file.split('.')[0] for file in filelist if file in ['French.json','Arabic.json', 'Japanese.json']]
CreateProperties = True
lemmatized = False
if lemmatized:
    netPath = 'files/lemmatized/networks/'
    folderframe = 'files/lemmatized/dataframes'
    folderdict = 'files/lemmatized/dictionaries'
    folderavg = 'files/lemmatized/avgproperties'
else:
    netPath = 'files/inflected/networks/'
    folderframe = 'files/inflected/dataframes'
    folderdict = 'files/inflected/dictionaries'
    folderavg = 'files/inflected/avgproperties'
for netName in languagelist:
    if CreateProperties:
        picsPath = 'pics/'
        langframe = csv2df(f'{folderframe}/{netName}.csv')
        mostfreq =langframe.unique_id.to_list()
        jsonfile = f'{folderdict}/{netName}.json'
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
    jsonname = f'{folderavg}/{netName}.json'
    dict2json(meanpropertiesDict,jsonname)
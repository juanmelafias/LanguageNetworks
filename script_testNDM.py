"""

	script_processNodes_onlyDutch.py: 

		This is the script that performs the main analysis on nodes of individual networks. Additional
		analyses are performed on the data produced by this script, as well as on complete networks. 

		The current analysis consists on loading an individual network, measuring properties of each
		individual node, performing PCA to visualize classes of nodes within the network. The analysis is
		repeated for different networks so that we can visualize how different classes of ndoes come to
		be. 

		Script last modified on 03/30/2021. 

		Some descriptors (as of 03/30/2021): 

			>> Datasets: 
				> DataDown: 20 participants. 
				> DataHI (Hearing Impairment): 20 participants. 
				> DataSLI (Specific Language Impairment): 20 participants. 
				> DataTD1 (Typical Development [1-2) years): 9 participants. 
				> DataTD2 (Typical Development [2-3) years): 4 participants. 
				> DataTD3 (Typical Development [3-4) years): 9 participants. 
				> DataTDLongDutch1 (Typical Development, longitudinal study): 1 participant, 18 snapshots, 17 valid ones. 

			>> Measuring now: 
				> 


			>> There is some coming and going related to networkx versions: 
				> Somehow, software @IFISC is not up to date... or whatever happens. 
				> I decided to keep two versions of the functions that compute stuff: 
					- One is executed in my laptop, the other one at IFISC. 
					- computeNetworksStatistics() --> computeNetworksStatistics_IFISC(). 
					- computeNetworksStatistics() --> computeNetworksStatistics_home(). 
				> But none of this is an issue because I'm not in the IFISC anymore. 

"""


# Importing relevant libraries: 
import numpy as np; 
import scipy.linalg as la; 
import matplotlib.pyplot as plt; 
import matplotlib as mplt; 
import os, sys; 
import networkx as nx; 
import helper as h; 
import loadHelper as lh; 
from copy import copy; 

# For 3D scatter: 
from mpl_toolkits.mplot3d import Axes3D; 



## Loading all available networks: 

location = "home"; 


# ########################################################################################################################
# ## Uncomment for syntax network: 

# # Metadata to load networks: 
# if (location=="home"): 
# 	dataPathMaster = "/home/brigan/Desktop/Research_IFISC/LanguageMorphospaces/Data"; 
# dataNames = ["DataDown", "DataHI", "DataSLI", "DataTD1", "DataTD2", "DataTD3", "DataTDLongDutch1_original"]; 
# dataFormats = ["txt", "sif", "sif", "sif", "sif", "sif", "sif"]; 
# # dataNames = ["DataDown"]; 
# # dataFormats = ["txt"]; 

# # Looping over folders, loading nets: 
# allNetworksDict = {}; 
# allNetworksNamesDict = {}; 
# for (dataName, dataFormat) in zip(dataNames, dataFormats): 
# 	dataPath = os.path.join(dataPathMaster, dataName); 
# 	if (dataFormat=="txt"): 
# 		(synNetDict, synNetNameList) = lh.loadAllTXTNetsFromPath(dataPath, False); 
# 	if (dataFormat=="sif"): 
# 		(synNetDict, synNetNameList) = lh.loadAllSIFNetsFromPath(dataPath); 
# 	allNetworksDict[dataName] = synNetDict; 
# 	allNetworksNamesDict[dataName] = synNetNameList; 

# # Choose a single network: 
# thisKey = "DataTD3"; 

# iNetwork = 5; 
# thisNetwork = allNetworksDict[thisKey][allNetworksNamesDict[thisKey][iNetwork]]; 


# #######################################################################################################################
# # Uncomment for randomly generated networks: 

# thisNetwork = nx.erdos_renyi_graph(200, 0.1); 
# # thisNetwork = nx.watts_strogatz_graph(200, 4, 0.05); 
# # thisNetwork = nx.barabasi_albert_graph(200, 2); 
# # thisNetwork = nx.bipartite.gnmk_random_graph(50,50,200); 


#######################################################################################################################
# Uncomment for CNB network: 

dataPath = "/home/brigan/Desktop/Research_CNB/Misc/CNB_net/Code/Output/"; 

# Reading edges: 
fIn	= open(dataPath + "edges.csv", 'r'); 
edges = []; 
nodes = []; 
allLines = fIn.read().splitlines(); 
nPapers = {}; 
nCollaborations = {}; 
for line in allLines: 
	thisEdge = line.split(', '); 
	edges += [(thisEdge[0], thisEdge[1])]; 

# Building network from edges: 
thisNetwork = nx.Graph(); 
thisNetwork.add_edges_from(edges); 



# ########################################################################################################################
# ## Uncomment for connectome network: 

# connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Connectome/"; 
# thisNetwork = nx.read_graphml(connectomeDataPath + "993675_repeated10_scale250.graphml"); 
# # thisNetwork = nx.read_graphml(connectomeDataPath + "958976_repeated10_scale250.graphml"); # Check out this network!! Compare to others! 
# # thisNetwork = nx.read_graphml(connectomeDataPath + "959574_repeated10_scale250.graphml"); # Check out this network!! Compare to others! 







######################################################################################################
## Perform analysis on nodes: 
## 


# Loading largest connected component and number of nodes: 
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 
nNodes = len(thisNetwork.nodes()); 


# Measure stuff from nodes: 
(nodeList, nodesProperties, includedProperties, excludedProperties) = h.computeNodesProperties(thisNetwork); 


nProperties = len(includedProperties); 
allPropertiesArray = np.zeros([nProperties, nNodes]); 
dictIStat = {}; 
for (iStat, statistic) in enumerate(includedProperties): 
	allPropertiesArray[iStat,:] = nodesProperties[statistic]; 
	dictIStat[statistic] = iStat; 





# # Standardizing distro: 
allStatisticsMean = np.mean(allPropertiesArray, 1); 
allStatisticsStd = np.std(allPropertiesArray, 1); 
allPropertiesArray = allPropertiesArray - np.transpose(np.repeat(np.array([allStatisticsMean]), nNodes, 0)); 
allPropertiesArray = np.divide(allPropertiesArray, np.transpose(np.repeat(np.array([allStatisticsStd]), nNodes, 0))); 

# Computing correlation matrix and diagonalizing: 
allStatisticsCov = np.cov(allPropertiesArray); 
# (eigVals, eigVects) = la.eig(allStatisticsCov); 	# ACHTUNG!! This seems to produce more imaginary component when it shouldn't... 
(eigVals, eigVects) = np.linalg.eig(allStatisticsCov); 
# eigVals = np.real(eigVals); 
# eigVects = np.real(eigVects); 

(varianceExplained, varianceExplained_cumul) = h.varianceExplained(eigVals); 
plt.figure(); 
plt.plot(varianceExplained); 

plt.figure(); 
plt.plot(varianceExplained_cumul); 


# plt.show(); 
# sys.exit(0); 


plt.figure(); 
plt.imshow(allStatisticsCov, interpolation="none"); 
plt.colorbar(); 

plt.figure(); 
plt.plot(eigVals); 

plt.figure(); 
plt.imshow(eigVects, interpolation="none", cmap="coolwarm"); 
plt.colorbar(); 


# Projecting data into eigenspace: 
allPropertiesArray_ = np.dot(np.transpose(eigVects), allPropertiesArray); 


## Using first three PCs as color coding: 
# Normalize components to [0,1]; 
valuesRGB0 = h.convertPC2RGB(allPropertiesArray_[0,:]); 
valuesRGB1 = h.convertPC2RGB(allPropertiesArray_[1,:]); 
valuesRGB2 = h.convertPC2RGB(allPropertiesArray_[2,:]); 
# Save hex color values to a list: 
nodeColor = []; 
for (iNode, node) in enumerate(nodeList): 
	nodeColor += [mplt.colors.to_hex([valuesRGB0[iNode], valuesRGB1[iNode], valuesRGB2[iNode]])]; 

 

# PC1-PC2: 
fig = plt.figure(); 
ax = fig.add_subplot(111); 
plt.scatter(allPropertiesArray_[0,:], allPropertiesArray_[1,:], c=nodeColor); 
# for (iWord, word) in enumerate(nodeList): 
# 	ax.annotate(word, (allPropertiesArray_[0,iWord], allPropertiesArray_[1,iWord])); 
	# plt.text(allPropertiesArray_[0,iWord], allPropertiesArray_[1,iWord], word); 
plt.xlabel("PC1"); 
plt.ylabel("PC2"); 


# PC1-PC3: 
fig = plt.figure(); 
plt.scatter(allPropertiesArray_[0,:], allPropertiesArray_[2,:], c=nodeColor); 
plt.xlabel("PC1"); 
plt.ylabel("PC3"); 

# PC1-PC2-PC3: 
fig = plt.figure(); 
ax = fig.add_subplot(111, projection='3d'); 
ax.scatter(allPropertiesArray_[0,:], allPropertiesArray_[1,:], allPropertiesArray_[2,:], c=nodeColor); 
ax.set_xlabel("PC1"); 
ax.set_ylabel("PC2"); 
ax.set_zlabel("PC3"); 


fig = plt.figure(); 
ax = fig.add_subplot(111); 
# nx.draw(thisNetwork, with_labels=True, pos=nx.circular_layout(thisNetwork), node_color=nodeColor); 
nx.draw(thisNetwork, with_labels=False, pos=nx.kamada_kawai_layout(thisNetwork), node_color=nodeColor); 
ax.set_aspect("equal"); 

fig = plt.figure(); 
ax = fig.add_subplot(111); 
# nx.draw(thisNetwork, with_labels=True, pos=nx.circular_layout(thisNetwork), node_color=nodeColor); 
nx.draw(thisNetwork, with_labels=False, node_color=nodeColor); 
ax.set_aspect("equal"); 






plt.show(); 
# sys.exit(0); 

# # Saving data in eigenspace: 
# np.savetxt(os.path.join(dataPathMaster, "Results/DataForPCA/dataEigenSpace.csv"), allPropertiesArray_, delimiter=", "); 
# np.savetxt(os.path.join(dataPathMaster, "Results/DataForPCA/eigenValues.csv"), eigVals, delimiter=", "); 
# np.savetxt(os.path.join(dataPathMaster, "Results/DataForPCA/eigenVectors.csv"), eigVects, delimiter=", "); 
# np.savetxt(os.path.join(dataPathMaster, "Results/DataForPCA/covariance.csv"), allStatisticsCov, delimiter=", "); 

# ## Saving data to files: 
# np.savetxt(os.path.join(dataPathMaster, "Results/DataForPCA/data.csv"), allPropertiesArray, delimiter=", "); 
# for key in netIndexes.keys(): 
# 	fOut = open(os.path.join(dataPathMaster, "Results/DataForPCA/labels"+key+".csv"), 'w'); 
# 	for (iIndex, index) in enumerate(netIndexes[key]): 
# 		fOut.write(str(index)); 
# 		if (iIndex < len(netIndexes[key])-1): 
# 			fOut.write(', '); 
# 	fOut.close(); 
# 	fOut = open(os.path.join(dataPathMaster, "Results/DataForPCA/netNames"+key+".csv"), 'w'); 
# 	for netName in allNetworksNamesDict[key]: 
# 		fOut.write(netName+'\n'); 
# 	fOut.close(); 

# print(netIndexesList); 
# np.savetxt(, netIndexesList, fmt='%d', delimiter=", "); 




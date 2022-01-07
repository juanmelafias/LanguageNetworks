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

########################################################################################################################
## Uncomment for syntax network: 

# Metadata to load networks: 
if (location=="home"): 
	dataPathMaster = "/home/brigan/Desktop/Research_IFISC/LanguageMorphospaces/Data"; 
if (location=="IFISC"): 
	dataPathMaster = "/home/luis/Desktop/Research_IFISC/LanguageMorphospaces/Data"; 
dataNames = ["DataDown", "DataHI", "DataSLI", "DataTD1", "DataTD2", "DataTD3", "DataTDLongDutch1_original"]; 
dataFormats = ["txt", "sif", "sif", "sif", "sif", "sif", "sif"]; 
# dataNames = ["DataDown"]; 
# dataFormats = ["txt"]; 

# Looping over folders, loading nets: 
allNetworksDict = {}; 
allNetworksNamesDict = {}; 
for (dataName, dataFormat) in zip(dataNames, dataFormats): 
	dataPath = os.path.join(dataPathMaster, dataName); 
	if (dataFormat=="txt"): 
		(synNetDict, synNetNameList) = lh.loadAllTXTNetsFromPath(dataPath, False); 
	if (dataFormat=="sif"): 
		(synNetDict, synNetNameList) = lh.loadAllSIFNetsFromPath(dataPath); 
	allNetworksDict[dataName] = synNetDict; 
	allNetworksNamesDict[dataName] = synNetNameList; 

# Choose a single network: 
thisKey = "DataTD3"; 
iNetwork = 5; 
thisNetwork = allNetworksDict[thisKey][allNetworksNamesDict[thisKey][iNetwork]]; 

# sys.exit(); 



#######################################################################################################################
# Uncomment for randomly generated networks: 

thisKey = "None"; 
otherNetwork = nx.erdos_renyi_graph(400, 0.1); 
# thisNetwork = nx.watts_strogatz_graph(400, 4, 0.15); 
# thisNetwork = nx.barabasi_albert_graph(200, 2); 



# #######################################################################################################################
# # Uncomment for CNB network: 

# thisKey = "None"; 
# dataPath = "/home/brigan/Desktop/Research_CNB/Misc/CNB_net/Code/Output/"; 

# # Reading edges: 
# fIn	= open(dataPath + "edges.csv", 'r'); 
# edges = []; 
# allLines = fIn.read().splitlines(); 
# for line in allLines: 
# 	thisEdge = line.split(', '); 
# 	edges += [(thisEdge[0], thisEdge[1])]; 

# # Building network from edges: 
# otherNetwork = nx.Graph(); 
# otherNetwork.add_edges_from(edges); 



# ########################################################################################################################
# ## Uncomment for connectome network: 

# thisKey = "None"; 
# connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Connectome/"; 
# thisNetwork = nx.read_graphml(connectomeDataPath + "993675_repeated10_scale250.graphml"); 
# # otherNetwork = nx.read_graphml(connectomeDataPath + "958976_repeated10_scale250.graphml"); # Check out this network!! Compare to others! 
# # thisNetwork = nx.read_graphml(connectomeDataPath + "959574_repeated10_scale250.graphml"); # Check out this network!! Compare to others! 






########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################




########################################################################################################################
## Perform analysis on nodes: 
## 

# Extracting largest connected component and number of nodes: 
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 
nNodes = len(thisNetwork.nodes()); 
Gcc = sorted(nx.connected_components(otherNetwork), key=len, reverse=True); 
otherNetwork = nx.Graph(otherNetwork.subgraph(Gcc[0])); 
otherNNodes = len(otherNetwork.nodes()); 


# Measure stuff from nodes: 
(nodeList, nodesStatistics) = h.computeNodesStatistics(thisNetwork); 
nStatistics = len(nodesStatistics.keys()); 
allStatisticsArray = np.zeros([nStatistics, nNodes]); 
dictIStat = {}; 
for (iStat, statistic) in enumerate(nodesStatistics.keys()): 
	allStatisticsArray[iStat,:] = nodesStatistics[statistic]; 
	dictIStat[statistic] = iStat; 
	print(statistic); 
	print(allStatisticsArray[iStat,:]); 


# Measure stuff from nodes: 
(otherNodeList, otherNodesStatistics) = h.computeNodesStatistics(otherNetwork); 
otherNStatistics = len(otherNodesStatistics.keys()); 
otherAllStatisticsArray = np.zeros([otherNStatistics, otherNNodes]); 
otherDictIStat = {}; 
for (iStat, statistic) in enumerate(otherNodesStatistics.keys()): 
	otherAllStatisticsArray[iStat,:] = otherNodesStatistics[statistic]; 
	otherDictIStat[statistic] = iStat; 
	print(statistic); 
	print(otherAllStatisticsArray[iStat,:]); 



# # Standardizing distro: 
allStatisticsMean = np.mean(allStatisticsArray, 1); 
allStatisticsStd = np.std(allStatisticsArray, 1); 
allStatisticsArray = allStatisticsArray - np.transpose(np.repeat(np.array([allStatisticsMean]), nNodes, 0)); 
allStatisticsArray = np.divide(allStatisticsArray, np.transpose(np.repeat(np.array([allStatisticsStd]), nNodes, 0))); 

# Computing correlation matrix and diagonalizing: 
allStatisticsCov = np.cov(allStatisticsArray); 
print(allStatisticsCov); 
(eigVals, eigVects) = la.eig(allStatisticsCov); 
eigVals = np.real(eigVals); 
eigVects = np.real(eigVects); 
if ((thisKey=="DataDown" and iNetwork==17) or (thisKey=="DataSLI" and iNetwork==16)): 
	eigVals = np.real(eigVals); 
	eigVects = np.real(eigVects); 

# # Standardizing distro: 
otherAllStatisticsMean = np.mean(otherAllStatisticsArray, 1); 
otherAllStatisticsStd = np.std(otherAllStatisticsArray, 1); 
otherAllStatisticsArray = otherAllStatisticsArray - np.transpose(np.repeat(np.array([otherAllStatisticsMean]), otherNNodes, 0)); 
otherAllStatisticsArray = np.divide(otherAllStatisticsArray, np.transpose(np.repeat(np.array([otherAllStatisticsStd]), otherNNodes, 0))); 


plt.figure(); 
plt.imshow(allStatisticsCov, interpolation="none"); 
plt.colorbar(); 

plt.figure(); 
plt.plot(eigVals); 

plt.figure(); 
plt.imshow(eigVects, interpolation="none", cmap="coolwarm"); 
plt.colorbar(); 


# Projecting data into eigenspace: 
allStatisticsArray_ = np.dot(np.transpose(eigVects), allStatisticsArray); 
otherAllStatisticsArray_ = np.dot(np.transpose(eigVects), otherAllStatisticsArray); 


## Using first three PCs as color coding: 
# Normalize components to [0,1]; 
valuesRGB0 = h.convertPC2RGB(allStatisticsArray_[0,:]); 
valuesRGB1 = h.convertPC2RGB(allStatisticsArray_[1,:]); 
valuesRGB2 = h.convertPC2RGB(allStatisticsArray_[2,:]); 
# Save hex color values to a list: 
nodeColor = []; 
for (iNode, node) in enumerate(nodeList): 
	nodeColor += [mplt.colors.to_hex([valuesRGB0[iNode], valuesRGB1[iNode], valuesRGB2[iNode]])]; 

## Using first three PCs as color coding: 
# Normalize components to [0,1]; 
otherValuesRGB0 = h.convertPC2RGB(otherAllStatisticsArray_[0,:]); 
otherValuesRGB1 = h.convertPC2RGB(otherAllStatisticsArray_[1,:]); 
otherValuesRGB2 = h.convertPC2RGB(otherAllStatisticsArray_[2,:]); 
# Save hex color values to a list: 
otherNodeColor = []; 
for (iNode, node) in enumerate(otherNodeList): 
	otherNodeColor += [mplt.colors.to_hex([otherValuesRGB0[iNode], otherValuesRGB1[iNode], otherValuesRGB2[iNode]])]; 

 

# # PC1-PC2: 
# fig = plt.figure(); 
# ax = fig.add_subplot(111); 
# plt.scatter(allStatisticsArray_[0,:], allStatisticsArray_[1,:], c=nodeColor); 
# # for (iWord, word) in enumerate(nodeList): 
# # 	ax.annotate(word, (allStatisticsArray_[0,iWord], allStatisticsArray_[1,iWord])); 
# 	# plt.text(allStatisticsArray_[0,iWord], allStatisticsArray_[1,iWord], word); 
# plt.xlabel("PC1"); 
# plt.ylabel("PC2"); 


# # PC1-PC3: 
# fig = plt.figure(); 
# plt.scatter(allStatisticsArray_[0,:], allStatisticsArray_[2,:], c=nodeColor); 
# plt.xlabel("PC1"); 
# plt.ylabel("PC3"); 

# PC1-PC2-PC3: 
fig = plt.figure(); 
ax = fig.add_subplot(111, projection='3d'); 
ax.scatter(allStatisticsArray_[0,:], allStatisticsArray_[1,:], allStatisticsArray_[2,:], c="black"); 
ax.scatter(otherAllStatisticsArray_[0,:], otherAllStatisticsArray_[1,:], otherAllStatisticsArray_[2,:], c="red"); 
ax.set_xlabel("PC1"); 
ax.set_ylabel("PC2"); 
ax.set_zlabel("PC3"); 


# fig = plt.figure(); 
# ax = fig.add_subplot(111); 
# # nx.draw(thisNetwork, with_labels=True, pos=nx.circular_layout(thisNetwork), node_color=nodeColor); 
# nx.draw(thisNetwork, with_labels=False, pos=nx.kamada_kawai_layout(thisNetwork), node_color=nodeColor); 
# ax.set_aspect("equal"); 

# fig = plt.figure(); 
# ax = fig.add_subplot(111); 
# # nx.draw(thisNetwork, with_labels=True, pos=nx.circular_layout(thisNetwork), node_color=nodeColor); 
# nx.draw(thisNetwork, with_labels=False, node_color=nodeColor); 
# ax.set_aspect("equal"); 




########################################################################################################################
## Tests for simultaneous diagonalization: 
## 	See AAND2011_lecture10.pdf @/home/brigan/Documents/BCCN/AAND2011/learningAAND/. 
## 

allStatisticsCov = np.cov(allStatisticsArray); 
otherAllStatisticsCov = np.cov(otherAllStatisticsArray); 
A = np.matrix(allStatisticsCov); 
B = np.matrix(otherAllStatisticsCov); 
(eigVals, lEigVectors, rEigVectors) = la.eig(B, A+B, left=True, right=True); 
eigVals = [eigVals[ii] for ii in range(eigVals.size)]; 
eigVects = rEigVectors; 
eigVals = np.real(eigVals); 
eigVects = np.real(eigVects); 

print(eigVects); 




# Projecting data into eigenspace: 
allStatisticsArray_ = np.dot(np.transpose(eigVects), allStatisticsArray); 
otherAllStatisticsArray_ = np.dot(np.transpose(eigVects), otherAllStatisticsArray); 

# Plotting eigenvalues and eigenvectors: 
plt.figure(); 
plt.plot(eigVals); 

plt.figure(); 
plt.imshow(eigVects, interpolation="none", cmap="coolwarm"); 
plt.colorbar(); 

# PC1-PClast: 
fig = plt.figure(); 
ax = fig.add_subplot(111); #, projection='3d'); 
ax.scatter(allStatisticsArray_[1,:], allStatisticsArray_[-1,:], c="black"); 
ax.scatter(otherAllStatisticsArray_[1,:], otherAllStatisticsArray_[-1,:], c="red"); 
ax.set_xlabel("PC1"); 
ax.set_ylabel("PC2"); 
# ax.set_zlabel("PC3"); 



plt.show(); 
# sys.exit(0); 

"""

	script_exploreConnectome.py: 

		Script to test ideas concerning brain connectomes. 

"""


# Importing relevant libraries: 
import numpy as np; 
import matplotlib.pyplot as plt; 
import matplotlib as mplt; 
import os, sys; 
import networkx as nx; 
import helper as h; 
import loadHelper as lh; 
from copy import copy; 
from sklearn.cluster import KMeans; 

# For 3D scatter: 
from mpl_toolkits.mplot3d import Axes3D; 



########################################################################################################################
## Uncomment for MRI connectome network (I have a lof of such connectomes): 

# Next networks are in MRI_234: 
connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_234/"; 
netName = "102008_repeated10_scale250.graphml"; 
# connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_1015/"; 
# netName = "205725_repeated10_scale500.graphml"; 

# Reading network and keeping largest connected component: 
net = nx.read_graphml(connectomeDataPath + netName); 
Gcc = sorted(nx.connected_components(net), key=len, reverse=True); 
net = nx.Graph(net.subgraph(Gcc[0])); 

# Reading node positions: 
nativePositions = {}; 
nativePositions_3D = {}; 
for node in net.nodes(): 
	nativePositions[node] = [net.nodes()[node]["dn_position_x"], 
								net.nodes()[node]["dn_position_y"]]; 
	nativePositions_3D[node] = [net.nodes()[node]["dn_position_x"], 
								net.nodes()[node]["dn_position_y"], 
								net.nodes()[node]["dn_position_z"]]; 

# Reading node regions: 
nodeCorticalSubcortDict = {}; 							# Cortical or subcortical location for each node. 
nodeCorticalSubcorticalInverseDict = {}; 				# Names of nodes in cortical & subcortical positions. 
nodeCorticalSubcorticalInverseDict["cortical"] = []; 
nodeCorticalSubcorticalInverseDict["subcortical"] = []; 
iNodeCorticalSubcorticalInverseDict = {}; 				# Index of nodes in cortical & subcortical positions. 
iNodeCorticalSubcorticalInverseDict["cortical"] = []; 
iNodeCorticalSubcorticalInverseDict["subcortical"] = []; 
nodeRegionsDict = {}; 									# Specific cortex or brain area to which each node belongs. 
nodeRegionInverseDict = {}; 
iNodeRegionInverseDict = {}; 
nodeHemisphereDict = {}; 								# Hemisphere to which each node belongs: 
nodeHemisphereInverseDict = {}; 
nodeHemisphereDict["left"] = []; 
nodeHemisphereDict["right"] = []; 
iNodeHemisphereInverseDict = {}; 
iNodeHemisphereInverseDict["left"] = []; 
iNodeHemisphereInverseDict["right"] = []; 
for (iNode, node) in enumerate(net.nodes()): 
	thisCorticalSubcortical = net.nodes()[node]["dn_region"]; 
	nodeCorticalSubcortDict[node] = thisCorticalSubcortical; 
	nodeCorticalSubcorticalInverseDict[thisCorticalSubcortical] += [node]; 
	iNodeCorticalSubcorticalInverseDict[thisCorticalSubcortical] += [iNode]; 

	thisNodeRegion = net.nodes()[node]["dn_fsname"]; 
	# if ('_' in thisNodeRegion): 
	# 	thisNodeRegion = thisNodeRegion.split('_')[0]; 
	nodeRegionsDict[node] = thisNodeRegion; 
	if (thisNodeRegion in nodeRegionInverseDict.keys()): 
		nodeRegionInverseDict[thisNodeRegion] += [node]; 
		iNodeRegionInverseDict[thisNodeRegion] += [iNode]; 
	else: 
		nodeRegionInverseDict[thisNodeRegion] = [node]; 
		iNodeRegionInverseDict[thisNodeRegion] = [iNode]; 

	thisHemisphere = net.nodes()[node]["dn_hemisphere"]; 
	nodeHemisphereDict[node] = thisHemisphere; 
	nodeHemisphereInverseDict[thisHemisphere] = node; 
	iNodeHemisphereInverseDict[thisHemisphere] = iNode; 



########################################################################################################################
## Performing network analysis: 


# Measuring node properties, select relevant features, normalize distro: 
(nodeList, propertiesDict, includedProperties, excludedProperties) = h.computeNodesProperties(net); 
allPropertiesArray = h.buildPropertiesArray(propertiesDict, includedProperties); 
allPropertiesArray = h.normalizeProperties(allPropertiesArray); 
print("Analysis includes the following properties: "); 
for (iP, thisProperty) in enumerate(includedProperties): 
	print('\t' + str(iP+1) + ": " + thisProperty); 


## Computing correlation matrix and diagonalizing: 
allStatisticsCov = np.cov(allPropertiesArray); 
(eigVals, eigVects) = np.linalg.eig(allStatisticsCov); 
# eigVals = np.real(eigVals); 
# eigVects = np.real(eigVects); 

## Projecting data into eigenspace: 
allPropertiesArray_ = np.dot(np.transpose(eigVects), allPropertiesArray); 

# Using first three PCs as color coding: 
# 	Normalize components to [0,1]; 
valuesRGB0 = h.convertPC2RGB(allPropertiesArray_[0,:]); 
valuesRGB1 = h.convertPC2RGB(allPropertiesArray_[1,:]); 
valuesRGB2 = h.convertPC2RGB(allPropertiesArray_[2,:]); 
# Save hex color values to a list: 
nodeColor = []; 
for (iNode, node) in enumerate(nodeList): 
	nodeColor += [mplt.colors.to_hex([valuesRGB0[iNode], valuesRGB1[iNode], valuesRGB2[iNode]])]; 



## A few plots: 

# PC1-PC2-PC3: 
fig = plt.figure(); 
ax = fig.add_subplot(111, projection='3d'); 
ax.scatter(allPropertiesArray_[0,:], allPropertiesArray_[1,:], allPropertiesArray_[2,:], c=nodeColor); 
ax.set_xlabel("PC1"); 
ax.set_ylabel("PC2"); 
ax.set_zlabel("PC3"); 

# Building array to plot with scatter in 3D: 
sortedPositions = []; 
for node in net.nodes(): 
	sortedPositions += [nativePositions_3D[node]]; 
sortedPositions = np.array(sortedPositions); 

# Proper 3D plot: 
fig = plt.figure(); 
ax = fig.gca(projection='3d'); 
ax.scatter(sortedPositions[:,0], sortedPositions[:,1], sortedPositions[:,2], s=100, ec="w", color=nodeColor); 
plt.xlabel("x-coordinate"); 
plt.ylabel("y-coordinate"); 
ax.set_zlabel("z-coordinate"); 




########################################################################################################################
########################################################################################################################
## Some further analysis: 
## 

nClusters = 5; 
kmeans = KMeans(nClusters).fit(allPropertiesArray_.T); 

# Coloring nodes according to their cluster: 
clusterStyles = {}; 
clusterStyles[0] = 'k'; 
clusterStyles[1] = 'r'; 
clusterStyles[2] = 'g'; 
clusterStyles[3] = 'b'; 
clusterStyles[4] = 'y'; 
clusterStyles[5] = 'm'; 
clusterStyles[6] = 'c'; 
clusterStyles[7] = 'tab:gray'; 

nodeClusterColor = []; 
for (iNode, node) in enumerate(nodeList): 
	nodeClusterColor += [clusterStyles[kmeans.labels_[iNode]]]; 



# Plotting in eigenspace: 
fig = plt.figure(); 
ax = fig.add_subplot(111, projection='3d'); 
ax.scatter(allPropertiesArray_[0,:], allPropertiesArray_[1,:], allPropertiesArray_[2,:], c=nodeClusterColor); 
ax.set_xlabel("PC1"); 
ax.set_ylabel("PC2"); 
ax.set_zlabel("PC3"); 

# Plotting in connectome space: 
fig = plt.figure(); 
ax = fig.gca(projection='3d'); 
ax.scatter(sortedPositions[:,0], sortedPositions[:,1], sortedPositions[:,2], s=100, ec="w", color=nodeClusterColor); 
plt.xlabel("x-coordinate"); 
plt.ylabel("y-coordinate"); 
ax.set_zlabel("z-coordinate"); 



plt.show(); 
sys.exit(0); 



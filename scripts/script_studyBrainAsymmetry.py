"""

	script_studyBrainAsymmetry.py: 

		This is a script to study symmetry and asymmetry across hemispheres in brain networks. 

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
import scipy.io; # To read .mat files! 


# For 3D scatter: 
from mpl_toolkits.mplot3d import Axes3D; 



########################################################################################################################
## Loading human connectomes derived from MRI: 


# # Next networks are in MRI_234: 
connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_1015/"; 
networkName = "100206_repeated10_scale500.graphml"; 
thisNetwork = nx.read_graphml(connectomeDataPath + networkName); 
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

# # For networks in MRI_1015: 
# connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_1015/"; 

nativePositions = {}; 
nativePositions_3D = {}; 
for node in thisNetwork.nodes(): 
	nativePositions[node] = [thisNetwork.nodes()[node]["dn_position_x"], 
								thisNetwork.nodes()[node]["dn_position_y"]]; 
	nativePositions_3D[node] = [thisNetwork.nodes()[node]["dn_position_x"], 
								thisNetwork.nodes()[node]["dn_position_y"], 
								thisNetwork.nodes()[node]["dn_position_z"]]; 



########################################################################################################################
## Performing network analysis: 

# Trivial network info: 
nNodes = len(thisNetwork.nodes()); 

# Measure stuff from nodes: 
(nodeList, nodesProperties, includedProperties, excludedProperties) = h.computeNodesProperties(thisNetwork); 
print("Analysis includes the following properties: "); 
for (iP, thisProperty) in enumerate(includedProperties): 
	print('\t' + str(iP+1) + ": " + thisProperty); 

nProperties = len(includedProperties); 
allPropertiesArray = np.zeros([nProperties, nNodes]); 
dictIStat = {}; 
for (iStat, statistic) in enumerate(includedProperties): 
	allPropertiesArray[iStat,:] = nodesProperties[statistic]; 


## Standardizing distro: 
allPropertiesArray = h.normalizeProperties(allPropertiesArray); 

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



# Computing node counterparts: 
iCounterNodeDict = {}; 
counterNodeDict = {}; 
counterDistanceDict = {}; 
zippedCounterparts = []; 
for targetNode in thisNetwork.nodes(): 
	(iCounterNode, counterNode, counterDistance, allDistances) = h.findCounterhemisphericNode(nativePositions_3D, targetNode); 
	iCounterNodeDict[targetNode] = iCounterNode; 
	counterNodeDict[targetNode] = counterNode; 
	counterDistanceDict[targetNode] = allDistances[iCounterNode]; 
	zippedCounterparts += [(targetNode, counterNode)]; 

print(counterDistanceDict); 

# Computing symmetry and asymmetry indexes: 
nodeSymmetryList = []; 
nodeAntisymmetryList = []; 
for (iNode, node) in enumerate(thisNetwork.nodes()): 
	targetNodeProperties = allPropertiesArray_[:,iNode]; 
	counterNodeProperties = allPropertiesArray_[:,iCounterNodeDict[node]]; 
	nodeSymmetry = np.sum(np.abs(np.subtract(targetNodeProperties, counterNodeProperties))); 
	nodeSymmetryList += [nodeSymmetry]; 
	nodeAntisymmetry = np.sum(np.abs(np.add(targetNodeProperties, counterNodeProperties))); 
	nodeAntisymmetryList += [nodeAntisymmetry]; 

Z_symmetry = max(nodeSymmetryList); 
Z_antisymmetry = max(nodeAntisymmetryList); 
nodeSymmetryColor = []; 
nodeAntisymmetryColor = []; 
nodeSymmetryList_ = []; 
nodeAntisymmetryList_ = []; 
for (iNode, node) in enumerate(thisNetwork.nodes()): 
	if (counterDistanceDict[node]<2.): 
		nodeSymmetryList_ += [nodeSymmetryList[iNode]]; 
		thisNodeSymmetryColor = nodeSymmetryList[iNode]/Z_symmetry; 
		nodeSymmetryColor += [[thisNodeSymmetryColor, thisNodeSymmetryColor, thisNodeSymmetryColor]]; 
		nodeAntisymmetryList_ += [nodeAntisymmetryList[iNode]]; 
		thisNodeAntisymmetryColor = nodeAntisymmetryList[iNode]/Z_antisymmetry; 
		nodeAntisymmetryColor += [[thisNodeAntisymmetryColor, thisNodeAntisymmetryColor, thisNodeAntisymmetryColor]]; 
	else: 
		nodeSymmetryColor += [[0., 0., 0.]]; 
		nodeAntisymmetryColor += [[0., 0., 0.]]; 





# Plotting asymmetry histogram: 
plt.figure(); 
plt.hist(nodeSymmetryList_, 100); 
plt.xlabel("Node symmetry"); 
plt.ylabel("Frequency"); 

# Plotting in network space in 2D if they have native coordinates: 
if "nativePositions" in locals(): 
	fig = plt.figure(); 
	ax = fig.add_subplot(111); 
	nx.draw(thisNetwork, with_labels=False, node_color=nodeSymmetryColor, pos=nativePositions, edge_color="tab:gray"); # Some connectomes might have a native position. 
	ax.set_aspect("equal"); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
	plt.title("Node symmetry"); 

# Plotting in network space in 3D if they have such native coordinates: 
if "nativePositions_3D" in locals(): 

	# Position cannot be given as a dictionary: 
	sortedPositions = []; 
	for node in thisNetwork.nodes(): 
		sortedPositions += [nativePositions_3D[node]]; 
	sortedPositions = np.array(sortedPositions); 

	# Proper plot: 
	fig = plt.figure(); 
	ax = fig.gca(projection='3d'); 
	ax.scatter(sortedPositions[:,0], sortedPositions[:,1], sortedPositions[:,2], s=100, ec="w", color=nodeSymmetryColor); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
	ax.set_zlabel("z-coordinate"); 
	plt.title("Node symmetry"); 

	# # Plot the edges (commented by now -- it takes too much RAM). 
	for edge in zippedCounterparts:
		xCoor = (nativePositions_3D[edge[0]][0], nativePositions_3D[edge[1]][0]); 
		yCoor = (nativePositions_3D[edge[0]][1], nativePositions_3D[edge[1]][1]); 
		zCoor = (nativePositions_3D[edge[0]][2], nativePositions_3D[edge[1]][2]); 
		plt.plot(xCoor, yCoor, zCoor, color="tab:gray"); 


# Plotting antisymmetry histogram: 
plt.figure(); 
plt.hist(nodeAntisymmetryList_, 100); 
plt.xlabel("Node antisymmetry"); 
plt.ylabel("Frequency"); 

# Plotting in network space in 2D if they have native coordinates: 
if "nativePositions" in locals(): 
	fig = plt.figure(); 
	ax = fig.add_subplot(111); 
	nx.draw(thisNetwork, with_labels=False, node_color=nodeAntisymmetryColor, pos=nativePositions, edge_color="tab:gray"); # Some connectomes might have a native position. 
	ax.set_aspect("equal"); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
	plt.title("Node antisymmetry"); 


# Plotting in network space in 3D if they have such native coordinates: 
if "nativePositions_3D" in locals(): 

	# Position cannot be given as a dictionary: 
	sortedPositions = []; 
	for node in thisNetwork.nodes(): 
		sortedPositions += [nativePositions_3D[node]]; 
	sortedPositions = np.array(sortedPositions); 

	# Proper plot: 
	fig = plt.figure(); 
	ax = fig.gca(projection='3d'); 
	ax.scatter(sortedPositions[:,0], sortedPositions[:,1], sortedPositions[:,2], s=100, ec="w", color=nodeAntisymmetryColor); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
	ax.set_zlabel("z-coordinate"); 
	plt.title("Node antisymmetry"); 




plt.show(); 
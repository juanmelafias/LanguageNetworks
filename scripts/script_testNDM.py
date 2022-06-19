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
import matplotlib.pyplot as plt; 
import matplotlib as mplt; 
import networkx as nx; 
from copy import copy; 
# For 3D scatter: 
from mpl_toolkits.mplot3d import Axes3D; 

# Importing libraries for I/O and system communication: 
import os, sys; 
import pickle as pkl; 
import scipy.io as sio; # To read .mat files! and .mnx files! 

# Importing functions for clustering: 
from sklearn.cluster import KMeans; 
import scipy.cluster.hierarchy as spc; 

# Importing homebrew libraries: 
import helper as h; 
import loadHelper as lh; 





########################################################################################################################
########################################################################################################################
#### Loading network: 
#### 

# Path to store properties -- common for all networks: 
netPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/NetworkProperties/"; 
basePicsPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Pics/"; 

netNames_humanMRI = ["MRI_234_993675", "MRI_234_958976", "MRI_234_100408", "MRI_234_959574", "MRI_234_100206"]; 
netNames_humanMRI += ["MRI_234_100307", "MRI_234_100610", "MRI_234_101006", "MRI_234_101107", "MRI_234_101309"]; 
netNames_humanMRI += ["MRI_234_101410", "MRI_234_101915"]; 

preteinProteinNets = ["proteinArabidopsisThaliana", "proteinMusMusculus"]; 

netNames = ["syntaxNetwork", "CNB_net", "collabNet", "macaqueBrain", "macaqueInterCortex", "catTract", "mouseVisual2"]; 
netNames += ["netCElegans", "netDeutscheAutobahn", "airports"]; 
netNames += netNames_humanMRI; 
# netNames += preteinProteinNets; 

netName = "randomWS"; 
picsPath = basePicsPath + "Pics_" + netName + '/'; 
if (not(os.path.isdir(picsPath))): 
	os.mkdir(picsPath); 

if ("random" not in netName): 
	(thisNetwork, metaDataDict) = lh.masterLoader(netName); 
	if ("nativePositions" in metaDataDict.keys()): 
		nativePositions = metaDataDict["nativePositions"]; 
	if ("nativePositions_3D" in metaDataDict.keys()): 
		nativePositions_3D = metaDataDict["nativePositions_3D"]; 
else: 
	args = {}; 
	args["nNodes"] = 200; 
	args["nNeighbors"] = 4; 
	args["pRewire"] = 0.05; 
	thisNetwork = lh.generateRandomNetwork(netName, args); 


# # Loading largest connected component and number of nodes: 
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 
nNodes = len(thisNetwork.nodes()); 
nEdges = thisNetwork.number_of_edges(); 

# # Uncomment for quick network representation: 
# fig = plt.figure(); 
# ax = fig.add_subplot(111); 
# nx.draw(thisNetwork, with_labels=False, pos=nx.kamada_kawai_layout(thisNetwork)); 
# ax.set_aspect("equal"); 

# print(nNodes); 
# print(nEdges); 

# plt.show(); 
# sys.exit(0);



########################################################################################################################
########################################################################################################################
#### Perform analysis on nodes: 
#### 

## Obtaining network propeties: 
# 	To avoid re-computing all the time, we now save network properties in individual files that can be accessed. 
# 	If computations have already been performed for some network, we read them from the files. 
# 	Otherwise, we need to compute and store them. 

fNeighborMean = True; 
fNeighborStd = True; 
if (("random" not in netName) and (os.path.isfile(netPath + netName + "_nodeList.csv")) 
								and (os.path.isfile(netPath + netName + "_properties.pkl"))): 
	# Files already exist with properties that have been computed. We can proceed with these: 
	# (nodeList, propertiesDict) = h.readNetworkProperties(netName, netPath); 
	(nodeList, propertiesDict) = h.readNetworkProperties(netName, netPath, fNeighborMean, fNeighborStd); 
	(includedProperties, excludedProperties) = h.findPathologicalProperties(propertiesDict); 
else: 
	# Properties have not been saved for this network and need to be computed: 
	(nodeList, propertiesDict, includedProperties, excludedProperties) = h.computeNodesProperties(thisNetwork, 
																						fNeighborMean, fNeighborStd); 
	if ("random" not in netName): 
		h.writeNetworkProperties(netName, netPath, nodeList, propertiesDict); 

# # Just in case, we keep these lines of code in case we wish to manually force the re-computation for some network: 
# (nodeList, propertiesDict, includedProperties, excludedProperties) = h.computeNodesProperties(thisNetwork); 
# # (nodeList, propertiesDict, includedProperties, excludedProperties) = h.computeNodesProperties(thisNetwork, False, False); 
# h.writeNetworkProperties(netName, netPath, nodeList, propertiesDict); 


# In either case, we retain only those properties that are not pathological. 
# We build a numpy array to work with them: 
includedPropertiesArray = h.buildPropertiesArray(propertiesDict, includedProperties); 
includedPropertiesArray = h.normalizeProperties(includedPropertiesArray); 


print("Analysis includes the following properties: "); 
for (iP, thisProperty) in enumerate(includedProperties): 
	print('\t' + str(iP+1) + ": " + thisProperty); 

# for (iP, thisProperty) in enumerate(includedProperties): 
# 	print('\t' + str(iP+1) + ": " + thisProperty); 
# 	print("\t\t" + str(includedPropertiesArray[iP])); 

## Computing correlation matrix and diagonalizing: 
allStatisticsCov = np.cov(includedPropertiesArray); 
(eigVals, eigVects) = np.linalg.eig(allStatisticsCov); 
eigVals = np.real(eigVals); 
eigVects = np.real(eigVects); 


# Computing PCs with information above noise level according to ref: 
# 	Donoho DL, Gavish M. 
# 	The optimal hard threshold for singular values is 4/√3. 
# 	arXiv preprint arXiv:1305.5870, (2013).
(noiseThreshold, nKeep) = h.computeComponentsAboveNoise(eigVals); 
print("Noise-trucating PC value is: " + str(noiseThreshold)); 
print("According to this, optimal number of PCs kept is: " + str(nKeep)); 
print("This is a fraction " + str(float(nKeep)/len(eigVals)) + " of eigenvalues. "); 

# Plotting covariance matrix: 
plt.figure(); 
plt.imshow(allStatisticsCov, interpolation="none"); 
plt.colorbar(); 

# Plotting eigenvectors: 
plt.figure(); 
plt.imshow(eigVects, interpolation="none", cmap="coolwarm"); 
plt.colorbar(); 


# Computing and plotting variance explained: 
(varianceExplained, varianceExplained_cumul) = h.varianceExplained(eigVals); 

plt.figure(); 
plt.plot(varianceExplained); 

plt.figure(); 
plt.plot(varianceExplained_cumul); 



## Projecting data into eigenspace: 
includedPropertiesArray_ = np.dot(np.transpose(eigVects), includedPropertiesArray); 

# Using first three PCs as color coding: 
# 	Normalize components to [0,1]; 
valuesRGB0 = h.convertPC2RGB(includedPropertiesArray_[0,:]); 
valuesRGB1 = h.convertPC2RGB(includedPropertiesArray_[1,:]); 
valuesRGB2 = h.convertPC2RGB(includedPropertiesArray_[2,:]); 
# Save hex color values to a list: 
nodeColor = []; 
for (iNode, node) in enumerate(nodeList): 
	nodeColor += [mplt.colors.to_hex([valuesRGB0[iNode], valuesRGB1[iNode], valuesRGB2[iNode]])]; 


# PC1-PC2: 
fig = plt.figure(); 
ax = fig.add_subplot(111); 
plt.scatter(includedPropertiesArray_[0,:], includedPropertiesArray_[1,:], c=nodeColor); 
plt.xlabel("PC1"); 
plt.ylabel("PC2"); 
plt.title("Nodes projected in PCs"); 
fig.savefig(picsPath + "projection_PCs1-2.pdf"); 

# PC1-PC3: 
fig = plt.figure(); 
plt.scatter(includedPropertiesArray_[0,:], includedPropertiesArray_[2,:], c=nodeColor); 
plt.xlabel("PC1"); 
plt.ylabel("PC3"); 
plt.title("Nodes projected in PCs"); 
fig.savefig(picsPath + "projection_PCs1-3.pdf"); 

# PC1-PC2-PC3: 
fig = plt.figure(); 
ax = fig.add_subplot(111, projection='3d'); 
ax.scatter(includedPropertiesArray_[0,:], includedPropertiesArray_[1,:], includedPropertiesArray_[2,:], c=nodeColor); 
ax.set_xlabel("PC1"); 
ax.set_ylabel("PC2"); 
ax.set_zlabel("PC3"); 
plt.title("Nodes projected in PCs"); 
fig.savefig(picsPath + "projection_PCs1-2-3.pdf"); 


# Plotting in network space: 
fig = plt.figure(); 
ax = fig.add_subplot(111); 
nx.draw(thisNetwork, with_labels=False, pos=nx.kamada_kawai_layout(thisNetwork), node_color=nodeColor, edge_color="tab:gray"); 
ax.set_aspect("equal"); 
plt.title("PC colors projected in network layout"); 
fig.savefig(picsPath + "networkColoredWithPCs_netLayout.pdf"); 

# Plotting in network space in 2D if they have native coordinates: 
if "nativePositions" in locals(): 
	fig = plt.figure(); 
	ax = fig.add_subplot(111); 
	nx.draw(thisNetwork, with_labels=False, node_color=nodeColor, pos=nativePositions, edge_color="tab:gray"); # Some connectomes might have a native position. 
	ax.set_aspect("equal"); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
	plt.title("PC colors projected in real space"); 
	fig.savefig(picsPath + "networkColoredWithPCs_geometry2D.pdf"); 


# Plotting in network space in 3D if they have such native coordinates: 
if "nativePositions_3D" in locals(): 

	# Position cannot be given as a dictionary: 
	sortedPositions = []; 
	for node in thisNetwork.nodes(): 
		sortedPositions += [nativePositions_3D[node]]; 
	sortedPositions = np.array(sortedPositions); 

	# Proper plot: 
	fig = plt.figure(); 
	ax = plt.axes(projection='3d'); 
	ax.scatter(sortedPositions[:,0], sortedPositions[:,1], sortedPositions[:,2], s=100, ec="w", color=nodeColor); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
	ax.set_zlabel("z-coordinate"); 
	plt.title("PC colors projected in real space"); 
	fig.savefig(picsPath + "networkColoredWithPCs_geometry3D.pdf"); 

	# # Plot the edges (commented by now -- it takes too much RAM). 
	# for edge in thisNetwork.edges():
	# 	xCoor = (nativePositions_3D[edge[0]][0], nativePositions_3D[edge[1]][0]); 
	# 	yCoor = (nativePositions_3D[edge[0]][1], nativePositions_3D[edge[1]][1]); 
	# 	zCoor = (nativePositions_3D[edge[0]][2], nativePositions_3D[edge[1]][2]); 
	# 	plt.plot(xCoor, yCoor, zCoor, color="tab:gray"); 


# Plotting nodes most similar to a target node: 
# iTarget = 155; 
iTarget = 10; 
(distanceToTarget, distanceToTarget_) = h.distanceToTargetNode(includedPropertiesArray_, iTarget); 
colorDistance = [[elem, elem, elem] for elem in distanceToTarget_]; 
fig = plt.figure(); 
ax = fig.add_subplot(111); 
if "nativePositions" in locals(): 
	nx.draw(thisNetwork, with_labels=False, node_color=colorDistance, pos=nativePositions, edge_color="tab:gray"); # Some connectomes might have a native position. 
	ax.set_aspect("equal"); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
else: 
	nx.draw(thisNetwork, with_labels=False, pos=nx.kamada_kawai_layout(thisNetwork), node_color=colorDistance, edge_color="tab:gray"); 
	ax.set_aspect("equal"); 

# Same, but in 3D if coordinates are provided: 
if "nativePositions_3D" in locals(): 
	fig = plt.figure(); 
	ax = plt.axes(projection='3d'); 
	ax.scatter(sortedPositions[:,0], sortedPositions[:,1], sortedPositions[:,2], s=100, ec="w", color=colorDistance); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
	ax.set_zlabel("z-coordinate"); 



########################################################################################################################
########################################################################################################################
## k-means clustering: 
## 

nClusters = 5; 
# nClusters = 8; 
kmeans = KMeans(nClusters).fit(includedPropertiesArray_.T); 

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
ax.scatter(includedPropertiesArray_[0,:], includedPropertiesArray_[1,:], includedPropertiesArray_[2,:], c=nodeClusterColor); 
ax.set_xlabel("PC1"); 
ax.set_ylabel("PC2"); 
ax.set_zlabel("PC3"); 
plt.title("Clusters (k-means) in eigenspace"); 
fig.savefig(picsPath + "kMeans_eigenspace.pdf"); 


fig = plt.figure(); 
ax = fig.add_subplot(111); 
if "nativePositions" in locals(): 
	nx.draw(thisNetwork, with_labels=False, node_color=nodeClusterColor, pos=nativePositions, edge_color="tab:gray"); # Some connectomes might have a native position. 
	ax.set_aspect("equal"); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
else: 
	nx.draw(thisNetwork, with_labels=False, pos=nx.kamada_kawai_layout(thisNetwork), node_color=nodeClusterColor, edge_color="tab:gray"); 
	ax.set_aspect("equal"); 

plt.title("Clusters (k-means) in network layout"); 
fig.savefig(picsPath + "kMeans_netLayout.pdf"); 


# Same, but in 3D if coordinates are provided: 
if "nativePositions_3D" in locals(): 
	fig = plt.figure(); 
	ax = plt.axes(projection='3d'); 
	ax.scatter(sortedPositions[:,0], sortedPositions[:,1], sortedPositions[:,2], s=100, ec="w", color=nodeClusterColor); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
	ax.set_zlabel("z-coordinate"); 

	plt.title("Clusters (k-means) in real space"); 
	fig.savefig(picsPath + "kMeans_geometry3D.pdf"); 







########################################################################################################################
########################################################################################################################
## Dendograms to visualize closeness between properties and data: 
## 

## Dendograms for properties:  

# From correlations to distances: 
pdist = spc.distance.pdist(allStatisticsCov); 
propertiesLinkage = spc.linkage(pdist, method='complete'); 

fig = plt.figure(); 
spc.dendrogram(propertiesLinkage, orientation="right", labels=includedProperties); 
plt.xlabel("Distance"); 
plt.ylabel("Node properties"); 
plt.title("Properties dendrogram"); 
fig.savefig(picsPath + "propertiesDendogram.pdf"); 


## Dendograms for data: 
nodesLinkage = spc.linkage(includedPropertiesArray_.T, 'ward'); 

distanceThreshold = 45; 
fig = plt.figure(); 
spc.dendrogram(nodesLinkage, orientation="right", color_threshold=distanceThreshold); 
plt.xlabel("Distance"); 
plt.ylabel("Nodes"); 
plt.title("Nodes dendrogram"); 
fig.savefig(picsPath + "nodesDendogram.pdf"); 


# Coloring according to clusters: 
# nodeClusters = spc.fcluster(nodesLinkage, distanceThreshold, criterion='distance'); 
nClusters = 5; 
nodeClusters = spc.fcluster(nodesLinkage, nClusters, criterion="maxclust"); 
nodeClusterColor = []; 
for (iNode, node) in enumerate(nodeList): 
	nodeClusterColor += [clusterStyles[nodeClusters[iNode]-1]]; 


# Plotting in eigenspace: 
fig = plt.figure(); 
ax = fig.add_subplot(111, projection='3d'); 
ax.scatter(includedPropertiesArray_[0,:], includedPropertiesArray_[1,:], includedPropertiesArray_[2,:], c=nodeClusterColor); 
ax.set_xlabel("PC1"); 
ax.set_ylabel("PC2"); 
ax.set_zlabel("PC3"); 
plt.title("Clusters (dendogram) in eigenspace"); 
fig.savefig(picsPath + "dendogramClusters_eigenspace.pdf"); 



fig = plt.figure(); 
ax = fig.add_subplot(111); 
if "nativePositions" in locals(): 
	nx.draw(thisNetwork, with_labels=False, node_color=nodeClusterColor, pos=nativePositions, edge_color="tab:gray"); # Some connectomes might have a native position. 
	ax.set_aspect("equal"); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
else: 
	nx.draw(thisNetwork, with_labels=False, pos=nx.kamada_kawai_layout(thisNetwork), node_color=nodeClusterColor, edge_color="tab:gray"); 
	ax.set_aspect("equal"); 
plt.title("Clusters (dendogram) in network layout"); 
fig.savefig(picsPath + "dendogramClusters_netLayout.pdf"); 



# Same, but in 3D if coordinates are provided: 
if "nativePositions_3D" in locals(): 
	fig = plt.figure(); 
	ax = plt.axes(projection='3d'); 
	ax.scatter(sortedPositions[:,0], sortedPositions[:,1], sortedPositions[:,2], s=100, ec="w", color=nodeClusterColor); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
	ax.set_zlabel("z-coordinate"); 
	plt.title("Clusters (dendogram) in real space"); 
	fig.savefig(picsPath + "dendogramClusters_geometry3D.pdf"); 


plt.show(); 
sys.exit(0); 


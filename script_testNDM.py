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
import os, sys; 
import networkx as nx; 
import pickle as pkl; 
from copy import copy; 
import scipy.io as sio; # To read .mat files! and .mnx files! 
from sklearn.cluster import KMeans; 

import helper as h; 
import loadHelper as lh; 

# For 3D scatter: 
from mpl_toolkits.mplot3d import Axes3D; 




# Path to store properties -- common for all networks: 
netPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/NetworkProperties/"; 




# ########################################################################################################################
# ## Uncomment for syntax network: 

# # Metadata to load networks: 
# dataPathMaster = "/home/brigan/Desktop/Research_IFISC/LanguageMorphospaces/Data"; 
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

# # thisNetwork = nx.erdos_renyi_graph(200, 0.1); 
# thisNetwork = nx.watts_strogatz_graph(200, 4, 0.05); 
# # thisNetwork = nx.barabasi_albert_graph(200, 2); 
# # thisNetwork = nx.bipartite.gnmk_random_graph(50,50,200); 


# #######################################################################################################################
# # Uncomment for CNB network: 

# dataPath = "/home/brigan/Desktop/Research_CNB/Misc/CNB_net/Code/Output/"; 
# netName = "CNB_net"; 

# # Reading edges: 
# fIn	= open(dataPath + "edges.csv", 'r'); 
# edges = []; 
# nodes = []; 
# allLines = fIn.read().splitlines(); 
# nPapers = {}; 
# nCollaborations = {}; 
# for line in allLines: 
# 	thisEdge = line.split(', '); 
# 	edges += [(thisEdge[0], thisEdge[1])]; 

# # Building network from edges: 
# thisNetwork = nx.Graph(); 
# thisNetwork.add_edges_from(edges); 


# #######################################################################################################################
# # Uncomment for Collaboration networks: 

# dataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Collaborations/"; 
# networkFileName = "ca-CSphd.mtx"; 

# fIn = open(dataPath + networkFileName, 'r'); 
# dL = fIn.read().splitlines(); 
# fIn.close(); 

# dL = dL[3::]; 
# edges = []; 
# for ll in dL: 
# 	thisSplitLine = ll.split(' '); 
# 	edges += [(int(thisSplitLine[0]), int(thisSplitLine[1]))]; 

# # Building network from edges: 
# thisNetwork = nx.Graph(); 
# thisNetwork.add_edges_from(edges); 



# #######################################################################################################################
# # Uncomment for Ecology networks: 

# dataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Ecology/"; 
# networkFileName = "eco-foodweb-baywet.edges"; 

# fIn = open(dataPath + networkFileName, 'r'); 
# dL = fIn.read().splitlines(); 
# fIn.close(); 

# dL = dL[2::]; 
# edges = []; 
# for ll in dL: 
# 	thisSplitLine = ll.split(' '); 
# 	edges += [(int(thisSplitLine[0]), int(thisSplitLine[1]))]; 

# # Building network from edges: 
# thisNetwork = nx.Graph(); 
# thisNetwork.add_edges_from(edges); 



########################################################################################################################
## Uncomment for MRI connectome network (I have a lof of such connectomes): 


# # Next networks are in MRI_234: 
connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_234/"; 
# thisNetwork = nx.read_graphml(connectomeDataPath + "993675_repeated10_scale250.graphml"); 
# netName = "MRI_234_993675"; 
# thisNetwork = nx.read_graphml(connectomeDataPath + "958976_repeated10_scale250.graphml"); # Check out this network!! Compare to others! 
# netName = "MRI_234_958976"; 
thisNetwork = nx.read_graphml(connectomeDataPath + "959574_repeated10_scale250.graphml"); # Check out this network!! Compare to others! 
netName = "MRI_234_959574"; 

# # Next networks are in MRI_1015: 
# connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_1015/"; 
# # thisNetwork = nx.read_graphml(connectomeDataPath + "101915_repeated10_scale500.graphml"); # Check out this network!! Compare to others! 
# thisNetwork = nx.read_graphml(connectomeDataPath + "987074_repeated10_scale500.graphml"); # Check out this network!! Compare to others! 

# ## Next networks are in MRI_Lobes: 
# connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_Lobes/"; 
# thisNetwork = nx.read_graphml(connectomeDataPath + "occipital-113922_connectome_scale500.graphml"); # Check out this network!! Compare to others! 

nativePositions = {}; 
nativePositions_3D = {}; 
for node in thisNetwork.nodes(): 
	nativePositions[node] = [thisNetwork.nodes()[node]["dn_position_x"], 
								thisNetwork.nodes()[node]["dn_position_y"]]; 
	nativePositions_3D[node] = [thisNetwork.nodes()[node]["dn_position_x"], 
								thisNetwork.nodes()[node]["dn_position_y"], 
								thisNetwork.nodes()[node]["dn_position_z"]]; 




# ########################################################################################################################
# ## Uncomment for Macaque brain: 

# connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Macaque/"; 
# thisNetwork = nx.read_graphml(connectomeDataPath + "rhesus_brain_1.graphml"); 
# thisNetwork = thisNetwork.to_undirected(); 


# # ########################################################################################################################
# # ## Uncomment for Drosophila: 

# connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Drosophila/"; 
# mat = scipy.io.loadmat(connectomeDataPath + "mac95.mat"); 
# for elem in mat.keys(): 
# 	print(elem); 
# 	print(mat[elem]); 


# sys.exit(0); 

# thisNetwork = nx.read_graphml(connectomeDataPath + "rhesus_brain_1.graphml"); 
# thisNetwork = thisNetwork.to_undirected(); 


# ########################################################################################################################
# ## Uncomment for Mouse Visual Cortex: 

# connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Mouse/VisualCortex/"; 
# thisNetwork = nx.read_graphml(connectomeDataPath + "mouse_visual.cortex_2.graphml"); 
# thisNetwork = thisNetwork.to_undirected(); 


# ########################################################################################################################
# ## Uncomment for C elegans: 

# connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Celegans/Celegans131/"; 

# # # Simple-to-load C elegans network: 
# # thisNetwork = nx.read_graphml(connectomeDataPath + "c.elegans_neural.male_1.graphml"); 
# # thisNetwork = thisNetwork.to_undirected(); 

# # # Complex-to-load C elegans network, but with positions! 
# # thisNetwork = nx.read_edgelist(connectomeDataPath + "C-elegans-frontal.txt", create_using=nx.Graph(), nodetype=int); 
# # nodePositions = np.genfromtxt(connectomeDataPath + "C-elegans-frontal-meta.csv", delimiter=',', skip_header=1, usecols=[2, 3]); 
# # nNodes = len(thisNetwork.nodes()); 
# # nativePositions = {}; 
# # for iNode in range(nNodes): 
# # 	nativePositions[iNode] = nodePositions[iNode,:]; 

# mat = sio.loadmat(connectomeDataPath + "celegans131.mat"); 
# thisNetwork = nx.convert_matrix.from_numpy_matrix(mat["celegans131matrix"]); 
# nativePositions = {}; 
# for node in thisNetwork.nodes(): 
# 	nativePositions[node] = mat["celegans131positions"][node,:]; 




# ########################################################################################################################
# ## Uncomment for Deutsche Autobahn: 

# dieAutobahnDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Autobahn/"; 


# mat = scipy.io.loadmat(dieAutobahnDataPath + "autobahn.mat"); 
# thisNetwork = nx.convert_matrix.from_numpy_matrix(mat["auto1168"]); 
# # for key in mat.keys(): 
# # 	print(key); 
# # 	print(mat[key]); 

# labeledCities = []; 
# for elem in np.squeeze(mat["auto1168labels"]): 
# 	labeledCities += [elem[0]]; 

# # # nativePositions = {}; 
# # # for node in thisNetwork.nodes(): 
# # # 	nativePositions[node] = mat["celegans131positions"][node,:]; 



# # sys.exit(); 



# ########################################################################################################################
# ## Uncomment for airport connections: 

# # Data was extracted from: https://www.dynamic-connectome.org/resources/

# airportDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Airport/"; 

# # Loading network: 
# mat = sio.loadmat(airportDataPath + "air500.mat"); 
# thisNetwork = nx.convert_matrix.from_numpy_matrix(mat["air500matrix"]); 
# nodeNames = np.squeeze(mat["air500labels"]); 
# # Reasigining node names to actual labels: 
# mapping = {}; 
# for (iName, name) in enumerate(nodeNames): 
# 	mapping[iName] = name[0]; 
# thisNetwork = nx.relabel_nodes(thisNetwork, mapping); 


# # Loading airpot metadata: 
# fIn = open(airportDataPath + "shorterMeta.csv", 'r'); 
# airportMeta = fIn.read().splitlines(); 
# fIn.close(); 
# nativePositions = {}; 
# for line in airportMeta: 
# 	splitLine = line.split(','); 
# 	nativePositions[splitLine[2]] = [float(splitLine[1]), float(splitLine[0])]; 








########################################################################################################################
########################################################################################################################
#### Perform analysis on nodes: 
#### 


# # Loading largest connected component and number of nodes: 
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 
nNodes = len(thisNetwork.nodes()); 

# print(nNodes); 
# sys.exit(); 


## Obtaining network propeties: 
# 	To avoid re-computing all the time, we now save network properties in individual files that can be accessed. 
# 	If computations have already been performed for some network, we read them from the files. 
# 	Otherwise, we need to compute and store them. 

if (os.path.isfile(netPath + netName + "_nodeList.csv") and os.path.isfile(netPath + netName + "_properties.pkl")): 
	# Files already exist with properties that have been computed. We can proceed with these: 
	(nodeList, propertiesDict) = h.readNetworkProperties(netName, netPath); 
	# (nodeList, propertiesDict) = h.readNetworkProperties(netName, netPath, False, False); 
	(includedProperties, excludedProperties) = h.findPathologicalProperties(propertiesDict); 
else: 
	# Properties have not been saved for this network and need to be computed: 
	(nodeList, propertiesDict, includedProperties, excludedProperties) = h.computeNodesProperties(thisNetwork); 
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

# PC1-PC3: 
fig = plt.figure(); 
plt.scatter(includedPropertiesArray_[0,:], includedPropertiesArray_[2,:], c=nodeColor); 
plt.xlabel("PC1"); 
plt.ylabel("PC3"); 

# PC1-PC2-PC3: 
fig = plt.figure(); 
ax = fig.add_subplot(111, projection='3d'); 
ax.scatter(includedPropertiesArray_[0,:], includedPropertiesArray_[1,:], includedPropertiesArray_[2,:], c=nodeColor); 
ax.set_xlabel("PC1"); 
ax.set_ylabel("PC2"); 
ax.set_zlabel("PC3"); 


# Plotting in network space: 
fig = plt.figure(); 
ax = fig.add_subplot(111); 
nx.draw(thisNetwork, with_labels=False, pos=nx.kamada_kawai_layout(thisNetwork), node_color=nodeColor, edge_color="tab:gray"); 
ax.set_aspect("equal"); 

# Plotting in network space in 2D if they have native coordinates: 
if "nativePositions" in locals(): 
	fig = plt.figure(); 
	ax = fig.add_subplot(111); 
	nx.draw(thisNetwork, with_labels=False, node_color=nodeColor, pos=nativePositions, edge_color="tab:gray"); # Some connectomes might have a native position. 
	ax.set_aspect("equal"); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 

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

	# # Plot the edges (commented by now -- it takes too much RAM). 
	# for edge in thisNetwork.edges():
	# 	xCoor = (nativePositions_3D[edge[0]][0], nativePositions_3D[edge[1]][0]); 
	# 	yCoor = (nativePositions_3D[edge[0]][1], nativePositions_3D[edge[1]][1]); 
	# 	zCoor = (nativePositions_3D[edge[0]][2], nativePositions_3D[edge[1]][2]); 
	# 	plt.plot(xCoor, yCoor, zCoor, color="tab:gray"); 


# Plotting nodes most similar to a target node: 
# iTarget = 155; 
iTarget = 100; 
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


# Same, but in 3D if coordinates are provided: 
if "nativePositions_3D" in locals(): 
	fig = plt.figure(); 
	ax = plt.axes(projection='3d'); 
	ax.scatter(sortedPositions[:,0], sortedPositions[:,1], sortedPositions[:,2], s=100, ec="w", color=nodeClusterColor); 
	plt.xlabel("x-coordinate"); 
	plt.ylabel("y-coordinate"); 
	ax.set_zlabel("z-coordinate"); 



plt.show(); 
sys.exit(0); 

# # Saving data in eigenspace: 
# np.savetxt(os.path.join(dataPathMaster, "Results/DataForPCA/dataEigenSpace.csv"), includedPropertiesArray_, delimiter=", "); 
# np.savetxt(os.path.join(dataPathMaster, "Results/DataForPCA/eigenValues.csv"), eigVals, delimiter=", "); 
# np.savetxt(os.path.join(dataPathMaster, "Results/DataForPCA/eigenVectors.csv"), eigVects, delimiter=", "); 
# np.savetxt(os.path.join(dataPathMaster, "Results/DataForPCA/covariance.csv"), allStatisticsCov, delimiter=", "); 

# ## Saving data to files: 
# np.savetxt(os.path.join(dataPathMaster, "Results/DataForPCA/data.csv"), includedPropertiesArray, delimiter=", "); 
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





# # Example for equal ratio: 
# fig = plt.figure(); 
# ax = fig.add_subplot(111); 
# nx.draw(thisNetwork, pos=nativePositions); 
# ax.set_aspect("equal"); 
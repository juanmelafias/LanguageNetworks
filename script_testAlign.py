"""

	script_testAlign.py: 

		Quick script to test and debug the align function. 

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


####################
## Loading networks: 

# Path to store properties -- common for all networks: 
netPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/NetworkProperties/"; 


netNames_humanMRI = ["MRI_234_993675", "MRI_234_958976", "MRI_234_100408", "MRI_234_959574", "MRI_234_100206"]; 
netNames_humanMRI += ["MRI_234_100307", "MRI_234_100610", "MRI_234_101006", "MRI_234_101107", "MRI_234_101309"]; 
netNames_humanMRI += ["MRI_234_101410", "MRI_234_101915"]; 
netNames = ["syntaxNetwork", "CNB_net", "collabNet", "catTract", "mouseVisual2", "netCElegans", "netDeutscheAutobahn"]; 
netNames += ["airports"]; 
# netNames += netNames_humanMRI; 

netsToLoad = netNames_humanMRI; 
# netsToLoad = ["CNB_net", "collabNet", "MRI_234_993675", "MRI_234_958976", "MRI_234_959574", "macaqueBrain"]; 
# netsToLoad += ["netCElegans", "airports"]; 
nNetworks = len(netsToLoad); 
networksList = []; 
for netName in netsToLoad: 
	(thisNet, thisMetaDataDict) = lh.masterLoader(netName); 
	Gcc = sorted(nx.connected_components(thisNet), key=len, reverse=True); 
	thisNet = nx.Graph(thisNet.subgraph(Gcc[0])); 
	networksList += [copy(thisNet)];


###########################################
## Loading or computing network properties:

fNeighborMean = False; 
fNeighborStd = False; 

includedPropertiesList = []; 
includedPropertiesArrayList = []; 
for (netName, net) in zip(netsToLoad, networksList): 
	print(netName); 
	if (("random" not in netName) and (os.path.isfile(netPath + netName + "_nodeList.csv")) 
									and (os.path.isfile(netPath + netName + "_properties.pkl"))): 
		# Files already exist with properties that have been computed. We can proceed with these: 
		(nodeList, propertiesDict) = h.readNetworkProperties(netName, netPath, fNeighborMean, fNeighborStd); 
		(includedProperties, excludedProperties) = h.findPathologicalProperties(propertiesDict); 
	else: 
		# Properties have not been saved for this network and need to be computed: 
		(nodeList, propertiesDict, includedProperties, excludedProperties) = h.computeNodesProperties(net, 
																							fNeighborMean, fNeighborStd); 
		if ("random" not in netName): 
			h.writeNetworkProperties(netName, netPath, nodeList, propertiesDict); 

	includedPropertiesArray = h.buildPropertiesArray(propertiesDict, includedProperties); 
	includedPropertiesArray = h.normalizeProperties(includedPropertiesArray); 

	includedPropertiesList += [copy(includedProperties)]; 
	includedPropertiesArrayList += [copy(includedPropertiesArray)]; 


##################################################
## Computing correlation matrix and diagonalizing: 

eigVectsList = []; 
for includedPropertiesArray in includedPropertiesArrayList: 
	allStatisticsCov = np.cov(includedPropertiesArray); 
	(eigVals, eigVects) = np.linalg.eig(allStatisticsCov); 
	eigVals = np.real(eigVals); 
	eigVects = np.real(eigVects); 
	eigVectsList += [copy(eigVects)]; 


#################################################### 
## Computing best matches and flipping if necessary: 
## 	First network is considered as refernece. 

refEigenbasis = eigVectsList[0]; 
refIncludedProperties = includedPropertiesList[0]; 
for iNet in range(1, nNetworks): 
	(bestMatch0to, bestMatchSign0to, bestMatchTo0, bestMatchSignTo0) = h.alignComponents(refEigenbasis, eigVectsList[iNet], 
																				refIncludedProperties, includedPropertiesList[iNet]); 
	print(bestMatchTo0); 


	# Flipping vectors in second eigenbasis to align with best match: 
	for ii in range(len(bestMatchTo0)): 
		eigVectsList[iNet][:,ii] *= bestMatchSign0to[ii]; 

otherEigenbasis = [eigVectsList[ii] for ii in range(1, nNetworks)]; 

h.plotEigenvectorProjections(refEigenbasis, otherEigenbasis); 

plt.show(); 
sys.exit(0); 


eigVectsProjection = np.dot(eigVects1.T, eigVects2); 

plt.figure(); 
plt.imshow(eigVectsProjection, interpolation="none"); 
plt.colorbar(); 

plt.figure(); 
plt.plot(eigVects1[:,0]); 


fig = plt.figure(); 
ax = fig.add_subplot(111); 
ax.set_aspect("equal"); 
# Plotting original axes: 
plt.plot([0, 1], [0, 0], 'k'); 
plt.plot([0, 0], [0, 1], 'k'); 
# Plotting projected axes: 
plt.plot([0, eigVectsProjection[1,1]], [0, eigVectsProjection[2,1]], 'r'); 
plt.plot([0, eigVectsProjection[1,2]], [0, eigVectsProjection[2,2]], 'r'); 


fig = plt.figure(); 
ax = fig.add_subplot(111, projection='3d'); 
# ax.set_aspect("equal"); 
# Plotting original axes: 
plt.plot([0, 1], [0, 0], [0, 0], 'k'); 
plt.plot([0, 0], [0, 1], [0, 0], 'k'); 
plt.plot([0, 0], [0, 0], [0, 1], 'k'); 
# Plotting projected axes: 
plt.plot([0, eigVectsProjection[1,1]], [0, eigVectsProjection[2,1]], [0, eigVectsProjection[3,1]], 'r'); 
plt.plot([0, eigVectsProjection[1,2]], [0, eigVectsProjection[2,2]], [0, eigVectsProjection[3,2]], 'r'); 
plt.plot([0, eigVectsProjection[1,3]], [0, eigVectsProjection[2,3]], [0, eigVectsProjection[3,3]], 'r'); 


# Finding best matches and correcting sign accordingly: 
referenceEigenbasis = eigVects0; 
propertiesInRefEigenbasis = includedProperties0; 
otherEigenbases = [eigVects1, eigVects2]; 
propertiesInOtherEigenbases = [includedProperties1, includedProperties2]; 
for (thisEigenbasis, thisProperties) in zip(otherEigenbases, propertiesInOtherEigenbases): 
	(bestMatch0to, bestMatchSign0to, bestMatchTo0, bestMatchSignTo0) = h.alignComponents(referenceEigenbasis, thisEigenbasis, 
																			propertiesInRefEigenbasis, thisProperties); 
	for ii in range(len(bestMatchTo0)): 
		thisEigenbasis[:,ii] *= bestMatchSignTo0[ii]; 

h.plotEigenvectorProjections(eigVects0, [eigVects1, eigVects2]); 



plt.show(); 
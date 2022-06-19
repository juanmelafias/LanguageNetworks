"""

	script_compareAllComplexity.py: 

		Script to measure and plot all complexities together. 

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


networkComplexitiesDict = {}; 
correctionFactorsDict = {}; 
correctionPerNodeDict = {}; 
nNodesDict = {}
correctedComplexitiesDict = {}; 



#######################################################################################################################
# Complexity for CNB network: 

print("Processing CNB network: "); 

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
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

# Computing and storing complexity: 
(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
networkComplexitiesDict["CNB"] = [netComplexity]; 
correctionFactorsDict["CNB"] = [correctionFactor]; 
correctedComplexitiesDict["CNB"] = [netComplexity*correctionFactor]; 
nNodes = len(thisNetwork.nodes()); 
nNodesDict["CNB"] = [nNodes]; 
correctionPerNodeDict["CNB"] = [correctionFactor/(nNodes**(1.5))]; 




#######################################################################################################################
# Complexity of largest connected component of Erdös-Renyi graph for varying connectivity: 

print("Erdös-Renyi networks: "); 

nNodesRandom = 200; 
pConnectList = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]; 
networkComplexitiesDict["ER"] = []; 
correctionFactorsDict["ER"] = []; 
correctedComplexitiesDict["ER"] = []; 
nNodesDict["ER"] = []; 
correctionPerNodeDict["ER"] = []; 
for pConnect in pConnectList: 
	# Building nework, keeping largest connected component: 
	thisNetwork = nx.erdos_renyi_graph(nNodesRandom, pConnect); 	
	Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
	thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

	# Computing and storing complexity: 
	(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
	networkComplexitiesDict["ER"] += [netComplexity]; 
	correctionFactorsDict["ER"] += [correctionFactor]; 
	correctedComplexitiesDict["ER"] += [netComplexity*correctionFactor]; 
	nNodes = len(thisNetwork.nodes()); 
	nNodesDict["ER"] += [nNodes]; 
	correctionPerNodeDict["ER"] += [correctionFactor/(nNodes**(1.5))]; 




#######################################################################################################################
# Complexity of Watts-Strogatz network with various rewiring connectivities: 

print("Processing Watts-Strogatz networks: "); 

nNeighbors = 4; 
pRewireList = [0.0001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]; 
networkComplexitiesDict["WS"] = []; 
correctionFactorsDict["WS"] = []; 
correctedComplexitiesDict["WS"] = []; 
nNodesDict["WS"] = []; 
correctionPerNodeDict["WS"] = []; 
for pRewire in pRewireList: 
	# Building nework -- always connected: 
	thisNetwork = nx.watts_strogatz_graph(nNodesRandom, nNeighbors, 0.05); 

	# Computing and storing complexity: 
	(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
	networkComplexitiesDict["WS"] += [netComplexity]; 
	correctionFactorsDict["WS"] += [correctionFactor]; 
	correctedComplexitiesDict["WS"] += [netComplexity*correctionFactor]; 
	nNodes = len(thisNetwork.nodes()); 
	nNodesDict["WS"] += [nNodes]; 
	correctionPerNodeDict["WS"] += [correctionFactor/(nNodes**(1.5))]; 


#######################################################################################################################
# Complexity of Barabasi-Albert (preferential attachment) graph: 

print("Processing Barabasi-Albert networks: "); 

nRepeat = 10; 
nNewAttachments = 2; 
networkComplexitiesDict["BA"] = []; 
correctionFactorsDict["BA"] = []; 
correctedComplexitiesDict["BA"] = []; 
nNodesDict["BA"] = []; 
correctionPerNodeDict["BA"] = []; 
for ii in range(nRepeat): 
	# Building nework -- always connected: 
	thisNetwork = nx.barabasi_albert_graph(nNodesRandom, nNewAttachments); 

	# Computing and storing complexity: 
	(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
	networkComplexitiesDict["BA"] += [netComplexity]; 
	correctionFactorsDict["BA"] += [correctionFactor]; 
	correctedComplexitiesDict["BA"] += [netComplexity*correctionFactor]; 
	nNodes = len(thisNetwork.nodes()); 
	nNodesDict["BA"] += [nNodes]; 
	correctionPerNodeDict["BA"] += [correctionFactor/(nNodes**(1.5))]; 


#######################################################################################################################
# Complexity of random bipartite graph: 

print("Processing random bipartite networks: "); 

nConnections = 400; 
networkComplexitiesDict["bipartite"] = []; 
correctionFactorsDict["bipartite"] = []; 
correctedComplexitiesDict["bipartite"] = []; 
nNodesDict["bipartite"] = []; 
correctionPerNodeDict["bipartite"] = []; 
for ii in range(nRepeat): 
	# Building nework -- always connected: 
	thisNetwork = nx.bipartite.gnmk_random_graph(int(nNodesRandom/2), int(nNodesRandom/2), nConnections); 
	Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); # Nothing guarantees connectedness! 
	thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

	# Computing and storing complexity: 
	(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
	networkComplexitiesDict["bipartite"] += [netComplexity]; 
	correctionFactorsDict["bipartite"] += [correctionFactor]; 
	correctedComplexitiesDict["bipartite"] += [netComplexity*correctionFactor]; 
	nNodes = len(thisNetwork.nodes()); 
	nNodesDict["bipartite"] += [nNodes]; 
	correctionPerNodeDict["bipartite"] += [correctionFactor/(nNodes**(1.5))]; 	


########################################################################################################################
# Complexity of MRI_234 human connectomes: 

print("Processing MRI_234 networks: "); 

# Loading nets and computing complexity: 
connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_234/"; 
allNames_MRI_234 = os.listdir(connectomeDataPath); 
nChoose = 10; 
toStudy = np.random.choice(allNames_MRI_234, nChoose); 

networkComplexitiesDict["MRI_234"] = []; 
correctionFactorsDict["MRI_234"] = []; 
correctedComplexitiesDict["MRI_234"] = []; 
nNodesDict["MRI_234"] = []; 
correctionPerNodeDict["MRI_234"] = []; 
for (iNet, netName) in enumerate(toStudy): 
	# print("Processing network " + str(iNet) + ": \n"); 
	thisNetwork = nx.read_graphml(connectomeDataPath + netName); 
	Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); # Apparently, these networks are *not* connected! 
	thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

	(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
	networkComplexitiesDict["MRI_234"] += [netComplexity]; 
	correctionFactorsDict["MRI_234"] += [correctionFactor]; 
	correctedComplexitiesDict["MRI_234"] += [netComplexity*correctionFactor]; 
	nNodes = len(thisNetwork.nodes()); 
	nNodesDict["MRI_234"] += [nNodes]; 
	correctionPerNodeDict["MRI_234"] += [correctionFactor/(nNodes**(1.5))]; 



########################################################################################################################
# Complexity of MRI_1015 human connectomes: 

print("Processing MRI_1015 networks: "); 

# Loading nets and computing complexity: 
connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_1015/"; 
allNames_MRI_1015 = os.listdir(connectomeDataPath); 
toStudy = np.random.choice(allNames_MRI_1015, nChoose); 

networkComplexitiesDict["MRI_1015"] = []; 
correctionFactorsDict["MRI_1015"] = []; 
correctedComplexitiesDict["MRI_1015"] = []; 
nNodesDict["MRI_1015"] = []; 
correctionPerNodeDict["MRI_1015"] = []; 
for (iNet, netName) in enumerate(toStudy): 
	# print("Processing network " + str(iNet) + ": \n"); 
	thisNetwork = nx.read_graphml(connectomeDataPath + netName); 
	Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
	thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

	(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
	networkComplexitiesDict["MRI_1015"] += [netComplexity]; 
	correctionFactorsDict["MRI_1015"] += [correctionFactor]; 
	correctedComplexitiesDict["MRI_1015"] += [netComplexity*correctionFactor]; 
	nNodes = len(thisNetwork.nodes()); 
	nNodesDict["MRI_1015"] += [nNodes]; 
	correctionPerNodeDict["MRI_1015"] += [correctionFactor/(nNodes**(1.5))]; 


########################################################################################################################
# Complexity of Macaque connectomes: 

print("Processing Macaque networks: "); 

# Loading nets and computing complexity: 
connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Macaque/"; 
thisNetwork = nx.read_graphml(connectomeDataPath + "rhesus_brain_1.graphml"); 
thisNetwork = thisNetwork.to_undirected(); 
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
networkComplexitiesDict["macaque"] = [netComplexity]; 
correctionFactorsDict["macaque"] = [correctionFactor]; 
correctedComplexitiesDict["macaque"] = [netComplexity*correctionFactor]; 
nNodes = len(thisNetwork.nodes()); 
nNodesDict["macaque"] = [nNodes]; 
correctionPerNodeDict["macaque"] = [correctionFactor/(nNodes**(1.5))]; 



########################################################################################################################
# Complexity of C elegans connectomes: 

print("Processing C elegans networks: "); 

# Loading nets and computing complexity: 
connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Celegans/"; 
thisNetwork = nx.read_graphml(connectomeDataPath + "c.elegans_neural.male_1.graphml"); 
thisNetwork = thisNetwork.to_undirected(); 
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
networkComplexitiesDict["celegans"] = [netComplexity]; 
correctionFactorsDict["celegans"] = [correctionFactor]; 
correctedComplexitiesDict["celegans"] = [netComplexity*correctionFactor]; 
nNodes = len(thisNetwork.nodes()); 
nNodesDict["celegans"] = [nNodes]; 
correctionPerNodeDict["celegans"] = [correctionFactor/(nNodes**(1.5))]; 

# There are two more celegans connectomes to be loaded from .mat files: 
mat = scipy.io.loadmat(connectomeDataPath + "Celegans131/celegans131.mat"); 
thisNetwork = nx.convert_matrix.from_numpy_matrix(mat["celegans131matrix"]); 
thisNetwork = thisNetwork.to_undirected(); 
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
networkComplexitiesDict["celegans"] += [netComplexity]; 
correctionFactorsDict["celegans"] += [correctionFactor]; 
correctedComplexitiesDict["celegans"] += [netComplexity*correctionFactor]; 
nNodes = len(thisNetwork.nodes()); 
nNodesDict["celegans"] += [nNodes]; 
correctionPerNodeDict["celegans"] += [correctionFactor/(nNodes**(1.5))]; 


mat = scipy.io.loadmat(connectomeDataPath + "Celegans277/celegans277.mat"); 
thisNetwork = nx.convert_matrix.from_numpy_matrix(mat["celegans277matrix"]); 
thisNetwork = thisNetwork.to_undirected(); 
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
networkComplexitiesDict["celegans"] += [netComplexity]; 
correctionFactorsDict["celegans"] += [correctionFactor]; 
correctedComplexitiesDict["celegans"] += [netComplexity*correctionFactor]; 
nNodes = len(thisNetwork.nodes()); 
nNodesDict["celegans"] += [nNodes]; 
correctionPerNodeDict["celegans"] += [correctionFactor/(nNodes**(1.5))]; 


########################################################################################################################
# Complexity of Deutsche Autobahn graph: 

print("Processing Deutsch Autobahn network: "); 

dieAutobahnDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Autobahn/"; 
mat = scipy.io.loadmat(dieAutobahnDataPath + "autobahn.mat"); 
thisNetwork = nx.convert_matrix.from_numpy_matrix(mat["auto1168"]); 
(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

networkComplexitiesDict["autobahn"] = [netComplexity]; 
correctionFactorsDict["autobahn"] = [correctionFactor]; 
correctedComplexitiesDict["autobahn"] = [netComplexity*correctionFactor]; 
nNodes = len(thisNetwork.nodes()); 
nNodesDict["autobahn"] = [nNodes]; 
correctionPerNodeDict["autobahn"] = [correctionFactor/(nNodes**(1.5))]; 



########################################################################################################################
# Complexity of airports graph: 

print("Processing airports network: "); 

airportDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Airport/"; 
mat = scipy.io.loadmat(airportDataPath + "air500.mat"); 
thisNetwork = nx.convert_matrix.from_numpy_matrix(mat["air500matrix"]); 
(netComplexity, correctionFactor) = h.computeNetworkComplexity(thisNetwork); 
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 

networkComplexitiesDict["airports"] = [netComplexity]; 
correctionFactorsDict["airports"] = [correctionFactor]; 
correctedComplexitiesDict["airports"] = [netComplexity*correctionFactor]; 
nNodes = len(thisNetwork.nodes()); 
nNodesDict["airports"] = [nNodes]; 
correctionPerNodeDict["airports"] = [correctionFactor/(nNodes**(1.5))]; 




########################################################################################################################
########################################################################################################################
## Plotting results: 
## 

print('\n'); 
print("Plotting cases: "); 

studyCases = ["CNB", "ER", "WS", "BA", "bipartite", "MRI_234", "MRI_1015", "macaque", "celegans", "autobahn", "airports"]; 


# Concatenating all quantities (not sure what for...):  
allComplexitiesToPlot = []; 
allCorrectionsToPlot = []; 
allCorrectedComplexitiesToPlot = []; 
allNNodesToPlot = []; 
for case in studyCases: 
	allComplexitiesToPlot += networkComplexitiesDict[case]; 
	allCorrectionsToPlot += correctionFactorsDict[case]; 
	allCorrectedComplexitiesToPlot += correctedComplexitiesDict[case]; 
	allNNodesToPlot += nNodesDict[case]; 


# Plotting allComplexities: 
fig = plt.figure(); 
plt.plot(allComplexitiesToPlot, 'o'); 
plt.xlabel("Network"); 
plt.ylabel("Uncorrected complexity"); 

# Plotting allCorrections: 
fig = plt.figure(); 
plt.plot(allCorrectionsToPlot, 'o'); 
plt.xlabel("Network"); 
plt.ylabel("Correction factor"); 

# Plotting allCorrectedComplexities: 
fig = plt.figure(); 
plt.plot(allCorrectedComplexitiesToPlot, 'o'); 
plt.xlabel("Network"); 
plt.ylabel("Corrected complexity"); 


# Plotting correction factor vs uncorrected complexity: 
fig = plt.figure(); 
for case in studyCases: 
	plt.plot(correctionFactorsDict[case], networkComplexitiesDict[case], 'o', label=case); 
plt.legend(loc='upper right'); 
plt.xlabel("Correction factor"); 
plt.ylabel("Uncorrected complexity"); 


# Plotting node-correction factor vs uncorrected complexity: 
fig = plt.figure(); 
for case in studyCases: 
	print(case); 
	print(correctionPerNodeDict[case]); 
	print(networkComplexitiesDict[case]); 
	plt.plot(correctionPerNodeDict[case], networkComplexitiesDict[case], 'o', label=case); 
plt.legend(loc='upper right'); 
plt.xlabel("Correction factor per node"); 
plt.ylabel("Uncorrected complexity"); 


# # Plotting nNodes vs uncorrected complexity: 
# fig = plt.figure(); 
# for case in studyCases: 
# 	plt.plot(nNodesDict[case], networkComplexitiesDict[case], 'o', label=case); 
# plt.legend(); 
# plt.xlabel("Number of nodes"); 
# plt.xlabel("Uncorrected complexity factor"); 


# # Plotting nNodes vs corrected complexity: 
# fig = plt.figure(); 
# for case in studyCases: 
# 	plt.plot(nNodesDict[case], correctedComplexitiesDict[case], 'o', label=case); 
# plt.legend(); 
# plt.xlabel("Number of nodes"); 
# plt.xlabel("Corrected complexity"); 


plt.show(); 



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



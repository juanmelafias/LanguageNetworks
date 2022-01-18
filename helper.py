"""

	helper.py: 

		With functions to assist in the analysis of syntactic networks. 

		List of functions contained herein: 

			>> loadNetworkFromTXT(), loadNetworkFromSIF(): Load one network file from its respective format files. 
			>> loadAllTXTNetsFromPath(), loadAllSIFNetsFromPath(): Loads all network files in a path. 
			>> computeAgeFromName(). 
			>> extractCommonNodes(). 
			>> computeIndexFromDevelopment(). 
			>> extractCommonNodesDegree(). 
			>> extractCommonBetweennessCentrality(). 
			>> extractCommonEigenvectorCentrality(). 
			>> computeCommonNodesStatistics(). 
			>> computeCoreSizesAndLargestCore(). 
			>> computeConnectedVocabularySize(). 
			>> computeNetworksStatistics(): Including ages from file name. For plotting! 
			>> computeNetworksStatistics_IFISC(): Ages from file name are not computed. For PCA! 
			>> computeNetworksStatistics_home(): Ages from file name are not computed. For PCA! 



"""

import numpy as np; 
import matplotlib.pyplot as plt; 
import random as rand; 
import os, sys; 
import networkx as nx; 
from copy import copy; 
from scipy.stats import entropy; 


def computeNodesProperties(net, fNeighborMean=True, fNeighborStd=True): 
	"""	computeNodesProperties function: 

			This function computes a series of properties of all the nodes of the network provided. Some of these
			properties might be problematic (e.g. because they are the same for all nodes, or they are ill-defined
			in the network). This function flags problematic properties so that they are not included in the analysis. 

			ACHTUNG!! 
				The measured properties and nodes must be in the same order to be able to plot things properly. 

			Inputs: 
				>> net: Upon whose nodes we wish to measure stuff. 
				>> fNeighborMean: Flag to indicate whether we want to incorporate average neighbor properties to the analysis. 
				>> fNeighborStd: Flag to indicate whether we want to incorporate std of neighbor properties to the analysis. 

			Returns: 
				<< nodeList: List of nodes as listed when called from the Graph() object. 
				<< nodesProperties: Dictionary containing all the measurements performed on the nodes. 
				<< includedProperties: List of non-problematic measurements. 
				<< excludedProperties: List of problematic measurements. 
					- includedProperties + excludedProperties = nodesProperties.keys(). 

	"""

	## Pre-processing the network: 

	# Extracting a list of nodes as they appear when called from the net object. 
	nodeList = net.nodes(); 
	nNodes = len(nodeList); 

	# ACHTUNG!! 
	# 	Extracting the largest connected component. 
	# 	In the future, this should be avoided. Problematic properties (e.g. average path length to each node) should
	# 	be automatically flagged and sorted out by this function. 
	thisGCC = max(nx.connected_components(net), key=len); 

	# ACHTUNG!! 
	# 	Some of the properties measured demand that networks have no self loops. 
	# 	At the moment this is left to keep the function working, but I should revise it at some point. 
	# 	Ideally, if a measure is problematic because of self-loops, this should be flagged and reported in excludedProperties. 
	netWOSL = copy(net); 
	netWOSL.remove_edges_from(nx.selfloop_edges(netWOSL)); 


	## Initializing properties measured upon nodes: 

	# A list of primary properties: 
	primaryProperties = ["degreeCentrality", "eigenvectorCentrality", "betweennessCentrality"]; 
	primaryProperties += ["closenessCentrality", "harmonicCentrality"]; 
	primaryProperties += ["clustering", "componentSize", "pagerank", "degree", "coreNumber", "onionLayer"]; 
	measuredProperties = copy(primaryProperties); 

	# These are complemented by average properties of the neighbors to measure if similar nodes connect (as in
	# assortativity) and standard deviation of properties of neighbors to measure whether connection is specific or
	# heterogeneous. 
	if (fNeighborMean): 
		for thisProperty in primaryProperties: 
			measuredProperties += [thisProperty+"_neighborMean"]; 
	if (fNeighborStd): 
		for thisProperty in primaryProperties: 
			measuredProperties += [thisProperty+"_neighborStd"]; 


	## Initializing the dictionary that will contain all properties: 
	nodesPropertiesDict = {}; 
	nodesProperties = {}; 
	for thisProperty in measuredProperties: 
		nodesPropertiesDict[thisProperty] = {}; 
		nodesProperties[thisProperty] = np.zeros([nNodes, 1]).squeeze(); 


	## Performing the actual computations: 

	# ACHTUNG!! 
	# 	Degree centrality: 
	# 	This is redundant as it is just the node degree normalized by the number of nodes in the network. 
	# 	Should be removed. Kept for the moment. 
	nodesPropertiesDict["degreeCentrality"] = nx.degree_centrality(net); 

	# Eigenvector centrality: 
	# 	ACH! Remind why this exception! 
	try: 
		nodesPropertiesDict["eigenvectorCentrality"] = nx.eigenvector_centrality(net); 
	except: 
		nodesPropertiesDict["eigenvectorCentrality"] = nx.eigenvector_centrality(net, max_iter=10000); 

	# Betweenness centrality: 
	nodesPropertiesDict["betweennessCentrality"] = nx.betweenness_centrality(net); 

	# Closeness centrality: 
	nodesPropertiesDict["closenessCentrality"] = nx.closeness_centrality(net); 

	# Harmonic centrality: 
	nodesPropertiesDict["harmonicCentrality"] = nx.harmonic_centrality(net); 

	# # ACHTUNG!! Excluded. It does not converge! 
	# # Katz centrality: 
	# try: 
	# 	nodesPropertiesDict["katzCentrality"] = nx.katz_centrality(net); 
	# except: 
	# 	nodesPropertiesDict["katzCentrality"] = nx.katz_centrality(net, max_iter=10000); 

	# Clustering coefficient: 
	nodesPropertiesDict["clustering"] = nx.clustering(net); 

	# thisEccentricity = nx.eccentricity(net); 
	
	# Page rank: 
	nodesPropertiesDict["pagerank"] = nx.pagerank(net); 

	# Node degree: 
	nodesPropertiesDict["degree"] = net.degree(); 

	# Size of largest k-core to which each node belongs: 
	nodesPropertiesDict["coreNumber"] = nx.core_number(netWOSL); 

	# Onion layer: order in which each node is removed when computing k-cores: 
	nodesPropertiesDict["onionLayer"] = nx.algorithms.core.onion_layers(net); 


	# # print("Average neighbor degree"); 
	# thisAND = nx.average_neighbor_degree(net); 

	# Sorting out properties in lists, which are more appropriate for building matrices and diagonalizing: 
	for (iNode, node) in enumerate(nodeList): 
		nodesProperties["degreeCentrality"][iNode] = nodesPropertiesDict["degreeCentrality"][node]; 
		nodesProperties["eigenvectorCentrality"][iNode] = nodesPropertiesDict["eigenvectorCentrality"][node]; 
		nodesProperties["betweennessCentrality"][iNode] = nodesPropertiesDict["betweennessCentrality"][node]; 
		nodesProperties["closenessCentrality"][iNode] = nodesPropertiesDict["closenessCentrality"][node]; 
		nodesProperties["harmonicCentrality"][iNode] = nodesPropertiesDict["harmonicCentrality"][node]; 
		nodesProperties["clustering"][iNode] = nodesPropertiesDict["clustering"][node]; 
		nodesProperties["componentSize"][iNode] = float(len(nx.node_connected_component(net, node)))/len(thisGCC); 
		nodesPropertiesDict["componentSize"][node] = nodesProperties["componentSize"][iNode]; 
		nodesProperties["pagerank"][iNode] = nodesPropertiesDict["pagerank"][node]; 
		nodesProperties["degree"][iNode] = nodesPropertiesDict["degree"][node]; 
		nodesProperties["coreNumber"][iNode] = nodesPropertiesDict["coreNumber"][node]; 
		nodesProperties["onionLayer"][iNode] = nodesPropertiesDict["onionLayer"][node]; 
		# nodesProperties["averageNeighborDegree"][iNode] = thisAND[node]; 


	# Finding out mean and standard deviation of properties over each node's neighbors: 
	if (fNeighborMean or fNeighborStd): 
		for (iNode, node) in enumerate(nodeList): 
			nodeNeighbors = [elem for elem in net.neighbors(node)]; 
			for thisProperty in primaryProperties: 
				neighborProperty = [nodesPropertiesDict[thisProperty][thisNeighbor] for thisNeighbor in nodeNeighbors]; 
				if (fNeighborMean): 
					nodesProperties[thisProperty+"_neighborMean"][iNode] = np.mean(neighborProperty); 
				if (fNeighborStd): 
					if (nodesPropertiesDict["degree"][node]>1): 
						nodesProperties[thisProperty+"_neighborStd"][iNode] = np.std(neighborProperty); 
					else: 
						nodesProperties[thisProperty+"_neighborStd"][iNode] = 0.; 


	# Reporting which properties are problematic: 
	# 	They can be problematic, e.g., because there is no variation. Then, they do not contribute to any PC. 
	# 	These would just have a zero on the eigenvectors, but algebra cannot handle these cases properly. 
	includedProperties = []; 
	excludedProperties = []; 
	for thisProperty in nodesProperties.keys(): 
		thisMean = np.mean(nodesProperties[thisProperty]); 
		thisStd = np.std(nodesProperties[thisProperty]); 
		if ((np.isnan(thisMean)) or (thisStd == 0.) or (thisStd/thisMean < 10e-11)): 
			excludedProperties += [thisProperty]; 
		else: 
			includedProperties += [thisProperty]; 

	return (nodeList, nodesProperties, includedProperties, excludedProperties); 


def convertPC2RGB(thisArray): 
	"""	convertPC2RGB function: 

			This function receives an array containing the values of a principal component and normalizes them to
			[0,1] so they can be used as an entry for an RGB color. 

			Inputs: 
				>> thisArray: Array with values across all nodes for a principal component. 

			Returns: 
				<< normalizedArray: Array normalized to [0,1]. 

	"""

	thisMin = min(thisArray); 
	thisMax = max(thisArray); 
	normalizedArray = [(elem - thisMin)/(thisMax-thisMin) for elem in thisArray]; 

	return normalizedArray; 


def varianceExplained(eigVals): 
	"""	varianceExplained function: 

			This function computes the percentage of variance explained by each component, as well as the cumulated
			percentage of variance. 

			Inputs: 
				>> eigVals: Eigenvalues under research. 

			Returns: 
				<< varianceExplained: Percentage of variance explained by each eigenvalue. 
				<< varianceExplained_cumul: Cumulative percentage of explained variance. 

	"""

	Z = np.sum(eigVals); 
	varianceExplained = eigVals/Z; 
	varianceExplained_cumul = np.cumsum(varianceExplained); 

	return (varianceExplained, varianceExplained_cumul); 


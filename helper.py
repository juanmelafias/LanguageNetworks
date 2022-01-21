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


# def normalizeProperties(netProperties, normalizeToStd=True): 
# 	"""	normalizeProperties function: 

# 	"""


def computeNetworkComplexity(net): 
	"""	computeNetworkComplexity function: 

			This function runs all steps needed to compute a network's complexity. This is: 
				(1) Computing network properties. 
				(2) Extracting useful properties (non-pathological ones). 
				(3) Standardizing distro of properties. 
				(4) Computing correlation matrices and diagonalizing. 
				(5) Computing index based on cumulative explained variance. 
				(6) Computing correcting factor as well. 

			Inputs: 
				>> net: From which we wish to measure the complexity. 

			Returns: 
				<< networkComplexity: Complexity index for this network based on accumulated explained variance. 
				<< correctionFactor: Trace of the covariance matrix before normalizing to std=1. 

	"""

	# Computing network properties: 
	nNodes = len(net.nodes()); 
	(nodeList, nodesProperties, includedProperties, excludedProperties) = computeNodesProperties(net); 
	nAllProperties = len(nodesProperties); 
	nProperties = len(includedProperties); 
	allPropertiesArray = np.zeros([nProperties, nNodes]); 
	dictIStat = {}; 
	for (iProperty, thisProperty) in enumerate(includedProperties): 
		allPropertiesArray[iProperty,:] = nodesProperties[thisProperty]; 

	# Standardizing distro of properties: 
	allPropertiesMean = np.mean(allPropertiesArray, 1); 
	allPropertiesStd = np.std(allPropertiesArray, 1); 
	allPropertiesArray_noStandard = copy(allPropertiesArray); 
	allPropertiesArray = allPropertiesArray - np.transpose(np.repeat(np.array([allPropertiesMean]), nNodes, 0)); 
	allPropertiesArray = np.divide(allPropertiesArray, np.transpose(np.repeat(np.array([allPropertiesStd]), nNodes, 0))); 

	# Computing correlation matrix and diagonalizing: 
	allPropertiesCov = np.cov(allPropertiesArray); 
	allPropertiesCov_noStandard = np.cov(allPropertiesArray_noStandard); 
	correctionFactor = np.trace(allPropertiesCov_noStandard)
	(eigVals, eigVects) = np.linalg.eig(allPropertiesCov); 
	# eigVals = np.real(eigVals); 
	# eigVects = np.real(eigVects); 

	# Computing complexity index: 
	(netVarianceExplained, netVarianceExplained_cumul) = varianceExplained(eigVals); 
	networkComplexity = 1.-sum(netVarianceExplained_cumul)/nAllProperties; 
	if (len(netVarianceExplained_cumul)==0): 
		networkComplexity = 0.; 

	return (networkComplexity, correctionFactor); 


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


def distanceToTargetNode(allPropertiesArray_, iTargetNode): 
	"""	distanceToTargetNode function: 

			This function computes the distance of all nodes to a target node in eigenspace coordinates. This can be
			used to plot which nodes are more similar to a given one. 

			Inputs: 
				>> allPropertiesArray_: Node properties projected in eigenspace. 
				>> iTargetNode: Index of the node with respect to which we wish to measure the distance. 

			Returns: 
				<< distanceToTarget: Array containing the distances to the target node in eigenspace. 
				<< distanceToTarget_: Same, but normalized to [0,1]; 
	
	"""

	nNodes = allPropertiesArray_.shape[1]; 
	distanceToTargetNode = [np.sum(np.abs(allPropertiesArray_[:,iNode] - allPropertiesArray_[:,iTargetNode])) for iNode in range(nNodes)]; 
	distanceToTargetNode = np.array(distanceToTargetNode).astype(float); 
	distanceToTargetNode_ = np.divide(distanceToTargetNode, max(distanceToTargetNode)); 

	return (distanceToTargetNode, distanceToTargetNode_); 
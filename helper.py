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


def computeNodesProperties(net, fVerbose=False): 
	"""	computeNodesProperties function: 

			This function computes a series of properties of all the nodes of the network provided. Some of these
			properties might be problematic (e.g. because they are the same for all nodes, or they are ill-defined
			in the network). This function flags problematic properties so that they are not included in the analysis. 

			ACHTUNG!! 
				The measured properties and nodes must be in the same order to be able to plot things properly. 

			Inputs: 
				>> net: Upon whose nodes we wish to measure stuff. 
				>> fVerbose=False: To indicate whether we want to read what is being computed at each moment. 
					> Not used at the moment. Decided to skip verbosity altogether. 

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
	primaryProperties += ["clustering", "componentSize", "coreNumber", "pagerank", "degree", "normalizedDegree"]; 
	measuredProperties = copy(primaryProperties); 

	# # These are complemented by average properties of the neighbors to measure if similar nodes connect (as in
	# # assortativity) and standard deviation of properties of neighbors to measure whether connection is specific or
	# # heterogeneous. 
	# for thisProperty in primaryProperties: 
	# 	measuredProperties += [thisProperty+"_neighborMean", thisProperty+"_neighborStd"]; 


	## Initializing the dictionary that will contain all properties: 
	nodesProperties = {}; 
	for thisProperty in measuredProperties: 
		nodesProperties[thisProperty] = np.zeros([nNodes, 1]).squeeze(); 


	## Performing the actual computations: 

	# ACHTUNG!! 
	# 	Degree centrality: 
	# 	This is redundant as it is just the node degree normalized by the number of nodes in the network. 
	# 	Should be removed. Kept for the moment. 
	thisDegreeCentrality = nx.degree_centrality(net); 

	# Eigenvector centrality: 
	# 	ACH! Remind why this exception! 
	try: 
		thisEigenvectorCentrality = nx.eigenvector_centrality(net); 
	except: 
		thisEigenvectorCentrality = nx.eigenvector_centrality(net, max_iter=10000); 

	# Betweenness centrality: 
	thisBetweennessCentrality = nx.betweenness_centrality(net); 

	# Closeness centrality: 
	thisClosenessCentrality = nx.closeness_centrality(net); 

	# Harmonic centrality: 
	thisHarmonicCentrality = nx.harmonic_centrality(net); 

	# # ACHTUNG!! Excluded. It does not converge! 
	# # Katz centrality: 
	# try: 
	# 	thisKatzCentrality = nx.katz_centrality(net); 
	# except: 
	# 	thisKatzCentrality = nx.katz_centrality(net, max_iter=10000); 

	# Clustering coefficient: 
	thisClustering = nx.clustering(net); 

	# Size of largest k-core to which each node belongs: 
	thisLargestKCore = nx.core_number(netWOSL); 

	# thisEccentricity = nx.eccentricity(net); 
	
	# Page rank: 
	thisPagerank = nx.pagerank(net); 

	# Node degree: 
	thisDegree = net.degree(); 


	# print("Average neighbor degree"); 
	thisAND = nx.average_neighbor_degree(net); 

	for (iNode, node) in enumerate(nodeList): 
		nodesProperties["degreeCentrality"][iNode] = thisDegreeCentrality[node]; 
		nodesProperties["eigenvectorCentrality"][iNode] = thisEigenvectorCentrality[node]; 
		nodesProperties["betweennessCentrality"][iNode] = thisBetweennessCentrality[node]; 
		nodesProperties["closenessCentrality"][iNode] = thisClosenessCentrality[node]; 
		nodesProperties["harmonicCentrality"][iNode] = thisHarmonicCentrality[node]; 
		nodesProperties["clustering"][iNode] = thisClustering[node]; 
		nodesProperties["componentSize"][iNode] = float(len(nx.node_connected_component(net, node)))/len(thisGCC); 
		nodesProperties["coreNumber"][iNode] = thisLargestKCore[node]; 
		nodesProperties["pagerank"][iNode] = thisPagerank[node]; 
		nodesProperties["degree"][iNode] = thisDegree[node]; 
		# nodesProperties["averageNeighborDegree"][iNode] = thisAND[node]; 

	nodesProperties["normalizedDegree"] = nodesProperties["degree"]*2/sum(nodesProperties["degree"]); 

	# for (iNode, node) in enumerate(nodeList): 
	# 	nodeNeighbors = net.neighbors(node); 
	# 	for thisProperty in primaryProperties: 
	# 		neighborProperty = nodesProperties[]


	# Reporting which statistics are problematic. 
	# These can be problematic, e.g., because there is no variation. Then, they do not contribute to any PC. 
	# These would just have a zero on the eigenvectors, but algebra cannot handle these cases properly. 
	includedProperties = []; 
	excludedProperties = []; 
	for statistic in nodesProperties.keys(): 
		thisMean = np.mean(nodesProperties[statistic]); 
		thisStd = np.std(nodesProperties[statistic]); 
		if ((np.isnan(thisMean)) or (thisStd == 0.) or (thisStd/thisMean < 10e-11)): 
			excludedProperties += [statistic]; 
		else: 
			includedProperties += [statistic]; 

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


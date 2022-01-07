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


def computeNodesStatistics(net): 
	"""	computeNodesStatistics function: 

			This function computes statistics for all the nodes of the network provided. 

			ACHTUNG!! 
				The measured statistics and nodes must be in the same order to be able to plot things properly. 

			Inputs: 
				>> net: Upon whose nodes we wish to measure stuff. 

			Returns: 
				<< nodesStatistics: stuff measured on each and every node. 

	"""

	nodeList = net.nodes(); 
	nNodes = len(nodeList); 
	allStatistics = ["averageNeighborDegree", "degreeCentrality", "eigenvectorCentrality", "betweennessCentrality"]; 
	# allStatistics += ["clustering", "componentSize", "coreNumber", "pagerank", "degree", "normalizedDegree"]; 
	## ACH! The next line is the one currently used for syntax network. This should be reviewed. 
	## 	The problem comes with "componentSize", which is the same for all if all are connected. 
	# allStatistics += ["componentSize", "coreNumber", "pagerank", "degree", "normalizedDegree"]; 
	## Component size is substituted by "clustering", which is troublesome for some sintex networks. 
	## Other kinds of networks have issues with coreNumber, because all nodes belong to the same largest k-core. 
	## We have to solve the invariant dimension issue. 
	allStatistics += ["clustering", "coreNumber", "pagerank", "degree", "normalizedDegree"]; 
	# allStatistics += ["clustering", "pagerank", "degree", "normalizedDegree"]; 
	nodesStatistics = {}; 
	for statistic in allStatistics: 
		nodesStatistics[statistic] = np.zeros([nNodes, 1]).squeeze(); 

	# Network without self loops: 
	netWOSL = copy(net); 
	netWOSL.remove_edges_from(nx.selfloop_edges(netWOSL)); 

	print("Average neighbor degree"); 
	thisAND = nx.average_neighbor_degree(net); 
	print("Degree centrality"); 
	thisDC = nx.degree_centrality(net); 
	print("Eigenvector centrality"); 
	try: 
		thisEC = nx.eigenvector_centrality(net); 
	except: 
		thisEC = nx.eigenvector_centrality(net, max_iter=1000); 
	print("Betweenness centrality"); 
	thisBC = nx.betweenness_centrality(net); 
	print("Clustering"); 
	thisClustering = nx.clustering(net); 
	print("Largest connected component"); 
	thisGCC = max(nx.connected_components(net), key=len); 
	print("Core number"); 
	thisKN = nx.core_number(netWOSL); 
	# thisEccentricity = nx.eccentricity(net); 
	print("Page rank"); 
	thisPR = nx.pagerank(net); 
	print("Degree"); 
	thisDegree = net.degree(); 

	for (iNode, node) in enumerate(nodeList): 
		nodesStatistics["averageNeighborDegree"][iNode] = thisAND[node]; 
		nodesStatistics["degreeCentrality"][iNode] = thisDC[node]; 
		nodesStatistics["eigenvectorCentrality"][iNode] = thisEC[node]; 
		nodesStatistics["betweennessCentrality"][iNode] = thisBC[node]; 
		nodesStatistics["clustering"][iNode] = thisClustering[node]; 
		# nodesStatistics["componentSize"][iNode] = float(len(nx.node_connected_component(net, node)))/len(thisGCC); 
		nodesStatistics["coreNumber"][iNode] = thisKN[node]; 
		nodesStatistics["pagerank"][iNode] = thisPR[node]; 
		nodesStatistics["degree"][iNode] = thisDegree[node]; 

	nodesStatistics["normalizedDegree"] = nodesStatistics["degree"]*2/sum(nodesStatistics["degree"]); 

	includedStatistics = []; 
	excludedStatistics = []; 
	for statistic in nodesStatistics.keys(): 
		thisMean = np.mean(nodesStatistics["normalizedDegree"]); 
		thisStd = np.std(nodesStatistics["normalizedDegree"]); 
		if ((thisStd == 0.) or (thisStd/thisMean < 10e-11)): 
			excludedStatistics += [statistic]; 
		else: 
			includedStatistics += [statistic]; 


	return (nodeList, nodesStatistics, includedStatistics, excludedStatistics); 


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


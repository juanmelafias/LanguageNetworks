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
import pickle as pkl; 
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

	fVerbose = True; 

	## Pre-processing the network: 

	if fVerbose: 
		print("Preparing network"); 

	# Extracting a list of nodes as they appear when called from the net object. 
	nodeList = net.nodes(); 
	nNodes = len(nodeList); 

	if fVerbose: 
		print("\tList of nodes extracted. "); 



	# ACHTUNG!! 
	# 	Extracting the largest connected component. 
	# 	In the future, this should be avoided. Problematic properties (e.g. average path length to each node) should
	# 	be automatically flagged and sorted out by this function. 
	thisGCC = max(nx.connected_components(net), key=len); 

	if fVerbose: 
		print("\tLargest connected component extracted. "); 

	# ACHTUNG!! 
	# 	Some of the properties measured demand that networks have no self loops. 
	# 	At the moment this is left to keep the function working, but I should revise it at some point. 
	# 	Ideally, if a measure is problematic because of self-loops, this should be flagged and reported in excludedProperties. 
	netWOSL = copy(net); 
	netWOSL.remove_edges_from(nx.selfloop_edges(netWOSL)); 

	if fVerbose: 
		print("\tSelf loops removed. "); 


	## Initializing properties measured upon nodes: 

	# A list of primary properties: 
	primaryProperties = ["degree", "eigenvectorCentrality", "betweennessCentrality"]; 
	## ACHTUNG!! 
	#	closenessVitality is interesting, but it is very heavy on computations. It is removed for tests. 
	# 	If removing one node breaks the network apart, closenessVitality results in an infinity... 
	# 	Fixed using hyperbolic tangent! This property seems to contribute a lot for some networks :) 
	primaryProperties += ["closenessCentrality", "harmonicCentrality", "componentSize", "pagerank", "coreNumber"]; 
	primaryProperties += ["onionLayer", "effectiveSize", "nodeCliqueNumber", "numberOfCliques"]; 
	primaryProperties += ["clustering", "squareClustering", "closenessVitality", "constraint"]; 
	# primaryProperties += ["clustering", "squareClustering", "constraint"]; 
	measuredProperties = copy(primaryProperties); 

	if fVerbose: 
		print("\tNetwork properties prepared. "); 

	# These are complemented by average properties of the neighbors to measure if similar nodes connect (as in
	# assortativity) and standard deviation of properties of neighbors to measure whether connection is specific or
	# heterogeneous. 
	if (fNeighborMean): 
		for thisProperty in primaryProperties: 
			measuredProperties += [thisProperty+"_neighborMean"]; 
	if (fNeighborStd): 
		for thisProperty in primaryProperties: 
			measuredProperties += [thisProperty+"_neighborStd"]; 

	if fVerbose: 
		print("\tNeighbor mean and std added or excluded. "); 


	## Initializing the dictionary that will contain all properties: 
	nodesPropertiesDict = {}; 
	nodesProperties = {}; 
	for thisProperty in measuredProperties: 
		nodesPropertiesDict[thisProperty] = {}; 
		nodesProperties[thisProperty] = np.zeros([nNodes, 1]).squeeze(); 

	if fVerbose: 
		print("\tDictionary initialized. "); 


	## Performing the actual computations: 

	if fVerbose: 
		print("Computing network stuff. "); 

	# ACHTUNG!! 
	# 	Degree centrality: 
	# 	This is redundant as it is just the node degree normalized by the number of nodes in the network. 
	# 	Should be removed. Kept for the moment. 
	if "degreeCentrality" in measuredProperties: 
		if fVerbose: 
			print("\tComputing degree centrality. "); 
		nodesPropertiesDict["degreeCentrality"] = nx.degree_centrality(net); 

	# Node degree: 
	if "degree" in measuredProperties: 
		if fVerbose: 
			print("\tComputing degree. "); 
		nodesPropertiesDict["degree"] = net.degree(); 

	# Eigenvector centrality: 
	# 	ACH! Remind why this exception! 
	if "eigenvectorCentrality" in measuredProperties: 
		if fVerbose: 
			print("\tComputing eigenvector centrality. "); 
		try: 
			nodesPropertiesDict["eigenvectorCentrality"] = nx.eigenvector_centrality(net); 
		except: 
			nodesPropertiesDict["eigenvectorCentrality"] = nx.eigenvector_centrality(net, max_iter=10000); 

	# Betweenness centrality: 
	if "betweennessCentrality" in measuredProperties: 
		if fVerbose: 
			print("\tComputing betweenness centrality. "); 
		nodesPropertiesDict["betweennessCentrality"] = nx.betweenness_centrality(net); 

	# Closeness centrality: 
	if "closenessCentrality" in measuredProperties: 
		if fVerbose: 
			print("\tComputing closeness centrality. "); 
		nodesPropertiesDict["closenessCentrality"] = nx.closeness_centrality(net); 

	# Harmonic centrality: 
	if "harmonicCentrality" in measuredProperties: 
		if fVerbose: 
			print("\tComputing harmonic centrality. "); 
		nodesPropertiesDict["harmonicCentrality"] = nx.harmonic_centrality(net); 

	# # ACHTUNG!! Excluded. It does not converge! 
	# # Katz centrality: 
	# try: 
	# 	nodesPropertiesDict["katzCentrality"] = nx.katz_centrality(net); 
	# except: 
	# 	nodesPropertiesDict["katzCentrality"] = nx.katz_centrality(net, max_iter=10000); 

	# Page rank: 
	if "pagerank" in measuredProperties: 
		if fVerbose: 
			print("\tComputing pagerank centrality. "); 
		nodesPropertiesDict["pagerank"] = nx.pagerank_numpy(net); 

	# Size of largest k-core to which each node belongs: 
	if "coreNumber" in measuredProperties: 
		if fVerbose: 
			print("\tComputing largest k-core. "); 
		nodesPropertiesDict["coreNumber"] = nx.core_number(netWOSL); 

	# Onion layer: order in which each node is removed when computing k-cores: 
	if "onionLayer" in measuredProperties: 
		if fVerbose: 
			print("\tComputing onion layer. "); 
		nodesPropertiesDict["onionLayer"] = nx.algorithms.core.onion_layers(net); 

	if "effectiveSize" in measuredProperties: 
		if fVerbose: 
			print("\tComputing effective size. "); 
		nodesPropertiesDict["effectiveSize"] = nx.effective_size(net); 

	if "nodeCliqueNumber" in measuredProperties: 
		if fVerbose: 
			print("\tComputing node clique number. "); 
		nodesPropertiesDict["nodeCliqueNumber"] = nx.node_clique_number(net); 

	if "numberOfCliques" in measuredProperties: 
		if fVerbose: 
			print("\tComputing number of maximal cliques. "); 
		nodesPropertiesDict["numberOfCliques"] = nx.number_of_cliques(net); 

	# Clustering coefficient: 
	if "clustering" in measuredProperties: 
		if fVerbose: 
			print("\tComputing clustering. "); 
		nodesPropertiesDict["clustering"] = nx.clustering(net); 

	if "squareClustering" in measuredProperties: 
		if fVerbose: 
			print("\tComputing square clustering. "); 
		nodesPropertiesDict["squareClustering"] = nx.square_clustering(net); 

	# Closeness vitality -- increase in distance between nodes when a node is removed: 
	if "closenessVitality" in measuredProperties: 
		if fVerbose: 
			print("\tComputing closeness vitality. "); 
		nodesPropertiesDict["closenessVitality"] = nx.closeness_vitality(net); 
		for node in nodesPropertiesDict["closenessVitality"]: 
			nodesPropertiesDict["closenessVitality"][node] = np.tanh(nodesPropertiesDict["closenessVitality"][node]); 

	if "constraint" in measuredProperties: 
		if fVerbose: 
			print("\tComputing node constraint. "); 
		nodesPropertiesDict["constraint"] = nx.constraint(net); 

	# thisEccentricity = nx.eccentricity(net); 


	if fVerbose: 
		print("Post-processing: "); 

	# Sorting out properties in lists, which are more appropriate for building matrices and diagonalizing: 
	
	for (iNode, node) in enumerate(nodeList): 
		for thisProperty in primaryProperties: 
			if (thisProperty != "componentSize"): 
				nodesProperties[thisProperty][iNode] = float(nodesPropertiesDict[thisProperty][node]); 
			if (thisProperty == "componentSize"): 
				nodesProperties["componentSize"][iNode] = float(len(nx.node_connected_component(net, node)))/len(thisGCC); 
				nodesPropertiesDict["componentSize"][node] = nodesProperties["componentSize"][iNode]; 

	# for thisProperty in primaryProperties: 
	# 	print(type(nodesProperties[thisProperty])); 

	# sys.exit(); 

	if fVerbose: 
		print("\tProperties sorted in lists. "); 

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

	if fVerbose: 
		print("\tProperties for neighbors computed. "); 


	# Finding out pathological properties: 
	(includedProperties, excludedProperties) = findPathologicalProperties(nodesProperties); 

	if fVerbose: 
		print("\tProblematic properties reported. "); 

	return (nodeList, nodesProperties, includedProperties, excludedProperties); 


def buildPropertiesArray(propertiesDict, includedProperties): 
	""" buildPropertiesArray function: 

			This function builds an array containing the relevant properties for the analysis. Node properties are
			usually stored in a dictionary, but arrays are better suited for the PCA analysis -- thus the need for
			this function. 

			Inputs: 
				>> propertiesDict: Dictionary containing the properties measured on the nodes. 
				>> includedProperties: List of properties to be included in the analysis. 

			Returns: 
				<< propertiesArray: Numpy array containing the relevant properties for the analysis. Order of nodes is
				carefully selected. 

	"""

	nProperties = len(includedProperties); 
	nNodes = len(propertiesDict[includedProperties[0]]); 
	propertiesArray = np.zeros([nProperties, nNodes]); 
	for (iStat, statistic) in enumerate(includedProperties): 
		propertiesArray[iStat,:] = propertiesDict[statistic]; 

	return propertiesArray; 


def findPathologicalProperties(nodesProperties): 
	"""	findPathologicalProperties function: 

			This function finds network properties that do not behave properly and must thus be removed from the analysis. 

			Properties can be problematic, e.g., because there is no variation. Then, they do not contribute to any
			Principal Component (PC). These would just have a zero on the eigenvectors, but python libraries do not
			handle these cases automatically. 

			Another possibility is that some of the properties are not defined for every node, or that they become
			infinity. For example, closenessVitality is the variation in distances between nodes when each node is
			removed. Some nodes split the network in 2, thus distance between nodes would be increased an infinite
			amount. For that specific property, this has been solved by taking the hyperbolic tangent (which maps
			[0, +inf] into [0, 1]). Other, similar cases should be addressed in a one-by-one manner. 

			Inputs: 
				>> nodesProperties: Dictionary with network properties stored as lists for each key. 

			Returns: 
				<< includedProperties: Non-pathological. To be included in the analysis. 
				<< excludedProperties: Pathological. To be excluded from the analysis. 

	"""

	includedProperties = []; 
	excludedProperties = []; 
	for thisProperty in nodesProperties.keys(): 
		thisMean = np.mean(nodesProperties[thisProperty]); 
		thisStd = np.std(nodesProperties[thisProperty]); 
		# Note that we exclude properties with very little variance to avoid numerical rounding errors. 
		if ((np.isnan(thisMean)) or (thisStd == 0.) or (thisStd/thisMean < 10e-11)): 
			excludedProperties += [thisProperty]; 
		else: 
			includedProperties += [thisProperty]; 

	return (includedProperties, excludedProperties); 


def normalizeProperties(netProperties, fNormalizeToStd=True): 
	"""	normalizeProperties function: 

			This function normalizes the list of network properties to have mean 0 and standard deviation 1. An option
			is provided to not to normalize to std=1. 

			Inputs: 
				>> netProperties: List of properties measured from a network. 
				>> normalizeToStd=True: Option to set standard deviation of properties to 1. 

			Returns: 
				<< netProperties: Properly normalized network properties. 

	"""

	nNodes = netProperties.shape[1]; 
	netPropertiesMean = np.mean(netProperties, 1); 
	netPropertiesStd = np.std(netProperties, 1); 
	netProperties = netProperties - np.transpose(np.repeat(np.array([netPropertiesMean]), nNodes, 0)); 
	if (fNormalizeToStd): 
		netProperties = np.divide(netProperties, np.transpose(np.repeat(np.array([netPropertiesStd]), nNodes, 0))); 

	return netProperties; 


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
	allPropertiesArray_noStandard = copy(allPropertiesArray); 
	allPropertiesArray = normalizeProperties(allPropertiesArray); 

	# Computing correlation matrix and diagonalizing: 
	allPropertiesCov = np.cov(allPropertiesArray); 
	allPropertiesCov_noStandard = np.cov(allPropertiesArray_noStandard); 
	correctionFactor = np.trace(allPropertiesCov_noStandard)
	(eigVals, eigVects) = np.linalg.eig(allPropertiesCov); 
	eigVals = np.real(eigVals); 
	eigVects = np.real(eigVects); 

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


def computeComponentsAboveNoise(eigVals): 
	"""	computeComponentsAboveNoise function: 

			This function uses the bound in [1] to estimate the number of components that are above noise, as well as
			the amount of variance considered signal. To this end, following [1], the median eigenvalue is the
			estimate of the noise across eigenvalues. Moving on, (4/√3) times this median marks the threshold below
			which the eigenvalues are pure noise. All eigenvalues above this should be considered as containing a signal. 

				[1] Donoho DL, Gavish M. The optimal hard threshold for singular values is 4/√3. arXiv preprint
				arXiv:1305.5870, (2013).

			Inputs: 
				>> eigVals: Eigenvalues to be analized. 

			Returns: 
				<< noiseThreshold: Below which eigenvalues contain only noise. 
				<< nKeep: Number of eigenvalues above noiseThreshold. 

	"""

	noiseThreshold = 4*np.median(eigVals)/np.sqrt(3); 
	nKeep = len([vv for vv in eigVals if vv>noiseThreshold]); 

	return (noiseThreshold, nKeep); 



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


def alignComponents(eigVects1, eigVects2, includedProperties1, includedProperties2): 
	"""	alignComponents function: 

			This function takes a set of two eigen-basis and searchers which components across sets aligns the most.
			This will allow us to compare components from two different networks. The process consists in proyecting
			components along each other, and each multiplied by -1. To make this properly, we need to take into
			account that two networks might have different sets of properties, and that projections along absent
			properties are 0. 

			ACHTUNG!! 
				- Still not implemented the case in which not all properties are the same! 

			ACHTUNG!! 
				- Review bestMatch12 and bestMatch21 carefully!! Some results indicate that I mixed these up. 

			Inputs: 
				>> eigVects1, eigVects2: Eigenvectors from each network. 
				>> includedProperties1, includedProperties2: Properties included in each network's analysis. 

			Returns: 
				<< bestMatch12, bestMatch21: Component in basis 2 along which each component from axis 1 has a largest projection. 
				<< bestMatch12, bestMatch21: Sign of that projection. 

	"""

	includedProperties1 = set(includedProperties1); 
	includedProperties2 = set(includedProperties2); 
	commonProperties = includedProperties2.intersection(includedProperties1); 
	nComponents = len(commonProperties); 

	# If both networks have the same set of properties included in the analysis, things are greatly facilitated: 
	if ((len(commonProperties) == len(includedProperties1)) and (len(commonProperties) == len(includedProperties1))): 
		bestMatch = []; 
		eigVectsProjection = np.dot(eigVects1.T, eigVects2); 
		eigVectsProjectionAbs = np.abs(eigVectsProjection); 
		eigVectsProjectionSign = np.sign(eigVectsProjection).astype(int); 

		bestMatch12 = np.argmax(eigVectsProjectionAbs, 0); 
		bestMatchSign12 = [eigVectsProjectionSign[bestMatch12[ii],ii] for ii in range(nComponents)]; 
		bestMatch21 = np.argmax(eigVectsProjectionAbs, 1); 
		bestMatchSign21 = [eigVectsProjectionSign[jj, bestMatch21[jj]] for jj in range(nComponents)]; 

	else: 
		print("ACHTUNG!!"); 

	# plt.figure(); 
	# plt.imshow(eigVectsProjection, interpolation="none"); 
	# plt.colorbar(); 

	# plt.figure(); 
	# plt.imshow(eigVectsProjectionSign, interpolation="none"); 
	# plt.colorbar(); 

	# plt.figure(); 
	# plt.imshow(eigVectsProjectionAbs, interpolation="none"); 
	# plt.colorbar(); 

	return (bestMatch12, bestMatchSign12, bestMatch21, bestMatchSign21); 


def plotEigenvectorProjections(referenceEigenbasis, otherEigenbases): 
	"""	plotEigenvectorProjections function: 

			This function plots the projection of the first 2 or 3 components of all otherEigenbases in the first 2 or 3
			components of the referenceEigenbasis. 

			ACHTUNG!! 
				- We assume that all eigenbases have the same components, or that non-common components have been
				  omitted. Otherwise, this will crash. 

			Inputs: 
				<< referenceEigenbasis: Upon which all other eigenbases will be plotted. 
				<< otherEigenbases: Which will be plotted upon referenceEigenbasis. 

	"""

	eigVectsProjections = [np.dot(referenceEigenbasis.T, thisEigenbasis) for thisEigenbasis in otherEigenbases]; 

	# Coloring by network: 
	fig = plt.figure(); 
	ax = fig.add_subplot(111); 
	ax.set_aspect("equal"); 
	# Plotting original axes: 
	plt.plot([0, 1], [0, 0], 'k'); 
	plt.plot([0, 0], [0, 1], 'k'); 
	# Plotting projected axes: 
	for projection in eigVectsProjections: 
		p = plt.plot([0, projection[1,1]], [0, projection[2,1]]); 
		projectionColor = p[0].get_color(); 
		plt.plot([0, projection[1,2]], [0, projection[2,2]], color=projectionColor); 

	fig = plt.figure(); 
	ax = fig.add_subplot(111, projection='3d'); 
	# ax.set_aspect("equal"); 
	# Plotting original axes: 
	plt.plot([0, 1], [0, 0], [0, 0], 'k'); 
	plt.plot([0, 0], [0, 1], [0, 0], 'k'); 
	plt.plot([0, 0], [0, 0], [0, 1], 'k'); 
	# Plotting projected axes: 
	for projection in eigVectsProjections: 
		p = plt.plot([0, projection[1,1]], [0, projection[2,1]], [0, projection[3,1]]); 
		projectionColor = p[0].get_color(); 
		plt.plot([0, projection[1,2]], [0, projection[2,2]], [0, projection[3,2]], color=projectionColor); 
		plt.plot([0, projection[1,3]], [0, projection[2,3]], [0, projection[3,3]], color=projectionColor); 


	# Coloring by component: 
	fig = plt.figure(); 
	ax = fig.add_subplot(111); 
	ax.set_aspect("equal"); 
	# Plotting original axes: 
	plt.plot([0, 1], [0, 0], 'k'); 
	plt.plot([0, 0], [0, 1], 'r'); 
	# Plotting projected axes: 
	for projection in eigVectsProjections: 
		plt.plot([0, projection[1,1]], [0, projection[2,1]], 'k'); 
		plt.plot([0, projection[1,2]], [0, projection[2,2]], 'r'); 

	fig = plt.figure(); 
	ax = fig.add_subplot(111, projection='3d'); 
	# ax.set_aspect("equal"); 
	# Plotting original axes: 
	plt.plot([0, 1], [0, 0], [0, 0], 'k'); 
	plt.plot([0, 0], [0, 1], [0, 0], 'r'); 
	plt.plot([0, 0], [0, 0], [0, 1], 'g'); 
	# Plotting projected axes: 
	for projection in eigVectsProjections: 
		plt.plot([0, projection[1,1]], [0, projection[2,1]], [0, projection[3,1]], 'k'); 
		plt.plot([0, projection[1,2]], [0, projection[2,2]], [0, projection[3,2]], 'r'); 
		plt.plot([0, projection[1,3]], [0, projection[2,3]], [0, projection[3,3]], 'g'); 

	return; 


########################################################################################################################
########################################################################################################################
## Functions for neuroscience analysis -- might move them to a different library in the future: 
##

def findCounterhemisphericNode(nodePositions, targetNode, fCentered=False): 
	"""	findCounterhemisphericNode function: 

			This function computes the node most likely to be the counterpart of the target in the opposite hemisphere. 

			Inputs: 
				>> nodePositions: Array with the position of each node in xyz-space. 
					- x-coordinate marks left-right. 
				>> iTargetNode: Node of which we wish to find the mirror counterpart. 
				>> fCentered=False: This indicates whether the 0 of the x-coordinate is at the brain's central sagittal plane. 

			Returns: 
				<< iCounterNode: Index of the most proximate mirror counterpart (counter node). 
				<< counterNode: Name of the counter node in the network. 
				<< counterDistance: Distance of the counter node to the actual mirror counterpart position. 
				<< allDistances: All distances along the x-axis (in case we wish to study other near neighbors). 

	"""

	allNodesNames = [nodeName for nodeName in nodePositions.keys()]; 
	iNodes = []; 
	for (iNode, node) in enumerate(allNodesNames): 
		iNodes += [iNode]; 
		if (node==targetNode): 
			iTarget = iNode; 

	# If x-coordinate is not centered, we need to do it: 
	center = 0.; 
	if (not(fCentered)): 
		x = [nodePositions[node][0] for node in allNodesNames]; 
		center = np.mean(x); 

	allDistances = [np.abs(nodePositions[candidateNode][0] + nodePositions[targetNode][0] - 2*center) + 
					np.abs(nodePositions[candidateNode][1] - nodePositions[targetNode][1]) +
					np.abs(nodePositions[candidateNode][2] - nodePositions[targetNode][2]) for candidateNode in allNodesNames]; 
	iCounterNode = np.argmin(allDistances); 
	counterNode = allNodesNames[iCounterNode]; 
	counterDistance = sum(abs(np.subtract(nodePositions[counterNode], nodePositions[targetNode]))); 

	return (iCounterNode, counterNode, counterDistance, allDistances); 





######################################################################################################################## 
######################################################################################################################## 
## I/O functions: 
## 

def saveNetworkProperties(netName, netPath, nodeList, propertiesDict): 
	""" saveNetworkProperties function: 

			This function saves all the information needed to recreate the analysis for a given network. The information
			saved includes: 
				 - nodeList: To make sure that nodes are recalled in the right order. 
				 - propertiesDict: Dictionary that stores all properties. Each property is accessed as a dictionary, but
				   each property is actually a list. The indexes that correspond to each node are given by nodeList. 

			Note that includedProperties and excludedProperties are not stored. The idea is to store all data that has
			been computed so that it doesn't need to be computed again. Choices regarding which properties to use
			should be made during each successive analysis. 

			Inputs: 
				>> netName: This is used to build the names of the files where the properties are stored. 
				>> netPath: Where to store the properties. 
				>> nodeList: To be stored. 
				>> propertiesDict: To be stored. 

	"""

	fOut = open(netPath + netName + "_nodeList.csv", 'w'); 
	for node in nodeList: 
		fOut.write(str(node) + '\n'); 
	fOut.close(); 

	with open(netPath + netName + "_properties.pkl", 'wb') as fOut:
	    pkl.dump(propertiesDict, fOut); 

	return; 

def writeNetworkProperties(netName, netPath, nodeList, propertiesDict): 
	""" writeNetworkProperties function: 

			This function calls saveNetworkProperties(). This is so we can call either indistinctly and we don't need to
			remember which of the two has been implemented. 

	"""

	saveNetworkProperties(netName, netPath, nodeList, propertiesDict); 
	return; 

def loadNetworkProperties(netName, netPath, fNeighborMean=True, fNeighborStd=True): 
	""" loadNetworkProperties function: 

			This function loads an existing list of nodes and the corresponding dictionary of properties. 

			Inputs: 
				>> netName: This is used to build the names of the files where the properties are stored. 
				>> netPath: Where to store the properties. 
				>> fNeighborMean=True, fNeighborStd=True: Flags indicating whether we load neighbor means and stds. 

			Returns: 
				<< nodeList: List of the network's nodes stored in the same order as properties have been saved. 
				<< propertiesDict: Dictionary storing one list for each network property. 
	"""

	fIn = open(netPath + netName + "_nodeList.csv", 'r'); 
	nodeList = fIn.readlines(); 
	fIn.close(); 
	nodeList = [node.split('\n')[0] for node in nodeList]; 

	with open(netPath + netName + "_properties.pkl", 'rb') as fIn:
		propertiesDict = pkl.load(fIn); 

	if (not(fNeighborMean) or not(fNeighborStd)): 
		allPropertiesArray = [key for key in propertiesDict.keys()]
		if (not(fNeighborMean)): 
			allPropertiesArray = [key for key in allPropertiesArray if "_neighborMean" not in key]; 
		if (not(fNeighborStd)): 
			allPropertiesArray = [key for key in allPropertiesArray if "_neighborStd" not in key]; 
		propertiesDict_ = copy(propertiesDict); 
		propertiesDict = {}; 
		for key in allPropertiesArray: 
			propertiesDict[key] = propertiesDict_[key]; 

	return (nodeList, propertiesDict); 

def readNetworkProperties(netName, netPath, fNeighborMean=True, fNeighborStd=True): 
	""" readNetworkProperties function: 

			This function calls loadNetworkProperties(). This is so we can call either indistinctly and we don't need to
			remember which of the two has been implemented. 

	"""

	(nodeList, propertiesDict) = loadNetworkProperties(netName, netPath, fNeighborMean, fNeighborStd); 
	return (nodeList, propertiesDict); 



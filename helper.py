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


def loadNetworkFromTXT(dataFileName, dataPath, produceCytoscape=True, dataPathOut="*"): 
	""" loadNetworkFromSIF function: 

			This function loads networks from a plain .txt format. The coding in this case is slightly different from
			the coding in SIF files: The kind of relationship between words is missing, edges appears as two words in a
			single line separated by a space. Single words are not listed at the end. 

			Inputs: 
				>> dataName, dataPath: to locate the network. 
                >> produceCytoscape: A boolean flag telling whether, besides reading the network, we wish to produce a
										file that can be interpreted by cytoscape. 
				>> dataPathOut: Indicating where to store the networks for cytoscape. 

			Returns: 
				<< syntacticNetwork: an object of class Graph from networkx with words as nodes and syntactic
										relationships as links. 			

	"""

	# Reading network files: 
	fIn = open(os.path.join(dataPath, dataFileName), 'r'); 
	dR = fIn.read(); 
	fIn.close(); 

	dL = [ll for ll in dR.split('\n') if ll!='']; 
	allNodes = []; 
	allEdges = []; 
	# Do we wish to build a .sif file? 
	if (produceCytoscape): 
		forSIFEdges = []; 
		forSIFNodes = []; 
		if (dataPathOut=="*"): 
			dataPathOut = dataPath; 
	for ll in dL: 
		sL = ll.split(' '); 
		if (len(sL)==2): 
			allNodes += [sL[0], sL[1]]; 
			allEdges += [(sL[0], sL[1])]
			if (produceCytoscape): 
				forSIFEdges += [sL[0] + "\tlink\t" + sL[1]]; 
		else: 
			allNodes += [sL[0]]; 
			if (produceCytoscape): 
				forSIFNodes += [sL[0]]; 

	# Building a network: 
	syntacticNetwork = nx.Graph(); 
	syntacticNetwork.add_nodes_from(allNodes); 
	syntacticNetwork.add_edges_from(allEdges); 

	# Output .sif file: 
	if (produceCytoscape): 
		forSIFEdges =  '\n'.join(forSIFEdges); 
		forSIFNodes =  '\n'.join(forSIFNodes); 
		forSIFFile = '\n'.join([forSIFEdges, forSIFNodes]); 

		fOut = open(dataPathOut+dataFileName.split(".txt")[0]+"_cyt.sif", 'w'); 
		fOut.write(forSIFFile); 
		fOut.close(); 


	return syntacticNetwork; 


def loadAllTXTNetsFromPath(dataPath, produceCytoscape=True, dataPathOut="*"): 
	""" loadAllTXTNetsFromPath function: 

			This function loads all the networks (in .sif format) within a given path. 

			Inputs: 
				>> dataName, dataPath: to locate the network. 
                >> produceCytoscape: A boolean flag telling whether, besides reading the network, we wish to produce a
										file that can be interpreted by cytoscape. 
				>> dataPathOut: Indicating where to store the networks for cytoscape. 

			Returns: 
				<< allNets: List with all the networks found in that path. 
				<< allNetsNames: names of all the networks found in that path -- this is, all file names without the extension. 

	"""

	# Getting data files: 
	filesList = [fName for fName in os.listdir(dataPath) if os.path.isfile(os.path.join(dataPath, fName)) and fName[-4:]==".txt"]; 

	# Looping over data files: 
	allNets = {}; 
	allNetsNames = []; 
	for fName in filesList: 
		thisNetName = fName.split(".txt")[0]; 
		allNetsNames += [thisNetName]; 
		allNets[thisNetName] = loadNetworkFromTXT(fName, dataPath, produceCytoscape, dataPathOut); 

	return (allNets, allNetsNames); 



def loadNetworkFromSIF(dataFileName, dataPath): 
	"""	loadNetworkFromSIF function: 

			This function loads a network from a given file (and path) which is assumed to be in the format provided by
			Coblijn. This is, each line is either "word \t relationship \t word" o just a word. The first case
			corresponds to syntactic relationships found in recordings. The second case corresponds to atomic
			expressions with no external relationships (e.g. "Wow!"). 

			Inputs: 
				>> dataName, dataPath: to locate the network. 

			Returns: 
				<< syntacticNetwork: an object of class Graph from networkx with words as nodes and syntactic
										relationships as links. 

	"""


	# Reading network files: 
	fIn = open(os.path.join(dataPath, dataFileName), 'r'); 
	dR = fIn.read(); 
	fIn.close(); 


	dR.replace('\r\n', '\n')
	dL = [ll for ll in dR.split('\n') if ll!='']; 
	allNodes = []; 
	allEdges = []; 
	for ll in dL: 
		sL = ll.split('\t'); 
		if (len(sL)==3): 
			allNodes += [sL[0], sL[2]]; 
			allEdges += [(sL[0], sL[2])]
		else: 
			allNodes += [sL[0]]; 

	# Building a network: 
	syntacticNetwork = nx.Graph(); 
	syntacticNetwork.add_nodes_from(allNodes); 
	syntacticNetwork.add_edges_from(allEdges); 

	return syntacticNetwork; 

def loadAllSIFNetsFromPath(dataPath): 
	""" loadAllSIFNetsFromPath function: 

			This function loads all the networks (in .sif format) within a given path. 

			Inputs: 
				>> dataPath: where to look for networks. 

			Returns: 
				<< allNets: List with all the networks found in that path. 
				<< allNetsNames: names of all the networks found in that path -- this is, all file names without the extension. 

	"""

	# Getting data files: 
	filesList = [fName for fName in os.listdir(dataPath) if os.path.isfile(os.path.join(dataPath, fName)) and fName[-4:]==".sif"]; 

	# Looping over data files: 
	allNets = {}; 
	allNetsNames = []; 
	for fName in filesList: 
		thisNetName = fName.split(".sif")[0]; 
		allNetsNames += [thisNetName]; 
		allNets[thisNetName] = loadNetworkFromSIF(fName, dataPath); 

	return (allNets, allNetsNames); 

def computeAgeFromName(name): 
	""" computeAgeFromName function: 

			This fucntion computes the age of a participant from the string that is used to name her or him. These
			strings end either with 5 or 6 digits which correspond (from right to left) to i) two digist for age in
			days, ii) two digits for age in months, and iii) one or two digist for age in years. 

			Inputs: 
				name: string used to name a participant. 

			Returns: 
				age: age of the participant in days. 

	"""

	nString = name[-6:]; 
	dd = nString[4:6]; 
	mm = nString[2:4]; 
	yy = nString[0:2]; 
	if (yy[0].isalpha()): 
		yy = yy[1:]; 
	age = 365*int(yy) + 30*int(mm)+int(dd); 

	return age; 

def extractCommonNodes(allNets): 
	""" extractCommonNodes function: 

			This function extracts all the nodes (words) which are common to all networks. 

			Inputs: 
				>> allNets: list with the network objects loaded somewhere else. 

			Returns: 
				<< commonNodes: list with the nodes (words) which are present in all networks. 

	"""

	commonNodes = allNets[0].nodes(); 
	for net in allNets[1:]: 
		commonNodes = np.intersect1d(commonNodes, net.nodes()); 

	return list(commonNodes); 

def computeIndexFromDevelopment(devFileName): 
	""" computeIndexFromDevelopment function: 

			This function extracts the index of the name along a development series. 

	"""

	iName = int(devFileName.split('_')[1]); 

	return iName; 

def extractCommonNodesDegree(allNets, commonNodes): 
	""" extractCommonNodesDegree function: 

			This function returns the degree of all common nodes for all networks available. 

			Inputs: 
				>> allNets: list with all networks being studied. 
				>> commonNodes: list with all nodes present in all networks. 

			Returns: 
				<< commonNodesDegree: degree of all nodes present in all networks. 

	"""

	nNets = len(allNets); 
	nCommon = len(commonNodes); 
	allNEdges = [net.number_of_edges() for net in allNets]; 
	commonNodesDegree = np.zeros([nNets, nCommon]); 
	for (iNet, net) in enumerate(allNets): 
		thisCommonNodesDegree = net.degree(commonNodes)
		commonNodesDegree[iNet,:] = [thisCommonNodesDegree[node] for node in commonNodes]; 
		commonNodesDegree[iNet,:] = np.true_divide(commonNodesDegree[iNet,:], allNEdges[iNet]); 

	return commonNodesDegree; 

def extractCommonBetweennessCentrality(allNets, commonNodes): 
	""" extractCommonBetweennessCentrality function: 

			This function extracts the betweenness centrality of all nodes that are common to all networks. 

			Inputs: 
				>> allNets: list with all networks being studied. 
				>> commonNodes: list with all nodes present in all networks. 

			Returns: 
				<< commonNodesBC: betweenness centrality of all nodes that are common to all networks. 

	"""

	nNets = len(allNets); 
	nCommon = len(commonNodes); 
	commonNodesBC = np.zeros([nNets, nCommon]); 
	for (iNet, net) in enumerate(allNets): 
		thisCommonNodeBC = nx.algorithms.centrality.betweenness_centrality(net); 
		commonNodesBC[iNet,:] = [thisCommonNodeBC[node] for node in commonNodes]; 

	return commonNodesBC; 

def extractCommonEigenvectorCentrality(allNets, commonNodes): 
	""" extractCommonEigenvectorCentrality function: 

			This function computes the eigenvector centrality for all nodes in each network, then extracts the
			centrality for the nodes that are common to all networks.

			ACHTUNG!! 
				Most EC is zero, which here becomes 10-17 or so, and it changes between different calls. 

			Inputs: 
				>> allNets: list with all networks being studied. 
				>> commonNodes: list with all nodes present in all networks. 

			Returns: 
				<< commonNodesEC: eigenvector centrality of all nodes that are common to all networks. 

	"""

	nNets = len(allNets); 
	nCommon = len(commonNodes); 
	commonNodesEC = np.zeros([nNets, nCommon]); 
	for (iNet, net) in enumerate(allNets): 
		thisCommonNodeEC = nx.algorithms.centrality.eigenvector_centrality_numpy(net); 
		commonNodesEC[iNet,:] = [thisCommonNodeEC[node] for node in commonNodes]; 

	return commonNodesEC; 


def computeCommonNodesStatistics(allNets): 
	""" computeCommonNodesStatistics function: 

			This function computes statistics from the common nodes for all networks. The idea is to extract efficiently
			everything that we might be interested in. While individual calls to the functions above would loop over the
			networks several times, this function only loops once. 

			ACHTUNG!! 
				More statistics should be added to this function. 
				By now: degree, betweenness centrality. 

			Inputs: 
				>> allNets: list with all networks being studied. 

			Returns: 
				<< commonNodesStatistics: A dictionary summarizing interesting statistics about all common nodes. 

	"""

	## Preliminary: 
	nNets = len(allNets); 
	allNEdges = [net.number_of_edges() for net in allNets]; 
	# Extract common nodes: 
	commonNodes = extractCommonNodes(allNets); 

	nCommon = len(commonNodes); 
	# Initialize dictionaries and arrays to store quantities: 
	allStatistics = ["degree", "betweennessCentrality", "eigenvectorCentrality"]; 
	commonNodesStatistics = {}; 
	for statistic in allStatistics: 
		commonNodesStatistics[statistic] = np.zeros([nNets, nCommon]); 

	# Loop over networks to compute statistics: 
	for (iNet, net) in enumerate(allNets): 
		# Betweenness centrality: 
		thisCommonNodeBC = nx.algorithms.centrality.betweenness_centrality(net); 
		commonNodesStatistics["betweennessCentrality"][iNet,:] = [thisCommonNodeBC[node] for node in commonNodes]; 

		# Degree: 
		thisCommonNodesDegree = net.degree(commonNodes)
		commonNodesStatistics["degree"][iNet,:] = [thisCommonNodesDegree[node] for node in commonNodes]; 
		commonNodesStatistics["degree"][iNet,:] = np.true_divide(commonNodesStatistics["degree"][iNet,:], allNEdges[iNet]); 

		# # Eigenvector centrality: 
		# #	ACHTUNG!! Most EC is zero, which here becomes 10-17 or so, and it changes between different calls. 
		# thisCommonNodeEC = nx.algorithms.centrality.eigenvector_centrality_numpy(net); 
		# commonNodesStatistics["eigenvectorCentrality"][iNet,:] = [thisCommonNodeEC[node] for node in commonNodes]; 

	return commonNodesStatistics; 

def computeCoreSizesAndLargestCore(net): 
	""" computeCommonNodesStatistics function: 

			This function computes the sizes of 2-, 3-, and 4-cores, as well as the number of the largest, non-empty
			k-core. 

			Inputs: 
				>> net: graph upon which we are going to compute stuff. 

			Returns: 
				<< k2CoreSize, k3CoreSize, k4CoreSize: sizes of 2-, 3-, and 4-cores of the graph. 
				<< kCoreLargest: k number of the largest, non-empty k-core -- usually between 2 and 4. 

	"""

	# Simplifying network to make sure there are not autoloops: 
	thisSimplerNet = copy(net); 
	# thisSimplerNet.remove_edges_from(thisSimplerNet.selfloop_edges()); 
	thisSimplerNet.remove_edges_from(nx.selfloop_edges(thisSimplerNet)); 



	maxKCore = 1; 
	fNextCore = True; 
	kCoreSizes = []; 
	while (fNextCore): 
		thisCore = nx.algorithms.core.k_core(thisSimplerNet, k=maxKCore+1); 
		thisCoreSize = len(thisCore.nodes()); 
		if (thisCoreSize): 
			kCoreSizes += [float(thisCoreSize)]; 
			maxKCore += 1; 
		else: 
			fNextCore = False; 

	while (len(kCoreSizes)<3): 
		kCoreSizes += [0.]; 

	return (kCoreSizes, maxKCore); 

def computeConnectedVocabularySize(net): 
	"""	computeConnectedVocabularySize function: 

			This function computes the proportion of vocabulary that belongs to the giant connected component. 

			Inputs: 
				>> net: graph upon which we are going to compute stuff. 

			Returns: 
				<< connectedVocabularySize: Fraction of nodes that belong to the giant connected component. 

	"""

	thisGCC = max(nx.connected_components(net), key=len); 
	connectedVocabularySize = float(len(thisGCC))/len(net.nodes()); 

	return connectedVocabularySize; 




def computeNetworksStatistics(allNets, allNames): 
	""" computeNetworksStatistics function: 

			This function computes a lot of statistics for the purpose of plotting them, not for performing
			PCA. Among the statistics computed is the network's age, which is read from the name of the
			original file. 

			Inputs: 
				>> allNets: List with all networks being studied. 
				>> allNames: List with the names of the networks so that they can be accessed when stored in
				dictionaries. 

			Returns: 
				<< netsStatistics: Dictionary summarizing relevant statistics for all networks. 

	"""

	## Preliminary: 
	nNets = len(allNets); 
	allStatistics = ["age", "vocabularySize", "connectedVocabularySize", "assortativity", "clustering"]; 
	allStatistics += ["coloringNumber", "connectedComponentSize", "coreSize2", "coreSize3"]; 
	allStatistics += ["coreSizeLargest", "cycleBasisSize", "cycleBasisAverageSize", "diameter"]; 
	allStatistics += ["pagerankEntropy", "jaccardSimilarityMean", "jaccardSimilarityEntropy"]; 
	allStatistics += ["maximalMatchingSize", "maximalIndependentSet", "nEdges", "degree"]; 
	netsStatistics = {}; 
	for statistic in allStatistics: 
		netsStatistics[statistic] = np.zeros([nNets, 1]).squeeze(); 

	for (iNet, net) in enumerate(allNets): 
		# Network name: 
		thisNetName = allNames[iNet][1]; 
		# Number of nodes and edges (needed several times below): 
		thisNNodes = net.number_of_nodes(); 
		thisEdges = net.number_of_edges(); 
		print("Processing network " + thisNetName + ": "); 

		lastStatistics = []; 

		# Age: 
		netsStatistics["age"][iNet] = computeAgeFromName(thisNetName); 

		## Vocabulary: 
		# Vocabulary size: 
		# print("Computing vocabulary size: "); 
		netsStatistics["vocabularySize"][iNet] = len(net.nodes()); 
		lastStatistics += [netsStatistics["vocabularySize"][iNet]]; 

		# Connected vocabulary size: 
		# print("Computing connected vocabulary: "); 
		netsStatistics["connectedVocabularySize"][iNet] = computeConnectedVocabularySize(net); 
		lastStatistics += [netsStatistics["connectedVocabularySize"][iNet]]; 

		# Assortativity: 
		# 	ACH! Is troublesome with new networkx version... 
		# print("Computing assortativity: "); 
		netsStatistics["assortativity"][iNet] = nx.algorithms.assortativity.degree_assortativity_coefficient(net); 
		lastStatistics += [netsStatistics["assortativity"][iNet]]; 

		# Clustering: 
		# print("Computing clustering: "); 
		netsStatistics["clustering"][iNet] = nx.algorithms.cluster.average_clustering(net); 
		lastStatistics += [netsStatistics["clustering"][iNet]]; 

		# Coloring number: 
		# print("Computing coloring number: "); 
		thisColoring = nx.coloring.greedy_color(net); 
		netsStatistics["coloringNumber"][iNet] = len(set(thisColoring.values())); 
		lastStatistics += [netsStatistics["coloringNumber"][iNet]]; 

		# Connected component size: 
		# print("Computing size of connected component: "); 
		thisLargestCC = max(nx.algorithms.components.connected_components(net), key=len); 
		netsStatistics["connectedComponentSize"][iNet] = float(len(thisLargestCC))/thisNNodes; 
		lastStatistics += [netsStatistics["connectedComponentSize"][iNet]]; 


		## kCoreSizes and largest kCore: 
		# print("Computing k-core stuff: "); 
		(kSizes, kLargest) = computeCoreSizesAndLargestCore(net); 
		kSizes = np.divide(kSizes, thisNNodes); 
		# print("\tSize of 2-core: "); 
		netsStatistics["coreSize2"][iNet] = kSizes[0]; 
		# print("\tSize of 3-core: "); 
		netsStatistics["coreSize3"][iNet] = kSizes[1]; 
		# netsStatistics["coreSize4"][iNet] = kSizes[2]; 
		# print("\tLargest k-core: "); 
		netsStatistics["coreSizeLargest"][iNet] = kLargest; 
		lastStatistics += [netsStatistics["coreSize2"][iNet]]; 
		lastStatistics += [netsStatistics["coreSize3"][iNet]]; 
		lastStatistics += [netsStatistics["coreSizeLargest"][iNet]]; 

		## Cycles: 
		# print("Computing cycle stuff: "); 
		thisCycleBasis = nx.algorithms.cycles.cycle_basis(net); 
		# print("\tSize of cycle basis: "); 
		netsStatistics["cycleBasisSize"][iNet] = len(thisCycleBasis); 
		# print("\tAverage size of cycles in basis: "); 
		netsStatistics["cycleBasisAverageSize"][iNet] = np.nan_to_num(np.mean([len(cycle) for cycle in thisCycleBasis])); 
		lastStatistics += [netsStatistics["cycleBasisSize"][iNet]]; 
		lastStatistics += [netsStatistics["cycleBasisAverageSize"][iNet]]; 
		# if (thisNetName=="PYeri160717"): 
		# 	# ACHTUNG!! 
		# 	# 	This network has no cycle basis. 
		# 	# 	But apparently everything is correctly mapped to 0. 
		# 	print(thisCycleBasis); 
		# 	print(netsStatistics["cycleBasisSize"][iNet]); 
		# 	print(len(thisCycleBasis)); 
		# 	print(netsStatistics["cycleBasisAverageSize"][iNet]); 

		## Diameter: 
		# print("Computing diameter: ")
		thisGCC = max(nx.connected_components(net), key=len); 
		thisGCC = net.subgraph(thisGCC); 

		# print("Computing network diameter: "); 
		netsStatistics["diameter"][iNet] = nx.algorithms.distance_measures.diameter(thisGCC); 
		lastStatistics += [netsStatistics["diameter"][iNet]]; 

		# # Efficiency: 
		# # 	ACH! Need myConda environment... 
		# netsStatistics["efficiency"][iNet] = nx.algorithms.global_efficiency(net); 

		## Pagerank entropy: 
		# 	I made up this one. This is the entropy of the eigenvalues resulting from applying the pagerank entropy. 
		pRank = nx.algorithms.link_analysis.pagerank_alg.pagerank(net); 
		# print("Computing entropy of pageranks: "); 
		netsStatistics["pagerankEntropy"][iNet] = entropy(list(pRank.values()))/np.log(2); 
		lastStatistics += [netsStatistics["pagerankEntropy"][iNet]]; 

		## Jaccard similarity: 
		# 	This measures, pairwise, the Jaccard similarity between the neighbors of two given nodes. 
		# 	Then the average and entropy of these numbers are computed. 
		# print("Computing Jaccard stuff: "); 
		jCoef = nx.algorithms.link_prediction.jaccard_coefficient(net, net.edges()); 
		jCoef_ = nx.algorithms.link_prediction.jaccard_coefficient(net); 
		goodJCoef = [elem[2] for elem in jCoef if elem[2]] + [elem[2] for elem in jCoef_ if elem[2]]; 
		# print("\tAverage Jaccard similarity: "); 
		netsStatistics["jaccardSimilarityMean"][iNet] = np.mean(goodJCoef); 
		# print("\tEntropy of Jaccard similarities: "); 
		netsStatistics["jaccardSimilarityEntropy"][iNet] = entropy(goodJCoef)/np.log(2); 
		lastStatistics += [netsStatistics["jaccardSimilarityMean"][iNet]]; 
		lastStatistics += [netsStatistics["jaccardSimilarityEntropy"][iNet]]; 

		## Maximal matching size: 
		# 	Maximal matching is a set of edges such that no node appears twice, and no more edges can be added without
		# 	violating this condition. 
		# print("Computing maximal matching size: "); 
		maximalMatching = nx.algorithms.matching.maximal_matching(net); 
		netsStatistics["maximalMatchingSize"][iNet] = float(len(maximalMatching))/thisEdges; 
		lastStatistics += [netsStatistics["maximalMatchingSize"][iNet]]; 

		## Maximal independent set: 
		# 	Maximal set of nodes such that another node cannot be added without inducing edges in the graph. 
		# print("computing maximal independent set: "); 
		maxIndependentSetSize = []; 
		# There are several possible such sets, and they're found by random, so repeat: 
		for ii in range(10): 
			thisMIS = nx.algorithms.mis.maximal_independent_set(net); 
			maxIndependentSetSize += [float(len(thisMIS))/thisNNodes]; 
		netsStatistics["maximalIndependentSet"][iNet] = np.mean(maxIndependentSetSize); 
		lastStatistics += [netsStatistics["maximalIndependentSet"][iNet]]; 

		## Edges and degree: 
		netsStatistics["nEdges"][iNet] = thisEdges; 
		lastStatistics += [netsStatistics["nEdges"][iNet]]; 
		degrees = [elem[1] for elem in net.degree()]; 
		netsStatistics["degree"][iNet] = float(sum(degrees))/thisNNodes; 
		lastStatistics += [netsStatistics["degree"][iNet]]; 

		if (any([np.isnan(elem) for elem in lastStatistics])): 
			print(lastStatistics); 


	return netsStatistics; 



def computeNetworksStatistics_IFISC(allNets, allNames): 
	""" computeNetworksStatistics function: 

			Similarly to the previous function, this one loops over all available networks and computes all desired,
			relevant statistics for the network. 

			ACHTUNG!! 
				More statistics should be added to this function. 
				By now: assortativity. 

			Inputs: 
				>> allNets: list with all networks being studied. 

			Returns: 
				<< netsStatistics: Dictionary summarizing relevant statistics for all networks. 

	"""

	## Preliminary: 
	nNets = len(allNets); 
	# allStatistics = ["age", "assortativity", "clustering", "coloringNumber", "connectedComponentSize", "coreSize2"]; 
	allStatistics = ["vocabularySize", "connectedVocabularySize", "assortativity", "clustering", "coloringNumber"]; 
	allStatistics += ["connectedComponentSize", "coreSize2", "coreSize3", "coreSizeLargest", "cycleBasisSize"]; 
	# allStatistics += ["efficiency", "pagerankEntropy"]; 
	allStatistics += ["cycleBasisAverageSize", "diameter", "pagerankEntropy", "jaccardSimilarityMean"]; 
	allStatistics += ["jaccardSimilarityEntropy", "maximalMatchingSize", "maximalIndependentSet"]; 
	netsStatistics = {}; 
	for statistic in allStatistics: 
		netsStatistics[statistic] = np.zeros([nNets, 1]).squeeze(); 

	for (iNet, net) in enumerate(allNets): 
		# Network name: 
		thisNetName = allNames[iNet][1]; 
		# Number of nodes and edges (needed several times below): 
		thisNNodes = net.number_of_nodes(); 
		thisEdges = net.number_of_edges(); 

		# # Age: 
		# netsStatistics["age"][iNet] = computeAgeFromName(thisNetName); 

		# Vocabulary size: 
		netsStatistics["vocabularySize"][iNet] = len(net.nodes()); 

		# Connected vocabulary size: 
		netsStatistics["connectedVocabularySize"][iNet] = computeConnectedVocabularySize(net); 

		# Assortativity: 
		# 	ACH! Is troublesome with new networkx version... 
		netsStatistics["assortativity"][iNet] = nx.algorithms.assortativity.degree_assortativity_coefficient(net); 

		# Clustering: 
		netsStatistics["clustering"][iNet] = nx.algorithms.cluster.average_clustering(net); 

		# Coloring number: 
		thisColoring = nx.coloring.greedy_color(net); 
		netsStatistics["coloringNumber"][iNet] = len(set(thisColoring.values())); 

		# Connected component size: 
		thisLargestCC = max(nx.algorithms.components.connected_components(net), key=len); 
		netsStatistics["connectedComponentSize"][iNet] = float(len(thisLargestCC))/thisNNodes; 


		## kCoreSizes and largest kCore: 
		(kSizes, kLargest) = computeCoreSizesAndLargestCore(net); 
		kSizes = np.divide(kSizes, thisNNodes); 
		netsStatistics["coreSize2"][iNet] = kSizes[0]; 
		netsStatistics["coreSize3"][iNet] = kSizes[1]; 
		# netsStatistics["coreSize4"][iNet] = kSizes[2]; 
		netsStatistics["coreSizeLargest"][iNet] = kLargest; 

		## Cycles: 
		thisCycleBasis = nx.algorithms.cycles.cycle_basis(net); 
		netsStatistics["cycleBasisSize"][iNet] = len(thisCycleBasis); 
		netsStatistics["cycleBasisAverageSize"][iNet] = np.nan_to_num(np.mean([len(cycle) for cycle in thisCycleBasis])); 

		# Diameter: 
		thisGCC = max(nx.connected_component_subgraphs(net), key=len); 
		netsStatistics["diameter"][iNet] = nx.algorithms.distance_measures.diameter(thisGCC); 

		# # Efficiency: 
		# # 	ACH! Need myConda environment... 
		# netsStatistics["efficiency"][iNet] = nx.algorithms.global_efficiency(net); 

		## Pagerank entropy: 
		# 	I made up this one. This is the entropy of the eigenvalues resulting from applying the pagerank entropy. 
		pRank = nx.algorithms.link_analysis.pagerank_alg.pagerank(net); 
		netsStatistics["pagerankEntropy"][iNet] = entropy(list(pRank.values()))/np.log(2); 

		## Jaccard similarity: 
		# 	This measures, pairwise, the Jaccard similarity between the neighbors of two given nodes. 
		# 	Then the average and entropy of these numbers are computed. 
		jCoef = nx.algorithms.link_prediction.jaccard_coefficient(net, net.edges()); 
		jCoef_ = nx.algorithms.link_prediction.jaccard_coefficient(net); 
		goodJCoef = [elem[2] for elem in jCoef if elem[2]] + [elem[2] for elem in jCoef_ if elem[2]]; 
		netsStatistics["jaccardSimilarityMean"][iNet] = np.mean(goodJCoef); 
		netsStatistics["jaccardSimilarityEntropy"][iNet] = entropy(goodJCoef)/np.log(2); 

		## Maximal matching size: 
		# 	Maximal matching is a set of edges such that no node appears twice, and no more edges can be added without
		# 	violating this condition. 
		maximalMatching = nx.algorithms.matching.maximal_matching(net); 
		netsStatistics["maximalMatchingSize"][iNet] = float(len(maximalMatching))/thisEdges; 

		## Maximal independent set: 
		# 	Maximal set of nodes such that another node cannot be added without inducing edges in the graph. 
		maxIndependentSetSize = []; 
		# There are several possible such sets, and they're found by random, so repeat: 
		for ii in range(10): 
			thisMIS = nx.algorithms.mis.maximal_independent_set(net); 
			maxIndependentSetSize += [float(len(thisMIS))/thisNNodes]; 
		netsStatistics["maximalIndependentSet"][iNet] = np.mean(maxIndependentSetSize); 



def computeNetworksStatistics_home(allNets, allNames): 
	""" computeNetworksStatistics function: 

			Similarly to the previous function, this one loops over all available networks and computes all desired,
			relevant statistics for the network. 

			Inputs: 
				>> allNets: list with all networks being studied. 

			Returns: 
				<< netsStatistics: Dictionary summarizing relevant statistics for all networks. 

	"""

	## Preliminary: 
	nNets = len(allNets); 
	# allStatistics = ["age", "assortativity", "clustering", "coloringNumber", "connectedComponentSize", "coreSize2"]; 
	allStatistics = ["vocabularySize", "connectedVocabularySize", "assortativity", "clustering", "coloringNumber"]; 
	allStatistics += ["connectedComponentSize", "coreSize2", "coreSize3", "coreSizeLargest", "cycleBasisSize"]; 
	# allStatistics += ["efficiency", "pagerankEntropy"]; 
	allStatistics += ["cycleBasisAverageSize", "diameter", "pagerankEntropy", "jaccardSimilarityMean"]; 
	allStatistics += ["jaccardSimilarityEntropy", "maximalMatchingSize", "maximalIndependentSet"]; 
	netsStatistics = {}; 
	for statistic in allStatistics: 
		netsStatistics[statistic] = np.zeros([nNets, 1]).squeeze(); 

	for (iNet, net) in enumerate(allNets): 
		# Network name: 
		thisNetName = allNames[iNet][1]; 
		# Number of nodes and edges (needed several times below): 
		thisNNodes = net.number_of_nodes(); 
		thisEdges = net.number_of_edges(); 
		print("Processing network " + thisNetName + ": "); 

		lastStatistics = []; 

		# # Age: 
		# netsStatistics["age"][iNet] = computeAgeFromName(thisNetName); 


		## Vocabulary: 
		# Vocabulary size: 
		# print("Computing vocabulary size: "); 
		netsStatistics["vocabularySize"][iNet] = len(net.nodes()); 
		lastStatistics += [netsStatistics["vocabularySize"][iNet]]; 

		# Connected vocabulary size: 
		# print("Computing connected vocabulary: "); 
		netsStatistics["connectedVocabularySize"][iNet] = computeConnectedVocabularySize(net); 
		lastStatistics += [netsStatistics["connectedVocabularySize"][iNet]]; 

		# Assortativity: 
		# 	ACH! Is troublesome with new networkx version... 
		# print("Computing assortativity: "); 
		netsStatistics["assortativity"][iNet] = nx.algorithms.assortativity.degree_assortativity_coefficient(net); 
		lastStatistics += [netsStatistics["assortativity"][iNet]]; 

		# Clustering: 
		# print("Computing clustering: "); 
		netsStatistics["clustering"][iNet] = nx.algorithms.cluster.average_clustering(net); 
		lastStatistics += [netsStatistics["clustering"][iNet]]; 

		# Coloring number: 
		# print("Computing coloring number: "); 
		thisColoring = nx.coloring.greedy_color(net); 
		netsStatistics["coloringNumber"][iNet] = len(set(thisColoring.values())); 
		lastStatistics += [netsStatistics["coloringNumber"][iNet]]; 

		# Connected component size: 
		# print("Computing size of connected component: "); 
		thisLargestCC = max(nx.algorithms.components.connected_components(net), key=len); 
		netsStatistics["connectedComponentSize"][iNet] = float(len(thisLargestCC))/thisNNodes; 
		lastStatistics += [netsStatistics["connectedComponentSize"][iNet]]; 


		## kCoreSizes and largest kCore: 
		# print("Computing k-core stuff: "); 
		(kSizes, kLargest) = computeCoreSizesAndLargestCore(net); 
		kSizes = np.divide(kSizes, thisNNodes); 
		# print("\tSize of 2-core: "); 
		netsStatistics["coreSize2"][iNet] = kSizes[0]; 
		# print("\tSize of 3-core: "); 
		netsStatistics["coreSize3"][iNet] = kSizes[1]; 
		# netsStatistics["coreSize4"][iNet] = kSizes[2]; 
		# print("\tLargest k-core: "); 
		netsStatistics["coreSizeLargest"][iNet] = kLargest; 
		lastStatistics += [netsStatistics["coreSize2"][iNet]]; 
		lastStatistics += [netsStatistics["coreSize3"][iNet]]; 
		lastStatistics += [netsStatistics["coreSizeLargest"][iNet]]; 

		## Cycles: 
		# print("Computing cycle stuff: "); 
		thisCycleBasis = nx.algorithms.cycles.cycle_basis(net); 
		# print("\tSize of cycle basis: "); 
		netsStatistics["cycleBasisSize"][iNet] = len(thisCycleBasis); 
		# print("\tAverage size of cycles in basis: "); 
		netsStatistics["cycleBasisAverageSize"][iNet] = np.nan_to_num(np.mean([len(cycle) for cycle in thisCycleBasis])); 
		lastStatistics += [netsStatistics["cycleBasisSize"][iNet]]; 
		lastStatistics += [netsStatistics["cycleBasisAverageSize"][iNet]]; 
		# if (thisNetName=="PYeri160717"): 
		# 	# ACHTUNG!! 
		# 	# 	This network has no cycle basis. 
		# 	# 	But apparently everything is correctly mapped to 0. 
		# 	print(thisCycleBasis); 
		# 	print(netsStatistics["cycleBasisSize"][iNet]); 
		# 	print(len(thisCycleBasis)); 
		# 	print(netsStatistics["cycleBasisAverageSize"][iNet]); 

		## Diameter: 
		# print("Computing diameter: ")
		thisGCC = max(nx.connected_components(net), key=len); 
		thisGCC = net.subgraph(thisGCC); 

		# print("Computing network diameter: "); 
		netsStatistics["diameter"][iNet] = nx.algorithms.distance_measures.diameter(thisGCC); 
		lastStatistics += [netsStatistics["diameter"][iNet]]; 

		# # Efficiency: 
		# # 	ACH! Need myConda environment... 
		# netsStatistics["efficiency"][iNet] = nx.algorithms.global_efficiency(net); 

		## Pagerank entropy: 
		# 	I made up this one. This is the entropy of the eigenvalues resulting from applying the pagerank entropy. 
		pRank = nx.algorithms.link_analysis.pagerank_alg.pagerank(net); 
		# print("Computing entropy of pageranks: "); 
		netsStatistics["pagerankEntropy"][iNet] = entropy(list(pRank.values()))/np.log(2); 
		lastStatistics += [netsStatistics["pagerankEntropy"][iNet]]; 

		## Jaccard similarity: 
		# 	This measures, pairwise, the Jaccard similarity between the neighbors of two given nodes. 
		# 	Then the average and entropy of these numbers are computed. 
		# print("Computing Jaccard stuff: "); 
		jCoef = nx.algorithms.link_prediction.jaccard_coefficient(net, net.edges()); 
		jCoef_ = nx.algorithms.link_prediction.jaccard_coefficient(net); 
		goodJCoef = [elem[2] for elem in jCoef if elem[2]] + [elem[2] for elem in jCoef_ if elem[2]]; 
		# print("\tAverage Jaccard similarity: "); 
		netsStatistics["jaccardSimilarityMean"][iNet] = np.mean(goodJCoef); 
		# print("\tEntropy of Jaccard similarities: "); 
		netsStatistics["jaccardSimilarityEntropy"][iNet] = entropy(goodJCoef)/np.log(2); 
		lastStatistics += [netsStatistics["jaccardSimilarityMean"][iNet]]; 
		lastStatistics += [netsStatistics["jaccardSimilarityEntropy"][iNet]]; 

		## Maximal matching size: 
		# 	Maximal matching is a set of edges such that no node appears twice, and no more edges can be added without
		# 	violating this condition. 
		# print("Computing maximal matching size: "); 
		maximalMatching = nx.algorithms.matching.maximal_matching(net); 
		netsStatistics["maximalMatchingSize"][iNet] = float(len(maximalMatching))/thisEdges; 
		lastStatistics += [netsStatistics["maximalMatchingSize"][iNet]]; 

		## Maximal independent set: 
		# 	Maximal set of nodes such that another node cannot be added without inducing edges in the graph. 
		# print("computing maximal independent set: "); 
		maxIndependentSetSize = []; 
		# There are several possible such sets, and they're found by random, so repeat: 
		for ii in range(10): 
			thisMIS = nx.algorithms.mis.maximal_independent_set(net); 
			maxIndependentSetSize += [float(len(thisMIS))/thisNNodes]; 
		netsStatistics["maximalIndependentSet"][iNet] = np.mean(maxIndependentSetSize); 
		lastStatistics += [netsStatistics["maximalIndependentSet"][iNet]]; 

		if (any([np.isnan(elem) for elem in lastStatistics])): 
			print(lastStatistics); 


	return netsStatistics; 



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


	return (nodeList, nodesStatistics); 


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


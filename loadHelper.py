"""

	loadHelper.py: 

		Functions to help load networks. Networks are loaded from different sources and they need different auxiliary
		protocols. These should be removed eventually. By now, all these functions are put in here so that we have a
		more clear code for the analysis itself in the helper.py file. 

"""

# Imports: 
import numpy as np; 
import os, sys; 
from copy import copy; 
import networkx as nx; 
import scipy.io as sio; # To read .mat files! and .mnx files! 



def masterLoader(netName): 
	"""	masterLoader function: 

			This function is a master loader that handles all the cases and exceptions in loading networks. Now
			everything is centralized here. 

			Inputs: 
				>> netName: Name of the network to be loaded. 

	""" 

	metaDataDict = {}; 

	if (netName == "syntaxNetwork"): 
		# Metadata to load networks: 
		dataPathMaster = "/home/brigan/Desktop/Research_IFISC/LanguageMorphospaces/Data"; 
		dataNames = ["DataDown", "DataHI", "DataSLI", "DataTD1", "DataTD2", "DataTD3", "DataTDLongDutch1_original"]; 
		dataFormats = ["txt", "sif", "sif", "sif", "sif", "sif", "sif"]; 
		# dataNames = ["DataDown"]; 
		# dataFormats = ["txt"]; 

		# Looping over folders, loading nets: 
		allNetworksDict = {}; 
		allNetworksNamesDict = {}; 
		for (dataName, dataFormat) in zip(dataNames, dataFormats): 
			dataPath = os.path.join(dataPathMaster, dataName); 
			if (dataFormat=="txt"): 
				(synNetDict, synNetNameList) = loadAllTXTNetsFromPath(dataPath, False); 
			if (dataFormat=="sif"): 
				(synNetDict, synNetNameList) = loadAllSIFNetsFromPath(dataPath); 
			allNetworksDict[dataName] = synNetDict; 
			allNetworksNamesDict[dataName] = synNetNameList; 

		# Choose a single network: 
		thisKey = "DataTD3"; 

		iNetwork = 5; 
		thisNetwork = allNetworksDict[thisKey][allNetworksNamesDict[thisKey][iNetwork]]; 

	if (netName == "CNB_net"): 
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

	if (netName == "collabNet"): 
		dataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Collaborations/"; 
		networkFileName = "ca-CSphd.mtx"; 

		fIn = open(dataPath + networkFileName, 'r'); 
		dL = fIn.read().splitlines(); 
		fIn.close(); 

		dL = dL[3::]; 
		edges = []; 
		for ll in dL: 
			thisSplitLine = ll.split(' '); 
			edges += [(int(thisSplitLine[0]), int(thisSplitLine[1]))]; 

		# Building network from edges: 
		thisNetwork = nx.Graph(); 
		thisNetwork.add_edges_from(edges); 



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


	if ("MRI" in netName): 
		netNameWords = netName.split('_'); 

		connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_" + netNameWords[1] + "/"; 
		thisNetwork = nx.read_graphml(connectomeDataPath + netNameWords[2] + "_repeated10_scale250.graphml"); 

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

		metaDataDict["nativePositions"] = nativePositions; 
		metaDataDict["nativePositions_3D"] = nativePositions_3D; 

	if (netName == "macaqueBrain"): 
		connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Macaque/"; 
		thisNetwork = nx.read_graphml(connectomeDataPath + "rhesus_brain_1.graphml"); 
		thisNetwork = thisNetwork.to_undirected(); 

	if (netName == "macaqueInterCortex"): 
		# Most boring network ever... 
		connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Macaque/"; 
		thisNetwork = nx.read_graphml(connectomeDataPath + "rhesus_interareal.cortical.network_2.graphml"); 
		thisNetwork = thisNetwork.to_undirected(); 

	if (netName == "catTract"): 
		connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Cat/"; 
		thisNetwork = nx.read_graphml(connectomeDataPath + "mixed.species_brain_1.graphml"); 
		thisNetwork = thisNetwork.to_undirected(); 


# # ########################################################################################################################
# # ## Uncomment for Drosophila: 

# connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Drosophila/"; 
# mat = scipy.io.loadmat(connectomeDataPath + "mac95.mat"); 
# for elem in mat.keys(): 
# 	print(elem); 
# 	print(mat[elem]); 


	if (netName == "mouseVisual2"): 
		# This is just a simple circuit. 
		connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Mouse/VisualCortex/"; 
		thisNetwork = nx.read_graphml(connectomeDataPath + "mouse_visual.cortex_2.graphml"); 
		thisNetwork = thisNetwork.to_undirected(); 


	if (netName == "netCElegans"): 
		connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Celegans/Celegans131/"; 
		netName = "netCElegans"; 

		# # Simple-to-load C elegans network: 
		# thisNetwork = nx.read_graphml(connectomeDataPath + "c.elegans_neural.male_1.graphml"); 
		# thisNetwork = thisNetwork.to_undirected(); 

		# # Complex-to-load C elegans network, but with positions! 
		# thisNetwork = nx.read_edgelist(connectomeDataPath + "C-elegans-frontal.txt", create_using=nx.Graph(), nodetype=int); 
		# nodePositions = np.genfromtxt(connectomeDataPath + "C-elegans-frontal-meta.csv", delimiter=',', skip_header=1, usecols=[2, 3]); 
		# nNodes = len(thisNetwork.nodes()); 
		# nativePositions = {}; 
		# for iNode in range(nNodes): 
		# 	nativePositions[iNode] = nodePositions[iNode,:]; 

		mat = sio.loadmat(connectomeDataPath + "celegans131.mat"); 
		thisNetwork = nx.convert_matrix.from_numpy_matrix(mat["celegans131matrix"]); 
		nativePositions = {}; 
		for node in thisNetwork.nodes(): 
			nativePositions[node] = mat["celegans131positions"][node,:]; 

		metaDataDict["nativePositions"] = nativePositions; 

	if (netName == "netDeutscheAutobahn"): 
		dieAutobahnDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Autobahn/"; 
		netName = "netDeutscheAutobahn"; 

		mat = sio.loadmat(dieAutobahnDataPath + "autobahn.mat"); 
		thisNetwork = nx.convert_matrix.from_numpy_matrix(mat["auto1168"]); 
		# for key in mat.keys(): 
		# 	print(key); 
		# 	print(mat[key]); 

		labeledCities = []; 
		for elem in np.squeeze(mat["auto1168labels"]): 
			labeledCities += [elem[0]]; 

		# # nativePositions = {}; 
		# # for node in thisNetwork.nodes(): 
		# # 	nativePositions[node] = mat["celegans131positions"][node,:]; 

	if (netName == "airports"): 
		# Data was extracted from: https://www.dynamic-connectome.org/resources/

		airportDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Airport/"; 
		netName = "airports"; 

		# Loading network: 
		mat = sio.loadmat(airportDataPath + "air500.mat"); 
		thisNetwork = nx.convert_matrix.from_numpy_matrix(mat["air500matrix"]); 
		nodeNames = np.squeeze(mat["air500labels"]); 
		# Reasigining node names to actual labels: 
		mapping = {}; 
		for (iName, name) in enumerate(nodeNames): 
			mapping[iName] = name[0]; 
		thisNetwork = nx.relabel_nodes(thisNetwork, mapping); 


		# Loading airpot metadata: 
		fIn = open(airportDataPath + "shorterMeta.csv", 'r'); 
		airportMeta = fIn.read().splitlines(); 
		fIn.close(); 
		nativePositions = {}; 
		for line in airportMeta: 
			splitLine = line.split(','); 
			nativePositions[splitLine[2]] = [float(splitLine[1]), float(splitLine[0])]; 

		metaDataDict["nativePositions"] = nativePositions; 

	if ("protein" in netName): 
		speciesName = netName.split("protein")[1]; 
		dataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/ProteinProtein/" + speciesName + '/'; 
		fileName = "interactions.dat"; 

		fIn = open(dataPath + fileName, 'r'); 
		dL = fIn.readlines()[1:]; 
		fIn.close(); 

		edges = []; 
		formerEdge = ('', ''); 
		for line in dL: 
			words = line.split('\t'); 
			newEdge = (words[0],  words[1]); 
			if (newEdge != formerEdge): 
				edges += [newEdge]; 
			formerEdge = newEdge; 


		# # Reading edges: 
		# fIn	= open(dataPath + "edges.csv", 'r'); 
		# edges = []; 
		# nodes = []; 
		# allLines = fIn.readlines(); 
		# for line in allLines: 
		# 	thisEdge = line.split(', '); 
		# 	edges += [(thisEdge[0], thisEdge[1])]; 

		# Building network from edges: 
		thisNetwork = nx.Graph(); 
		thisNetwork.add_edges_from(edges); 

	return (thisNetwork, metaDataDict); 


def generateRandomNetwork(netName, args): 
	"""	generateRandomNetwork function: 

			This function generates random networks. It reads the kind of network to be generated from netName and
			needed parameters from args. 

			Inputs: 
				>> netName: "random"+specs, where specs indicates the kind of random network to be generated. 
				>> args: Dictionary storing the parameters for network generation. 
					> For "ER": nNodes, pConnect. 
					> For "WS": nNodes, nNeighbors, pRewire. 

			Returns: 
				<< thisNetwork: Resulting random network. 

	"""

	if ("ER" in netName): 
		nNodes = args["nNodes"]; 
		pConnect = args["pConnect"]; 
		thisNetwork = nx.erdos_renyi_graph(nNodes, pConnect); 

	if ("WS" in netName): 
		nNodes = args["nNodes"]; 
		nNeighbors = args["nNeighbors"]; 
		pRewire = args["pRewire"]; 
		thisNetwork = nx.watts_strogatz_graph(nNodes, nNeighbors, pRewire); 

	if ("BA" in netName): 
		nNodes = args["nNodes"]; 
		nNewAttachments = args["nNewAttachments"]; 
		thisNetwork = nx.barabasi_albert_graph(nNodesRandom, nNewAttachments); 

	return thisNetwork; 





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


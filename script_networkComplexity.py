"""

	script_networkComplexity.py: 

		The more complex a network is, the more dimensions should be needed to explain its features. Accordingly, more
		eigenvalues would be needed to account for a desired amount of variability. This script explores NDMs to
		measure network complexity in this way. 

		As a first test example, we will measure network complexity for randomly generated networks over a range of some
		parameter (which will be probability of connection or rewiring). 

"""


# Importing relevant libraries: 
import numpy as np; 
import scipy.linalg as la; 
import matplotlib.pyplot as plt; 
import matplotlib as mplt; 
import os, sys; 
import networkx as nx; 
import helper as h; 
import loadHelper as lh; 
from copy import copy; 

# For 3D scatter: 
from mpl_toolkits.mplot3d import Axes3D; 



## Loading all available networks: 

location = "home"; 


# #######################################################################################################################
# # Uncomment for randomly generated networks: 

# thisKey = "None"; 
# # thisNetwork = nx.erdos_renyi_graph(200, 0.1); 
# thisNetwork = nx.watts_strogatz_graph(200, 4, 0.10); 
# # thisNetwork = nx.barabasi_albert_graph(200, 2); 


nRepeats = 50; 
nNodesTarget = 100; 

# pList = np.arange(0.01, 0.2, 0.005); 
# pList = np.arange(0.001, 0.051, 0.001); # Nice for Watts-Strogatz transition. 
pList = np.arange(0.001, 1, 0.05); # Over a longer range. 
networkComplexity = []; 
networkComplexityStd = []; 
allNetworkComplexities = []; 
allPRewire = []; 
# Looping over networks: 
for pRewire in pList: 
	print("p: " + str(pRewire)); 

	thisNetworkComplexity = []; 
	# Looping over number of repeats: 
	nNodesStart = nNodesTarget; 
	for it in range(nRepeats): 

		# Building the network: 
		
		# # Uncomment for watts-strogatz: 
		# thisNetwork = nx.watts_strogatz_graph(nNodesStart, 4, pRewire); 
		# Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 	
		# thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 
		# nNodes = len(thisNetwork.nodes()); 

		# Uncomment for erdos-renyi: 
		# nNodes = 1; 
		# while(nNodes==1): 
		thisNetwork_ = nx.erdos_renyi_graph(nNodesStart, pRewire); 
		connectedComponents = sorted(nx.connected_components(thisNetwork_), key=len, reverse=True); 	

		connectedComponentsComplexity = []; 
		for thisSubGraph in connectedComponents: 
			thisNetwork = nx.Graph(thisNetwork_.subgraph(thisSubGraph)); 
			nNodes = len(thisNetwork.nodes()); 
			if (nNodes==1): 
				# Subgraph has a unique node: 
				nC = 0; 
			else: 
				# Measuring stuff from network nodes: 
				(nodeList, nodesStatistics, includedStatistics, excludedStatistics) = h.computeNodesProperties(thisNetwork); 
				nAllStatistics = len(nodesStatistics); 
				nStatistics = len(includedStatistics); 
				# Loading measurements to matrix to compute covariances and diagonalize: 
				allStatisticsArray = np.zeros([nStatistics, nNodes]); 
				dictIStat = {}; 
				for (iStat, statistic) in enumerate(includedStatistics): 
					allStatisticsArray[iStat,:] = nodesStatistics[statistic]; 
					dictIStat[statistic] = iStat; 


				# Standardizing distro: 
				allStatisticsMean = np.mean(allStatisticsArray, 1); 
				allStatisticsStd = np.std(allStatisticsArray, 1); 
				allStatisticsArray_noStandard = copy(allStatisticsArray); 
				allStatisticsArray = allStatisticsArray - np.transpose(np.repeat(np.array([allStatisticsMean]), nNodes, 0)); 
				allStatisticsArray = np.divide(allStatisticsArray, np.transpose(np.repeat(np.array([allStatisticsStd]), nNodes, 0))); 


				# Computing correlation matrix and diagonalizing: 
				allStatisticsCov = np.cov(allStatisticsArray); 
				allStatisticsCov_noStandard = np.cov(allStatisticsArray_noStandard); 
				correctionFactor = sum(np.diag(allStatisticsCov_noStandard)); 
				(eigVals, eigVects) = np.linalg.eig(allStatisticsCov); 
				eigVals = np.real(eigVals); 
				eigVects = np.real(eigVects); 

				# Computing complexity index: 
				(varianceExplained, varianceExplained_cumul) = h.varianceExplained(eigVals); 
				nC = 1.-sum(varianceExplained_cumul)/nAllStatistics; 
				nC = (1.-sum(varianceExplained_cumul)/nAllStatistics)*correctionFactor; 
				if (len(varianceExplained_cumul)==0): 
					nC = 0.; 

			connectedComponentsComplexity += [nC]; 

		nC = np.mean(connectedComponentsComplexity); 
		thisNetworkComplexity += [nC]; 
		allNetworkComplexities += [nC]; 
		allPRewire += [pRewire]; 

	# print(thisNetworkComplexity); 
	# print(np.mean(thisNetworkComplexity)); 
	# print(np.std(thisNetworkComplexity)); 
	# sys.exit(0); 

	networkComplexity += [np.mean(thisNetworkComplexity)]; 
	networkComplexityStd += [np.std(thisNetworkComplexity)]; 


plt.figure(); 
plt.scatter(allPRewire, allNetworkComplexities); 
plt.errorbar(pList, networkComplexity, networkComplexityStd, color='r'); 

plt.show(); 



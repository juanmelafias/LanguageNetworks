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
import matplotlib.pyplot as plt; 
import matplotlib as mplt; 
import os, sys; 
import networkx as nx; 
import helper as h; 
import loadHelper as lh; 
from copy import copy; 

# For 3D scatter: 
from mpl_toolkits.mplot3d import Axes3D; 


########################################################################################################################
## Uncomment for Macaque brain: 

connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Macaque/"; 
network1 = nx.read_graphml(connectomeDataPath + "rhesus_brain_1.graphml"); 
network1 = network1.to_undirected(); 


########################################################################################################################
## Uncomment for MRI connectome network (I have a lof of such connectomes): 

# # Next networks are in MRI_234: 
connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_234/"; 
networksToLoad = ["993675_repeated10_scale250.graphml", "958976_repeated10_scale250.graphml", "959574_repeated10_scale250.graphml", 
					"100206_repeated10_scale250.graphml", "100307_repeated10_scale250.graphml", "100408_repeated10_scale250.graphml", 
					"100610_repeated10_scale250.graphml", "101006_repeated10_scale250.graphml", "101107_repeated10_scale250.graphml", 
					"101309_repeated10_scale250.graphml"]; 

networks2 = [nx.read_graphml(connectomeDataPath + netName) for netName in networksToLoad]; 




# ## Computing network complexities: 

# allComplexities = []; 
# allCorrections = []; 

# # For macaque: 
# (netComplexity_mac, correctionFactor_mac) = h.computeNetworkComplexity(network1); 
# allComplexities += [netComplexity_mac]; 
# allCorrections += [correctionFactor_mac]; 


# # For humans: 
# netComplexity_hum = []; 
# correctionFactor_hum = []; 
# for (iNet, net) in enumerate(networks2): 
# 	print(iNet); 
# 	(thisNetComplexity_hum, thisCorrectionFactor_hum) = h.computeNetworkComplexity(net); 
# 	allComplexities += [thisNetComplexity_hum]; 
# 	allCorrections += [thisCorrectionFactor_hum]; 


# # Plotting results: 
# plt.figure(); 
# plt.plot(allComplexities, 'o'); 

# plt.figure(); 
# plt.plot(allCorrections, 'o'); 

# plt.figure(); 
# plt.plot(allCorrections, allComplexities, 'o'); 

# plt.figure(); 
# plt.plot(np.multiply(allCorrections, allComplexities), 'o'); 

# plt.show(); 





## Measuring complexity for a lot of human connectomes: 

allNames_MRI_234 = os.listdir(connectomeDataPath); 

# Loading nets and computing complexity: 
netComplexity_hum = []; 
correctionFactor_hum = []; 
for (iNet, netName) in enumerate(allNames_MRI_234[0:100]): 
	print("Processing network " + str(iNet) + ". "); 
	net = nx.read_graphml(connectomeDataPath + netName); 
	Gcc = sorted(nx.connected_components(net), key=len, reverse=True); 
	net = nx.Graph(net.subgraph(Gcc[0])); 
	(thisNetComplexity_hum, thisCorrectionFactor_hum) = h.computeNetworkComplexity(net); 
	netComplexity_hum += [thisNetComplexity_hum]; 
	correctionFactor_hum += [thisCorrectionFactor_hum]; 

# Plotting results: 
plt.figure(); 
plt.plot(netComplexity_hum, 'o'); 

plt.figure(); 
plt.plot(correctionFactor_hum, 'o'); 

plt.figure(); 
plt.plot(correctionFactor_hum, netComplexity_hum, 'o'); 


plt.show(); 



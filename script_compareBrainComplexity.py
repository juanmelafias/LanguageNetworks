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



########################################################################################################################
## Uncomment for Macaque brain: 

connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Macaque/"; 
network1 = nx.read_graphml(connectomeDataPath + "rhesus_brain_1.graphml"); 
network1 = network1.to_undirected(); 


########################################################################################################################
## Uncomment for MRI connectome network (I have a lof of such connectomes): 

# # Next networks are in MRI_234: 
connectomeDataPath = "/home/brigan/Desktop/Research_CNB/Networks/Networks/Human/MRI_234/"; 
networksToLoad = ["993675_repeated10_scale250", "958976_repeated10_scale250", "959574_repeated10_scale250", 
					"100206_repeated10_scale250", "100307_repeated10_scale250", "100408_repeated10_scale250", 
					"100610_repeated10_scale250", "101006_repeated10_scale250", "101107_repeated10_scale250", 
					"101309_repeated10_scale250"]; 

networks2 = [nx.read_graphml(connectomeDataPath + netName + ".graphml") for netName in networksToLoad]; 




# ## Computing network complexities: 

# # For macaque: 
# (netComplexity_mac, correctionFactor_mac) = h.computeNetworkComplexity(network1); 


# # For humans: 
# netComplexity_hum = []; 
# correctionFactor_hum = []; 
# for net in networks2: 
# 	(thisNetComplexity_hum, thisCorrectionFactor_hum) = h.computeNetworkComplexity(net); 
# 	netComplexity_hum += [thisNetComplexity_hum]; 
# 	correctionFactor_hum += [thisCorrectionFactor_hum]; 


# # Plotting results: 
# plt.figure(); 
# plt.plot([netComplexity_mac]+netComplexity_hum, 'o'); 

# plt.figure(); 
# plt.plot([correctionFactor_mac]+correctionFactor_hum, 'o'); 

# plt.show(); 





## Measuring complexity for a lot of human connectomes: 

allNames_MRI_234 = os.listdir(connectomeDataPath); 

# Loading nets and computing complexity: 
netComplexity_hum = []; 
correctionFactor_hum = []; 
for (iNet, netName) in enumerate(allNames_MRI_234): 
	print("Processing network " + str(iNet) + ": \n"); 
	net = nx.read_graphml(connectomeDataPath + netName); 
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



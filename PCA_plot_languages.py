# Importing relevant libraries:
# import pandas as pd
# Importing libraries for I/O and system communication:
import os

import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
import numpy as np
import scipy.cluster.hierarchy as spc

# Importing homebrew libraries:
import helper as h
from utils import build_properties_array_languages

# from copy import copy;
# For 3D scatter:
# from mpl_toolkits.mplot3d import Axes3D;


# import pickle as pkl;
# import scipy.io as sio; # To read .mat files! and .mnx files!

# Importing functions for clustering:
# from sklearn.cluster import KMeans;
# import scipy.cluster.hierarchy as spc;


# import loadHelper as lh;


clusterStyles = {}
clusterStyles[0] = "k"
clusterStyles[1] = "r"
clusterStyles[2] = "g"
clusterStyles[3] = "b"
clusterStyles[4] = "y"
clusterStyles[5] = "m"
clusterStyles[6] = "c"
clusterStyles[7] = "tab:gray"

primaries = True

(valid1, arraymeanproperties, dict_pathologies) = build_properties_array_languages(
	"files/inflected/networks/", primaries
)
(
	valid2,
	arraymeanpropertieslemma,
	dict_pathologies_lemma,
) = build_properties_array_languages("files/lemmatized/networks/", primaries)
arraymeanproperties = h.normalizeProperties(arraymeanproperties)
arraymeanpropertieslemma = h.normalizeProperties(arraymeanpropertieslemma)
if primaries:
	folderpics = "languagesmeanprimaries"
else:
	folderpics = "languagesmeanneighbours"
paths = [f"files/inflected/{folderpics}/", f"files/lemmatized/{folderpics}/"]
filelist = os.listdir("./files/inflected/dictionaries/")
nodeList = [
	file.split(".")[0]
	for file in filelist
]
valid_keys = [valid1, valid2]

for indexlang, includedPropertiesArray in enumerate(
	[arraymeanproperties, arraymeanpropertieslemma]
):
	picsPath = paths[indexlang]
	includedProperties = valid_keys[indexlang]
	allStatisticsCov = np.cov(includedPropertiesArray)
	print(includedPropertiesArray.shape)
	(eigVals, eigVects) = np.linalg.eig(allStatisticsCov)
	eigVals = np.real(eigVals)
	eigVects = np.real(eigVects)

	# Computing PCs with information above noise level according to ref:
	# 	Donoho DL, Gavish M.
	# 	The optimal hard threshold for singular values is 4/√3.
	# 	arXiv preprint arXiv:1305.5870, (2013).
	(noiseThreshold, nKeep) = h.computeComponentsAboveNoise(eigVals)
	print("Noise-trucating PC value is: " + str(noiseThreshold))
	print("According to this, optimal number of PCs kept is: " + str(nKeep))
	print(
		"This is a fraction " + str(float(nKeep) / len(eigVals)) + " of eigenvalues. "
	)

	# Plotting covariance matrix:
	plt.figure()
	plt.savefig(picsPath+'covariance_matrix.pdf', bbox_inches = 'tight')
	#plt.colorbar()

	# Plotting eigenvectors:
	plt.figure()
	plt.imshow(eigVects, interpolation="none", cmap="coolwarm")
	#plt.colorbar()

	# Computing and plotting variance explained:
	(varianceExplained, varianceExplained_cumul) = h.varianceExplained(eigVals)

	plt.figure()
	plt.plot(varianceExplained)
	

	plt.figure()
	plt.plot(varianceExplained_cumul)
	plt.savefig(picsPath+'accum_variance.pdf', bbox_inches = 'tight')

	## Projecting data into eigenspace:
	includedPropertiesArray_ = np.dot(np.transpose(eigVects), includedPropertiesArray)
	print(includedPropertiesArray_.shape)

	'''
		# Using first three PCs as color coding:
		# 	Normalize components to [0,1];
		
		valuesRGB0 = h.convertPC2RGB(includedPropertiesArray_[0, :])
		valuesRGB1 = h.convertPC2RGB(includedPropertiesArray_[1, :])
		valuesRGB2 = h.convertPC2RGB(includedPropertiesArray_[2, :])
		
		# Save hex color values to a list:
		nodeColor = []
		for (iNode, node) in enumerate(nodeList):
		
			nodeColor += [
				mplt.colors.to_hex(
					[valuesRGB0[iNode], valuesRGB1[iNode], valuesRGB2[iNode]]
				)
			]
	'''
			## Dendograms for properties:
		# Coloring nodes according to their cluster: 
	clusterStyles = {}; 
	clusterStyles[0] = 'k'; 
	clusterStyles[1] = 'r'; 
	clusterStyles[2] = 'g'; 
	clusterStyles[3] = 'b'; 
	clusterStyles[4] = 'y'; 
	clusterStyles[5] = 'm'; 
	clusterStyles[6] = 'c'; 
	clusterStyles[7] = 'tab:gray'; 

	# From correlations to distances: 
	pdist = spc.distance.pdist(allStatisticsCov); 
	propertiesLinkage = spc.linkage(pdist, method='complete'); 

	fig = plt.figure(); 
	spc.dendrogram(propertiesLinkage, orientation="right", labels=includedProperties); 
	plt.xlabel("Distance"); 
	plt.ylabel("Node properties"); 
	plt.title("Properties dendrogram"); 
	fig.savefig(picsPath + "propertiesDendogram.pdf"); 


	## Dendograms for data: 
	nodesLinkage = spc.linkage(includedPropertiesArray_.T, 'ward'); 

	distanceThreshold = 45; 
	fig = plt.figure(); 
	spc.dendrogram(nodesLinkage, orientation="right", color_threshold=distanceThreshold); 
	plt.xlabel("Distance"); 
	plt.ylabel("Nodes"); 
	plt.title("Nodes dendrogram"); 
	fig.savefig(picsPath + "nodesDendogram.pdf"); 


	# Coloring according to clusters: 
	# nodeClusters = spc.fcluster(nodesLinkage, distanceThreshold, criterion='distance'); 
	nClusters = 5; 
	nodeClusters5 = spc.fcluster(nodesLinkage, 5, criterion="maxclust"); 
	nodeColor = []; 
	for (iNode, node) in enumerate(nodeList): 
		nodeColor += [clusterStyles[nodeClusters5[iNode]-1]]; 
	# PC1-PC2:
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.scatter(
		includedPropertiesArray_[0, :], includedPropertiesArray_[1, :], c=nodeColor
	)
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	plt.title("Nodes projected in PCs")
	fig.savefig(picsPath + "projection_PCs1-2.pdf", bbox_inches = 'tight')

	# PC1-PC3:
	fig = plt.figure()
	plt.scatter(
		includedPropertiesArray_[0, :], includedPropertiesArray_[2, :], c=nodeColor
	)
	plt.xlabel("PC1")
	plt.ylabel("PC3")
	plt.title("Nodes projected in PCs")
	fig.savefig(picsPath + "projection_PCs1-3.pdf", bbox_inches = 'tight')

	# PC1-PC2-PC3:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.scatter(
		includedPropertiesArray_[0, :],
		includedPropertiesArray_[1, :],
		includedPropertiesArray_[2, :],
		c=nodeColor,
	)
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")
	plt.title("Nodes projected in PCs")
	fig.savefig(picsPath + "projection_PCs1-2-3.pdf", bbox_inches = 'tight')

	# Plotting in network space:
	""""
	fig = plt.figure(); 
	ax = fig.add_subplot(111); 
	nx.draw(thisNetwork, with_labels=False, pos=nx.kamada_kawai_layout(thisNetwork), node_color=nodeColor, edge_color="tab:gray"); 
	ax.set_aspect("equal"); 
	plt.title("PC colors projected in network layout"); 
	fig.savefig(picsPath + "networkColoredWithPCs_netLayout.pdf");
	"""

	pdist = spc.distance.pdist(allStatisticsCov)
	propertiesLinkage = spc.linkage(pdist, method="complete")

	fig = plt.figure()
	spc.dendrogram(propertiesLinkage, orientation="right", labels=includedProperties)
	plt.xlabel("Distance")
	plt.ylabel("Node properties")
	plt.title("Properties dendrogram")
	fig.savefig(picsPath + "propertiesDendogram.pdf", bbox_inches = 'tight')

	## Dendograms for data:
	nodesLinkage = spc.linkage(includedPropertiesArray_.T, "ward")

	distanceThreshold = 45
	print(nodesLinkage.shape)
	fig = plt.figure()
	spc.dendrogram(
		nodesLinkage,
		orientation="right",
		color_threshold=distanceThreshold,
		labels=nodeList,
	)
	plt.xlabel("Distance")
	plt.ylabel("Nodes")
	plt.title("Nodes dendrogram")
	fig.savefig(picsPath + "nodesDendogram.pdf", bbox_inches = 'tight')

	# Coloring according to clusters:
	# nodeClusters = spc.fcluster(nodesLinkage, distanceThreshold, criterion='distance');
	nClusters = 5
	nodeClusters = spc.fcluster(nodesLinkage, nClusters, criterion="maxclust")
	nodeClusterColor = []
	for (iNode, node) in enumerate(nodeList):
		nodeClusterColor += [clusterStyles[nodeClusters[iNode] - 1]]

	# Plotting in eigenspace:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.scatter(
		includedPropertiesArray_[0, :],
		includedPropertiesArray_[1, :],
		includedPropertiesArray_[2, :],
		c=nodeClusterColor,
	)
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")
	plt.title("Clusters (dendogram) in eigenspace")
	fig.savefig(picsPath + "dendogramClusters_eigenspace.pdf", bbox_inches = 'tight')

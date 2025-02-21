# Importing relevant libraries:
# import pandas as pd
# Importing libraries for I/O and system communication:
import os

import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
import numpy as np
import pandas as pd
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
filelist = os.listdir('./files/inflected/dictionaries/')
languagelist = [file.split('.')[0] for file in filelist if file not in ['Ancient_Greek']]

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
	x = np.arange(len(includedProperties))
	plt.imshow(allStatisticsCov, interpolation="none"); 
	plt.xticks(x, includedProperties,fontsize = 8,rotation = 90)
	plt.yticks(x, includedProperties,fontsize = 8)
	plt.savefig(picsPath+'covariance_matrix.pdf', bbox_inches = 'tight',dpi = 150)
	plt.colorbar()

	# Plotting eigenvectors:
	plt.figure()
	x = np.arange(len(includedProperties))
	plt.imshow(eigVects, interpolation="none", cmap="coolwarm")
	plt.colorbar()

	# Computing and plotting variance explained:
	(varianceExplained, varianceExplained_cumul) = h.varianceExplained(eigVals)


	plt.figure()
	plt.plot(varianceExplained,marker = 'o')
	plt.ylabel('%')
	plt.xlabel('PCs')
	plt.title('Explained variance')
	plt.savefig(picsPath+'explained_variance.pdf', bbox_inches = 'tight')
	

	plt.figure()
	plt.plot(varianceExplained_cumul,marker = 'o')
	plt.ylabel('%')
	plt.xlabel('PCs')
	plt.title('Explained variance')
	#plt.xticks(x, includedProperties,fontsize = 8)
	plt.savefig(picsPath+'accum_variance.pdf', bbox_inches = 'tight')

	## Projecting data into eigenspace:
	includedPropertiesArray_ = np.dot(np.transpose(eigVects), includedPropertiesArray)
	print(includedPropertiesArray_.shape)
	if indexlang == 0: #inflected
		np.save('eigvects_inlangcomp.npy',eigVects)
	else:	#lemmatized
		eigVects_inlangcomp = np.load("eigvects_inlangcomp.npy")
		includedPropertiesArray_inlangcomp = np.dot(np.transpose(eigVects_inlangcomp), includedPropertiesArray); 
		print(f'sum is {np.dot(eigVects_inlangcomp[1,:], eigVects[1,:])}')
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

	fig = plt.figure(figsize=(3,4),dpi = 200); 
	spc.dendrogram(propertiesLinkage, orientation="right", labels=includedProperties); 
	plt.xlabel("Distance"); 
	
	plt.ylabel("Node properties"); 
	plt.title("Properties dendrogram"); 
	
	fig.savefig(picsPath + "propertiesDendogram.pdf", bbox_inches = 'tight', dpi = 200); 

	
	## Dendograms for data: 
	nodesLinkage = spc.linkage(includedPropertiesArray_.T, 'ward'); 

	distanceThreshold = 45; 
	fig = plt.figure(figsize=(4.8,8.4),dpi = 200); 
	
	spc.dendrogram(nodesLinkage, orientation="right", color_threshold=distanceThreshold, leaf_font_size=13, labels=languagelist); 
	
	plt.xlabel("Distance"); 
	plt.ylabel("Nodes"); 
	plt.title("Nodes dendrogram"); 
	

	fig.savefig(picsPath + "nodesDendogram.pdf", dpi =200,bbox_inches = 'tight'); 


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
	fig.savefig(picsPath + "propertiesDendogram.pdf", bbox_inches = 'tight', dpi = 150)

	## Dendograms for data:
	nodesLinkage = spc.linkage(includedPropertiesArray_.T, "ward")

	distanceThreshold = 45
	

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
	fig.savefig(picsPath + "dendogramClusters_eigenspace.pdf", bbox_inches = 'tight', dpi = 150)
	dflangs = pd.DataFrame()
	dflangs['languages'] = nodeList
	dflangs['pc1'] = includedPropertiesArray_[0, :]
	dflangs['pc2'] = includedPropertiesArray_[1, :]
	dflangs['pc3'] = includedPropertiesArray_[2, :]
	if indexlang == 1:
		dflangs['pc1_i'] = includedPropertiesArray_inlangcomp[0,:]
		dflangs['pc2_i'] = includedPropertiesArray_inlangcomp[1,:]
		dflangs['pc3_i'] = includedPropertiesArray_inlangcomp[2,:]
	iol_list = ['inflected','lemmatized']
	dflangs['nc5'] = nodeColor
	dflangs['iol'] = iol_list[indexlang]
	if primaries:
		dflangs['prim_or_neigh'] = 'primaries'
	else:
		dflangs['prim_or_neigh'] = 'neighbours'
	dflangs.to_csv(picsPath + 'dflangcomp.csv')
	
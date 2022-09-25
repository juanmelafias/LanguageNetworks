"""

	quickNetworkBuilder.py

		This is a brief script to build a syntax network out of .conll files. 

"""


import numpy as np; 
import matplotlib.pyplot as plt; 
import networkx as nx; 
import os, sys; 
from copy import copy; 
from mpl_toolkits.mplot3d import Axes3D; 
from pathlib import Path



# Naming paths and files: 
root = Path("./stb.conll")
pathName = root.absolute() 
filePath = root / Path("stb.conll"); 

# Loading all lines: 
fIn = open(filePath,encoding = "utf8"); 
allLines = fIn.read().split('\n'); 
fIn.close(); 


## Let us read just the first 10 sentences: 

nSentences = 100; 
dicConnections = {}; # Dictionary to store connections. 

# Each sentence is coded as a block of lines, each containing the information of a single word. 
# Each block ends with a blank line that marks the end of the sentence. 
iLine = 0; 		# Index to iterate over lines in the file. 
iSentence = 0; 	# Index to keep track of processed sentences. 
fGo = True; 
dicSentenceIDs = {}; 	# To keep track of words within sentence. 
sentenceHeads = []; 
while (fGo): 

	thisLine = allLines[iLine]; 
	print(thisLine); 

	if (thisLine != ''): 
		# This line codes a valid word. We need to process it: 
		wordInfo = thisLine.split('\t'); 	# Info is separated by tabs. 
		# Adding word ID to dicSenteceID: 
		dicSentenceIDs[wordInfo[0]] = wordInfo[1]; 
		# Adding the head of this word to sentenceConnections: 
		sentenceHeads += [wordInfo[6]]; 

	else: 
		# End of sentence! 

		# Making sure that all words are entries of dicConnections: 
		for key in dicSentenceIDs.keys(): 
			if (dicSentenceIDs[key] not in dicConnections.keys()): 
				dicConnections[dicSentenceIDs[key]] = []; 

		# Adding connections to dicConnections: 
		for (iWord, head) in enumerate(sentenceHeads): 
			# If the head is 0, this is the root node of the tree or a punctuation symbol. We don't process this! 
			if (head != '0'): 
				dicConnections[dicSentenceIDs[str(iWord+1)]] += [dicSentenceIDs[head]]; 
				dicConnections[dicSentenceIDs[head]] += [dicSentenceIDs[str(iWord+1)]]; 

		# Clearing sentence buffers: 
		dicSentenceIDs = {}; 
		sentenceHeads = []; 

		iSentence += 1; 

	iLine += 1; 
	if (iSentence==nSentences): 
		break; 



# Properly building the network: 
net = nx.Graph(); 
net.add_nodes_from(dicConnections.keys()); 
for (k, v) in dicConnections.items():
    net.add_edges_from(([(k, t) for t in v])); 

plt.figure(); 
nx.draw(net, with_labels=False, pos=nx.kamada_kawai_layout(net)); 

plt.show(); 




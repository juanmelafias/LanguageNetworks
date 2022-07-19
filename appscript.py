from telnetlib import WONT
import pandas as pd
import numpy as np; 
import matplotlib.pyplot as plt; 
import matplotlib as mplt; 
import networkx as nx; 
from copy import copy; 
# For 3D scatter: 
from mpl_toolkits.mplot3d import Axes3D; 
import plotly.graph_objects as go

# Importing libraries for I/O and system communication: 
import os, sys; 
from pathlib import Path
import pickle as pkl; 
import scipy.io as sio; # To read .mat files! and .mnx files! 

# Importing functions for clustering: 
from sklearn.cluster import KMeans;  
import scipy.cluster.hierarchy as spc; 
import plotly.express as px

# Importing homebrew libraries: 
import helper as h; 
import loadHelper as lh; 

#importing functions

from utils import csv2df,json2dict,load_network,connect,get_traces, adjust_trace_colors
from constants import SERVER, DATABASE, USERNAME, PASSWORD, DRIVERS, DRIVER
root = os.getcwd()
filelist = os.listdir('./files/inflected/dictionaries/')
languagelist = [file.split('.')[0] for file in filelist]
for bool in [False,True]:
    for language in languagelist:
	#Loading data
        print(language)
        cnxn = connect(SERVER, DATABASE, USERNAME, PASSWORD, DRIVER)
        root = os.getcwd()
        netName = language
        lemmatized = bool
        if lemmatized:
            iol = 'lemmatized'
        else:
            iol='inflected'
        picsPath = f'./pics/{iol}/{netName}/'
        #picsPath.mkdir(picsPath, exist_ok = True)
        image_path = root / Path("pics") / Path(iol) / Path(language)
        image_path.mkdir(parents=True, exist_ok=True)
        langframe = csv2df(f'files/{iol}/dataframes/{netName}.csv')
        mostfreq =langframe.unique_id.to_list()
        thisNetwork = load_network(f'files/{iol}/dictionaries/{netName}.json')
        thisNetwork=thisNetwork.subgraph(mostfreq)
        word_list = langframe.word.to_list()
        POS_list = langframe.POS.to_list()

        indexes = [i for i in range(1,len(word_list)+1,1)]
        dict_rank = dict(zip(word_list,indexes))
        dict_pos = dict(zip(word_list,POS_list))

        netPath=f'./files/{iol}/networks/'
        Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
        thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 
        nNodes = len(thisNetwork.nodes());

        nEdges = thisNetwork.number_of_edges(); 
        nodeList = [node for node in thisNetwork.nodes()]

        dict_words = dict(zip(langframe['unique_id'].to_list(),langframe['word'].to_list()))
        wordList = [dict_words[node] for node in nodeList]
        rankList = [dict_rank[word] for word in wordList]
        POSList = [dict_pos[word] for word in wordList]
        ranking = range(1,len(wordList)+1,1)
        df = pd.DataFrame()
        df['id_palabra'] = np.array(nodeList) 
        df['palabra'] = pd.Series(wordList)
        #pd.DataFrame(data = [np.array(nodeList),np.array(wordList)], columns = ['id_palabra', 'palabra'])
        df['language'] = netName
        df['POS'] = pd.Series(POSList)
        df['ranking'] = np.array(rankList)
        if lemmatized:
            df['lemmatized'] = 'yes'
        else:
            df['lemmatized'] = 'no'

        fNeighborMean = True; 
        fNeighborStd = True; 
        if (("random" not in netName) and (os.path.isfile(netPath + netName + "_nodeList.csv")) 
                                        and (os.path.isfile(netPath + netName + "_properties.pkl"))): 
            # Files already exist with properties that have been computed. We can proceed with these: 
            # (nodeList, propertiesDict) = h.readNetworkProperties(netName, netPath); 
            (nodeList, propertiesDict) = h.readNetworkProperties(netName, netPath, fNeighborMean, fNeighborStd); 
            (includedProperties, excludedProperties) = h.findPathologicalProperties(propertiesDict); 
        else: 
            # Properties have not been saved for this network and need to be computed: 
            (nodeList, propertiesDict, includedProperties, excludedProperties) = h.computeNodesProperties(thisNetwork, 
                                                                                                fNeighborMean, fNeighborStd); 
            if ("random" not in netName): 
                h.writeNetworkProperties(netName, netPath, nodeList, propertiesDict); 
        # # Just in case, we keep these lines of code in case we wish to manually force the re-computation for some network: 
        # (nodeList, propertiesDict, includedProperties, excludedProperties) = h.computeNodesProperties(thisNetwork); 
        # # (nodeList, propertiesDict, includedProperties, excludedProperties) = h.computeNodesProperties(thisNetwork, False, False); 
        # h.writeNetworkProperties(netName, netPath, nodeList, propertiesDict); 


        # In either case, we retain only those properties that are not pathological. 
        # We build a numpy array to work with them: 
        includedPropertiesArray = h.buildPropertiesArray(propertiesDict, includedProperties); 
        includedPropertiesArray = h.normalizeProperties(includedPropertiesArray); 


        print("Analysis includes the following properties: "); 
        for (iP, thisProperty) in enumerate(includedProperties): 
            print('\t' + str(iP+1) + ": " + thisProperty); 
        # for (iP, thisProperty) in enumerate(includedProperties): 
        # 	print('\t' + str(iP+1) + ": " + thisProperty); 
        # 	print("\t\t" + str(includedPropertiesArray[iP])); 

        ## Computing correlation matrix and diagonalizing: 
        allStatisticsCov = np.cov(includedPropertiesArray); 
        (eigVals, eigVects) = np.linalg.eig(allStatisticsCov); 
        eigVals = np.real(eigVals); 
        eigVects = np.real(eigVects); 
        (noiseThreshold, nKeep) = h.computeComponentsAboveNoise(eigVals); 
        print("Noise-trucating PC value is: " + str(noiseThreshold)); 
        print("According to this, optimal number of PCs kept is: " + str(nKeep)); 
        print("This is a fraction " + str(float(nKeep)/len(eigVals)) + " of eigenvalues. "); 

        '''
        # Plotting covariance matrix: 
        plt.figure(); 
        plt.imshow(allStatisticsCov, interpolation="none"); 
        plt.colorbar(); 

        # Plotting eigenvectors: 
        plt.figure(); 
        plt.imshow(eigVects, interpolation="none", cmap="coolwarm"); 
        plt.colorbar(); 


        # Computing and plotting variance explained: 
        (varianceExplained, varianceExplained_cumul) = h.varianceExplained(eigVals); 

        plt.figure(); 
        plt.plot(varianceExplained); 

        plt.figure(); 
        plt.plot(varianceExplained_cumul); 
        '''

        nodeList = [node for node in thisNetwork.nodes()]
        ## Projecting data into eigenspace: 
        includedPropertiesArray_ = np.dot(np.transpose(eigVects), includedPropertiesArray); 

        # Using first three PCs as color coding: 
        # 	Normalize components to [0,1]; 
        valuesRGB0 = h.convertPC2RGB(includedPropertiesArray_[0,:]); 
        valuesRGB1 = h.convertPC2RGB(includedPropertiesArray_[1,:]); 
        valuesRGB2 = h.convertPC2RGB(includedPropertiesArray_[2,:]); 

        df['pc1']=pd.Series(includedPropertiesArray_[0,:]); 
        df['pc1']=df['pc1'].apply(lambda x: np.round(x,4))
        df['pc2']=pd.Series(includedPropertiesArray_[1,:]); 
        df['pc2']=df['pc2'].apply(lambda x: np.round(x,4)) 
        df['pc3']=pd.Series(includedPropertiesArray_[2,:]); 
        df['pc3']=df['pc3'].apply(lambda x: np.round(x,4)) 
        df['rgb1']=pd.Series(valuesRGB0)  
        df['rgb1']=df['rgb1'].apply(lambda x: np.round(x,4))
        df['rgb2']=pd.Series(valuesRGB1)  
        df['rgb2']=df['rgb2'].apply(lambda x: np.round(x,4))
        df['rgb3']=pd.Series(valuesRGB2) 
        df['rgb3']=df['rgb3'].apply(lambda x: np.round(x,4))

        cursor = cnxn.cursor()
        # Insert Dataframe into SQL Server:

        # Save hex color values to a list: 

        nodeColor = []; 
        for (iNode, node) in enumerate(nodeList): 
            nodeColor += [mplt.colors.to_hex([valuesRGB0[iNode], valuesRGB1[iNode], valuesRGB2[iNode]])]; 
        # PC1-PC2-PC3: 

        '''
        fig = plt.figure(); 
        ax = fig.add_subplot(111, projection='3d'); 
        ax.scatter(includedPropertiesArray_[0,:], includedPropertiesArray_[1,:], includedPropertiesArray_[2,:], c=nodeColor); 
        ax.set_xlabel("PC1"); 
        ax.set_ylabel("PC2"); 
        ax.set_zlabel("PC3"); 
        plt.title("Nodes projected in PCs"); 
        fig.savefig(picsPath + "projection_PCs1-2-3.pdf"); 
        '''
        #plotting graphs
        
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


        ## Dendograms for data: 
        nodesLinkage = spc.linkage(includedPropertiesArray_.T, 'ward'); 

       


        # Coloring according to clusters: 
        # nodeClusters = spc.fcluster(nodesLinkage, distanceThreshold, criterion='distance'); 
        nClusters = 5; 
        nodeClusters5 = spc.fcluster(nodesLinkage, 5, criterion="maxclust"); 
        nodeClusters4 = spc.fcluster(nodesLinkage, 4, criterion="maxclust"); 
        nodeClusters3 = spc.fcluster(nodesLinkage, 3, criterion="maxclust"); 
        nodeClusters2 = spc.fcluster(nodesLinkage, 2, criterion="maxclust");

        nodeClusterColor = []; 
        for (iNode, node) in enumerate(nodeList): 
            nodeClusterColor += [clusterStyles[nodeClusters5[iNode]-1]]; 

        df['nc5'] = nodeClusters5
        df['nc5'] = df['nc5'].apply(lambda x: str(x))
        df['nc4'] = nodeClusters4
        df['nc4'] = df['nc4'].apply(lambda x: str(x))
        df['nc3'] = nodeClusters3
        df['nc3'] = df['nc3'].apply(lambda x: str(x))
        df['nc2'] = nodeClusters2
        df['nc2'] = df['nc2'].apply(lambda x: str(x))
        '''
        fig = px.scatter_3d(df, x='pc1', y='pc2', z='pc3',
                            color='nc5', symbol='palabra')
        fig.show()
        
        node_trace,edge_trace = get_traces(thisNetwork)
        node_trace = adjust_trace_colors(node_trace,thisNetwork)
        fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
        title='<br>Network graph made with Python',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002 ) ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )
        fig.show()
        '''
        df.to_csv(f'files/{iol}/dfplot/{netName}.csv')

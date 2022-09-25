import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
import common.helper as h
from common.utilsstreamlit import read_plot_info
from msilib.schema import Component
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import networkx as nx
from pylab import colorbar
from matplotlib.lines import Line2D
import plotly.io as pio
pio.kaleido.scope.mathjax = None
from common.utils import csv2df,load_network,plotly_graph,json2dict
from turtle import color, width

def plot_fractions_per_tc(fracposoverpos, fracposovertc, labels,tc, path):
    fig, ax = plt.subplots()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(
        x - width / 2, fracposoverpos,edgecolor = 'black', width = width, label="frac POS over all words", color='#0b8eab'
    )
    rects2 = ax.bar(
        x + width / 2, fracposovertc,edgecolor = 'black', width = width, label="frac over TC", color='#ffc152'
    )

    plt.ylabel("Relative frequency", fontsize=12)
    plt.title(f"Distribution of POS tags across TC {tc}")
    #ax.set_xticks(x, labels)
    ax.legend()

    '''
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    '''

    plt.xticks(x, labels,fontsize = 8)
    #plt.yticks(fontsize=12)
    #plt.show()
    plt.savefig(path, dpi=200, bbox_inches = 'tight')
    #plt.close(fig)

def plot_fractions_per_pos(fractcoverpos,fractcoverwords,labels,pos, path):
    fig, ax = plt.subplots()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(
        x + width / 2 , fractcoverpos,edgecolor = 'black', width = width, label="frac over POS", color='#ffc152'
    )
    rects2 = ax.bar(
        x - width / 2, fractcoverwords,edgecolor = 'black', width = width, label="frac TC over words", color='#0b8eab'
    )

    plt.ylabel("Relative frequency", fontsize=12)
    plt.title(f"Distribution of {pos} POS tag across TCs")
    #ax.set_xticks(x, labels)
    ax.legend()

    '''
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    '''

    plt.xticks(x, labels)
    plt.xlabel("Topological community")
    #plt.yticks(fontsize=12)
    #plt.show()
    plt.savefig(path, dpi=200,bbox_inches = 'tight')
    #plt.close(fig)

def plot_entropy(entropy,labels,tag,path):
    fig, ax = plt.subplots()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(
        x , entropy,edgecolor = 'black', width = width, color='#0b8eab'
    )
    

    plt.ylabel("Entropy", fontsize=12)
    plt.title(f"Distribution of entropy {tag}")
    #ax.set_xticks(x, labels)


    '''
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    '''

    plt.xticks(x, labels,fontsize = 8)
    plt.savefig(path, dpi=200,bbox_inches = 'tight')

def language_analysis(netName,lemmatized,fNeighborMean = True, fNeighborStd = False ):
    if lemmatized:
        iol = 'lemmatized'
        netPath = 'files/lemmatized/networks/'
        folderframe = 'files/lemmatized/dfplot/'
        saving = 'files/lemmatized/analysis/'
    else:
        iol = 'inflected'
        netPath = 'files/inflected/networks/'
        folderframe = 'files/inflected/dfplot/'
        saving = 'files/inflected/analysis/'
    picsdir = f'{saving}{netName}'
    if not os.path.exists(picsdir):
        os.mkdir(picsdir)
    #Reading saved properties from the network
    (nodeList, propertiesDict) = h.readNetworkProperties(netName, netPath, fNeighborMean, fNeighborStd); 
    (includedProperties, excludedProperties) = h.findPathologicalProperties(propertiesDict);  
    goodkeys = [key for key in propertiesDict.keys()][:17]
    goodkeys.append('clustering_neighborMean')

    #Starting by eliminating closeness vitality, which is probably broken

    goodprops = {key:propertiesDict[key] for key in goodkeys if key not in ['closenessVitality','componentSize']}
    #loading our dataframes with all the  info
    dfplot = read_plot_info(netName,0,iol)
    #geting our pos tags
    postags = dfplot.groupby(by = 'POS').mean().index
    #getting our list of nodes
    langframe = csv2df(f'files/{iol}/dataframes/{netName}.csv')
    mostfreq =langframe.unique_id.to_list()
    thisNetwork = load_network(f'files/{iol}/dictionaries/{netName}.json')
    thisNetwork=thisNetwork.subgraph(mostfreq)
    Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
    thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0]));
    nodeList = thisNetwork.nodes()
    flist = []
    plist = []
    for key in goodprops.keys():
        dfanova = pd.DataFrame()
        prop = goodprops[key]
        zipdict = dict(zip(nodeList,prop))
        orderednodes = dfplot['id_palabra'].to_list()
        #adding node properties to our dataframe as columns
        dfplot[key] = pd.Series([zipdict[node] for node in orderednodes])
        valuesperpos = []
        #creating the distributions per pos and per tag
        for tag in postags:
            values = dfplot[dfplot['POS'] == tag][key].to_list()
            valuesperpos.append(values)
            
        #plotting the results
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        bp = ax.violinplot(valuesperpos)
        ax.set_xticks([i for i in range(1,len(postags)+1,1)])
        ax.set_xticklabels(postags)
        plt.title(f'Distribution per POS of {key}')
        plt.xlabel('POS tag')
        plt.ylabel(key)
        finaldir = picsdir + '/violinperpos'
        if not os.path.exists(finaldir):
            os.mkdir(finaldir)
        plt.savefig(f'{finaldir}/{key}.pdf',bbox_inches = 'tight')
        
        '''fvalue,pvalue = stats.f_oneway(valuesperpos[0], valuesperpos[1], valuesperpos[2], valuesperpos[3], valuesperpos[4], 
        valuesperpos[5], valuesperpos[6], valuesperpos[7], valuesperpos[8], valuesperpos[9], valuesperpos[10], 
        valuesperpos[11], valuesperpos[12] )
        print(f"f-value is {fvalue} and p-value is {pvalue} for {key} property")'''
        '''flist.append(fvalue)
        plist.append(pvalue)'''
    '''dfvil = pd.DataFrame()
    dfvil['prop'] = [prop for prop in goodprops.keys()]   
    dfvil['f-value'] = flist
    dfvil['p-value'] = plist
    dfvil.to_csv('p_f_values_pos.csv')
'''
    postags = dfplot.groupby(by = 'POS').mean().index
    for tag in postags:
        deg_plot = np.array(dfplot[dfplot['POS']==tag]['degree'].to_list()).reshape((-1,1))
        deg_mean_plot = np.array(dfplot[dfplot['POS']==tag]['degree_neighborMean'].to_list())
        model = LinearRegression()
        model.fit(deg_plot,deg_mean_plot)
        r_sq = model.score(deg_plot,deg_mean_plot)
        y_pred = deg_plot*model.coef_ + model.intercept_
        fig = plt.figure()
        plt.scatter(deg_plot,deg_mean_plot)
        plt.plot(deg_plot,y_pred,color = 'r')
        plt.legend(['nodes','LS fit'])
        plt.title(f'Assortativity for {tag}')
        plt.xlabel('node degree')
        plt.ylabel('mean neighbor degree')
        finaldir = picsdir+'/assortperpos'
        if not os.path.exists(finaldir):
            os.mkdir(finaldir)
        if tag != 'AUX':
            fig.savefig(f'{finaldir}/{tag}.pdf',bbox_inches = 'tight')
        else:
            fig.savefig(f'{finaldir}/auxi.pdf',bbox_inches = 'tight')
    legendgraph = {1:'OuterPeriphery',2:'InnerPeriphery', 3:'SuperCore',4:'InerConnectors',5:'OuterConnectors'}
    for tc in range(1,6,1):
        deg_plot = np.array(dfplot[dfplot['nc5']==tc]['degree'].to_list()).reshape((-1,1))
        deg_mean_plot = np.array(dfplot[dfplot['nc5']==tc]['degree_neighborMean'].to_list())
        model = LinearRegression()
        model.fit(deg_plot,deg_mean_plot)
        r_sq = model.score(deg_plot,deg_mean_plot)
        y_pred = deg_plot*model.coef_ + model.intercept_
        fig = plt.figure()
        plt.scatter(deg_plot,deg_mean_plot)
        plt.plot(deg_plot,y_pred,color = 'r')
        plt.legend(['nodes','LS fit'])
        plt.title(f'Assortativity for TC {legendgraph[tc]}')
        plt.xlabel('node degree')
        plt.ylabel('mean neighbor degree')
        finaldir = picsdir+'/assortpertc'
        if not os.path.exists(finaldir):
            os.mkdir(finaldir)
        fig.savefig(f'{finaldir}/{tc}.png',bbox_inches = 'tight')
    #plotting the network, standard procedure for loading them up
    langframe = csv2df(f'files/{iol}/dataframes/{netName}.csv')
    mostfreq =langframe.unique_id.to_list()
    thisNetwork = load_network(f'files/{iol}/dictionaries/{netName}.json')
    thisNetwork=thisNetwork.subgraph(mostfreq)
    Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
    thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0]));
    #Now we have to follow the order of g.nodes(), so we will have to rearrange
    #the colors in the dataframe in that order
    nodeList = [int(node) for node in thisNetwork.nodes()]
    dictpalabras = dict(zip(dfplot['id_palabra'].to_list(),dfplot['palabra'].to_list()))
    realpalabras = [dictpalabras[node] for node in nodeList]
    colors2 = dict(zip(dfplot['id_palabra'].to_list(),dfplot['nc2']))
    colors3 = dict(zip(dfplot['id_palabra'].to_list(),dfplot['nc3']))
    colors4 = dict(zip(dfplot['id_palabra'].to_list(),dfplot['nc4']))
    colors5 = dict(zip(dfplot['id_palabra'].to_list(),dfplot['nc5']))
    realcolors2 = [colors2[node] for node in nodeList]
    realcolors3 = [colors3[node] for node in nodeList]
    realcolors4 = [colors4[node] for node in nodeList]
    realcolors5 = [colors5[node] for node in nodeList]
    colors2 = {1:'b',2:'g',3:'r',4:'c',5:'m'}
    truecolors2 = [colors2[tc] for tc in realcolors2]
    truecolors3 = [colors2[tc] for tc in realcolors3]
    truecolors4 = [colors2[tc] for tc in realcolors4]
    truecolors5 = [colors2[tc] for tc in realcolors5]
    colorsinverse2 = {value:key for key,value in colors2.items()}
    for colorset in [realcolors2,realcolors3, realcolors4,realcolors5]:
        plotly_graph(thisNetwork,colorset,realpalabras,True,picsdir)
    #same violin plots as before, but know grouping by topological communities, nc= 5 has been chosem
    for key in goodprops.keys():
        prop = goodprops[key]
        zipdict = dict(zip(nodeList,prop))
        orderednodes = dfplot['id_palabra'].to_list()
        dfplot[key] = pd.Series([zipdict[node] for node in orderednodes])
        valuesperpos = []
        topcoms = [i for i in range(1,6,1)]
        for topcom in topcoms:
            values = dfplot[dfplot['nc5'] == topcom][key].to_list()
            valuesperpos.append(values)
        

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        bp = ax.violinplot(valuesperpos)
        ax.set_xticks([i for i in range(1,len(topcoms)+1,1)])
        ax.set_xticklabels(topcoms)
        plt.title(f'Distribution per topological community of {key}')
        plt.xlabel('topological community')
        plt.ylabel(key)
        #plt.show()
        finaldir = picsdir+'/violinpertc'
        if not os.path.exists(finaldir):
            os.mkdir(finaldir)
        fig.savefig(f'{finaldir}/{key}.png',bbox_inches = 'tight')
        fvalue,pvalue = stats.f_oneway(valuesperpos[0], valuesperpos[1], valuesperpos[2], valuesperpos[3], valuesperpos[4] )
        #print(f"f-value is {fvalue} and p-value is {pvalue} for {key} property")
    #Here we are going to study the distribution of words by part of speech through the topological communities
    cpertcandpos=dfplot.groupby(by=['nc5','POS']).count()['palabra']
    cpertc = dfplot.groupby(by=['nc5']).count()['palabra']
    cperpos = dfplot.groupby(by=['POS']).count()['palabra']
    dict_cpertc = dict(cpertc)
    dict_cperpos = dict(cperpos)
    dict_cpertcandpos= dict(cpertcandpos)
    #dict_fracposovertc tells us the fraction that a particular part of speech represents across all other POS in a particular TC
    dict_fracposovertc = {key:(dict_cpertcandpos[key]/dict_cpertc[key[0]]) for key in dict_cpertcandpos} 
    #dict_fracposoverpos tells us the fraction that a particular part of speech in a particular TC over all words ibelonging to that POS
    dict_fracposoverpos = {key:(dict_cpertcandpos[key]/dict_cperpos[key[1]]) for key in dict_cpertcandpos}
    rel_freq_pos_over_total = dict((dfplot.groupby(by='POS').count()['palabra']/len(dfplot)).transpose())
    entropy_H = []
    for tc in range(1,6,1):
        list_fracposovertc = []
        list_fracposoverpos = []
        list_relfreposovertotal = []
        list_labelpos = []
        for key in dict_fracposovertc.keys():
            if tc == key[0]:
                pos = key[1]
                list_fracposovertc.append(dict_fracposovertc[key])
                list_fracposoverpos.append(dict_fracposoverpos[key])
                list_labelpos.append(pos)
                list_relfreposovertotal.append(rel_freq_pos_over_total[key[1]])
        finaldir = picsdir+'/posacrosstc'
        if not os.path.exists(finaldir):
            os.mkdir(finaldir)
        path = f'{finaldir}/tc{tc}.pdf'
        plot_fractions_per_tc(list_relfreposovertotal, list_fracposovertc, list_labelpos,tc,path)
        entropy_H.append(-np.dot(np.array(list_fracposovertc),np.log(np.array(list_fracposovertc))))
    path = f'{finaldir}/entropyH.pdf'
    plot_entropy(entropy_H,['1','2','3','4','5'],'H',path)
    #Now we do the same over each POS tag, to see how they distribute acroos topological communities
    entropy_S = []
    fractcoverwords = [value/len(dfplot) for value in dict_cpertc.values()] 
    for tag in postags:
        list_fractcoverpos = []
        total_labels =[str(i) for i in range(1,6,1)]
        list_labelstc = []
        def_frac = [0, 0, 0, 0, 0]
        for key in dict_fracposovertc.keys():
            tc = key[0]
            if tag == key[1]:    
                list_fractcoverpos.append(dict_fracposoverpos[key])
                list_labelstc.append(int(tc))
        for index,value in enumerate(list_labelstc):
            def_frac[value-1] = list_fractcoverpos[index] 
        finaldir = picsdir+'/tcacrosspos'
        if not os.path.exists(finaldir):
            os.mkdir(finaldir)
        path = f'{finaldir}/pos{tag}.pdf'   
        plot_fractions_per_pos(def_frac,fractcoverwords,total_labels,tag,path)
        entropy_S.append(-np.dot(np.array(list_fractcoverpos),np.log(np.array(list_fractcoverpos))))
    path = f'{finaldir}/entropyS.pdf'
    plot_entropy(entropy_S,postags,'S',path)
    nodeList = dfplot['id_palabra'].to_list()
#Here we are going to load the dictionary from which the network was orignially created, in order to see the connections each node makes
    jsonname = f"files/{iol}/dictionaries/{netName}.json"
    connections = json2dict(jsonname, transform_keys=True)
    #We filter here since we only want connections from nodes that actually are in the network(top 500 words, 1st gcc)
    connections = {key:connections[key] for key in connections if key in nodeList}
    #We prepare a dicitionary to map words connected to topologies
    dict_palabranc5 = dict(zip(dfplot['id_palabra'].to_list(),dfplot['nc5'].to_list()))
    #dict_palabranc5 = {int(key):value for key,value in dict_palabranc5.items()}
    dict_palabranc5.keys()
    list_connectedtc={}
#Here we form a dictionary storing the number of times each word is connected to each topology, list_conectedtc(This was a mistake, is not a list)
    for node in nodeList:
        connections[node] = list(set(connections[node]))
        connections[node] = [id for id in connections[node] if id != node]
        nodesconnected = connections[node]
        topologies = [dict_palabranc5[id] for id in nodesconnected if id in nodeList]
        
        list_connectedtc[node] = (topologies)
        #We are going to store the number of times each word is connected to each tc in this variables
        cnxnstc1 = []
        cnxnstc2 = []
        cnxnstc3 = []
        cnxnstc4 = []
        cnxnstc5 = []
        cnxnstotal = []
        #print([key for key in list_connectedtc.keys()])
    for node in nodeList:
        cnxnlist = []
        total_connections = len(list_connectedtc[node])
        for i in range(1,6,1):
            list_count_tc = [tc for tc in list_connectedtc[node] if tc == i]
            cnxnlist.append(len(list_count_tc))
        cnxnstc1.append(cnxnlist[0])
        cnxnstc2.append(cnxnlist[1])
        cnxnstc3.append(cnxnlist[2])
        cnxnstc4.append(cnxnlist[3])
        cnxnstc5.append(cnxnlist[4])
        cnxnstotal.append(total_connections)
    #Incorporating this data to our df
    dfplot['cnxnstc1'] = pd.Series(cnxnstc1)
    dfplot['cnxnstc2'] = pd.Series(cnxnstc2)
    dfplot['cnxnstc3'] = pd.Series(cnxnstc3)
    dfplot['cnxnstc4'] = pd.Series(cnxnstc4)
    dfplot['cnxnstc5'] = pd.Series(cnxnstc5)
    dfplot['cnxnstotal'] = pd.Series(cnxnstotal)


    #list_connectedtc = {key:[dict_palabranc5[node] for node in value] for key,value in connections.items()}
    #list_connectedtc

    #Calculating frequencies of connection to each community
    dfplot['rel_cnxnstc1'] = dfplot['cnxnstc1']/dfplot['cnxnstotal']
    dfplot['rel_cnxnstc2'] = dfplot['cnxnstc2']/dfplot['cnxnstotal']
    dfplot['rel_cnxnstc3'] = dfplot['cnxnstc3']/dfplot['cnxnstotal']
    dfplot['rel_cnxnstc4'] = dfplot['cnxnstc4']/dfplot['cnxnstotal']
    dfplot['rel_cnxnstc5'] = dfplot['cnxnstc5']/dfplot['cnxnstotal']
    #gettin here total connections to each community by POS tag
    thickness = dfplot.groupby(by='POS').sum()[['cnxnstc1','cnxnstc2','cnxnstc3','cnxnstc4','cnxnstc5','cnxnstotal']]
    #Now getting the frequency of connections to each tc by POS
    thickness['rel_cnxnstc1'] = thickness['cnxnstc1']/thickness['cnxnstotal']
    thickness['rel_cnxnstc2'] = thickness['cnxnstc2']/thickness['cnxnstotal']
    thickness['rel_cnxnstc3'] = thickness['cnxnstc3']/thickness['cnxnstotal']
    thickness['rel_cnxnstc4'] = thickness['cnxnstc4']/thickness['cnxnstotal']
    thickness['rel_cnxnstc5'] = thickness['cnxnstc5']/thickness['cnxnstotal']
    #Using data calculated above to form our bipartite network with edges' width set as relative frequency of connections

    thick_plot = thickness[['rel_cnxnstc1','rel_cnxnstc2','rel_cnxnstc3','rel_cnxnstc4','rel_cnxnstc5']]
    asd=dict(thick_plot.transpose())
    #labels = [label for label in asd['ADJ'].index]
    dict_conn_thickness ={}
    dict_def_thickness = {}
    dict_conn ={}
    tcs = [1,2,3,4,5]
    for key in asd:
        dict_conn_thickness[key] = asd[key].to_list()
        dict_def_thickness[key] = []
        dict_conn[key] = []
        for index,value in enumerate(dict_conn_thickness[key]):
            if value>0:
                dict_conn[key].append(index+1)
                dict_def_thickness[key].append(value)

                
            
    g = nx.Graph()
    #g.add_nodes_from(dict_def_thickness.keys())
    colors = {'ADV':'b','CCONJ':'g','VERB':'r','ADJ':'c',
        'PRON':'m','DET':'y','ADP':'k','PROPN':'w','SCONJ':'tab:purple',
        'NOUN':'tab:orange','NUM':'tab:pink','PART':'tab:brown','AUX':'tab:olive'}
    colors2 = ['b','g','r','c','m']
    for k in dict_conn.keys():
        for i,v in enumerate(tcs):
            g.add_edge(k,v,color= colors2[i],weight=dict_conn_thickness[k][i]*10)
    edges = g.edges()
    degrees = dict(thickness['cnxnstotal'].transpose())
    minignodelist = [node for node in g.nodes]
    list_nodesize = [degrees[key] if key in postags else 300 for key in minignodelist]        
    pos = nx.bipartite_layout(g,postags,align = 'horizontal',aspect_ratio=3.5/3)
    colors = [g[u][v]['color'] for u,v in edges]
    weights = [g[u][v]['weight'] for u,v in edges]
    plt.figure()
    nx.draw(g,pos,width=weights, edge_color = colors, node_size = list_nodesize, with_labels = True,font_size=10)
    finaldir = picsdir+'/networks/'
    if not os.path.exists(finaldir):
            os.mkdir(finaldir)
    plt.savefig(finaldir + 'posvstc.pdf',bbox_inches = 'tight')
    #gettin here total connections to each community by TC
    thickness_tc = dfplot.groupby(by='nc5').sum()[['cnxnstc1','cnxnstc2','cnxnstc3','cnxnstc4','cnxnstc5','cnxnstotal']]
    #Now getting the frequency of connections to each tc by TC
    thickness_tc['rel_cnxnstc1'] = thickness_tc['cnxnstc1']/thickness_tc['cnxnstotal']
    thickness_tc['rel_cnxnstc2'] = thickness_tc['cnxnstc2']/thickness_tc['cnxnstotal']
    thickness_tc['rel_cnxnstc3'] = thickness_tc['cnxnstc3']/thickness_tc['cnxnstotal']
    thickness_tc['rel_cnxnstc4'] = thickness_tc['cnxnstc4']/thickness_tc['cnxnstotal']
    thickness_tc['rel_cnxnstc5'] = thickness_tc['cnxnstc5']/thickness_tc['cnxnstotal']
    #Using data calculated above to form our bipartite network with edges' width set as relative frequency of connections

    thick_plot = thickness_tc[['rel_cnxnstc1','rel_cnxnstc2','rel_cnxnstc3','rel_cnxnstc4','rel_cnxnstc5']]
    asd=dict(thick_plot.transpose())
    labels = [label for label in asd[1].index]
    dict_conn_thickness ={}
    dict_def_thickness = {}
    dict_conn ={}
    tcs = [1,2,3,4,5]
    for key in asd:
        dict_conn_thickness[key] = asd[key].to_list()
        dict_def_thickness[key] = []
        dict_conn[key] = []
        for index,value in enumerate(dict_conn_thickness[key]):
            if value>0:
                dict_conn[key].append(index+1)
                dict_def_thickness[key].append(value)

                
            
    g = nx.Graph()
    #g.add_nodes_from(dict_def_thickness.keys())
    colors = {'ADV':'b','CCONJ':'g','VERB':'r','ADJ':'c',
        'PRON':'m','DET':'y','ADP':'k','PROPN':'w','SCONJ':'tab:purple',
        'NOUN':'tab:orange','NUM':'tab:pink','PART':'tab:brown','AUX':'tab:olive'}
    colors2 = ['b','g','r','c','m']
    tc_str =['1','2','3','4','5']
    for k in asd.keys():
        for i,v in enumerate(tc_str):
            g.add_edge(k,v,color= colors2[i],weight=dict_conn_thickness[k][i]*10)
    edges = g.edges()
    degrees = dict(thickness['cnxnstotal'].transpose())
    minignodelist = [node for node in g.nodes]
    list_nodesize = [degrees[key] if key in postags else 300 for key in minignodelist]        
    pos = nx.bipartite_layout(g,tc_str,align = 'horizontal',aspect_ratio=3.5/3)
    colors = [g[u][v]['color'] for u,v in edges]
    weights = [g[u][v]['weight'] for u,v in edges]
    plt.figure()
    nx.draw(g,pos,width=weights, edge_color = colors, node_size = list_nodesize, with_labels = True,font_size=10)
    plt.savefig(picsdir+'/networks/tcvstc.pdf',bbox_inches = 'tight')


if __name__== '__main__':
    filelist = os.listdir('./files/inflected/dictionaries/')
    languagelist = [file.split('.')[0] for file in filelist]
    for lemmatized in [False, True]:
        for netName in languagelist[50]:
            print(netName)
            language_analysis(netName, lemmatized)
import networkx as nx
from pylab import colorbar
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utilsstreamlit import read_plot_info
netName = 'Spanish'
iol = 'inflected'
dfplot = read_plot_info(netName,0,iol)

#plotting the network, standard procedure for loading them up
from utils import csv2df,load_network
langframe = csv2df(f'files/{iol}/dataframes/{netName}.csv')
mostfreq =langframe.unique_id.to_list()
thisNetwork = load_network(f'files/{iol}/dictionaries/{netName}.json')
thisNetwork=thisNetwork.subgraph(mostfreq)
Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0]));
#Now we have to follow the order of g.nodes(), so we will have to rearrange
#the colors in the dataframe in that order
nodeList = [int(node) for node in thisNetwork.nodes()]
dictpalabras = dict(zip(dfplot['id_palabra'].to_list(),dfplot['nc2'].to_list()))
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
colorset = truecolors5
edge_x = []
edge_y = []
G = thisNetwork
import networkx as nx
from pylab import colorbar
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utilsstreamlit import read_plot_info
netName = 'Spanish'
iol = 'inflected'
dfplot = read_plot_info(netName,0,iol)

#plotting the network, standard procedure for loading them up
from utils import csv2df,load_network
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
pc2 = dict(zip(dfplot['id_palabra'].to_list(),dfplot['pc2']))
pc1 = dict(zip(dfplot['id_palabra'].to_list(),dfplot['pc1']))
realpc1 = [pc1[node] for node in nodeList]
realpc2 = [pc2[node] for node in nodeList]
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
colorset = truecolors5
edge_x = []
edge_y = []
G = thisNetwork
pos = nx.spring_layout(G)
for node in nodeList:
    G.nodes[node]['pos'] = (pos[node][0],pos[node][1])
colordict = {1:'blue',2:'yellow',3:'green',4:'red',5:'brown'}
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))
node_trace.marker.color = [colordict[color] for color in realcolors5]
node_trace.text = realpalabras
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
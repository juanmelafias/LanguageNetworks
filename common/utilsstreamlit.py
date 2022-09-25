import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from common.utils import load_network

def display_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: Pandas dataframe to be converted to AgGrid

    Returns:
        data: Modified pandas dataframe

    This module allows the app user to edit a dataframe in real time and save those
    changes in memory using the AgGrid module.
    """
    # Construct GridOptionsBuilder dict
    gd = GridOptionsBuilder.from_dataframe(df)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(editable=True, groupable=True)
    #sel_mode = st.radio("Selection type", options=["single", "multiple"])
    gd.configure_selection(selection_mode="multiple", use_checkbox=True)
    gridoptions = gd.build()
    
    Table = AgGrid(
        df,
        gridOptions=gridoptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        height=500,
        allow_unsafe_jscode=True,
        theme="fresh",
    )
    # here only selected rows are used to generate reports. The problem is that
    # the returned value is a list of dictionaries, not a df, so we need to transform it back again
    data = pd.DataFrame(Table["selected_rows"])
    

    """
    Table = AgGrid(
        df,
        gridOptions=gridoptions,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        height=500,
        allow_unsafe_jscode=True,
        theme="fresh",
    )
    data = Table["data"]
    """
    return data

def read_plot_info(language,nwords,iol):
    
    df = pd.read_csv(f'files/{iol}/dfplot/{language}.csv')
    df = df.drop(labels = ['Unnamed: 0'],axis = 1).sort_values(by='ranking')
    if nwords != 0:
        df = df.iloc[0:nwords]
    return df
def plotly_graph(G,colors,palabras,trans,display_legend):
	nodeList = [int(node) for node in G.nodes()]
	edge_x = []
	edge_y = []
	pos = nx.spring_layout(G)
	colordict = {1:'blue',2:'yellow',3:'green',4:'red',5:'brown'}
	realcolors = [colordict[color] for color in colors]
	if len(set(colors)) == 2:
		legendgraph = {'blue':'Periphery','yellow':'Core'}
	elif len(set(colors)) == 3:
		legendgraph = {'blue':'Periphery','yellow':'SuperCore', 'green':'OuterCore'}
	elif len(set(colors)) == 4:
		legendgraph = {'blue':'Periphery','yellow':'SuperCore', 'green':'InnerConnectors','red':'OuterConnectors'}
	else:
		legendgraph = {'blue':'OuterPeriphery','yellow':'InnerPeriphery', 'green':'SuperCore','red':'InerConnectors','brown':'OuterConnectors'}
	labels = [legendgraph[color] for color in realcolors] 
	
	for node in nodeList:
		G.nodes[node]['pos'] = (pos[node][0],pos[node][1])

		
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
		mode='lines',
		name='edges')

	node_x = []
	node_y = []
	for node in G.nodes():
		x, y = G.nodes[node]['pos']
		node_x.append(x)
		node_y.append(y)
	dfgo = pd.DataFrame()
	dfgo['x'] = node_x
	dfgo['y'] = node_y
	dfgo['color'] = realcolors
	dfgo['palabras'] = palabras
	dfgo['trans'] = trans
	node_traces = []
	for color in legendgraph.keys():
		name = legendgraph[color]
		dfsel = dfgo[dfgo['color'] == color]
		print(dfsel['palabras'])
		print(color)
		node_trace = go.Scatter(
			x=dfsel['x'], y=dfsel['y'],
			mode='markers',
			showlegend=True,
			marker=dict(
				# colorscale options
				#'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
				#'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
				#'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
				#colorscale='YlGnBu',
				#reversescale=True,
				size=10,
				line_width=2)
				)
		if display_legend:
			node_trace.name = name
		node_trace.text = (dfsel['palabras']+', '+dfsel['trans']).to_list()
		node_trace.marker.color = dfsel['color'].to_list()
		node_traces.append(node_trace)
	
	
	fig = go.Figure(data=[edge_trace]+node_traces,
				layout=go.Layout(
					title='Network colored according to clusters',
					titlefont_size=16,
					showlegend=True,
					hovermode='closest',
					margin=dict(b=20,l=5,r=5,t=40),
					annotations=[ dict(
						text="Network colored according to clusters",
						showarrow=False,
						xref="paper", yref="paper",
						x=0.005, y=-0.002 ) ],
					xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
					yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
					)
	
		
	st.plotly_chart(fig)
    
def whole_network_plotter(netName,iol,now,noc):
	dfplot = read_plot_info(netName,now,iol)

	mostfreq =dfplot.id_palabra.to_list()
	thisNetwork = load_network(f'files/{iol}/dictionaries/{netName}.json')
	thisNetwork=thisNetwork.subgraph(mostfreq)
	Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True); 
	thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0])); 
	#Now we have to follow the order of g.nodes(), so we will have to rearrange
	#the colors in the dataframe in that order
	nodeList = [int(node) for node in thisNetwork.nodes()]
	dictpalabras = dict(zip(dfplot['id_palabra'].to_list(),dfplot['palabra'].to_list()))
	realpalabras = [dictpalabras[node] for node in nodeList]
	cols = [col for col in dfplot.columns]
	if 'trans' in cols:
		dict_trans = dict(zip(dfplot['id_palabra'].to_list(),dfplot['trans'].to_list()))
		real_trans = [dict_trans[node] for node in nodeList]
	else:
		real_trans = ['no trans' for node in nodeList]
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
	dict_colors = {2:realcolors2,3:realcolors3,4:realcolors4,5:realcolors5}
	plotly_graph(thisNetwork,dict_colors[noc],realpalabras,real_trans,display_legend=False)
import numpy as np; 
import matplotlib.pyplot as plt; 
import networkx as nx; 
from copy import copy; 
from mpl_toolkits.mplot3d import Axes3D; 
from pathlib import Path
import pandas as pd
from utils import network_gen, dict2json

routes = pd.read_csv('rutascorpus.csv')
routes = routes.drop('Unnamed: 0',axis = 1)

linelimit = 40000
selection = routes[routes['num_lines']>linelimit]
rutas = selection.route.to_list()

for item in rutas[-8:]:
    nets,frames=network_gen(item,linelimit)
    archivo = item.split('/')[-2][:-4].split('-')[0].replace('UD_','')
    frames = pd.DataFrame(frames).transpose()
    frames.to_csv(f'dataframes/{archivo}.csv')
    networkname =  f'dictionaries/{archivo}.json'
    dict2json(nets,networkname)
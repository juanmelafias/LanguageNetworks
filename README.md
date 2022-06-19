# Language Networks

This repository is intented for the creation and Analysis of syntactical Language Networks. Syntactical networks are built using syntactic relationships between words, differently from adjacency networks, which are built from words which are adjacent in a sentence. 

The repo stores lots of code and files, most of them might be useful or not

To get all set up to run the scripts run ./setup.sh on your command line

## Files
Here are stored files derived from the Language Annotated Corpora. If such files are to be generated again from scratch, it will be needed the UsefulData folder, located in the repo root directory. This can be obtained by getting in touch with me.

The Files folder contains to folders itself: inflected and lemmatized. Each contains the same kind of data but for networks constructed from inflected and lemmatized forms respectively. 

### dictionaries and dataframes

So, starting from the .conllu files with the syntacticlly annotated data stored in UsefulData, we can generate the first data about our network. This can be done by running **NetworkGenerator.py**. This calls the file_generator() function stored in **utils.py**. This function takes three parameters: *routes_file*, which is a string with the location of a csv file wich contains the location and basic info of the most important corpus of each language in the UsefulData folder; *num_lines*, which sets the threshold of minimum number of lines for a corpus to be read, and generated files from; and *lemmatized*, which if sets to True creates networks based on lemmatized forms intead of inflected.

This script will create two kind of files: "{name_of_language}.json" and "{name_of_language}.csv" files. The former will be stored in the ***dictionaries/*** folder, since when read back it is expected to be a dictionary where the keys are the nodes of the network, and its values are the other each node is connected to. The latter will be stored in the ***dataframes/***, since whenever is read back is intended to be a Pandas DataFrame with four columns: *word_id*, which gives the node in the network; *word*, which maps the id to the word; *POS*, which gives the part of speech tag of the word, helping thus differencing homographs; and *count*, which provides the number of times each word appears in the corpus in the first *num_lines* lines.

### networks and avgproperties 

Once the files in ***dictionaries/*** and ***dataframes/*** are generated, we can start calculating the properties of the network. This is done massively for all languages stored in ***dictionaries/*** with the aid of **meanproperties.py**. This script allows to compute the properties for each node of the network and the average properties of the node. If your properties are already calculated and you only need to compute the average by reading them first, set CreateProperties to False. Before going any further it is important to state that data is extracted from the files in ***dictionaries/*** and ***dataframes/*** with the functions *json2dict* and *csv2df* in **utils.py**. *json2dict* is pretty standard, simply recuperates the exactly same dictionary that was generated with the connections in the network, but *csv2df* does not retrieve the exact same df it came from. It eliminates the interjections, symbols, punctuation marks and vocatives from the dataframe first, and then sorts the words by its count number, only to finally retrieve the most repeated *wordsnumber* words, which is set to 500 by default. With the function *load_networks* from **utils.py** we create a networkx network, from which we create a subgraph with only the words in the retrieved dataframe in *csv2df*. After that, the greatest connected component of the network is created, and at that point the properties are calculated and then saved using the functions from **helper.py** *computeNodesProperties* and *writeNetworkProperties*. The results are stored in ***networks/***. After that the properties of each network are read from those files, and its node avegrage values are computed and stored in ***avgproperties/***

### languagesmean

In this folder are stored the results from running **PCA_plot_languages.py**. This script is also responsible for computing the numpy array that stores the data with the average properties for all Languages. By tuning the Boolean *primaries* we store the results for the primary properties plus the mean degree of the neighbours in the directory ***languagesmeanprimaries/***. On the contrary, if we wish to compute results based on all 48 properties, we shall set *primaries* to false and results will be stored in ***languagesmeanneighbours/
***

## Notebooks

They are ther primary for experimentation. Ignore them

## App

Folder for future visualization app with streamlit

## csvroutes

csv useful for certain parts of the code

## reqs

Requirements of python packages necessary to run the app
## Scripts
This folder stores a series of python scripts useful for Network Analysys taken from brigan/NodeMorphospaces Github repository


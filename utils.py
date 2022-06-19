from json import loads as cargar
import json
import pandas as pd
import numpy as np
import shutil
import os
import networkx as nx
import helper as h; 
#import loadHelper as lh;
def build_properties_array_languages(netPath,primaries = True):
    """
    Args:
        netPath: Path of directory where to find language files with mean properties
    Returns:
        properties_array_languages: np.array of shape(valid_properties,num_languages) 
        with valid properties <= 15., with mean properties for each language
        dict_pathologies: dictionary with languages as keys and Pathological features 
        as values.

    """
 
    filelist = os.listdir('./files/inflected/dictionaries/')
    languagelist = [file.split('.')[0] for file in filelist if file not in ['Japanese.json','Arabic.json','French.json']]
    fNeighborMean = True; 
    fNeighborStd = True; 
    dict_pathologies = {}
    list_excluded = set(); 
    
    for column,netName in enumerate(languagelist):
        
        meanpropertiesDict, excludedProperties = build_language_mean_dict(netName,netPath,fNeighborMean,fNeighborStd)
        dict_pathologies[netName] = excludedProperties
        for prop in excludedProperties:
            list_excluded.add(prop)
    if primaries:
        first17 = [key for key in meanpropertiesDict.keys()][:17]
        valid_keys = [key for key in first17 if key not in list_excluded]
        
    else: 
        all = [key for key in meanpropertiesDict.keys()]
        valid_keys = [key for key in all if key not in list_excluded]
    num_cols = len(valid_keys)
    properties_array_languages = np.zeros([num_cols,len(languagelist)])
    print(list_excluded)
    for column,netName in enumerate(languagelist):
        meanpropertiesDict, excludedProperties = build_language_mean_dict(netName,netPath,fNeighborMean,fNeighborStd)
        meanpropertieslist=[meanpropertiesDict[key] for key in valid_keys]
        properties_array_languages[:,column] = meanpropertieslist
    return valid_keys, properties_array_languages, dict_pathologies


def build_language_mean_dict(netName,netPath,fNeighborMean,fNeighborStd):
    
    """
    Args:
        netPath: Path of directory where to find language files with mean properties
        netName: Name of language where to find the properties
        fNeighbourMean/Std: Boolean that indicates wether to compute Neighbour properties
    Returns:
        meanpropertiesDict:dictionary with properties
        excludedproperties: list with excluded properties

    """
    (nodeList, propertiesDict) = h.readNetworkProperties(netName, netPath, fNeighborMean, fNeighborStd); 
    (includedProperties, excludedProperties) = h.findPathologicalProperties(propertiesDict); 
    meanpropertiesDict = {key:propertiesDict[key].mean() for key in propertiesDict.keys()}
    return meanpropertiesDict,excludedProperties
    


def stats_language(lemmatized = False):
    if lemmatized:
        dirdf = 'dataframeslemma/'
        dirdict = 'dictionarieslemma/'
    else:
        dirdf = 'dataframes/'
        dirdict = 'dictionaries/'
    csv2df()
def load_network(jsonfile):
    net = json2dict(jsonfile)
    g = nx.Graph()
    g.add_nodes_from(net.keys())
    for (k, v) in net.items():
        g.add_edges_from(([(k, t) for t in v]))
    return g


def file_generator(routesfile='csvroutes/rutascorpus.csv', numlines=50000, lemmatized=False):

    if lemmatized:
        folderdict = 'files/lemmatized/dictionaries'
        folderframe = 'files/lemmatized/dataframes'
        shutil.rmtree(f'./{folderdict}/')
        shutil.rmtree(f'./{folderframe}/')
        os.mkdir(f'./{folderdict}')
        os.mkdir(f'./{folderframe}')
    else:
        folderdict = 'files/inflected/dictionaries'
        folderframe = 'files/inflected/dataframes'
        shutil.rmtree(f'./{folderdict}/')
        shutil.rmtree(f'./{folderframe}/')
        os.mkdir(f'./{folderdict}')
        os.mkdir(f'./{folderframe}')

    routes = pd.read_csv(routesfile)
    routes = routes.drop('Unnamed: 0', axis=1)
    selection = routes[routes['num_lines'] > numlines]
    rutas = selection.route.to_list()
    for file in rutas:
        dic, frame = network_gen(
            file, linelimit=numlines, lemmatized=lemmatized)
        name = file.split('/')[3].split('-')[0].replace('UD_', '')

        dictlocation = f'{folderdict}/{name}.json'
        framelocation = f'{folderframe}/{name}.csv'

        dict2json(dic, dictlocation)
        frame = pd.DataFrame(frame).transpose()
        print(frame['count'].sum())
        frame.to_csv(framelocation)


def csv2df(csvname,	wordsnumber=500):
    df = pd.read_csv(csvname).rename(columns={'Unnamed: 0': 'unique_id'})
    df = df.loc[df['POS'] != 'PUNCT']
    df = df[df['POS'] != 'INTJ']
    df = df[df['POS'] != 'SYM']
    df = df[df['POS'] != 'X']
    df = df.sort_values('count', ascending=False).iloc[:wordsnumber]
    return df


def json2dict(jsonname,transform_keys = True):

    with open(jsonname, 'r') as f:
        a = f.readline()

    dicti = cargar(a)
    if transform_keys:
        dicti = {int(key): dicti[key] for key in dicti.keys()}
    return dicti


def network_gen(filePath, linelimit=40000, lemmatized=False):
    fIn = open(filePath, encoding="utf8")
    allLines = fIn.read().split('\n')
    fIn.close()
    len(allLines)
    # Let us read just the first 10 sentences:
    # Each sentence is coded as a block of lines, each containing the information of a single word.
    # Each block ends with a blank line that marks the end of the sentence.
    id2word = {}
    word2id = {}
    nSentences = 100
    dicConnections = {}  # Dictionary to store connections.
    SentenceGlobal = []
    unique_id = 0
    # Each sentence is coded as a block of lines, each containing the information of a single word.
    # Each block ends with a blank line that marks the end of the sentence.
    iLine = 0 		# Index to iterate over lines in the file.
    iSentence = 0 	# Index to keep track of processed sentences.
    fGo = True
    dicSentenceIDs = {} 	# To keep track of words within sentence.
    sentenceHeads = []
    limit = 20000
    dicSentencePOS = {}
    if lemmatized:
        wordindex = 2
    else:
        wordindex = 1
    while (fGo):

        thisLine = allLines[iLine]
        # print(thisLine);

        if (thisLine != ''):
            if (thisLine[0] != '#'):
                # This line codes a valid word. We need to process it:
                wordInfo = thisLine.split('\t') 	# Info is separated by tabs.
                # Adding word ID to dicSenteceID:
                if ('-' not in wordInfo[0]) & ('.' not in wordInfo[0]):
                    dicSentenceIDs[wordInfo[0]] = wordInfo[wordindex].lower()
                    dicSentencePOS[wordInfo[0]] = wordInfo[3]
                # Adding the head of this word to sentenceConnections:
                    if wordInfo[3] in ['CACA']:
                        sentenceHeads += '0'
                    else:
                        sentenceHeads += [wordInfo[6]]

        elif (thisLine == ''):
            # End of sentence!

            # Making sure that all words are entries of dicConnections:
            for key in dicSentenceIDs.keys():
                # first we check if words are worthy of being part of the metwork according to their POS tag
                # On top of this we will be storing the words unique id to later form the dictionary for the network
                if (dicSentencePOS[key] not in ['CACA']):
                    # Once that is checked, we look for it in the dictionary of homograph words, if it is not there, we add it
                    if (dicSentenceIDs[key] not in word2id.keys()):
                        unique_id += 1
                        word2id[dicSentenceIDs[key]] = {
                            'POS': [dicSentencePOS[key]], 'id': [unique_id]}
                        id2word[unique_id] = {
                            'word': dicSentenceIDs[key], 'POS': dicSentencePOS[key], 'count': 1}
                        dicConnections[unique_id] = []
                        SentenceGlobal += [unique_id]
                    # If the word is already in the homograph dictionary but the recorded word has a different pos tag, we add it along with the new tag
                    elif (dicSentenceIDs[key] in word2id.keys()) & (dicSentencePOS[key] not in word2id[dicSentenceIDs[key]]['POS']):
                        unique_id += 1
                        word2id[dicSentenceIDs[key]
                                ]['POS'] += [dicSentencePOS[key]]
                        word2id[dicSentenceIDs[key]]['id'] += [unique_id]
                        id2word[unique_id] = {
                            'word': dicSentenceIDs[key], 'POS': dicSentencePOS[key], 'count': 1}
                        dicConnections[unique_id] = []
                        SentenceGlobal += [unique_id]
                    # If the word exists in the homograph dictionary with its right POS tag, we retrieve the word id and add it to the count
                    else:
                        for index, tag in enumerate(word2id[dicSentenceIDs[key]]['POS']):
                            if tag == dicSentencePOS[key]:
                                retrieved_id = word2id[dicSentenceIDs[key]
                                                       ]['id'][index]
                        id2word[retrieved_id]['count'] += 1
                        SentenceGlobal += [retrieved_id]
                else:
                    SentenceGlobal += [0]

            # Adding connections to dicConnections:
            for (iWord, head) in enumerate(sentenceHeads):
                # If the head is 0, this is the root node of the tree or a punctuation symbol. We don't process this!
                if (head != '0'):
                    # We have to retrieve
                    idword = SentenceGlobal[iWord]
                    idhead = SentenceGlobal[int(head)-1]
                    dicConnections[idword] += [idhead]
                    dicConnections[idhead] += [idword]
                    # preguntar a Luis si no se quedan repetidas las palabras aquí

            # Clearing sentence buffers:
            dicSentenceIDs = {}
            sentenceHeads = []
            dicSentencePOS = {}
            SentenceGlobal = []

            if (iLine > linelimit):
                fGo = False
            iSentence += 1

        iLine += 1
    return dicConnections, id2word


def dict2json(dicti, jsonname):

    import json
    # create json object from dictionary
    json = json.dumps(dicti)

    # open file for writing, "w"
    f = open(jsonname, "w")

    # write json object to file
    f.write(json)

    # close file
    f.close()

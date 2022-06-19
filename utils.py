import os
import shutil
from json import loads as cargar

import networkx as nx
import numpy as np
import pandas as pd

import helper as h


# import loadHelper as lh;
def build_properties_array_languages(netPath, primaries=True):
    """
    Args:
                netPath: Path of directory where to find language files with
                mean properties
    Returns:
                properties_array_languages: np.array of shape(valid_properties,
                num_languages)with valid properties <= 16, with mean properties
                for each language dict_pathologies: dictionary with languages as
                keys and Pathological features as values.

    """

    filelist = os.listdir("./files/inflected/dictionaries/")
    languagelist = [
        file.split(".")[0]
        for file in filelist
        if file not in ["Japanese.json", "Arabic.json", "French.json"]
    ]
    fNeighborMean = True
    fNeighborStd = True
    dict_pathologies = {}
    list_excluded = set()

    for column, netName in enumerate(languagelist):

        meanpropertiesDict, excludedProperties = build_language_mean_dict(
            netName, netPath, fNeighborMean, fNeighborStd
        )
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
    properties_array_languages = np.zeros([num_cols, len(languagelist)])
    print(list_excluded)
    for column, netName in enumerate(languagelist):
        meanpropertiesDict, excludedProperties = build_language_mean_dict(
            netName, netPath, fNeighborMean, fNeighborStd
        )
        meanpropertieslist = [meanpropertiesDict[key] for key in valid_keys]
        properties_array_languages[:, column] = meanpropertieslist
    return valid_keys, properties_array_languages, dict_pathologies


def build_language_mean_dict(netName, netPath, fNeighborMean, fNeighborStd):

    """
    Args:
                netPath: Path of directory where to find language files with mean
                properties netName: Name of language where to find the properties
                fNeighbourMean/Std: Boolean that indicates wether to compute
                Neighbour properties
    Returns:
                meanpropertiesDict:dictionary with properties
                excludedproperties: list with excluded properties

    """
    (nodeList, propertiesDict) = h.readNetworkProperties(
        netName, netPath, fNeighborMean, fNeighborStd
    )
    (includedProperties, excludedProperties) = h.findPathologicalProperties(
        propertiesDict
    )
    meanpropertiesDict = {
        key: propertiesDict[key].mean() for key in propertiesDict.keys()
    }
    return meanpropertiesDict, excludedProperties


def stats_language(lemmatized=False):
    if lemmatized:
        dirdf = "files/lemmatized/dataframes/"
        dirdict = "files/lemmatized/dictionaries/"
    else:
        dirdf = "files/inflected/dataframes/"
        dirdict = "files/inflected/dictionaries/"
    listdf = os.listdir(dirdf)
    listdict = os.listdir(dirdict)
    stats = {"languages": [], "number_nodes": [], "number_edges": []}
    for index, routedf in enumerate(listdf):
        lang = routedf.split(".")[0]
        routedict = listdict[index]
        df = csv2df(dirdf + routedf)
        mostfreq = df.unique_id.to_list()
        thisNetwork = load_network(dirdict + routedict)
        thisNetwork = thisNetwork.subgraph(mostfreq)
        Gcc = sorted(nx.connected_components(thisNetwork), key=len, reverse=True)
        thisNetwork = nx.Graph(thisNetwork.subgraph(Gcc[0]))
        nNodes = len(thisNetwork.nodes())
        nEdges = thisNetwork.number_of_edges()
        stats["languages"].append(lang)
        stats["number_nodes"].append(nNodes)
        stats["number_edges"].append(nEdges)

    return stats


def load_network(jsonfile):
    """
        Args:
            jsonfile: string, route to jsonfile containing a dictionary
            with network connections

    Returns:
            g: networkx.Graph object, formed with the elements of the
            dictionary read the json file
    """
    net = json2dict(jsonfile)
    g = nx.Graph()
    g.add_nodes_from(net.keys())
    for (k, v) in net.items():
        g.add_edges_from(([(k, t) for t in v]))
    return g


def file_generator(
    routesfile="csvroutes/rutascorpus.csv", numlines=50000, lemmatized=False
):

    """
    Args:
                routesfile: string, route .csv file that contains routes
                to other files
                numlines: integer, limit that serves as filter to select only
                corpora with a number of lines above the limit
                lemmatized: Boolean, tells wether to take inflected
                or lemmatized forms

    The file generator function takes care of the storage of the
        objects returned by the network generator function, and does
        it massively for all the corpora in a certain
    directory that has a number of lines superior to numlines
    """
    if lemmatized:
        folderdict = "files/lemmatized/dictionaries"
        folderframe = "files/lemmatized/dataframes"
        shutil.rmtree(f"./{folderdict}/")
        shutil.rmtree(f"./{folderframe}/")
        os.mkdir(f"./{folderdict}")
        os.mkdir(f"./{folderframe}")
    else:
        folderdict = "files/inflected/dictionaries"
        folderframe = "files/inflected/dataframes"
        shutil.rmtree(f"./{folderdict}/")
        shutil.rmtree(f"./{folderframe}/")
        os.mkdir(f"./{folderdict}")
        os.mkdir(f"./{folderframe}")

    routes = pd.read_csv(routesfile)
    routes = routes.drop("Unnamed: 0", axis=1)
    selection = routes[routes["num_lines"] > numlines]
    rutas = selection.route.to_list()
    for file in rutas:
        dic, frame = network_gen(file, linelimit=numlines, lemmatized=lemmatized)
        name = file.split("/")[3].split("-")[0].replace("UD_", "")

        dictlocation = f"{folderdict}/{name}.json"
        framelocation = f"{folderframe}/{name}.csv"

        dict2json(dic, dictlocation)
        frame = pd.DataFrame(frame).transpose()
        print(frame["count"].sum())
        frame.to_csv(framelocation)


def csv2df(csvname, wordsnumber=500):
    """
    Args:
                csvname = string, route to csv file of a certain language with
                four columns: unique_id, word, POS and count
                wordsnumber: number of rows-words- wanted from the csv to be
                written onto the DataFrame
    Returns:
                df: Pandas df with rows sorted in descending order,
                with invalid words removed
    """
    df = pd.read_csv(csvname).rename(columns={"Unnamed: 0": "unique_id"})
    df = df.loc[df["POS"] != "PUNCT"]
    df = df[df["POS"] != "INTJ"]
    df = df[df["POS"] != "SYM"]
    df = df[df["POS"] != "X"]
    df = df.sort_values("count", ascending=False).iloc[:wordsnumber]
    return df


def json2dict(jsonname, transform_keys=True):
    """
    Args:
                jsonname = string, route to json file.
                transform_keys = Boolean, tells wether to convert output
                dictionary keys into strings, necessasry when the
                json file contains a language naetwork
    Returns:
                dicti: dictionary with the information contained in the json.
    """
    with open(jsonname, "r") as f:
        a = f.readline()

    dicti = cargar(a)
    if transform_keys:
        dicti = {int(key): dicti[key] for key in dicti.keys()}
    return dicti


def network_gen(filePath, linelimit=40000, lemmatized=False):
    """
    Args:
                filePath: string, route to .conllu file.
                linelimit: integer, limit that tells when to start reading the
                last sentence. This allows to control in some way
                that all languages have the same number of lines read
                lemmatized: Boolean, tells wether to take inflected or
                lemmatized forms
    Returns:
                dicConnections: Dictionary with words_id's as keys and its
                connections-referred to as word_id's too- as the values.
                id2word: pd.DataFrame with four columns: unique:id, word,
                POS and count.
    """

    fIn = open(filePath, encoding="utf8")
    allLines = fIn.read().split("\n")
    fIn.close()
    len(allLines)
    # Let us read just the first 10 sentences:
    # Each sentence is coded as a block of lines, each containing
    #  the information of a single word.
    # Each block ends with a blank line that marks the end of
    # the sentence.
    id2word = {}
    word2id = {}
    nSentences = 100
    dicConnections = {}  # Dictionary to store connections.
    SentenceGlobal = []
    unique_id = 0
    # Each sentence is coded as a block of lines, each containing
    #  the information of a single word.
    # Each block ends with a blank line that marks the end of the
    #  sentence.
    iLine = 0  # Index to iterate over lines in the file.
    iSentence = 0  # Index to keep track of processed sentences.
    fGo = True
    dicSentenceIDs = {}  # To keep track of words within sentence.
    sentenceHeads = []
    limit = 20000
    dicSentencePOS = {}
    if lemmatized:
        wordindex = 2
    else:
        wordindex = 1
    while fGo:

        thisLine = allLines[iLine]
        # print(thisLine);

        if thisLine != "":
            if thisLine[0] != "#":
                # This line codes a valid word. We need to process it:
                wordInfo = thisLine.split("\t")  # Info is separated by tabs.
                # Adding word ID to dicSenteceID:
                if ("-" not in wordInfo[0]) & ("." not in wordInfo[0]):
                    dicSentenceIDs[wordInfo[0]] = wordInfo[wordindex].lower()
                    dicSentencePOS[wordInfo[0]] = wordInfo[3]
                    # Adding the head of this word to sentenceConnections:
                    if wordInfo[3] in ["CACA"]:
                        sentenceHeads += "0"
                    else:
                        sentenceHeads += [wordInfo[6]]

        elif thisLine == "":
            # End of sentence!

            # Making sure that all words are entries of dicConnections:
            for key in dicSentenceIDs.keys():
                # first we check if words are worthy of being part of the
                #  metwork according to their POS tag
                # On top of this we will be storing the words unique id
                #  to later form the dictionary for the network
                if dicSentencePOS[key] not in ["CACA"]:
                    # Once that is checked, we look for it in the dictionary
                    # of homograph words, if it is not there, we add it
                    if dicSentenceIDs[key] not in word2id.keys():
                        unique_id += 1
                        word2id[dicSentenceIDs[key]] = {
                            "POS": [dicSentencePOS[key]],
                            "id": [unique_id],
                        }
                        id2word[unique_id] = {
                            "word": dicSentenceIDs[key],
                            "POS": dicSentencePOS[key],
                            "count": 1,
                        }
                        dicConnections[unique_id] = []
                        SentenceGlobal += [unique_id]
                    # If the word is already in the homograph dictionary but the
                    # recorded word has a different pos tag, we add it along with
                    #  the new tag
                    elif (dicSentenceIDs[key] in word2id.keys()) & (
                        dicSentencePOS[key] not in word2id[dicSentenceIDs[key]]["POS"]
                    ):
                        unique_id += 1
                        word2id[dicSentenceIDs[key]]["POS"] += [dicSentencePOS[key]]
                        word2id[dicSentenceIDs[key]]["id"] += [unique_id]
                        id2word[unique_id] = {
                            "word": dicSentenceIDs[key],
                            "POS": dicSentencePOS[key],
                            "count": 1,
                        }
                        dicConnections[unique_id] = []
                        SentenceGlobal += [unique_id]
                    # If the word exists in the homograph dictionary with its
                    # right POS tag,
                    # we retrieve the word id and add it to the count
                    else:
                        for index, tag in enumerate(
                            word2id[dicSentenceIDs[key]]["POS"]
                        ):
                            if tag == dicSentencePOS[key]:
                                retrieved_id = word2id[dicSentenceIDs[key]]["id"][index]
                        id2word[retrieved_id]["count"] += 1
                        SentenceGlobal += [retrieved_id]
                else:
                    SentenceGlobal += [0]

            # Adding connections to dicConnections:
            for (iWord, head) in enumerate(sentenceHeads):
                # If the head is 0, this is the root node of the tree
                #  or a punctuation symbol. We don't process this!
                if head != "0":
                    # We have to retrieve
                    idword = SentenceGlobal[iWord]
                    idhead = SentenceGlobal[int(head) - 1]
                    dicConnections[idword] += [idhead]
                    dicConnections[idhead] += [idword]

            # Clearing sentence buffers:
            dicSentenceIDs = {}
            sentenceHeads = []
            dicSentencePOS = {}
            SentenceGlobal = []

            if iLine > linelimit:
                fGo = False
            iSentence += 1

        iLine += 1
    return dicConnections, id2word


def dict2json(dicti, jsonname):
    """
    Args:
                dicti: Dictionary, possibly containing network information
                jsonname: string containing the path desired to store the
                dictionary

    The dict2json saves a dictionary into a .json file.
    """

    # create json object from dictionary
    json = json.dumps(dicti)

    # open file for writing, "w"
    f = open(jsonname, "w")

    # write json object to file
    f.write(json)

    # close file
    f.close()

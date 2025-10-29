# What are Language Syntax Networks
Words in a sentence form dependency relationships based on language-specific rules. Hierarchically linking these words creates syntactic trees (fig. a), where one word acts as the “head,” pointing to another. Combining trees from multiple sentences and ignoring directedness produces complex graphs (fig. b)
![Syntactic Trees and Graph: “Some of the content in this topic may not be
applicable to some languages. You can create SQL queries in one of two ANSI SQL query modes:
ANSI-89 describes the traditional Jet SQL syntax”](./shared/treedef.png)

# How these networks were formed

[Universal dependencies](https://universaldependencies.org/) hosts syntactically annotated data from over 150 languages. The results included here encompass just over 50 of those.

These networks can be explored in page 2
![Syntactic Network of French Language. Colours represent topological communities](./shared/network.png)

# From networks to syntactic embeddings

A novel [approach](https://arxiv.org/abs/2409.02317) was used to study the networks. Since each word is a node in the network, several properties can be computed for all nodes/words. This creates a vector representation of words of a given language.
![Vector representation of words in Spanish projected onto the first principal components](./shared/wordsspanishpc.PNG)

# Communities in the network

Having vector representations of words makes clustering possible. Words in the same clusters share similar syntactic properties. Checking clusters across languages reveals universally preserved language structures. Communities are represented in different colours across languages. 

You can explore visualizations by esploring different number of clusters and comparing accross languages.

# Words across languages

Clusters/topological communities allows us to compare words in the same language. Words across languages can still be compared though. We just to project them onto the same eigenspace.

This is what has been done in the home page. It is specially useful to compare cognates across similar languages


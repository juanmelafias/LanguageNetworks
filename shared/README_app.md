# Welcome!

I built this app to display some findings of my Master's [thesis](https://arxiv.org/abs/2503.06724). You can contact me [here](https://www.linkedin.com/in/juansoriapostigo/)

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

A novel [approach](https://arxiv.org/abs/2409.02317) was used to study the networks. Since each word is a node in the network, several properties can be computed for all nodes/words. This creates a vector representation of words of a given language. You can explore this embeddings [here](?page=page_5)
![Vector representation of words in Spanish projected onto the first principal components](./shared/wordsspanishpc.PNG)

The graph properties used to create these networks are the following. For each node the mean of the neighbours is also considered

**Primary properties**

- Degree
- Eigenvector Centrality
- Betweenness Centrality
- Closeness centrality
- Harmonic Centrality
- Pagerank
- Core Number
- Onion layer
- Effective size
- Node clique number
- Number of cliques
- Clustering
- Square clustering
- Constraint
- Component Size

# Communities in the network

Having vector representations of words makes clustering possible. Words in the same clusters share similar syntactic properties. Checking clusters across languages reveals universally preserved language structures. Communities are represented in different colours across languages. 

You can explore this in the [syntax network plotter](?page=page_2)

# Words across languages

Clusters/topological communities allows us to compare words in the same language. Words across languages can still be compared though. We just to project them onto the same eigenspace.

![Embeddings of Adposition of French, Spanish and Italian](./shared/adp_romance.png)


This is what has been done in the home page. It is specially useful to compare cognates/words with similar syntactic functions across similar languages. You can try in this [page](?page=page_6)

# Mean properties of Languages

We can compare languages by computing the average properties of their nodes and doing some clustering on them. Common sense tells us that similar languages may have similar syntactic properties. You can check if this holds in the [Mean Language Properties page](?page=page_4)
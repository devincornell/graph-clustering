import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def cluster_mcl(G,desiredClusters,verbose=False,weightAttr='weight'):
    '''Will use the MCL algorithm to cluster the graph G where numClusters 
    is the approximate number of clusters desired.'''
    alpha = 0.1 # growth parameter
    acceptable = 1 # convergence critera will look for desiredClusters +- accept
    max_r = 30
    min_r = 1.01

    # init mcl params
    e = 2 # inflation coeff (element-wise square followed by column-wise normalization)
    r = 1.01 # expansion coeff (evolves markov chain)
    a = 1.0 # self-loops to eliminate some weirdness from clustering results
    
    if verbose: print('Starting search for clusters.')
    keepGoing = True
    solnFound = False
    while keepGoing:
        if verbose: print('Running MCL with r=%f.' % r)
        A, groups = mcl(G,r,e,a,verbose=False,weightAttr=weightAttr) # perform mcl to see the resulting clusters
        clusters_found = np.size(groups)
        if verbose: print('Found %f clusters.' % clusters_found)


        # modify parameters to see if a different result can be obtained
        #r = r + (desiredClusters - clusters_found)*alpha
        if desiredClusters > clusters_found:
            r = r + alpha
        else:
            r = r - alpha

        # check if convergence criteria has been reached (or out of bounds was reached)
        if clusters_found in range(desiredClusters-acceptable,desiredClusters+acceptable+1):
            keepGoing = False
            solnFound = True
        elif r > max_r or r < min_r:
            if verbose:    
                if r > max_r:
                    print('reached maximum r value.')
                if r < min_r:
                    print('reached minimum r value.')

            keepGoing = False
            solnFound = False

    if not solnFound:
        if verbose: print('Couldnt find a solution. Change max_r and min_r to get more results.')
        return None
    else:
        print('Found a solution with %d clusters.' % clusters_found)
        return groupAttractors(A,G,groups) # gets dict of node-cluster pairs

def analyze_mcl(G,rRange=np.arange(2,30),e=2,a=1.0,showPlot=False,weightAttr='weight',filename=None):
    numGroups = np.zeros((np.size(rRange),1))
    i = 0
    for r in rRange:
        result = mcl(G,r,e,a,verbose=False,weightAttr=weightAttr)
        if result == None:
            numGroups[i] = 0
        else:

            numGroups[i] = np.size(result[1])
        i = i + 1

    if showPlot:
        plt.plot(rRange,numGroups)
        plt.title('MCL Clusters Discovered Across Expansion Parameter')
        
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    
    return numGroups, rRange


def mcl(G,r=3,e=2,a=1.0,verbose=False,weightAttr='weight'):
    '''Performs MCL on undirected graph G using power parameter e and inflation parameter r.'''
    # https://www.cs.ucsb.edu/~xyan/classes/CS595D-2009winter/MCL_Presentation2.pdf
    max_iter = 100
    convThresh = 10**-4
    thresh = 10**-1
    if verbose: print('starting mcl')

    # get matrix from graph and add loopbacks
    assert(not G.is_directed())
    A = np.matrix(nx.adjacency_matrix(G,weight=weightAttr).A)
    np.fill_diagonal(A,a)

    # normalize A
    A = np.divide(A,np.tile(np.sum(A,axis=0),(np.size(A,0),1)))
    iter = 1
    keepGoing = True
    while keepGoing:
        prevA = A.copy()

        # expand
        A = A**e

        # inflate
        A = np.power(A,r) # element-wise power
        A = np.divide(A,np.tile(np.sum(A,axis=0),(np.size(A,0),1))) # normalize

        if verbose: print('iteration',iter)

        # evaluate convergence criterion
        g = np.subtract(A,prevA)
        if np.sum(np.abs(g)) < convThresh:
            keepGoing = False
            if verbose: print('converged in this many iterations:',iter)
    
        if iter > max_iter:
            keepGoing = False

        iter = iter + 1

    # find attractors and cluster into groups
    colProj = np.transpose(np.sum(A,axis=1)) # groups show up in this
    groups = np.argwhere(colProj > thresh)[:,1]
    if verbose: print('num clusters found:',np.size(groups))

    return A, groups

def groupAttractors(A,G,groups):
    # make lists of lists to describe clusters
    thresh = 10**-1
    clusterDict = {}
    nodes = G.nodes().copy()
    gn = 1

    for g in groups:
        cluster = np.argwhere(A[g,:] > thresh)[:,1]
        if np.size(cluster) > 0:
            for n in nodes:
                if n in list(cluster):
                    clusterDict[n] = gn
        gn = gn + 1

    return clusterDict

def ut_getWeightClusterGraph(k,n,uc,uo):
    G = nx.relaxed_caveman_graph(k,n,0.15)
    
    # add weights to intra-cluster edges
    edges = G.edges()
    eWeights = {}
    for e in edges:
        eWeights[e] = float(np.random.normal(uc,1.0,1))
    nx.set_edge_attributes(G,'weight',eWeights)

    # add edges between clusters (weaker than within clusters)
    nedges = nx.non_edges(G)
    eWeights = {}
    for e in nedges:
        G.add_edge(e[0],e[1],weight=float(np.random.normal(uo,1.0,1)))

    return G

if __name__ == '__main__':
    #uc = 30.0 # cluster edge mean
    #uo = 4.0 # cluster edge mean
    G = ut_getWeightClusterGraph(5,133,4,30)
    
    analyze_mcl(G,showPlot=True,weightAttr='weight')
    #classLabels = cluster_mcl(G,5,verbose=True)
    #print(classLabels)
    #nx.set_node_attributes(G,'cluster',classLabels)
    #print(np.sum(np.abs(classLabels)))

    nx.write_gexf(G,'randomGraph.gexf')


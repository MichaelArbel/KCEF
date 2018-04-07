import numpy as np

from sklearn.model_selection import KFold

def create_splits(Y, n_split):
    kf = KFold(n_splits = n_split)
    splits = kf.split(Y)
    split = []
    count = 0
    for train_fold, test_fold in splits:
        if count ==0: # Making sure all the splits have the same size
            N = len(train_fold)
        split.append([train_fold[:N], test_fold[:N]])
        count  +=1
    return split

def median_heuristic(data, graph_node):
    if len(graph_node[0][1])>0:
        X = data[:,graph_node[0][1]]
        X = (X - np.mean(X, axis = 0))/np.std(X, axis =0)
        n,d = X.shape
        dist = np.matmul(X,X.T)
        diag_dist = np.outer(np.diagonal(dist), np.ones([1, n]))
        dist = diag_dist + diag_dist.T - 2*dist
        iu1 = np.triu_indices(n, k=1)
        sigma_x = np.sqrt( np.median(dist[iu1]))
        return sigma_x
    else:
        return 1.

def make_graph(graph_type, d):
    graph = []
    if graph_type == "full":
        for i in range(d):
            graph.append([[i], range(i)])
        return graph
    elif graph_type == "marginal":
        graph = [[range(d), []]]
        return graph

    elif graph_type == "markov":
        graph.append([[0], []])
        for i in range(d-1):
            graph.append([[i+1], [i]])
        return graph
    




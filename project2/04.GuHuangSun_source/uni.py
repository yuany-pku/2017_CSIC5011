'''
Computes Google's PageRank, HITS Authority and Hub;
Stores the above information in txt files which are fed to Gephi for visulization.
'''

import scipy.io as sio
import numpy as np

def normalize(x):
    '''
    Normalizes the 1-norm of all rows of a matrix to 1
    '''
    col_sum = np.sum(x, axis=1, keepdims=True)
    x = x / col_sum
    return x

# Read from file
contents = sio.loadmat('univ_cn.mat')
unis = contents['univ_cn']
W = np.asarray(contents['W_cn'])
rank = contents['rank_cn']

# Get the names of the uni's
n = W.shape[1]
unis = [unis[:,i][0][0].split('.')[0].upper() for i in range(n)]

# Compute the pagerank
D = np.sum(W, axis=1, keepdims=True)
indices = [i for i in range(n) if D[i]>0]
P1 = np.zeros((n,n))
for i in indices:
    P1[i,:] = W[i,:] / D[i]

def PageRank(alpha):
    '''
    Gets the top universities according to pagerank
    '''
    Pa = P1 * alpha + (1 - alpha) * np.ones((n,n)) / n
    Pa = normalize(Pa)
    evals,evecs = np.linalg.eig(Pa.T)
    first_index = np.argsort(np.abs(evals))[-1]
    first_evec = evecs[:,first_index]
    PageRank_score = np.abs(first_evec/np.sum(first_evec))
    indicies = np.argsort(PageRank_score)[::-1]
    Uni_PageRank = [unis[i] for i in indicies]
    return PageRank_score, Uni_PageRank

PageRankScore, PageRankRanking = PageRank(0.85)

# HITS Authority
u,s,v = np.linalg.svd(W)
first_index = np.argsort(np.abs(s))[-1]
auth_score = v[:,first_index] / np.sum(v[:,first_index])
auth_ind = np.argsort(auth_score)[::-1]
auth_rank = [unis[i] for i in auth_ind]

# HITS Hub
hub_score = u[:,first_index] / np.sum(u[:,first_index])
hub_ind = np.argsort(hub_score)[::-1]
hub_rank = [unis[i] for i in hub_ind]

print('Actual ranking: \n', unis)
print('PageRank ranking: \n', PageRankRanking)
print('HITS authority ranking: \n', auth_rank)
print('HITS hub ranking: \n', hub_rank)

# Create nodes and edges for gephi
nodes = ['id,label,pagerank,authoriy,hub\n']
for i,j,k,l,m in zip(range(n), unis, PageRankScore, auth_score, hub_score):
    nodes.append('{},"{}",{},{},{}\n'.format(i+1, j, k, l, m))
with open('nodes.csv', 'w') as f:
    for i in nodes:
        f.write(i)

edges = ['Source,Target,Weight\n']
for i in range(n):
    for j in range(n):
        if W[i,j] > 0:
            edges.append('{},{},{}\n'.format(i+1,j+1,W[i,j]))
with open('edges.csv', 'w') as f:
    for i in edges:
        f.write(i)

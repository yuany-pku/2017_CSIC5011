
function [C1,C2,C3] = SpectralClustering(W, k)
% spectral clustering algorithm
% input: adjacency matrix W; number of cluster k 
% return: cluster column C1,C2,C3 by Unnormalized Laplacian(L), Nomalized
% Laplacian(D^(-1/2)LD^(-1/2), Trasition matrix(D^-1W) respectively         

% calculate degree matrix
degs = sum(W, 2);
D = sparse(1:size(W, 1), 1:size(W, 2), degs);

% compute unnormalized Laplacian
L = D - W;
d1 = diag(D);
d2 = d1.^(-1/2);
D1 = diag(d2);
LN = D1 * L * D1;
P = inv(D) * L;

% compute the eigenvectors corresponding to the k smallest eigenvalues
% diagonal matrix V is NcutL's k smallest magnitude eigenvalues 
% matrix Q whose columns are the corresponding eigenvectors.
[Q1,V1] = eigs(L, k, 'SA');
[Q2,V2] = eigs(LN, k, 'SA');
[Q3,V3] = eigs(P, k,'SR');
% use the k-means algorithm to cluster V row-wise
% C will be a n-by-1 matrix containing the cluster number for each data point
C1 = kmeans(Q1,k);
C2 = kmeans(Q2,k);
C3 = kmeans(Q3,k);

end
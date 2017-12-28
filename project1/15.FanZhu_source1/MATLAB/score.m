
function V = score(C, K)
% this function is to find the K cluster of undirect graph with
% adjacency matrix A

% V = score(A,K)

% Input:
% A --- adjacency matrix
% K --- number of clusters

% Output
% V --- cell, each entry is the set of node


% coauthor graph ---(3607,3607) 0-1 matrix A
% A = load('jiashun/coauthorship/coauthorAdj.txt');

% plot the giant coauthor graph

bgcogi = biograph(triu(C),[],'Showarrows','off','NodeAutoSize','off','EdgeType','segmented','NodeAutoSize','off','LayoutType','radial');

% first K leading eigenvetors of A
[Xi,~] = eigs(C,K);

% construct matrix R
for i = 1:K-1
    R(:,i) = Xi(:,i+1) ./ Xi(:,1);
end

% k-means
V = kmeans(R,K);





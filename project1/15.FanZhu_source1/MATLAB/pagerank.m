
function aut = pagerank(G,k,a)
% Input: G is citition matrix, PaperCitrePaper or AuthorCiteAuthor
%        k is num of authorities you want 
%        a is a parameter for adjusting transition probabilty matrix
% Output: Top-k nodes of largest authority
n = size(G,1);
d = G*ones(n,1);
D = diag(d);%degree matrix
P = pinv(D) * G; %transition probabilty matrix
E = 1/n * ones(n,1) * ones(1,n);
Pa = a * P + (1-a) * E; %positive adjusted transition probabilty matrix

[va,~] = eigs(Pa,1);

%sort
[~,index]=sort(abs(va),'descend');
aut = index(1:k);
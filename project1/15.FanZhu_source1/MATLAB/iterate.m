
function [topaut,tophub] = iterate(A,k)
% A is the adjacency matrix of PaperCitePaper or AuthorCiteAuthor
% k is the num of authorities and hubs we want to explore

[va, ~] = eigs(A'*A,1);
[vh, ~] = eigs(A *A',1);

% sort
[~, inda] = sort(abs(va),'descend');
[~, indh] = sort(abs(vh),'descend');

topaut = inda(1:k);
tophub = indh(1:k);
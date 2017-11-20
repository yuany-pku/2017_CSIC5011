load D:\¸Û¿Æ¿Î³Ì\pca\project2\univ_cn.mat W_cn univ_cn rank_cn
v = rank_cn;        % research rank of universities
webpage = univ_cn;  % webpage of universities in mainland china
W = W_cn;           % Link weight matrix

D = sum(W,2);
n = length(D);
idnz = find(D>0);
T = zeros(n,n);
T(idnz,idnz) = diag(1./D(idnz)) * W(idnz,idnz);

alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9]; % alpha=0.85 is Google's PageRank choice
for i=1:length(alpha),
	T1 = alpha(i) * T + (1-alpha(i))*ones(n,1)*ones(1,n)/n;
    [evec,eval] = eigs(T1',1);
	[score_page(:,i)]=evec/sum(evec);  % pagerank score
end
% PageRank
idp = ones(9,76);
for i=1:length(alpha),
    [~,id]=sort(score_page(:,i),'descend');
    idp(i,:)=id;
end
webpage{id(1:5)}
score_out = D; % out-degree score
score_in = sum(W,1)'; % in-degree score
score_research = max(v)-v; % research score

% HITS rank
[U,S,V] = svds(W);
u1 = U(:,1)/sum(U(:,1));
v1 = V(:,1)/sum(V(:,1));
[~,idu]=sort(u1,'descend');
[~,idv]=sort(v1,'descend');
webpage{idu(1:5)}   % Hub Ranking
webpage{idv(1:5)}  % Authority Ranking

% corr between actual ranking and Pagerank
% S present Spearman coefficient; K represent Kendall coefficient
coeff_S1=[];
coeff_K1=[];
pval_S1 =[];
pval_K1 =[];
for i=1:length(alpha),
    [coeff_S1(i),pval_S1(i)] = corr(idp(i,:)' , rank_cn , 'type' , 'Spearman');
    [coeff_K1(i),pval_K1(i)] = corr(idp(i,:)' , rank_cn , 'type' , 'Kendall');
end
% corr between actual ranking and Authority ranking
[coeff_S2,pval_S2] = corr(rank_cn , idv , 'type' , 'Spearman'); 
[coeff_K2,pval_K2] = corr(rank_cn , idv , 'type' , 'Kendall');  

% corr between actual ranking and Hub ranking
[coeff_S3,pval_S3] = corr(idu , rank_cn , 'type' , 'Spearman'); 
[coeff_K3,pval_K3] = corr(idu , rank_cn , 'type' , 'Kendall');  
f = figure('Position',[200 200 400 150]); 

% draw a table of corr between actual ranking and Pagerank
f = figure;
data=[coeff_S1; coeff_K1];
colnames = {'0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.85','0.9'};
rownames = {'Spearman', 'Kendall'};
t = uitable(f,'Data',data, 'Position',[20 20 262 204]);
t.ColumnName = {'0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.85','0.9'};
t.ColumnEditable = true;
t.RowName = rownames;
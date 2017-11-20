clear all;
clc ;

load univ_cn.mat  W_cn univ_cn rank_cn

v = rank_cn;        % research rank of universities
webpage = univ_cn;  % webpage of universities in mainland china
W = W_cn;           % Link weight matrix

D = sum(W,2);
n = length(D);
idnz = find(D>0);
T = zeros(n,n);
T(idnz,idnz) = diag(1./D(idnz)) * W(idnz,idnz);

alpha = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.85 0.9]; % alpha=0.85 is Google's PageRank choice

for i=1:length(alpha),
	T1 = alpha(i) * T + (1-alpha(i))*ones(n,1)*ones(1,n)/n;
    [evec,eval] = eigs(T1',1);
	[score_page(:,i)]=evec/sum(evec);  % pagerank score
end

% PageRank
[~,id0]=sort(score_page(:,8),'descend');
webpage{id0(1:10)}

score_out = D; % out-degree score
score_in = sum(W,1)'; % in-degree score
score_research = max(v)-v; % research score

% Authority ranking
[~,id1] = sort(score_in,'descend');
webpage{id1(1:10)}
% Hub ranking
[~,id2] = sort(score_out,'descend');
webpage{id2(1:10)}

%-------------------------------------
%-------------------------------------
%-------------------------------------

% HITS rank
[U,S,V] = svds(W);
u1 = U(:,1)/sum(U(:,1));
v1 = V(:,1)/sum(V(:,1));
[~,id3]=sort(v1,'descend');
[~,id4]=sort(u1,'descend');
webpage{id3(1:10)}   % Authority Ranking
webpage{id4(1:10)}   % Hub Ranking


%-------------------------------------
%-------------------------------------
%-------------------------------------


%for kendall
%Pagerank, pageranking
[rho0, pressure0] = corr(v, id0, 'type', 'Kendall' )
%Pagerank, authority ranking
[rho1, pressure1] = corr(v, id1, 'type', 'Kendall' )
%Pagerank, hub ranking
[rho2, pressure2] = corr(v, id2, 'type', 'Kendall' )
%hits, authority ranking
[rho3, pressure3] = corr(v, id3, 'type', 'Kendall' )
%hits, hub ranking
[rho4, pressure4] = corr(v, id4, 'type', 'Kendall' )
rho_kendall = [rho0, rho1, rho2, rho3, rho4]

%for Spearman
%Pagerank, pageranking
[rho0, pressure0] = corr(v, id0, 'type', 'Spearman' )
%Pagerank, authority ranking
[rho1, pressure1] = corr(v, id1, 'type', 'Spearman' )
%Pagerank, hub ranking
[rho2, pressure2] = corr(v, id2, 'type', 'Spearman' )
%hits, authority ranking
[rho3, pressure3] = corr(v, id3, 'type', 'Spearman' )
%hits, hub ranking
[rho4, pressure4] = corr(v, id4, 'type', 'Spearman' )
rho_spearman = [rho0, rho1, rho2, rho3, rho4]

plot(rho_spearman, '+')
hold on;
plot(rho_kendall, 'd')


%-------------------------------------
%-------------------------------------
%-------------------------------------
num = 100;
for i = 1:num + 1
    alpha(i) = (i - 1)/num;
end % alpha=0.85 is Google's PageRank choice

for i=1:length(alpha),
	T1 = alpha(i) * T + (1-alpha(i))*ones(n,1)*ones(1,n)/n;
    [evec,eval] = eigs(T1',1);
	[score_page(:,i)]=evec/sum(evec);  % pagerank score
end
for i = 1:length(alpha)
    [~,id]=sort(score_page(:,i),'descend');
    iddd(:, i) = id;
    [rhodd, pressureddd] = corr(v, id, 'type', 'Kendall' );
    [rhoddd, pressureddd] = corr(v, id, 'type', 'Spearman' );
    kendall_dd(i) = rhodd;
    spaerman_dd(i) = rhoddd;
end
plot(alpha, spaerman_dd, '+')
hold on;
plot(alpha, kendall_dd, 'd')
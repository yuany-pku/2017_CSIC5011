%--------------------------------------------------------------------------
% CSIC 5011 FinalProject
% Dec. 10, 2017
% Bu Qi
%--------------------------------------------------------------------------
load Agedata.mat Pair_Compar
load Groundtruth.mat Age

edg_num = size( Pair_Compar, 1 );
%--------------------------------------------------------------------------
NodeNum = 30; % number of individuals
%--------------------------------------------------------------------------
% Hodgerank
input = zeros( edg_num, 2 );
for row = 1 : edg_num
    if Pair_Compar( row, 4 ) == 1
        input( row, 1 ) = Pair_Compar( row, 2 );
        input( row, 2 ) = Pair_Compar( row, 3 );
    end
    if Pair_Compar( row, 4 ) == -1
        input( row, 1 ) = Pair_Compar( row, 3 );
        input( row, 2 ) = Pair_Compar( row, 2 );
    end
end
[score,totalIncon,harmIncon]=Hodgerank(input, NodeNum);
%--------------------------------------------------------------------------
% PageRank
% construct the link weight matrix
W = zeros( NodeNum );
for row = 1 : edg_num
    if Pair_Compar( row, 4 ) == -1
        W(Pair_Compar( row, 2 ), Pair_Compar( row, 3 )) = W(Pair_Compar( row, 2 ), Pair_Compar( row, 3 ))+1;
    end
    if Pair_Compar( row, 4 ) == 1
        W(Pair_Compar( row, 3 ), Pair_Compar( row, 2 )) = W(Pair_Compar( row, 3 ), Pair_Compar( row, 2 ))+1;
    end
end
D = sum(W,2);
n = length(D);
idnz = find(D>0);
T = zeros(n,n);
T(idnz,idnz) = diag(1./D(idnz)) * W(idnz,idnz);
alpha = .85;
T1 = alpha * T + (1-alpha)*ones(n,1)*ones(1,n)/n;
[evec,eval] = eigs(T1',1);
score_page=evec/sum(evec);  % pagerank score
%--------------------------------------------------------------------------
% compare different rankings
% true rank
[~, id] = sort( Age(:, 2), 'descend');
% Hodgerank
[~,id_Hodge_1]=sort(score(:, 1), 'descend');
[~,id_Hodge_2]=sort(score(:, 2), 'descend');
[~,id_Hodge_3]=sort(score(:, 3), 'descend');
[~,id_Hodge_4]=sort(score(:, 4), 'descend');
% PageRank
[~,id_Page]=sort(score_page, 'descend');
% HITS rank
[U,S,V] = svds(W);
u1 = U(:,1)/sum(U(:,1));
v1 = V(:,1)/sum(V(:,1));
[~,idu]=sort(u1,'descend');
[~,idv]=sort(v1,'descend');
% preprocess for comparison
id_1 = zeros(NodeNum, 1);
id_2 = zeros(NodeNum, 1);
id_3 = zeros(NodeNum, 1);
id_4 = zeros(NodeNum, 1);
id_P = zeros(NodeNum, 1); 
for item = 1 : size(id, 1)
    id_1( find( id_Hodge_1==id(item)) ) = item;
    id_2( find( id_Hodge_2==id(item)) ) = item;
    id_3( find( id_Hodge_3==id(item)) ) = item;
    id_4( find( id_Hodge_4==id(item)) ) = item;
    id_P( find( id_Page==id(item)) ) = item;
end
id = linspace( 1, NodeNum, NodeNum )';
% spearman comparison
[rho_1] = spearman(id_1, id);
[rho_2] = spearman(id_2, id);
[rho_3] = spearman(id_3, id);
[rho_4] = spearman(id_4, id);
[rho_5] = spearman(id_P, id);
%--------------------------------------------------------------------------
%{
figure
bar(score(:, 1))
xlabel('individuals', 'FontSize', 14)
ylabel('Hodgerank Value', 'FontSize', 14)
hold off
figure
bar(score_page)
xlabel('individuals', 'FontSize', 14)
ylabel('PageRank Value', 'FontSize', 14)
hold off
%}
%--------------------------------------------------------------------------
% consider random networks
num_repeat = 100;
% ER network
percent = linspace( .1, .9, 21 );
per_ER = percent;
tot_ER = zeros( length(per_ER), num_repeat );
har_ER = zeros( length(per_ER), num_repeat );
for step_num = 1 : num_repeat
    tot_inconsist = zeros( size(percent) );
    har_inconsist = zeros( size(percent) );
    ind = 0;
    for perc = percent
        ind = ind + 1;
        input_ER = [];
        L = round( perc*NodeNum*(NodeNum-1)/2 );
        A = gen_ER(NodeNum, L);
        for row = 1 : edg_num
            if A( Pair_Compar( row, 2 ), Pair_Compar( row, 3 ) ) == 1
                if Pair_Compar( row, 4 ) == 1
                    input_ER = [ input_ER; [Pair_Compar( row, 2 ), Pair_Compar( row, 3 )] ];
                end
                if Pair_Compar( row, 4 ) == -1
                    input_ER = [ input_ER; [Pair_Compar( row, 3 ), Pair_Compar( row, 2 )] ];
                end
            end
        end
        [~ , t, h]=Hodgerank(input_ER, NodeNum);
        tot_inconsist( ind ) = t(1);
        har_inconsist( ind ) = h(1);
    end 
    tot_ER(:, step_num) = tot_inconsist;
    har_ER(:, step_num) = har_inconsist;
end
% BA network        
m0 = 15;
time_num = 15;
m_list = linspace(2, 15, 10);
percent = (m0 + m_list .* time_num) * 2 / (NodeNum * (NodeNum - 1));
per_BA = percent;
tot_BA = zeros( length(per_BA), num_repeat );
har_BA = zeros( length(per_BA), num_repeat );
for step_num = 1 : num_repeat
    tot_inconsist = zeros( size(percent) );
    har_inconsist = zeros( size(percent) );
    ind = 0;
    for m = m_list
        ind = ind + 1;
        input_BA = [];
        A = gen_BA(m0, m, time_num);
        for row = 1 : edg_num
            if A( Pair_Compar( row, 2 ), Pair_Compar( row, 3 ) ) == 1
                if Pair_Compar( row, 4 ) == 1
                    input_BA = [ input_BA; [Pair_Compar( row, 2 ), Pair_Compar( row, 3 )] ];
                end
                if Pair_Compar( row, 4 ) == -1
                    input_BA = [ input_BA; [Pair_Compar( row, 3 ), Pair_Compar( row, 2 )] ];
                end
            end
        end
        [~ , t, h]=Hodgerank(input_BA, NodeNum);
        percent( ind ) = sum(sum( A )) / (NodeNum*(NodeNum-1));
        tot_inconsist( ind ) = t(1);
        har_inconsist( ind ) = h(1);
    end
    tot_BA(:, step_num) = tot_inconsist;
    har_BA(:, step_num) = har_inconsist;
end
% figure plot
figure
errorbar( per_ER, mean(tot_ER,2), std(tot_ER,0,2), 'LineWidth', 2 )
hold on
errorbar( per_ER, mean(har_ER,2), std(har_ER,0,2), 'LineWidth', 2 )
hold on
errorbar( per_BA, mean(tot_BA,2), std(tot_BA,0,2), 'LineWidth', 2 )
hold on
errorbar( per_BA, mean(har_BA,2), std(har_BA,0,2), 'LineWidth', 2 )
hold off
set(gca, 'LineWidth', 2, 'FontSize', 14)
xlabel( 'connectivity' )
ylabel( 'Inconsistency' )
ylim([0, 0.22])
legend('total inconsistency ER' ,'harmonic inconsistency ER', 'total inconsistency BA', 'harmonic inconsistency BA')
grid on
%--------------------------------------------------------------------------
% results
[id_1, id_2, id_3, id_4, id_P, id]
[rho_1, rho_2, rho_3, rho_4, rho_5]
totalIncon'
harmIncon'



%--------------------------------------------------------------------------
% functions built in
function [score,totalIncon,harmIncon]=Hodgerank(incomp, NodeNum)

% function [score,totalIncon,harmIncon]=Hodgerank(incomp)
%   Find the Hodge Decomposition of pairwise ranking data with four models:
%       model1: Uniform noise model, Y_hat(i,j) = 2 pij -1
%       model2: Bradley-Terry, Y_hat(i,j) = log(abs(pij+eps))-log(abs(1-pij-eps))
%       model3: Thurstone-Mosteller, Y_hat(i,j) ~ norminv(abs(1-pij-eps));
%       model4: Arcsin, Y_hat4(i,j) = asin(2*pij-1)
%
%   Input: 
%       incomp: n-by-2 matrix, the first column is the video with better quality 
%           and the second line is the video with worse quality.
%   Output:
%       score: 16-by-4 score matrix of 16 videos and 4 models
%       totalIncon: 4-by-1 total inconsistency
%       harmIncon: 4-by-1 harmonic inconsistency

%   by Qianqian Xu, Yuan Yao
%       CAS and Peking University
%       August, 2012
%
%   Reference:
%       Qianqian Xu, Qingming Huang, Tingting Jiang, Bowei Yan, Weisi Lin 
%   and Yuan Yao, "HodgeRank on Random Graphs for Subjective Video Quality  
%   Assessment", IEEE Transaction on Multimedia, vol. 14, no. 3, pp. 844-
%   857, 2012.

NodeNum;
if ~isempty(incomp)
    count=zeros(NodeNum,NodeNum);
    eps=1e-4;
    sigma=1;

    for i = 1:size(incomp,1),
        for k = 0:NodeNum-1,
            for l = 0:NodeNum-1,
                if ((mod(incomp(i,1),NodeNum)==k) && (mod(incomp(i,2),NodeNum)==l)),
                    if k==0,k=NodeNum;end
                    if l==0,l=NodeNum;end
                    count(k,l)=count(k,l)+1;
                end
            end
        end
    end
end

score_model1_rg=[]; % score of each item for model1
score_model2_rg=[]; % score of each item for model2
score_model3_rg=[]; % score of each item for model3
score_model4_rg=[]; % score of each item for model4
res_rg=[];  % total inconsistency
harmonic_res=[];
curl_res=[];

c0=count;
thresh=0;
G=((c0+c0')>thresh);

edges=[];
d0=[];
d1=[];
triangles=[];
k=0;
GG=[];

% Pairwise comparison skew-symmetric matrices
Y_hat1=zeros(NodeNum,NodeNum);
Y_hat2=zeros(NodeNum,NodeNum);
Y_hat3=zeros(NodeNum,NodeNum);
Y_hat4=zeros(NodeNum,NodeNum);

% Edge flow vectors
y_hat1=[];
y_hat2=[];
y_hat3=[];
y_hat4=[];
for j=1:(NodeNum-1),
    for i=(j+1):NodeNum,
        if (G(i,j)>0),
            pij=c0(i,j)/(c0(i,j)+c0(j,i));
            % Uniform noise model
            Y_hat1(i,j)=2*pij-1;
            % Bradley-Terry model
            Y_hat2(i,j)=log(abs(pij+eps))-log(abs(1-pij-eps));
            % Thurstone-Mostelle model
            Y_hat3(i,j)=-sqrt(2)*sigma*norminv(abs(1-pij-eps));
            %arcsin model
            Y_hat4(i,j)=asin(2*pij-1);
            edges=[edges, [i;j]];
            k=k+1;
            GG(k)=c0(i,j)+c0(j,i);   %-------------------------------weight=number of raters-
            %              GG(k)=1;  %------------------------------------------- -unweighted
            y_hat1(k)=Y_hat1(i,j);
            y_hat2(k)=Y_hat2(i,j);
            y_hat3(k)=Y_hat3(i,j);
            y_hat4(k)=Y_hat4(i,j);

        end
    end
end

for i=1:NodeNum,
    for j=(i+1):NodeNum,
        for k=(j+1):NodeNum,
            if ((G(i,j)>0)&&(G(j,k)>0)&&(G(k,i)>0)),
                triangles = [triangles, [i,j,k]'];
            end
        end
    end
end
Y=[];

numEdge=size(edges,2);
numTriangle=size(triangles,2);

d0=zeros(numEdge,NodeNum);
for k=1:numEdge,
    d0(k,edges(1,k))=1;
    d0(k,edges(2,k))=-1;
end

d1 = zeros(numTriangle,numEdge);
for k=1:numTriangle,
    index=find(edges(1,:)==triangles(2,k) & edges(2,:)==triangles(1,k));
    d1(k,index)=1;
    index=find(edges(1,:)==triangles(3,k) & edges(2,:)==triangles(2,k));
    d1(k,index)=1;
    index=find(edges(1,:)==triangles(3,k) & edges(2,:)==triangles(1,k));%----------------------------edited by Yan on 11th, Mar
    d1(k,index)=-1;
end

% 0-degree Laplacian (graph Laplacian)
L0=[];
% d0_star is the conjugate of d0
d0_star = d0'*diag(GG);
L0=d0_star*d0;

% L1 Laplacian (upper part of 1-Laplacian)
% d1_star is the conjugate of d1
d1_star = diag(1./GG)*d1';
L1=d1*d1_star;


% Find divergence and global score for model I
div1 = d0_star*y_hat1';
x_global_m1=lsqr(L0,div1);
score_model1_rg=log(x_global_m1+1);

% Find curl and harmonic components for model II
curl1=d1*y_hat1';
curl_m1=lsqr(L1,curl1);
harmonic_m1=y_hat1'-d0*x_global_m1-d1_star*curl_m1;    % i.e. \hat{Y}^{(2)} in the paper ACM-MM11
res_rg(1,:) = (y_hat1'-d0*x_global_m1)'*diag(GG)*(y_hat1'-d0*x_global_m1)/(y_hat1*diag(GG)*y_hat1');
harmonic_res(1,:) = (harmonic_m1)'*diag(GG)*(harmonic_m1)/(y_hat1*diag(GG)*y_hat1');


div2 = d0_star*y_hat2';
x_global_m2=lsqr(L0,div2);
score_model2_rg=log(exp(x_global_m2));
curl2=d1*y_hat2';
curl_m2=lsqr(L1,curl2);
harmonic_m2=y_hat2'-d0*x_global_m2-d1_star*curl_m2;
res_rg(2,:) = (y_hat2'-d0*x_global_m2)'*diag(GG)*(y_hat2'-d0*x_global_m2)/(y_hat2*diag(GG)*y_hat2');
harmonic_res(2,:) = (harmonic_m2)'*diag(GG)*(harmonic_m2)/(y_hat2*diag(GG)*y_hat2');


div3 = d0_star*y_hat3';
x_global_m3=lsqr(L0,div3);
score_model3_rg=log(exp(x_global_m3));
curl3=d1*y_hat3';
curl_m3=lsqr(L1,curl3);
harmonic_m3=y_hat3'-d0*x_global_m3-d1_star*curl_m3;
res_rg(3,:) = (y_hat3'-d0*x_global_m3)'*diag(GG)*(y_hat3'-d0*x_global_m3)/(y_hat3*diag(GG)*y_hat3');
harmonic_res(3,:) = (harmonic_m3)'*diag(GG)*(harmonic_m3)/(y_hat3*diag(GG)*y_hat3');


div4 = d0_star*y_hat4';
x_global_m4=lsqr(L0,div4);
score_model4_rg=sin(x_global_m4);
curl4=d1*y_hat4';
curl_m4=lsqr(L1,curl4);
harmonic_m4=y_hat4'-d0*x_global_m4-d1_star*curl_m4;
res_rg(4,:) = (y_hat4'-d0*x_global_m4)'*diag(GG)*(y_hat4'-d0*x_global_m4)/(y_hat4*diag(GG)*y_hat4');
harmonic_res(4,:) = (harmonic_m4)'*diag(GG)*(harmonic_m4)/(y_hat4*diag(GG)*y_hat4');


score = [score_model1_rg,score_model2_rg,score_model3_rg,score_model4_rg];
totalIncon = res_rg;
harmIncon = harmonic_res; 

end

function [rho] = spearman(X, Y)
n = length(X);
d_square = sum((X-Y).^2);
rho = 1 - d_square * 6 / (n*(n^2-1)); 
end
%--------------------------------------------------------------------------
% generate a ER network with N nodes
% and L edges over N*(N-1)/2 possible edges
function [A] = gen_ER(N, L)
    if L < (N-1) || L > (N*(N-1)/2)
        fprintf('not applicable \n');
        A = [];
        return
    end
    A = zeros(N);
    B = OrderAllEdge(A);
    while sum(sum(A)) ~= 2*L
        while chkconnect(A) == 0
            %pick L edges from N*(N-1)/2 possible ones
            %generate a sequence of ordered edges
            A0 = zeros(N);
            Edge = sort(randsample(N*(N-1)/2 , L));
            for l = 1 : L
                k = Edge(l);
                a = B(k, 1);
                b = B(k, 2);
                A0(a, b) = 1;
                A0(b, a) = 1;
            end
            A = A0;
        end
    end
end
%order all possible edges of an N*N adjacency matrix A
%give out a matrix
% B = [Node 1, Node 2, Link(1,2)]
%Node 1 from 1 to N-1
%Node 2 from i+1 to N
%Totally there are N*(N-1)/2 edges
function [B] = OrderAllEdge(A)
    N = size(A, 1);
    B = [];
    for i = 1 : (N-1)
        for j = (i+1) : N
            B = [B ; [i, j, A(i, j)]];
        end
    end
end
%input an n*n adjacency matrix A undirected
%check whether it is connected (x=1) or not (x=0)
function [x] = chkconnect(A)
    x = 1;
    [a, b] = size (A); 
    A0 = A;
    for i = 2 : a-1
        A0 = A0 + A^i/factorial(i);
    end
    for j = 1 : a*b
        if A0(j) == 0  
            x = 0;
        end
    end
end
%--------------------------------------------------------------------------
% generate a BA network
% undirected network
% start with a small number (m0) of nodes forming a ring with m0 edges
% at each step we add a new node with m<=m0 edges that link the new 
% node to m different nodes in the system
% the distribution of the edges is proportional to the degree of nodes
% finally after t steps we have m0+t nodes and m0+m*t edges
function [A] = gen_BA( m0, m, t )
N = m0 + t;
A = zeros( N );
% start with a ring
for x = 1 : m0-1
   A(x, x+1) = 1;
   A(x+1, x) = 1;
end
A(1, m0) = 1;
A(m0, 1) = 1;
% add nodes and edges
for node = m0 + 1 : N
    D = sum( A ); % degrees of nodes
    L = sum( D ); % number of edges * 2
    %for edge = 1 : m
    while sum(sum(A)) < L+2*m
        edge_num = randi( L ); 
        node_num = 1;
        while edge_num - sum( D( 1:node_num ) ) > 0
            node_num = node_num + 1;
        end
        A( node, node_num ) = 1;
        A( node_num, node ) = 1;
    end
end
% randomly swap some node indices
for swap_num = 1 : N
    A0 = A;
    x = randi( N );
    y = randi( N );
    for node = 1 : N
        if node~=x && node~=y && x~=y
            A0(x, node) = A(y, node);
            A0(node, x) = A(node, y);
            A0(y, node) = A(x, node);
            A0(node, y) = A(node, x);
        end
    end
    A = A0;
end
end






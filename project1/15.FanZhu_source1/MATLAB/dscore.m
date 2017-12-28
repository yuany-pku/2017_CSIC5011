
function  [Rl,Rr,cidx,clu1,clu2,clu3] = dscore(C, K)
% this function is to find the K cluster of directed graph with
% adjacency matrix C

% V = score(C, aulist, K)

% Input:
% A --- adjacency matrix
% K --- number of clusters

% Output
% V --- cell, each entry is the set of node

% coauthor graph ---(3607,3607) 0-1 matrix A
% A = load('jiashun/coauthorship/coauthorAdj.txt');

% plot the giant coauthor graph

% bgcogi = biograph(triu(C),[],'Showarrows','off','NodeAutoSize','off','EdgeType','segmented','NodeAutoSize','off','LayoutType','radial');

% first K leading eigenvetors of A
[U,~,V] = svds(C,K,'L'); m = size(C,1);

N1 = sign(C*C'); N2 = sign(C'*C);

[~,indx1] = graphconncomp(sparse(N1));
[~,indx2] = graphconncomp(sparse(N2));

% Define R
mod1 = mode(indx1); mod2 = mode(indx2);
id1 = find(indx1 == mod1); id2 = find(indx2 == mod2);
Rl = zeros(m,K-1); Rr = zeros(m,K-1);
for i = 1: K-1
    Rl(id1,i) = sign(U(id1,i+1)./U(id1,1)).*min(abs(U(id1,i+1)./U(id1,1)),log(m));
    Rr(id2,i) = sign(V(id2,i+1)./V(id2,1)).*min(abs(V(id2,i+1)./V(id2,1)),log(m));
end

% four sets
ints = intersect(id1,id2);
intd12 = setdiff(id1,id2);
intd21 = setdiff(id2,id1);
intu = union(id1,id2);
intdu = setdiff(1:m,intu);

% k-means
cidxl = kmeans(Rl,K);
plot(Rl(cidxl==1,1),Rl(cidxl==1,2),'r.', ...
             Rl(cidxl==2,1),Rl(cidxl==2,2),'b*',Rl(cidxl==3,1),Rl(cidxl==3,2),'g+');
         
figure;
cidxr = kmeans(Rr,K);
 plot(Rr(cidxr==1,1),Rr(cidxr==1,2),'r.', ...
             Rr(cidxr==2,1),Rr(cidxr==2,2),'b*',Rr(cidxr==3,1),Rr(cidxr==3,2),'g+');

Rls = Rl(ints,:);
Rrs = Rr(ints,:);
cidx = kmeans([Rls,Rrs],3);

% find the cluster
cd1 = find(cidx == 1); n1 = length(cd1);
cd2 = find(cidx == 2); n2 = length(cd2);
cd3 = find(cidx == 3); n3 = length(cd3);

% point
clu1 = ints(cd1); clu2 = ints(cd2); clu3 = ints(cd3);

nn = length(cidx) -1;

rs1 = sum(Rl(ints(cd1),:))/n1;
rs2 = sum(Rl(ints(cd2),:))/n2;
rs3 = sum(Rl(ints(cd3),:))/n3;
rz1 = sum(Rr(ints(cd1),:))/n1;
rz2 = sum(Rr(ints(cd2),:))/n2;
rz3 = sum(Rr(ints(cd3),:))/n3;

for i = 1:length(intd12)
    s1 = norm(Rl(intd12(i),:) - rs1);
    s2 = norm(Rl(intd12(i),:) - rs2);
    s3 = norm(Rl(intd12(i),:) - rs3);
    [~, j] = min([s1,s2,s3]);
    nn = nn + 1;
    cidx(nn + 1) = j;
    if j == 1
        clu1(end+1) = intd12(i);
    elseif j==2
        clu2(end+1) = intd12(i);
    else
        clu3(end+1) = intd12(i);
    end
end 

for i = 1:length(intd21)
    s1 = norm(Rr(intd21(i),:) - rz1);
    s2 = norm(Rr(intd21(i),:) - rz2);
    s3 = norm(Rr(intd21(i),:) - rz3);
    [~, j] = min([s1,s2,s3]);
    nn = nn + 1;
    cidx(nn + 1) = j;
    if j == 1
        clu1(end+1) = intd21(i);
    elseif j==2
        clu2(end+1) = intd21(i);
    else
        clu3(end+1) = intd21(i);
    end
end

NN = sign(C+C');
cc1 = clu1; cc2 = clu2; cc3 = clu3;

for i = 1:length(intdu)
    s1 = sum(NN(intdu(i),cc1));
    s2 = sum(NN(intdu(i),cc2));
    s3 = sum(NN(intdu(i),cc3));
    [~,j] = max([s1,s2,s3]);
    nn = nn + 1;
    cidx(nn + 1) = j;
    if j == 1
        clu1(end+1) = intdu(i);
    elseif j==2
        clu2(end+1) = intdu(i);
    else
        clu3(end+1) = intdu(i);
    end
end 
   





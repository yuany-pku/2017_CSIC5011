% Final project of CSIC 5011 (Fall 2017) at HKUST
% Guiyu Cao (gcaoaa@connect.ust.hk) and Yi-Su Lo (yloab@connect.ust.hk)

clear all;
clc;
load('karate.mat');

% % % % % % % % % % % % % % % % % % % % % 
% Problem A
% % % % % % % % % % % % % % % % % % % % % 
% construct matrix D
rank_num = 34;
dd = sum(A,2);
for i = 1:rank_num
    for j = 1:rank_num
        if (i == j) 
            D(i,j) = dd(i);
        else
            D(i,j) = 0;
        end
    end
end

% construct matrix L(unormalized graph Laplacian)
L = D - A;

% construct matrix h_L(normalized graph Laplacian)
% D^(-1/2)
for i = 1:rank_num
    for j = 1:rank_num
        if (i == j) 
            D_2(i,j) = 1.0/sqrt(D(i,j));
        else
            D_2(i,j) = 0;
        end
    end
end
h_L = D_2*L*D_2;

% spectral clustering via Cheeger vector
[evec,eval] = eig(h_L);
Cheeger_vector = evec(:,2);
for i = 1:rank_num
    if(Cheeger_vector(i) < 0)
        c0_Cheeger(i) = 0;
    else
        c0_Cheeger(i) = 1;
    end
end
% plot((c0 - c0_Cheeger' ))

% % % % % % % % % % % % % % % % % % % % % 
% Problem B
% % % % % % % % % % % % % % % % % % % % % 
% construct Markov transition matrix P
% D^(-1)
for i = 1:rank_num
    for j = 1:rank_num
        if (i == j) 
            D_1(i,j) = 1.0/D(i,j);
        else
            D_1(i,j) = 0;
        end
    end
end
P = D_1*A;
[EigV, EigD] = eig(P);
stationary_pi = EigV(:,1);

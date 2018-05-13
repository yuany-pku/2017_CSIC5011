function [cov,temp,iter]=m_estimator(X)
%   [cov,temp,iter]=m_estimator(X)
%
%
%   Tyler's M-estimator for covariance
%       X: data matrix with each row representing a point
%       cov: the estimated covariance
%   Reference
%       Robust subspace recovery by geodesically convex optimization,
%       Teng Zhang, http://arxiv.org/pdf/1206.1386v2

% Teng Zhang
%   Princeton University
%

[N,D]=size(X);
initcov=eye(D);
oldcov=initcov-1;
cov=initcov;
iter=1;
eps=10^-10;%regularization parameter
while norm((oldcov-cov),'fro')>10^-8 & iter<100
temp=X*(cov+eps*eye(D))^-1;

d=sum(temp.*conj(X),2);
oldcov=cov;
%temp=((real(d./log(d))+eps*ones(N,1)).^-1);
temp=((real(d)+eps*ones(N,1)).^-1);
cov=X'.*repmat(((temp)),1,D)'*X/N;
cov=cov/trace(cov);
iter=iter+1;
end
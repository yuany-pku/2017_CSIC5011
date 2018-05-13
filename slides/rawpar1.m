clear; tic;

% Parallel Analysis Program For Raw Data and Data Permutations.

%  This program conducts parallel analyses on data files in which
%  the rows of the data matrix are cases/individuals and the
%  columns are variables; There can be no missing values;

%  You must also specify:
%   -- the # of parallel data sets for the analyses;
%   -- the desired percentile of the distribution of random
%      data eigenvalues;
%   -- whether principal components analyses or principal axis/common
%      factor analysis are to be conducted, and
%   -- whether normally distributed random data generation or 
%      permutations of the raw data set are to be used in the
%      parallel analyses;

%  WARNING: Permutations of the raw data set are time consuming;
%  Each parallel data set is based on column-wise random shufflings
%  of the values in the raw data matrix using Castellan's (1992, 
%  BRMIC, 24, 72-77) algorithm; The distributions of the original 
%  raw variables are exactly preserved in the shuffled versions used
%  in the parallel analyses; Permutations of the raw data set are
%  thus highly accurate and most relevant, especially in cases where
%  the raw data are not normally distributed or when they do not meet
%  the assumption of multivariate normality (see Longman & Holden,
%  1992, BRMIC, 24, 493, for a Fortran version); If you would
%  like to go this route, it is perhaps best to (1) first run a 
%  normally distributed random data generation parallel analysis to
%  familiarize yourself with the program and to get a ballpark
%  reference point for the number of factors/components;
%  (2) then run a permutations of the raw data parallel analysis
%  using a small number of datasets (e.g., 10), just to see how long
%  the program takes to run; then (3) run a permutations of the raw
%  data parallel analysis using the number of parallel data sets that
%  you would like use for your final analyses; 1000 datasets are 
%  usually sufficient, although more datasets should be used
%  if there are close calls.


% The "load" command can be used to read a raw data file

% The raw data matrix must be named "raw"


%  These next commands generate artificial raw data 
%  (50 cases) that can be used for a trial-run of
%  the program, instead of using your own raw data; 
%  Just run this whole file; However, make sure to
%  delete these commands before attempting to run your own data.

% Start of artificial data commands.
com = randn(500,3);
raw = randn(500,9);
raw(:,1:3) = raw(:,1:3) + [ com(:,1) com(:,1) com(:,1) ];
raw(:,4:6) = raw(:,4:6) + [ com(:,2) com(:,2) com(:,2) ];
raw(:,7:9) = raw(:,7:9) + [ com(:,3) com(:,3) com(:,3) ];
% End of artificial data commands.


ndatsets  = 100  ; % Enter the desired number of parallel data sets here

percent   = 95  ; % Enter the desired percentile here

% Specify the desired kind of parellel analysis, where:
% 1 = principal components analysis
% 2 = principal axis / common factor analysis
kind = 1 ;

% Enter either
%  1 for normally distributed random data generation parallel analysis, or
%  2 for permutations of the raw data set (more time consuming).
randtype = 2 ;

%the next command can be used to set the state of the random # generator
randn('state',1953125)

%%%%%%%%%%%%%%% End of user specifications %%%%%%%%%%%%%%%


[ncases,nvars] = size(raw);

% principal components analysis & random normal data generation
if (kind == 1 & randtype == 1)
realeval = flipud(sort(eig(corrcoef(raw))));
for nds = 1:ndatsets; evals(:,nds) = eig(corrcoef(randn(ncases,nvars)));end
end

% principal components analysis & raw data permutation
if (kind == 1 & randtype == 2)
realeval = flipud(sort(eig(corrcoef(raw))));
for nds = 1:ndatsets; 
x = raw;
for lupec = 2:nvars;
    col = x(randperm(ncases),lupec);
    x(:,lupec) = col;
%for luper = 1:(ncases -1);
%k = fix( (ncases - luper + 1) * rand(1) + 1 )  + luper - 1;
%d = x(luper,lupec);
%x(luper,lupec) = x(k,lupec);
%x(k,lupec) = d;end;end;
end
evals(:,nds) = eig(corrcoef(x));end;end

% PAF/common factor analysis & random normal data generation
if (kind == 2 & randtype == 1)
r = corrcoef(raw);
smc = 1 - (1 ./ diag(inv(r)));
for ii=1:size(r,1);r(ii,ii) = smc(ii,1);end;
realeval = flipud(sort(eig(r)));
for nds = 1:ndatsets; 
r = corrcoef(randn(ncases,nvars));
smc = 1 - (1 ./ diag(inv(r)));
for ii=1:size(r,1);r(ii,ii) = smc(ii,1);end;
evals(:,nds) = eig(r);
end;end

% PAF/common factor analysis & raw data permutation
if (kind == 2 & randtype == 2)
r = corrcoef(raw);
smc = 1 - (1 ./ diag(inv(r)));
for ii=1:size(r,1);r(ii,ii) = smc(ii,1);end;
realeval = flipud(sort(eig(r)));
for nds = 1:ndatsets; 
x = raw;
for lupec = 1:nvars;
for luper = 1:(ncases -1);
k = fix( (ncases - luper + 1) * rand(1) + 1 )  + luper - 1;
d = x(luper,lupec);
x(luper,lupec) = x(k,lupec);
x(k,lupec) = d;end;end;
r = corrcoef(x);
smc = 1 - (1 ./ diag(inv(r)));
for ii=1:size(r,1);r(ii,ii) = smc(ii,1);end;
evals(:,nds) = eig(r);
end;end


evals = flipud(sort(evals,1));
means = (mean(evals,2));   % mean eigenvalues for each position.
evals = sort(evals,2);     % sorting the eigenvalues for each position.
percentiles = (evals(:,round((percent*ndatsets)/100)));  % percentiles.
pvals = sum(evals>(realeval*ones(1,ndatsets)),2)/ndatsets; % p-values of observed random eigenvalues greater than real eigenvalues

format short
disp([' ']);disp(['PARALLEL ANALYSIS ']); disp([' '])
if (kind == 1 & randtype == 1);
disp(['Principal Components Analysis & Random Normal Data Generation' ]);disp([' ']);end
if (kind == 1 & randtype == 2);
disp(['Principal Components Analysis & Raw Data Permutation' ]);disp([' ']);end
if (kind == 2 & randtype == 1);
disp(['PAF/Common Factor Analysis & Random Normal Data Generation' ]);disp([' ']);end
if (kind == 2 & randtype == 2);
disp(['PAF/Common Factor Analysis & Raw Data Permutation' ]);disp([' ']);end
disp(['Variables  = ' num2str(nvars) ]);
disp(['Cases      = ' num2str(ncases) ]);
disp(['Datsets    = ' num2str(ndatsets) ]);
disp(['Percentile = ' num2str(percent) ]); disp([' '])
disp(['Raw Data Eigenvalues, & Mean & Percentile Random Data Eigenvalues']);disp([' '])
disp(['      Root   Raw Data   P-values    Means   Percentiles' ])
disp([(1:nvars).'  realeval  pvals means  percentiles]);

if kind == 2;
disp(['Warning: Parallel analyses of adjusted correlation matrices' ]);
disp(['e.g., with SMCs on the diagonal, tend to indicate more factors' ]);
disp(['than warranted (Buja, A., & Eyuboglu, N., 1992, Remarks on parallel' ]);
disp(['analysis. Multivariate Behavioral Research, 27, 509-540).' ]);
disp(['The eigenvalues for trivial, negligible factors in the real' ]);
disp(['data commonly surpass corresponding random data eigenvalues' ]);
disp(['for the same roots. The eigenvalues from parallel analyses' ]);
disp(['can be used to determine the real data eigenvalues that are' ]);
disp(['beyond chance, but additional procedures should then be used' ]);
disp(['to trim trivial factors.' ]);disp([' ' ]);
disp(['Principal components eigenvalues are often used to determine' ]);
disp(['the number of common factors. This is the default in most' ]);
disp(['statistical software packages, and it is the primary practice' ]);
disp(['in the literature. It is also the method used by many factor' ]);
disp(['analysis experts, including Cattell, who often examined' ]);
disp(['principal components eigenvalues in his scree plots to determine' ]);
disp(['the number of common factors. But others believe this common' ]);
disp(['practice is wrong. Principal components eigenvalues are based' ]);
disp(['on all of the variance in correlation matrices, including both' ]);
disp(['the variance that is shared among variables and the variances' ]);
disp(['that are unique to the variables. In contrast, principal' ]);
disp(['axis eigenvalues are based solely on the shared variance' ]);
disp(['among the variables. The two procedures are qualitatively' ]);
disp(['different. Some therefore claim that the eigenvalues from one' ]);
disp(['extraction method should not be used to determine' ]);
disp(['the number of factors for the other extraction method.' ]);
disp(['The issue remains neglected and unsettled.' ]);disp([' ']);disp([' ']);end

plot ( (1:nvars).',[realeval means  percentiles],'-*')
xlabel('Index'); ylabel('Eigenvalues'); title ('Real and Random-Data Eigenvalues')
set(get(gca,'YLabel'),'Rotation',90.0) % for the rotation of the YLabel
set(gca,'XTick', 1:nvars)
set(gca,'FontName','Times', 'FontSize',16, 'fontweight', 'normal' );
legend('real data eigenvalues','mean random eigenvalues','percentile random eigenvalues'); legend boxoff; 
textobj = findobj('type', 'text'); set(textobj, 'fontunits', 'points'); 
set(textobj,'FontName','Times', 'fontsize', 16); set(textobj, 'fontweight', 'normal');





disp(['time for this problem = ', num2str(toc) ]); disp([' '])


 


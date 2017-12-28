
function [ARI,VI] = arivi(clus1,clus2)
%% Adjusted Rand Index
n = length(unique(clus1));
m = length(clus1);

sa = 0; sb = 0;
for i = 1:n
    c1{i} = clus1 == i;
    a(i) = sum(c1{i});
    sa = sa + nchoosek(a(i),2);
    c2{i} = clus2 == i;
    b(i) = sum(c2{i});
    sb = sb + nchoosek(b(i),2);
end

s = 0;
for i = 1:n
    for j = 1:n
        nn(i,j) = sum(c1{i} .* c2{j});
        s = s + nn(i,j)*(nn(i,j)-1)/2;
    end
end

ARI = (s - 2*(sa * sb)/(m*(m-1))) / (0.5*(sa+sb) - (2*sa*sb/(m*(m - 1))));

%% Variation of information
for i=1:n
    p(i) = a(i)/n;
    q(i) = b(i)/n;
end
t = 0;
for i=1:n
    for j=1:n
        r(i,j) = nn(i,j)/n;
        if r(i,j)
            t = t - r(i,j)*(log(r(i,j)/p(i)) + log(r(i,j)/q(j)));
        end
    end
end
VI = t;
end


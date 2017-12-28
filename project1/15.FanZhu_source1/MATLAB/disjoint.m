
W = load('Jiashun/coauthorship/coauthorAdj.txt');

% load the author list
fid = fopen('Jiashun/authorList.txt');

tline = fgetl(fid); i = 0;
while ischar(tline)
    i = i + 1;
    au{i} = tline;
    tline = fgetl(fid);
end

fclose(fid);


% the disjoint component
[~,s] = graphconncomp(sparse(W));

k = 10; v = s;
for i = 1: k
    a(i) = mode(v);
    b = find(v~= a(i));
    v = v(b);
end

for i = 2:k
    t = find( s == a(i));
    bg = biograph(triu(W(t,t)),[],'Showarrows','off','NodeAutoSize','off','EdgeType','segmented','NodeAutoSize','off','LayoutType','radial');
    for j = 1:length(t)
        bg.nodes(j).Label = au{t(j)};
        bg.nodes(j).Shape = 'circle';
        bg.nodes(j).Color = [1,0,0];
    end
    view(bg);
end
    
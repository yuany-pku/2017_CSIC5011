
% load the adjacency matrix
N = load('jiashun/citation/authorCitAdjGiant.txt');

% load the authors list
fid = fopen('jiashun/citation/authorListCitGiant.txt');

tline = fgetl(fid); i = 0;
while ischar(tline)
    i = i + 1;
    aulist1{i} = tline;
    tline = fgetl(fid);
end

fclose(fid);

% D-score detect the communities
[Rl,Rr,cidx,clu1,clu2,clu3] = dscore(N, 3);


% view the communities
K1 = N(clu1,clu1);

% aut and hub
[topaut,tophub] = iterate(K1,20);
topint = intersect(topaut,tophub);
inta = setdiff(topaut,topint);
intb = setdiff(tophub,topint);
int = union(topaut,tophub);
K11 = K1(int,int);

% community 1
bgk1 = biograph(K11,[],'NodeAutoSize','off','EdgeType','straight','LayoutType','radial');

for i = 1:length(topint)
    t = find(int==topint(i));
    bgk1.nodes(t).Label = aulist1{clu1(topint(i))};
bgk1.nodes(t).Color = [1,0,0];
bgk1.nodes(t).Shape = 'house';
bgk1.nodes(t).FontSize = 30;
end
for i = 1:length(inta)
    t = find(int==inta(i));
    bgk1.nodes(t).Label = aulist1{clu1(inta(i))};
bgk1.nodes(t).Color = [0,1,0];
bgk1.nodes(t).Shape = 'circle';
bgk1.nodes(t).FontSize = 20;
end

for i = 1:length(intb)
    t = find(int==intb(i));
    bgk1.nodes(t).Label = aulist1{clu1(intb(i))};
bgk1.nodes(t).Color = [0,0,1];
bgk1.nodes(t).Shape = 'rect';
bgk1.nodes(t).FontSize = 20;
end

view(bgk1);

% community 1
K2 = N(clu2,clu2);
[topaut,tophub] = iterate(K2,20);
topint = intersect(topaut,tophub);
inta = setdiff(topaut,topint);
intb = setdiff(tophub,topint);
int = union(topaut,tophub);
K22 = K2(int,int);


bgk2 = biograph(K22,[],'NodeAutoSize','off','EdgeType','straight','LayoutType','radial');


for i = 1:length(topint)
    t = find(int==topint(i));
    bgk2.nodes(t).Label = aulist1{clu2(topint(i))};
bgk2.nodes(t).Color = [1,0,0];
bgk2.nodes(t).Shape = 'house';
bgk2.nodes(t).FontSize = 30;
end

for i = 1:length(inta)
    t = find(int==inta(i));
    bgk2.nodes(t).Label = aulist1{clu2(inta(i))};
bgk2.nodes(t).Color = [0,1,0];
bgk2.nodes(t).Shape = 'circle';
bgk2.nodes(t).FontSize = 20;
end

for i =1:length(intb)
    t = find(int==intb(i));
    bgk2.nodes(t).Label = aulist1{clu2(intb(i))};
bgk2.nodes(t).Color = [0,0,1];
bgk2.nodes(t).Shape = 'rect';
bgk2.nodes(t).FontSize = 20;

end

view(bgk2);


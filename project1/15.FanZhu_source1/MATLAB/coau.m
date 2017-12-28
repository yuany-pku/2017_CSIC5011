
% load the adjacency matrix
D =load('Jiashun/coauthorship/coauthorAdjThreshGiant.txt');

% load the authors list
fid = fopen('Jiashun/coauthorship/authorListCoauthorThreshGiant.txt');

tline = fgetl(fid); i = 0;
while ischar(tline)
    i = i + 1;
    aulist{i} = tline;
    tline = fgetl(fid);
end

fclose(fid);

% SCORE or Spectral Clustering detect the communities.
V = score(D, 3); % SCORE
[C1,C2,C3] = SpectralClustering(D, 3); % Spectral clustering

% choose to view
bg = biograph(triu(D),[],'Showarrows','off','NodeAutoSize','off','EdgeType','segmented','NodeAutoSize','off','LayoutType','radial');

for i = 1:size(D,1)
    if sum(D(i,:)) >= 8
        bg.nodes(i).Label = aulist{i};
        bg.nodes(i).FontSize = 20;
    else
        bg.nodes(i).Label = '';
    end
    if V(i) == 1
       bg.nodes(i).Shape = 'circle';
       bg.nodes(i).Color = [1 0 0];
    elseif V(i) == 2
       bg.nodes(i).Shape = 'diamond'; 
       bg.nodes(i).Color = [0 1 0];
    else 
        bg.nodes(i).Color = [1 0 1]; 
    end
end
view(bg)
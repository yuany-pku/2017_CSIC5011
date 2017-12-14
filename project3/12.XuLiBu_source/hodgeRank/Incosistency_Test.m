%Let's write a loop to calculate each annotator's inconsistency.
%We would like to know if the results might be improved if we leave out
%some 'dishonest' annotators;
rng(3317);
id_comp = csvread('C:\Fall2017-HKUST\CSIC5011-Yaoyuan\Final Project\Age\age_3column.csv');
annotator_id = unique(id_comp(:,1),'rows');
curlinconsistency = zeros(size(annotator_id,1),1);
harminconsistency = zeros(size(annotator_id,1),1);

for i=1:size(annotator_id,1),
    comp_rows = id_comp(id_comp(:,1)==i,2:3);
    [curlinconsistency(i,1),harminconsistency(i,1)] = Hodgerank_age(comp_rows);
end
id = 1:size(annotator_id,1);
id_select = id(curlinconsistency(:,1)<quantile(curlinconsistency(:,1),0.75));
compare_select = [];
for i=1:size(id_select,2),
    comp_rows = id_comp(id_comp(:,1)==id_select(i),2:3);
    compare_select = vertcat(comp_rows,compare_select);
end

[Score,TotalIncon,HarmIncon] = Hodgerank(compare_select);


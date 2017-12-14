import scTDA
import pylab
import random
random.seed(42)

pylab.rcParams["patch.force_edgecolor"] = True
pylab.rcParams['patch.facecolor'] = 'k'

t = scTDA.TopologicalRepresentation('tcgapanglioma', lens='mds', metric='euclidean')

t.save('tcgapanglioma_mds_25_0.40', 25, 0.40);

c = scTDA.UnrootedGraph('tcgapanglioma_mds_25_0.40', 'tcgapanglioma.no_subsampling.tsv', groups=False)
c.draw('GABBR1|2550', weight=20);
c.draw('EGFR|1956', weight = 20);

c.draw('timepoint',weight=30, table=False);


##rooted graph

c = scTDA.RootedGraph('tcgapanglioma_mds_25_0.40', 'tcgapanglioma.no_subsampling.tsv', posgl=False, groups = False)

c.show_statistics()

c.draw('EGFR|1956',weight=30, table=False);
c.draw('GABBR1|2550', weight=30);

c.draw('_CDR');

c.plot_CDR_correlation()

c.draw('timepoint',weight=30, table=False);

c.plot_rootlane_correlation()


#Add annotation and save the gene table
splic = []
rna = []

f = open('GO0008380_RNA_spliciing.tsv', 'r')
for nb, line in enumerate(f):
    if nb > 0:
        sp = line.split('\t')
        splic.append(sp[2])
f.close()
splic = list(set(splic))

f = open('GO0044822_polyA_RNA_binding.tsv', 'r')
for nb, line in enumerate(f):
    if nb > 0:
        sp = line.split('\t')
        rna.append(sp[2])
f.close()
rna = list(set(rna))

annotations = {'Splicing': splic, 'RNA_binding': rna}

c.save(n=1000, filtercells=0, filterexp=0.0, annotation=annotations);
 #select genes
g = c.cellular_subpopulations(1.9) #cut off is set be be 1.9

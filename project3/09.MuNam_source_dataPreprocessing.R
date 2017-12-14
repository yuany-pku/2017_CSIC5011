setwd("~/Dropbox/Kevin/FinalProject/")
pg = read.delim("GBMLGG.rnaseqv2__illuminahiseq_rnaseqv2_RSEM_genes_normalized__data.data.txt")

rownames(pg) = pg$HybridizationREF; pg = pg[,-1]
apply(pg, 1, mean) -> pgmn

pg = pg[pgmn > 1,]

apply(pg, 1, sd) -> pgsd

hist(log2(pgsd), breaks = 100)
plot(density(log2(pgsd ), bw = 0.5))

pgsdz  = scale(log2(pgsd))

pgs = pg[pgsdz > 2,]

dm = t(scale(t(pgs)))

pca = prcomp(dm)

dan = read.delim("tcgadata.txt", header = F)
names(dan) = c("SID", "NCell", "Source", "Grade")
dan$"NID" = gsub("-",".",dan$SID)
dan$NID = gsub("data/","",dan$NID)
dan$NID = gsub(".txt","",dan$NID)


df = data.frame(NID = names(pgs), pc1 = pca$rotation[,1], pc2 = pca$rotation[,2], stringsAsFactors = F)
dan = merge(dan , df)
dan$Grade = as.factor(dan$Grade)

myc = c('#d7191c','#fdae61','#ffffbf','#abdda4','#2b83ba')
jc = c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000")
library(ggplot2)
ggplot(aes(pc1, pc2, color = Grade), data = dan[rev(1:nrow(dan)),]) + 
  geom_point()+ theme_classic() +
  #scale_color_manual(values = jc[4:8]) + 
  ylim(-0.1, 0.15) + labs(x = "PC1, 23.5%", y = "PC2, 17.9%") 


stopifnot( identical(names(pg), dan$NID))

pgout = cbind(dan[,4:5], log2(t(pg) + 1))
pgout$ID = paste("D", pgout$Grade, sep = "")
pgout$ID = paste(pgout$ID, pgout$Source, 0:700, sep = "_")
pgout = pgout[,c(17685, 2,1, 3:17684)]
names(pgout)[1:3] = c("ID", "timepoint", "lib")

#Save out, will be feed to scTDA.TopologicalRepresentation
write.table(pgout, file = "tcgapanglioma.no_subsampling.tsv", sep = "\t", quote =F ,row.names = F)
write.table(log2(t(pgs) + 1), file = "tcgapanglioma.mapper.tsv", sep = "\t", quote =F ,row.names = F)



#boxplots
mybox = function(i){
  cbind(t(pgs)[,i], dan) ->mydf
  names(mydf)[1] = "mygene"
  gn = rownames(pgs)[i]
  mydf$Grade = as.integer(mydf$Grade)
  #boxplot(log2(mydf$mygene + 1) ~ mydf$Grade, xlab = "WHO Grade", ylab = paste(gn, "Expression (log2RPKM)"))
  p = ggplot(aes(as.factor(Grade),log2(mygene + 1 )), data = mydf)  + 
    geom_boxplot(aes(fill = as.factor(Grade)),show.legend = F, width = 0.5) +#geom_jitter(width = 0.3, cex = 0.5, alpha = 0.5) +
    theme_classic() + labs(x = "WHO Grade", y = paste(gn, "Expression (log2RPKM)"))
  show(p)
  #abline(lm(log2(mydf$mygene + 1) ~ mydf$Grade), lwd = 2, lty = 2, col = "tomato")
}

for (i in 1:100) mybox(i)


tmp = t(pg["CCDC140|151278",])
mydf = cbind(tmp, dan)


boxplot(mydf$mygene ~ mydf$Grade,outline=FALSE, xlab = "Grade", ylab = "Expression of CCDC140 (log2RPKM)")

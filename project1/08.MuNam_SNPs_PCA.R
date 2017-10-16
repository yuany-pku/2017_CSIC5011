setwd("~/Dropbox/Kevin/COMP/")
library(data.table)
library(vegan)
library(psych)
library(reshape2)
gnty =fread("HGDP_SNP_0129.txt")
names(gnty)[1] = "snp"

mp = read.delim("HGDP_Map.txt", stringsAsFactors = F, header =F)
names(mp) = c("snp", "chr", "pos")

#%remove SNPs on M, X, XY, Y
mp = mp[!mp$chr %in% c("M", "X", "XY", "Y"),]
gnty = gnty[gnty$snp %in% mp$snp,]

smp = read.delim("HGDP_SampleList.txt", stringsAsFactors = F, header = F)
names(smp) = "ID"

ans = read.delim("~/Dropbox/Kevin/COMP/SampleAnnotation_revised.txt", stringsAsFactors = F)

anss = ans[ans$ID %in% smp$ID,]
rownames(anss) = anss$ID
anss = anss[names(gnty)[-1],]
stopifnot(identical(gnty$snp, mp$snp))
gnty =cbind(mp, gnty[,-1])

snpst = read.delim("HGDP_SNP_stat.txt")
snpst = snpst[mp$snp,]

stopifnot(identical(gnty$snp, rownames(snpst)))

gnty = gnty[snpst$MS <1,]
mp = mp[snpst$MS <1,]
snpst = snpst[snpst$MS <1,]

#% save the data. this is the same with professor Yao provided, but has more annotation
write.table(gnty, file = "ceph_hgdp_minor_code_XNA.betterAnnotated.csv", quote =F, sep = ",", row.names = F)
write.table(anss, file = "ceph_hgdp_minor_code_XNA.sampleInformation.csv", quote =F, sep = ",", row.names = F)
#%end saving
snpst$sd = apply(snpst[,1:3], 1, function(x) sum(x==0))

gnty = gnty[snpst$sd <2,]
mp = mp[snpst$sd <2,]
snpst = snpst[snpst$sd <2,]


#just run this ones
xf = t(scale(t(gnty[,4:1046])))
stopifnot(dim(xf)[2] == 1043)

pcza = prcomp(xf, rank. = 2)
plot(pcza$rotation[,1], pcza$rotation[,2], col = as.factor(anss$region), cex = 0.5,
     xlab = paste("PC1 (", round(summary(pcza)$importance[2,1]*100,2), "%)"),
     ylab = paste("PC2 (", round(summary(pcza)$importance[2,2]*100,2), "%)"),
     pch = 21)
legend("topleft", legend =levels(as.factor(anss$region)),col = 1:7, pch = 19 , bty = "n", cex = 0.6)

plot(-pcza$rotation[,1], pcza$rotation[,3], col = as.factor(anss$region), cex = 0.5,
     xlab = paste("PC1 (", round(summary(pcza)$importance[2,1]*100,2), "%)"),
     ylab = paste("PC3 (", round(summary(pcza)$importance[2,3]*100,2), "%)"))
legend("bottomleft", legend =levels(as.factor(anss$region)),col = 1:7, pch = 19 , bty = "n", cex = 0.6)

plot(pcza$rotation[,2], pcza$rotation[,3], col = as.factor(anss$region), cex = 0.5,
     xlab = paste("PC2 (", round(summary(pcza)$importance[2,2]*100,2), "%)"),
     ylab = paste("PC3 (", round(summary(pcza)$importance[2,3]*100,2), "%)"))
legend("bottomleft", legend =levels(as.factor(anss$region)),col = 1:7, pch = 19 , bty = "n", cex = 0.6)

###--end run----
mypca = function(mat, nf, gr){
  xfs = mat[sample(1:nrow(mat), nf),]
  
  ##suggest from jbb
  xfst = t(xfs)
  xfstz = scale(xfst)
  xfstz = xfstz[,apply(xfstz, 2, function(x) sum(is.na(x))) ==0] #remove snps that are all the same to the 1043 cases.
  xfsz = t(xfstz)
  pcz = prcomp(xfsz, rank. = 2)
  plot(pcz$rotation[,1], pcz$rotation[,2], col = as.factor(gr), cex = 0.7,
       xlab = paste("PC1 (", round(summary(pcz)$importance[2,1]*100,2), "%)"),
       ylab = paste("PC2 (", round(summary(pcz)$importance[2,2]*100,2), "%)"))
  legend("topleft", legend =levels(as.factor(gr)),col = 1:7, pch = 19 , bty = "n", cex = 0.4)
  return(pcz)
}



pc2 = mypca(mat = gnty[,4:1046], nf = 5000, gr = anss$region)
pc3 = mypca(mat = gnty[,4:1046], nf = 50000, gr = anss$region)
pc4 = mypca(mat = gnty[,4:1046], nf = 1043, gr = anss$region)

prrcstat = function(mat1, mat2){ ##KEY function. it defines how similar two PCA results are.
  X = mat1
  Y= mat2
  
  xc = scale(X, scale = F, center = T) #centered but not scaled
  yc = scale(Y, scale = F, center = T) #centered but not scaled
  
  C = t(yc) %*% xc
  
  s = svd(C)
  
  D = 1-tr(diag(s$d))^2 /(tr(t(xc) %*%xc) * tr(t(yc) %*% yc))
  #X = UDV', that is, C =  s$u %*% diag(s$d) %*% t(s$v), in other words,
  ## U = s$u, D = diag(s$d), V = t(u$v)
  
  #suppose y = f(x) = pA'x + b, then
  #A = VU' = s$v %*% t(s$u)
  #p = tr(d)/tr(t(xc) %*% xc)
  #b = y0 - p %*% t(A) %*% x0
  
  #most important
  #D is the distance (dissimilarity) of two set of coordinates: 
  #D(X, Y) =1-tr(diag(s$d))^2 /(tr(t(xc) %*%xc) * tr(t(yc) %*% yc))
  #1-D is the similarity
  #1-D = tr(diag(s$d))^2 /(tr(t(xc) %*%xc) * tr(t(yc) %*% yc))
  return(D)
}

mypro = function(x, y){
  prpca = procrustes( x$rotation[,1:2],y$rotation[,1:2])
  plot(prpca)
  points(prpca$X, pch=21,  cex = 0.7,col = as.factor(anss$region))
  legend("bottomleft", legend =levels(as.factor(anss$region)),col = 1:7, pch = 19 , bty = "n", cex = 0.6)
  
  print(summary(prpca))
  smlt = 1-prrcstat(prpca$Yrot, prpca$X)
  print(paste("Similarity:",smlt))
  protest( x$rotation[,1:2],y$rotation[,1:2], permutations = 10000)
  return(smlt)
}
mypro(pc4, pcza)

wps = data.frame(nsnp = c(5,10,25,50, 100,250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000),
                 similarity1 = 0, similarity2 = 0,similarity3 = 0,similarity4 = 0,similarity5 = 0)
for (i in 1:nrow(wps)){
  tmppc = mypca(mat = gnty[,4:1046], nf = wps$nsnp[i], gr = anss$region)
  tmppr = mypro(tmppc, pcza)
  wps$similarity1[i] = tmppr
}
for (i in 1:nrow(wps)){
  tmppc = mypca(mat = gnty[,4:1046], nf = wps$nsnp[i], gr = anss$region)
  tmppr = mypro(tmppc, pcza)
  wps$similarity2[i] = tmppr
}
for (i in 1:nrow(wps)){
  tmppc = mypca(mat = gnty[,4:1046], nf = wps$nsnp[i], gr = anss$region)
  tmppr = mypro(tmppc, pcza)
  wps$similarity3[i] = tmppr
}
for (i in 1:nrow(wps)){
  tmppc = mypca(mat = gnty[,4:1046], nf = wps$nsnp[i], gr = anss$region)
  tmppr = mypro(tmppc, pcza)
  wps$similarity4[i] = tmppr
}
for (i in 1:nrow(wps)){
  tmppc = mypca(mat = gnty[,4:1046], nf = wps$nsnp[i], gr = anss$region)
  tmppr = mypro(tmppc, pcza)
  wps$similarity5[i] = tmppr
}

wps = rbind(wps, c(488890,1,1,1,1,1)) #total: 488890

plot(log10(wps$nsnp), wps$similarity, type = "o", pch = 20,
     xlab = "Number of SNPs (log10)", ylab = "Similarity")
segments(log10(1043),0,log10(1043),wps$similarity[wps$nsnp ==1000], lty = 2, col = "red")

library(reshape2)
melt(wps, id.vars = "nsnp") ->wpsm
boxplot(value ~ nsnp , data = wpsm, xlab = "Number of SNPs", ylab = "Similarity")


#use sorted SNPs
snpanova = read.csv("171003_ANOVA_7region.csv", stringsAsFactors = F)[,1:3]
snpanova = snpanova[order(snpanova$F_statistic, decreasing = T),]
snpanova = snpanova[snpanova$SNP %in% gnty$snp,]

wpsanova = data.frame(nsnp = c(5,10,25,50, 100,250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000),similarity = 0)
for (i in 1:nrow(wpsanova)){
  ns = wpsanova$nsnp[i]
  tmppc = mypca(mat = gnty[gnty$snp %in% snpanova$SNP[1:ns],4:1046], nf = ns, gr = anss$region)
  tmppr = mypro(tmppc, pcza)
  wpsanova$similarity[i] = tmppr
}

plot(1:15, wpsanova$similarity, type = "o", pch = 20, ylim = c(0,1), 
     xaxt = "n", xlab = "Number of SNPs", ylab = "Similarity")
axis(side = 1, at = 1:15, labels = wpsanova$nsnp)


#Application to real samples

hg19map = read.delim("HGDP_Map.hg19.bed", stringsAsFactors = F, header =F)
names(hg19map) = c("chr", "start", "end", "snp")
hg19map$id = paste(hg19map$chr, hg19map$start, sep = ":")

pcapred = function(genotypeFile){
  p1346 = read.delim(genotypeFile, stringsAsFactors = F, header =F)
  names(p1346) = c("chr", "start", "base", "depth", "P1346")
  p1346$id = paste(p1346$chr, p1346$start, sep = ":")
  
  p1346m = merge(hg19map[,4:5], p1346, by = "id")
  
  subgnty = gnty[gnty$snp %in% p1346m$snp,]
  subgnty = merge(subgnty, p1346m[,c(2,7)])
  
  pca4pred = mypca(subgnty[,4:1047], nrow(subgnty), gr = as.factor(c(anss$region, "T")))
  points(x = pca4pred$rotation[1044,1], y=pca4pred$rotation[1044,2], pch = 21, col = "black", cex = 1.5, bg='orange')
  
}
pcapred(genotypeFile = "BTS1346.SNP.genotype.txt")
pcapred(genotypeFile = "BTS1370.SNP.genotype.txt")
pcapred(genotypeFile = "BTS1475.SNP.genotype.txt")

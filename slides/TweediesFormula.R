library(splines)
n0 = 100000
n_neg = 5000
n_pos = 5000
N = n0+n_neg+n_pos
z = c(rnorm(n0), rnorm(n_pos,2,1),rnorm(n_neg,-2,1)) 
bins = seq(min(z)-.1,max(z)+.1, len = 100)
h = hist(z, bins, plot = F)
x = h$m
g  = glm(h$c~ns(x,df=7),family = poisson)
ss = splinefun(x,log(dnorm(x)/g$fit),method = "natural")
mu_T = -ss(z,deriv=1)
#ss = splinefun(x,g$fit),method = "natural")
#mu_T = z+ss(z,deriv=1)
mu_JS = (1-(N-2)/(t(z)%*%z))*z
#generate figure
#png(filename='z_mu.png',res=600, width = 20, height = 10, units="cm")
par(mfrow=c(1,2))
plot(z,mu_JS,pch=20)
abline(0,1,col='red')
title('James-stein Estimator')
plot(z,mu_T,pch=20)
title('Tweedie formula')
abline(0,1,col='red')
#dev.off()


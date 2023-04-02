##### Serie Temporal #######
GMD_tmp <- read.table("GMD_tmp.csv", header=TRUE, sep=";", dec=".")
# Datos desde 2013-04 hasta 202206 de GMD
serie_GMD = ts(GMD_tmp, start = c(2013, 4), frequency = 12)
print(serie_GMD)

par(mfrow=c(1,1))
boxplot(serie_GMD, ylab = "", xlab = "GMD", col = "lightblue", horizontal = TRUE)
boxplot(serie_GMD~cycle(serie_GMD), xlab = "Meses del Año", ylab = "GMD", col = "lightblue")
boxplot(serie_GMD~round(time(serie_GMD),0), xlab = "Años", ylab = "GMD", col = "lightblue")

plot(serie_GMD, xlab = "Años", main = "Evolución de GMD", col = "blue", lwd=2)

serie_GMD_descc <- decompose(serie_GMD)
plot(serie_GMD_descc, xlab = "Año")

par(mfrow=c(2,1))
plot(serie_GMD, xlab = "Año", main = "Serie Original")
plot(log(serie_GMD), xlab = "Año", main = "Serie tras logaritmo", col = "blue")

dif1.serie_GMD = diff(serie_GMD)
par(mfrow=c(1,1))
plot(dif1.serie_GMD, xlab = "Año", main ="Serie GMD tras diferenciación con valor previo", col = "blue")

tsstationary_GMD = diff(dif1.serie_GMD, lag=12)
plot(tsstationary_GMD, xlab = "Año", main ="Serie GMD tras diferenciación con valor previo", col = "blue")

par(mfrow=c(2,1))
acf(tsstationary_GMD, lag.max=34)
pacf(tsstationary_GMD, lag.max=34)


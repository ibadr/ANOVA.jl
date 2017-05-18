using RDatasets
using GLM
using ANOVA

iris = dataset("datasets","iris")
mdl = lm(@formula(SepalLength ~ Species), iris)
anova(mdl)

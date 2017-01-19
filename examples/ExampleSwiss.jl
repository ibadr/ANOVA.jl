using RDatasets
using GLM
using ANOVA

swiss = dataset("datasets","swiss")
mdl = lm(Fertility ~ Agriculture + Education + Examination, swiss)
anova(mdl)

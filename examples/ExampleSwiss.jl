using RDatasets
using GLM
using ANOVA

swiss = dataset("datasets","swiss")
mdl = lm(@formula(Fertility ~ Agriculture + Education + Examination), swiss)
anova(mdl)

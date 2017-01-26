using GLM
using StatsBase: RegressionModel

# Similar to typealiases in GLM
typealias BlasReal Union{Float32,Float64}
typealias FP AbstractFloat
typealias FPVector{T<:FP} DenseArray{T,1}

# Initial code imported from pull request https://github.com/JuliaStats/GLM.jl/pull/70/
# See also https://github.com/JuliaStats/GLM.jl/pull/65

effects(mod::RegressionModel) = effects(mod.model)
effects{T<:BlasReal,V<:FPVector}(
  mod::LinearModel{LmResp{V},DensePredQR{T}}) =
    (mod.pp.qr[:Q]'*mod.rr.y)[1:size(mod.pp.X,2)]

type AnovaTable{T<:BlasReal}
  Df::Vector{T}
  SS::Vector{T}
  MS::Vector{T}
  F::Vector{T}
  PrF::Vector{T}
  model::RegressionModel
end

anova(mod::RegressionModel) = anova(mod.model)
function anova{T<:BlasReal,V<:FPVector}(mod::LinearModel{LmResp{V},DensePredQR{T}})
  eff = effects(mod)
  hasIntercept = isapprox(mean(mod.pp.X[:,1])-1.0,0.0)
  if hasIntercept
    k = 1
  else
    k = 0
  end
  n = size(mod.pp.X,2) + 1 - k
  DF = zeros(T,n)
  SS = zeros(T,n)
  MS = zeros(T,n)
  fstat = zeros(T,n-1)
  pval = zeros(T,n-1)
  DF[end] = dof_residual(mod)
  SS[end] = deviance(mod.rr)
  MS[end] = SS[end]/DF[end]

  @inbounds for i in 1:n-1
    DF[i]=1
    SS[i]=sumabs2(eff[i+k])
    MS[i]=SS[i]/DF[i]
    fstat[i]=MS[i]/MS[end]
    pval[i]=ccdf(FDist(DF[i], DF[end]), fstat[i])
  end

  return AnovaTable(
    DF,SS,MS,fstat,pval,mod
  )
end

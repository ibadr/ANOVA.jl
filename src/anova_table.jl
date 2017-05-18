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

function effects{T<:BlasReal,V<:FPVector,C<:Any}(
  mod::LinearModel{LmResp{V},DensePredChol{T,C}})
  # TODO this is hackish, and recomputes Q1 with every call
  R = cholfact!(mod.pp)[:U]
  Q1 = mod.pp.X / R
  return Q1' * mod.rr.y
end

type AnovaTable{T<:BlasReal}
  termnames::Vector{Any}
  Df::Vector{T}
  SS::Vector{T}
  MS::Vector{T}
  F::Vector{T}
  PrF::Vector{T}
end

function anova(mod::RegressionModel)
  eff = effects(mod)
  with_intercept = all(mod.model.pp.X[:,1] .== 1.0)
  T = eltype(mod.model.pp.X)
  termnames = mod.mf.terms.terms
  assign = mod.mm.assign
  df_residual = dof_residual(mod)
  devnc = deviance(mod.model.rr)
  return _anova(eff,with_intercept,T,termnames,assign,df_residual,devnc)
end
function anova{M<:LinearModel}(mod::M)
  eff = effects(mod)
  with_intercept = all(mod.pp.X[:,1] .== 1.0)
  T = eltype(mod.pp.X)
  if with_intercept
    termnames = ["x$i" for i in 1:size(mod.pp.X,2)-1]
  else
    termnames = ["x$i" for i in 1:size(mod.pp.X,2)]
  end
  assign = 0:1:size(mod.pp.X,2)-1
  df_residual = dof_residual(mod)
  devnc = deviance(mod.rr)
  return _anova(eff,with_intercept,T,termnames,assign,df_residual,devnc)
end

function _anova(eff,with_intercept,T,termnames,assign,df_residual,devnc)
  k = with_intercept ? 1 : 0
  unq_assign = unique(assign)
  n = length(unq_assign) + 1 - k
  DF = zeros(T,n)
  SS = zeros(T,n)
  MS = zeros(T,n)
  fstat = zeros(T,n-1)
  pval = zeros(T,n-1)
  DF[end] = df_residual
  SS[end] = devnc
  MS[end] = SS[end]/DF[end]

  @inbounds for i in 1:n-1
    v = unq_assign[i+k]
    mask = assign .== v
    DF[i] = sum(mask)
    SS[i] = sumabs2(eff[mask])
    MS[i] = SS[i]/DF[i]
    fstat[i] = MS[i]/MS[end]
    pval[i] = ccdf(FDist(DF[i], DF[end]), fstat[i])
  end
  return AnovaTable(termnames,DF,SS,MS,fstat,pval)
end

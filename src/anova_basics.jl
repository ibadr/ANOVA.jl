using GLM

# Initial code imported from pull request https://github.com/JuliaStats/GLM.jl/pull/70/
# See also https://github.com/JuliaStats/GLM.jl/pull/65

effects(mod::RegressionModel)=(mod.model.pp.qr[:Q]'*mod.model.rr.y)[1:size(mod.model.pp.X,2)]
effects(mod::LmMod{DensePredQR{Float64}})=(mod.pp.qr[:Q]'*mod.rr.y)[1:size(mod.pp.X,2)]

type ANOVAtest
  SSH::Float64
  SSE::Float64
  MSH::Float64
  MSE::Float64
  dfH::Int
  dfE::Int
  fstat::Float64
  pval::Float64    #regular pvalue or -log10pval
  log10pval::Bool
end

#this is a test for a group of terms from a LmMod derived from a formula and dataframe
function ANOVAtest(mod::RegressionModel,terms::Array{Int,1}; log10pval=false)
  #terms is arrary of number for each term in the model to be tested together, starting with the intercept at 0
  eff=effects(mod)  #get effect for each coefficient
  ind=findin(mod.mm.assign,terms)
  eff=eff[ind]
  SSH=sum(Abs2Fun(),eff)
  dfH=length(eff)
  MSH=SSH/dfH
  SSE=deviance(mod.model.rr)
  dfE=df_residual(mod.model.pp)
  MSE=SSE/dfE
  fstat=MSH/MSE
  pval=ccdf(FDist(dfH, dfE), fstat)
  if log10pval pval= -log10(pval) end
  return ANOVAtest(SSH,SSE,MSH,MSE,dfH,dfE,fstat,pval,log10pval)
end

#this is a test for a single term from a LmMod derived from a formula and dataframe
function ANOVAtest(mod::RegressionModel,term::Int; log10pval=false)
  #term an integer for a single term in model to be tested, starting with the intercept at 0
  eff=effects(mod)  #get effect for each coefficient
  ind=findin(mod.mm.assign,term)
  eff=eff[ind]
  SSH=eff[1]*eff[1]
  dfH=length(eff)
  MSH=SSH/dfH
  SSE=deviance(mod.model.rr)
  dfE=df_residual(mod.model.pp)
  MSE=SSE/dfE
  fstat=MSH/MSE
  pval=ccdf(FDist(dfH, dfE), fstat)
  if log10pval pval= -log10(pval) end
  return ANOVAtest(SSH,SSE,MSH,MSE,dfH,dfE,fstat,pval,log10pval)
end


#this is a test for a group of coefficients/columns of X based on a LmMod derived without a formula and dataframe
function ANOVAtest(mod::LmMod{DensePredQR{Float64}},cols::Array{Int,1}; log10pval=false)
  #cols a vector of position numbers (column number of X) to be grouped together with the intercept starting at 1
  #this does not refer the terms of a model defined by a formula if a term has >1 DF
  eff=effects(mod)  #get effect for each coefficient
  eff=eff[cols]
  SSH=sum(Abs2Fun(),eff)
  dfH=length(eff)
  MSH=SSH/dfH
  SSE=deviance(mod.rr)
  dfE=df_residual(mod.pp)
  MSE=SSE/dfE
  fstat=MSH/MSE
  pval=ccdf(FDist(dfH, dfE), fstat)
  if log10pval pval= -log10(pval) end
  return ANOVAtest(SSH,SSE,MSH,MSE,dfH,dfE,fstat,pval,log10pval)
end

#this is a test for a single coefficient/columns of X based on a LmMod derived without a formula and dataframe
function ANOVAtest(mod::LmMod{DensePredQR{Float64}},col::Int; log10pval=false)
  #col means the position number of the column of X with the intercept starting at 1
  #this does not refer the terms of a model defined by a formula if a term has >1 DF
  eff=effects(mod)  #get effect for each coefficient
  eff=eff[col]
  SSH=eff[1]*eff[1]
  dfH=length(eff)
  MSH=SSH/dfH
  SSE=deviance(mod.rr)
  dfE=df_residual(mod.pp)
  MSE=SSE/dfE
  fstat=MSH/MSE
  pval=ccdf(FDist(dfH, dfE), fstat)
  if log10pval pval= -log10(pval) end
  return ANOVAtest(SSH,SSE,MSH,MSE,dfH,dfE,fstat,pval,log10pval)
end

function Base.show(io::IO,at::ANOVAtest)
  if at.log10pval
      println("              ","DF",'\t',"SS",'\t',"MS",'\t',"F",'\t',"log10pval")
  else
      println("              ","DF",'\t',"SS",'\t',"MS",'\t',"F",'\t',"pval")
  end
  println("Hypothesis    ",round(at.dfH,3),'\t',round(at.SSH,3),'\t',round(at.MSH,3),'\t',round(at.fstat,3),'\t',at.pval)
  println("Residuals     ",round(at.dfE,3),'\t', round(at.SSE,3),'\t', round(at.MSE,3))
end

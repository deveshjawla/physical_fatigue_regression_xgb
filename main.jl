# using Distributed
using XGBoost
nsteps = 1000
input_size = 7
# addprocs(num_chains; exeflags=`--project`)
experiment_name = "supervised_learning"
using DelimitedFiles, DataFrames, Statistics
include("DataUtils.jl")
pool_x = readdlm("pool_x.csv", ',', Float32)
pool_y = readdlm("pool_y.csv", ',', Float32)
test_x = readdlm("test_x.csv", ',', Float32)
test_y = readdlm("test_y.csv", ',', Float32)

x,y = (Array{Float64}(permutedims(pool_x)), vec(pool_y))

bst = xgboost((x, y); num_round=nsteps, max_depth=6, XGBoost.regression()...)
mkpath("./$(experiment_name)/convergence_statistics")

ŷ_test = predict(bst, permutedims(test_x))

writedlm("./$(experiment_name)/ŷ_test.csv", ŷ_test, ',')

test_y=vec(test_y)
mse_ = mse(test_y, ŷ_test)
rmse_ = rmse(test_y, ŷ_test)
r2_ = r2_score(test_y, ŷ_test)
r2_adjusted_ = adjusted_r2_score(test_y, ŷ_test, input_size)
writedlm("./$(experiment_name)/classification_performance.csv", [["RMSE", "MSE", "R2", "R2_adjusted"] [rmse_, mse_, r2_, r2_adjusted_]], ',')

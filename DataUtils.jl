using Random

function data_balancing(data_xy::DataFrame; balancing::String, positive_class_label=1, negative_class_label=2)::DataFrame
    negative_class = data_xy[data_xy[:, end].==negative_class_label, :]
    positive_class = data_xy[data_xy[:, end].==positive_class_label, :]
    size_positive_class = size(positive_class)[1]
    size_normal = size(negative_class)[1]
    multiplier = div(size_normal, size_positive_class)
    leftover = mod(size_normal, size_positive_class)
    if balancing == "undersampling"
        data_xy = vcat(negative_class[1:size(positive_class)[1], :], positive_class)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    elseif balancing == "generative"
        new_positive_class = vcat(repeat(positive_class, outer=multiplier - 1), positive_class[1:leftover, :], positive_class)
        data_x = select(new_positive_class, Not([:label]))
        data_y = select(new_positive_class, [:label])
        new_positive_class = mapcols(x -> x + x * rand(collect(-0.05:0.01:0.05)), data_x)
        new_positive_class = hcat(data_x, data_y)
        data_xy = vcat(negative_class, new_positive_class)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    elseif balancing == "none"
        nothing
    end
    # data_x = Matrix(data_xy)[:, 1:end-1]
    # data_y = data_xy.target
    return data_xy
end

# A handy helper function to normalize our dataset.
function standardize(x, mean_, std_)
    return (x .- mean_) ./ (std_ .+ convert(eltype(std_), 0.000001))
end

# A handy helper function to normalize our dataset.
function scaling(x, max_, min_)
    return (x .- min_) ./ (max_ - min_)
end

function pool_test_maker(pool::DataFrame, test::DataFrame, n_input::Int)::Tuple{Tuple{Array{Float32, 2}, Array{Int, 2}}, Tuple{Array{Float32, 2}, Array{Int, 2}}}
    pool = Matrix{Float32}(permutedims(pool))
    test = Matrix{Float32}(permutedims(test))
    pool_x = pool[1:n_input, :]
    pool_y = pool[end, :]
    # pool_max = maximum(pool_x, dims=1)
    # pool_mini = minimum(pool_x, dims=1)
    # pool_x = scaling(pool_x, pool_max, pool_mini)
    pool_mean = mean(pool_x, dims=2)
    pool_std = std(pool_x, dims=2)
    pool_x = standardize(pool_x, pool_mean, pool_std)

    test_x = test[1:n_input, :]
    test_y = test[end, :]
    # test_x = scaling(test_x, pool_max, pool_mini)
    test_x = standardize(test_x, pool_mean, pool_std)


    pool_y = permutedims(pool_y)
    test_y = permutedims(test_y)
    pool = (pool_x, pool_y)
    test = (test_x, test_y)
    return pool, test
end

function pool_test_maker_xgb(pool::DataFrame, test::DataFrame, n_input::Int)::Tuple{Tuple{Array{Float32, 2}, Array{Int, 2}}, Tuple{Array{Float32, 2}, Array{Int, 2}}}
    pool = Matrix{Float32}(permutedims(pool))
    test = Matrix{Float32}(permutedims(test))
    pool_x = pool[1:n_input, :]
    pool_y = pool[end, :]
	pool_y = Int.(pool_y) .- 1
    # pool_max = maximum(pool_x, dims=1)
    # pool_mini = minimum(pool_x, dims=1)
    # pool_x = scaling(pool_x, pool_max, pool_mini)
    pool_mean = mean(pool_x, dims=2)
    pool_std = std(pool_x, dims=2)
    pool_x = standardize(pool_x, pool_mean, pool_std)

    test_x = test[1:n_input, :]
    test_y = test[end, :]
	test_y = Int.(test_y) .- 1
    # test_x = scaling(test_x, pool_max, pool_mini)
    test_x = standardize(test_x, pool_mean, pool_std)


    pool_y = permutedims(pool_y)
    test_y = permutedims(test_y)
    pool = (pool_x, pool_y)
    test = (test_x, test_y)
    return pool, test
end


# Function to split samples.
function split_data(df; at=0.70)
    r = size(df, 1)
	df=df[shuffle(axes(df, 1)), :]
    index = Int(round(r * at))
    pool = df[1:index, :]
    test = df[(index+1):end, :]
    return pool, test
end

using EvalMetrics
function performance_stats(ground_truth_, predictions_)
	ground_truth = deepcopy(Int.(vec(ground_truth_)))
    predictions = deepcopy(Int.(vec(predictions_)))
    ground_truth[ground_truth.==2] .= 0
    predictions[predictions.==2] .= 0
	cm = ConfusionMatrix(ground_truth, predictions)
    f1 = f1_score(cm)
    mcc = matthews_correlation_coefficient(cm)
    acc = accuracy(cm)
    fpr = false_positive_rate(cm)
    # fnr = fnr(cm)
    # tpr = tpr(cm)
    # tnr = tnr(cm)
    prec = precision(cm)
    recall = true_positive_rate(cm)
	threat_score = EvalMetrics.threat_score(cm)
    return acc, mcc, f1, fpr, prec, recall, threat_score, cm
end

function accuracy_multiclass(true_labels, predictions)
    return mean(true_labels .== predictions)
end


# function balanced_accuracy_score(y_pred, y_true)
# 	C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         per_class = np.diag(C) / C.sum(axis=1)
#     if np.any(np.isnan(per_class)):
#         warnings.warn("y_pred contains classes not in y_true")
#         per_class = per_class[~np.isnan(per_class)]
#     score = np.mean(per_class)
#     if adjusted:
#         n_classes = len(per_class)
#         chance = 1 / n_classes
#         score -= chance
#         score /= 1 - chance
#     return score
# end

"""
    mse(y_pred, y_true)

Mean Squared Error. Calculated as `sum((y_true .- y_pred).^2) / length(y_true)` based on provided `y_pred` and `y_true`.
"""
function mse(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    return sum((y_true .- y_pred).^2) / length(y_true)
end

"""
    mse(y_pred, y_true)

Mean Squared Error. Calculated as `sum((y_true .- y_pred).^2) / length(y_true)` based on provided `y_pred` and `y_true`.
"""
function rmse(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    return sqrt(sum((y_true .- y_pred).^2) / length(y_true))
end


"""
    r2_score(y_pred, y_true)

Calculates the r2 (Coefficient of Determination) score for the provided `y_pred` and `y_true`.
Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
A constant model that always predicts the expected value of y, disregarding the input features, would get a r2_score of `0.0`.
"""
function r2_score(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    mean = sum(y_true) / length(y_true)
    ss_total = sum((y_true .- mean).^2)
    return 1 - ss_res/(ss_total + eps(eltype(y_pred)))
end

"""
    adjusted_r2_score(y_pred, y_true, n)

Modified version of `r2_score` that has been adjusted for the number of predictors in the model. Here the argument `n` is for the number of predictors(or independent variables in X). 

See also: [`r2_score`](@ref)
"""
function adjusted_r2_score(y_pred, y_true, n)		# n -> number of predictors(independent variables in X)
    @assert length(y_true) == length(y_pred)
    score = r2_score(y_pred, y_true)
    epsilon = eps(eltype(y_pred)(1.0))
    return 1 - ((1 - score) * (length(y_true) -1)) / abs(( length(y_true) - n - 1 + epsilon))
end
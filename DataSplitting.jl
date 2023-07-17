using DataFrames, DelimitedFiles, CSV, Statistics

PATH = @__DIR__
cd(PATH)

df = CSV.read("fatigue_data.csv", DataFrame, header=1)

rename!(df, :fatigue => :label)


# df = select(df, Not(:temperature_celsius))
rename!(df, "Gender (0=F 1=M)" => :gender)
include("DataUtils.jl")

function pool_test_maker(df::DataFrame, n_input::Int)::Tuple{Tuple{Array{Float32, 2}, Array{Float32, 2}}, Tuple{Array{Float32, 2}, Array{Float32, 2}}}
	# df = select(df, [:eda_scl_usiemens,:pulse_rate_bpm, :Age, :label])
	pool, test = split_data(df)
    pool = Matrix{Float32}(permutedims(pool))
    test = Matrix{Float32}(permutedims(test))
    pool_x = pool[1:n_input, :]
    pool_y = pool[end, :]

	pool_max = maximum(pool_y)
    pool_min = minimum(pool_y)
    pool_y = scaling(pool_y, 20, 6)
	
	
	
    # pool_mean = mean(pool_x, dims=2)
    # pool_std = std(pool_x, dims=2)
    # pool_x = standardize(pool_x, pool_mean, pool_std)
	# pool_x = vcat(pool_x, permutedims(pool[n_input, :]))
	
    test_x = test[1:n_input, :]
    test_y = test[end, :]
    test_y = scaling(test_y, 20, 6)
	
    # test_x = standardize(test_x, pool_mean, pool_std)
	# test_x = vcat(test_x, permutedims(test[n_input, :]))


    pool_y = permutedims(pool_y)
    test_y = permutedims(test_y)
	println(size(test_x), size(pool_x))
    pool = (pool_x, pool_y)
    test = (test_x, test_y)
    return pool, test
end

pool, test = pool_test_maker(df, 7)
writedlm("pool_x.csv", pool[1], ',')
writedlm("pool_y.csv", pool[2], ',')
writedlm("test_x.csv", test[1], ',')
writedlm("test_y.csv", test[2], ',')

using Random
function train_validate_test(df; v=0.6, t=0.8)
	    r = size(df, 1)
	    val_index = Int(round(r * v))
	    test_index = Int(round(r * t))
		df=df[shuffle(axes(df, 1)), :]
	    train = df[1:val_index, :]
	    validate = df[(val_index+1):test_index, :]
	    test = df[(test_index+1):end, :]
	    return train, validate, test
	end

	train,test,validate=train_validate_test(df)
	train = data_balancing(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
	test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)
	validate = data_balancing(validate, balancing="undersampling", positive_class_label=1, negative_class_label=2)
	

	CSV.write("./train.csv", train)
	CSV.write("./test.csv", test)
	CSV.write("./validate.csv", validate)
	
# Install required packages
using Pkg;
Pkg.add(["DataFrames", "DataFramesMeta", "CSV", "Statistics", "MLJ", "Flux"]);
# Use required packages
using DataFrames, DataFramesMeta, CSV, Statistics, MLJ, Flux, Flux.Zygote;
# Call predefined functions
include("predefined_functions_ML.jl");
# Import all the datasets and combined them into a single ML dataframe
fire_df = DataFrame();
for (yr, peatflg) in zip(2008:1:2015, ["_peatland", "_non_peatland"]
    df1 = DataFrame(CSV.File("../data/data_" * string(yr) * peatflg * ".csv")
    append!(fire_df, df1)
end;
show(first(fire_df, 6), allcols = true)
yr_biweek = unique(fire_df[:, [:year, :biweek]]);
grid_yr_biweek = unique(fire_df[:, [:id, :year, :biweek]]);

# Grid level Peatland
# With Ecosys outputs of WTD and soil moisture as features i.e. ANN (WTHR+HYDROL)
# Artificial neural network grid level
grid_list = unique(fire_df[fire_df[:, :min_peat_depth] .> 0, :id]);
model_label = "Artificial Neural Network Regression";
for (r1, r2) in zip(1:100:1101, 100:100:1200)
    model_perform_df_all_mods = DataFrame();
    test_train_predict_df_all_mods = DataFrame();
    model_perform_df_f = DataFrame();
    test_train_predict_df_f = DataFrame();
    peat_desc = "Peatland";
    ecosys_desc = "With Ecosys";
    for grid_id in grid_list[r1 : r2]
        x = fire_df[fire_df[:, :id] .== grid_id, [4, 5, 6, 7, 8, 10, 17, 20]];
        y = convert(Array{Float64, 1}, fire_df[fire_df[:, :id] .== grid_id, 23]);
        N = size(x)[2];
        nnet_mod = Flux.Chain(Flux.Dense(N, N * 4, relu), Flux.Dense(N * 4, N * 2, relu), Flux.Dense(N * 2, 1));
        for yr in 2008:2015
            getindices = fire_df[fire_df[:, :id] .== grid_id, :year];
            train = findall(q -> q .!= yr, getindices);
            test = findall(q -> q .== yr, getindices);
            model_perform_df_f, test_train_predict_df_f = flux_model_train(nnet_mod, x, y, train, test, 100, 200, length(train));
            model_perform_df_f[:, :grid_id] .= grid_id;
            model_perform_df_f[:, :test_year] .= yr;
            model_perform_df_f[:, :peat_desc] .= peat_desc;
            model_perform_df_f[:, :ecosys_desc] .= ecosys_desc;
            append!(model_perform_df_all_mods, model_perform_df_f);
            sort!(test_train_predict_df_f, :idx);
            test_train_predict_df_f = hcat(grid_yr_biweek[(grid_yr_biweek[:, :year] .== yr) .& (grid_yr_biweek[:, :id] .== grid_id), :], test_train_predict_df_f[:, 3:5]);
            test_train_predict_df_f[:, :test_year] .= yr;
            test_train_predict_df_f[:, :peat_desc] .= peat_desc;
            test_train_predict_df_f[:, :ecosys_desc] .= ecosys_desc;
            append!(test_train_predict_df_all_mods, test_train_predict_df_f);
        end
    end
    model_perform_df_all_mods[:, :model_name] .= model_label;
    test_train_predict_df_all_mods[:, :model_name] .= model_label;
    CSV.write("../outputs/" * model_label * "_model_perform_df_all_mods_" * string(r1)
    * "_"  * string(r2) * "_"  * ecosys_desc * "_"  * peat_desc * ".csv", model_perform_df_all_mods);
    CSV.write("../outputs/" * model_label * "_test_train_predict_df_all_mods_" * string(r1)
    * "_"  * string(r2) * "_"  * ecosys_desc * "_"  * peat_desc * ".csv", test_train_predict_df_all_mods);
end
for (r1, r2) in zip(1201, 1256)
    model_perform_df_all_mods = DataFrame();
    test_train_predict_df_all_mods = DataFrame();
    model_perform_df_f = DataFrame();
    test_train_predict_df_f = DataFrame();
    peat_desc = "Peatland";
    ecosys_desc = "With Ecosys";
    for grid_id in grid_list[r1 : r2]
        x = fire_df[fire_df[:, :id] .== grid_id, [4, 5, 6, 7, 8, 10, 17, 20]];
        y = convert(Array{Float64, 1}, fire_df[fire_df[:, :id] .== grid_id, 23]);
        N = size(x)[2];
        nnet_mod = Flux.Chain(Flux.Dense(N, N * 4, relu), Flux.Dense(N * 4, N * 2, relu), Flux.Dense(N * 2, 1));
        for yr in 2008:2015
            getindices = fire_df[fire_df[:, :id] .== grid_id, :year];
            train = findall(q -> q .!= yr, getindices);
            test = findall(q -> q .== yr, getindices);
            model_perform_df_f, test_train_predict_df_f = flux_model_train(nnet_mod, x, y, train, test, 100, 200, length(train));
            model_perform_df_f[:, :grid_id] .= grid_id;
            model_perform_df_f[:, :test_year] .= yr;
            model_perform_df_f[:, :peat_desc] .= peat_desc;
            model_perform_df_f[:, :ecosys_desc] .= ecosys_desc;
            append!(model_perform_df_all_mods, model_perform_df_f);
            sort!(test_train_predict_df_f, :idx);
            test_train_predict_df_f = hcat(grid_yr_biweek[(grid_yr_biweek[:, :year] .== yr) .& (grid_yr_biweek[:, :id] .== grid_id), :], test_train_predict_df_f[:, 3:5]);
            test_train_predict_df_f[:, :test_year] .= yr;
            test_train_predict_df_f[:, :peat_desc] .= peat_desc;
            test_train_predict_df_f[:, :ecosys_desc] .= ecosys_desc;
            append!(test_train_predict_df_all_mods, test_train_predict_df_f);
        end
    end
    model_perform_df_all_mods[:, :model_name] .= model_label;
    test_train_predict_df_all_mods[:, :model_name] .= model_label;
    CSV.write("../outputs/" * model_label * "_model_perform_df_all_mods_" * string(r1)
    * "_"  * string(r2) * "_"  * ecosys_desc * "_"  * peat_desc * ".csv", model_perform_df_all_mods);
    CSV.write("../outputs/" * model_label * "_test_train_predict_df_all_mods_" * string(r1)
    * "_"  * string(r2) * "_"  * ecosys_desc * "_"  * peat_desc * ".csv", test_train_predict_df_all_mods);
end

# Without Ecosys outputs of WTD and soil moisture as features i.e. ANN (WTHR)
# Artificial neural network grid level
grid_list = unique(fire_df[fire_df[:, :min_peat_depth] .> 0, :id]);
model_label = "Artificial Neural Network Regression";
for (r1, r2) in zip(1:100:1101, 100:100:1200)
    model_perform_df_all_mods = DataFrame();
    test_train_predict_df_all_mods = DataFrame();
    model_perform_df_f = DataFrame();
    test_train_predict_df_f = DataFrame();
    peat_desc = "Peatland";
    ecosys_desc = "Without Ecosys";
    for grid_id in grid_list[r1 : r2]
        x = fire_df[fire_df[:, :id] .== grid_id, [4, 5, 6, 7, 8, 10]];
        y = convert(Array{Float64, 1}, fire_df[fire_df[:, :id] .== grid_id, 23]);
        N = size(x)[2];
        nnet_mod = Flux.Chain(Flux.Dense(N, N * 4, relu), Flux.Dense(N * 4, N * 2, relu), Flux.Dense(N * 2, 1));
        for yr in 2008:2015
            getindices = fire_df[fire_df[:, :id] .== grid_id, :year];
            train = findall(q -> q .!= yr, getindices);
            test = findall(q -> q .== yr, getindices);
            nnet_mod = Flux.Chain(Flux.Dense(N, N * 4, relu), Flux.Dense(N * 4, N * 2, relu), Flux.Dense(N * 2, 1));
            model_perform_df_f, test_train_predict_df_f = flux_model_train(nnet_mod, x, y, train, test, 100, 200, length(train));
            model_perform_df_f[:, :grid_id] .= grid_id;
            model_perform_df_f[:, :test_year] .= yr;
            model_perform_df_f[:, :peat_desc] .= peat_desc;
            model_perform_df_f[:, :ecosys_desc] .= ecosys_desc;
            append!(model_perform_df_all_mods, model_perform_df_f);
            sort!(test_train_predict_df_f, :idx);
            test_train_predict_df_f = hcat(grid_yr_biweek[(grid_yr_biweek[:, :year] .== yr) .& (grid_yr_biweek[:, :id] .== grid_id), :], test_train_predict_df_f[:, 3:5]);
            test_train_predict_df_f[:, :test_year] .= yr;
            test_train_predict_df_f[:, :peat_desc] .= peat_desc;
            test_train_predict_df_f[:, :ecosys_desc] .= ecosys_desc;
            append!(test_train_predict_df_all_mods, test_train_predict_df_f);
        end
    end
    model_perform_df_all_mods[:, :model_name] .= model_label;
    test_train_predict_df_all_mods[:, :model_name] .= model_label;
    CSV.write("../outputs/" * model_label * "_model_perform_df_all_mods_" * string(r1)
    * "_"  * string(r2) * "_"  * ecosys_desc * "_"  * peat_desc * ".csv", model_perform_df_all_mods);
    CSV.write("../outputs/" * model_label * "_test_train_predict_df_all_mods_" * string(r1)
    * "_"  * string(r2) * "_"  * ecosys_desc * "_"  * peat_desc * ".csv", test_train_predict_df_all_mods);
end
for (r1, r2) in zip(1201, 1256)
    model_perform_df_all_mods = DataFrame();
    test_train_predict_df_all_mods = DataFrame();
    model_perform_df_f = DataFrame();
    test_train_predict_df_f = DataFrame();
    peat_desc = "Peatland";
    ecosys_desc = "Without Ecosys";
    for grid_id in grid_list[r1 : r2]
        x = fire_df[fire_df[:, :id] .== grid_id, [4, 5, 6, 7, 8, 10]];
        y = convert(Array{Float64, 1}, fire_df[fire_df[:, :id] .== grid_id, 23]);
        N = size(x)[2];
        nnet_mod = Flux.Chain(Flux.Dense(N, N * 4, relu), Flux.Dense(N * 4, N * 2, relu), Flux.Dense(N * 2, 1));
        for yr in 2008:2015
            getindices = fire_df[fire_df[:, :id] .== grid_id, :year];
            train = findall(q -> q .!= yr, getindices);
            test = findall(q -> q .== yr, getindices);
            nnet_mod = Flux.Chain(Flux.Dense(N, N * 4, relu), Flux.Dense(N * 4, N * 2, relu), Flux.Dense(N * 2, 1));
            model_perform_df_f, test_train_predict_df_f = flux_model_train(nnet_mod, x, y, train, test, 100, 200, length(train));
            model_perform_df_f[:, :grid_id] .= grid_id;
            model_perform_df_f[:, :test_year] .= yr;
            model_perform_df_f[:, :peat_desc] .= peat_desc;
            model_perform_df_f[:, :ecosys_desc] .= ecosys_desc;
            append!(model_perform_df_all_mods, model_perform_df_f);
            sort!(test_train_predict_df_f, :idx);
            test_train_predict_df_f = hcat(grid_yr_biweek[(grid_yr_biweek[:, :year] .== yr) .& (grid_yr_biweek[:, :id] .== grid_id), :], test_train_predict_df_f[:, 3:5]);
            test_train_predict_df_f[:, :test_year] .= yr;
            test_train_predict_df_f[:, :peat_desc] .= peat_desc;
            test_train_predict_df_f[:, :ecosys_desc] .= ecosys_desc;
            append!(test_train_predict_df_all_mods, test_train_predict_df_f);
        end
    end
    model_perform_df_all_mods[:, :model_name] .= model_label;
    test_train_predict_df_all_mods[:, :model_name] .= model_label;
    CSV.write("../outputs/" * model_label * "_model_perform_df_all_mods_" * string(r1)
    * "_"  * string(r2) * "_"  * ecosys_desc * "_"  * peat_desc * ".csv", model_perform_df_all_mods);
    CSV.write("../outputs/" * model_label * "_test_train_predict_df_all_mods_" * string(r1)
    * "_"  * string(r2) * "_"  * ecosys_desc * "_"  * peat_desc * ".csv", test_train_predict_df_all_mods);
end

using CSV, XLSX, DelimitedFiles, DataFrames, StatsBase

xf = XLSX.readxlsx("Physical_fatigue_Devesh_label.xlsx")

names_xf = XLSX.sheetnames(xf)
sh = xf["in"]
df = DataFrame(XLSX.readtable("Physical_fatigue_Devesh_label.xlsx", "in"; header = true, infer_eltypes=true))
types_df = map(eltype, eachcol(df))
names(df)
renamed_df = rename(df, "Gender (0=F 1=M)" => :gender_0f_1m)
renamed_df = rename(df, "fitness level(years training)" => :years_training, "sleeping hours(hours)"=> :slept_hours, "Borg_Test"=>:fatigue)

CSV.write("fatigue_data.csv", renamed_df)
df= CSV.read("fatigue_data.csv", DataFrame, header=1)

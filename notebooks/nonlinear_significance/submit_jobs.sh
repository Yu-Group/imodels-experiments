sims=("boolean_linear_dgp" 
	"normal_ar1_linear_dgp" 
	# "normal_block_cor_indep_signal_linear_dgp"
	# "normal_block_cor_linear_dgp" 
	"normal_linear_dgp" 
	"normal_linear_lss_dgp" 
	"normal_lss_dgp" 
	"normal_poly_dgp")

for sim in "${sims[@]}"
do
	sbatch --job-name=${sim} submit_python_parallel_savio3.sh ${sim}
done
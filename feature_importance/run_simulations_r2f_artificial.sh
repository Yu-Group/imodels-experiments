sims=("r2f.boolean_linear_dgp" 
	"r2f.normal_ar1_linear_dgp" 
	"r2f.normal_block_cor_indep_signal_linear_dgp"
	"r2f.normal_block_cor_linear_dgp" 
	"r2f.normal_linear_dgp" 
	"r2f.normal_linear_lss_dgp" 
	"r2f.normal_lss_dgp" 
	"r2f.normal_poly_dgp"
	"r2f.normal_poly_int_dgp")

for sim in "${sims[@]}"
do
  sbatch --job-name=${sim} submit_simulation_job.sh ${sim}
done

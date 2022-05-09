sims=("enhancer_linear_dgp" 
	"enhancer_lss_dgp" 
	"enhancer_linear_lss_dgp"
	"enhancer_permute_nonsignal_linear_dgp" 
	"enhancer_permute_nonsignal_lss_dgp" 
	"enhancer_permute_nonsignal_linear_lss_dgp" 
	"enhancer_static_linear_dgp" 
	"enhancer_static_lss_dgp")

for sim in "${sims[@]}"
do
	sbatch --job-name=${sim} submit_python_parallel_savio3.sh ${sim}
done
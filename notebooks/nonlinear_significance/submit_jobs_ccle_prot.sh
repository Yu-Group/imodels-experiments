sims=("ccle_prot_linear_dgp" 
	"ccle_prot_lss_dgp" 
	"ccle_prot_linear_lss_dgp"
	"ccle_prot_permute_nonsignal_linear_dgp" 
	"ccle_prot_permute_nonsignal_lss_dgp" 
	"ccle_prot_permute_nonsignal_linear_lss_dgp")

for sim in "${sims[@]}"
do
	sbatch --job-name=${sim} submit_python_parallel_savio3.sh ${sim}
done

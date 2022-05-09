sims=("tcga_linear_dgp" 
	"tcga_lss_dgp" 
	"tcga_linear_lss_dgp"
	"tcga_permute_nonsignal_linear_dgp"
	"tcga_permute_nonsignal_lss_dgp"
	"tcga_permute_nonsignal_linear_lss_dgp"
	"tcga_static_linear_dgp"
	"tcga_static_lss_dgp")

for sim in "${sims[@]}"
do
	sbatch --job-name=${sim} submit_python_parallel_savio3.sh ${sim}
done

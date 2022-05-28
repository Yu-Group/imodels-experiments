sims=("r2f.ccle_prot_cart_dgp"
	"r2f.ccle_prot_linear_dgp"
	"r2f.ccle_prot_linear_lss_dgp"
	"r2f.ccle_prot_logistic_dgp"
	"r2f.ccle_prot_lss_dgp" 
	"r2f.ccle_prot_poly_int_dgp"
	"r2f.fmri_augmented_dgp" 
	"r2f.fmri_linear_dgp" 
	"r2f.fmri_linear_lss_dgp"
	"r2f.fmri_logistic_dgp"
	"r2f.fmri_lss_dgp" 
	"r2f.fmri_poly_int_dgp"
	"r2f.german_linear_dgp"  
	"r2f.german_linear_lss_dgp"
	"r2f.german_logistic_dgp" 
	"r2f.german_lss_dgp"
	"r2f.german_poly_int_dgp"
	"r2f.ukbb_linear_dgp"  
	"r2f.ukbb_linear_lss_dgp"
	"r2f.ukbb_logistic_dgp" 
	"r2f.ukbb_lss_dgp" 
	"r2f.ukbb_poly_int_dgp"
	"r2f.r2f_algorithmic_choices.german_linear_dgp"  
	"r2f.r2f_algorithmic_choices.german_lss_dgp"
	"r2f.r2f_algorithmic_choices.german_poly_int_dgp"
	"r2f.r2f_algorithmic_choices.ukbb_linear_dgp"  
	"r2f.r2f_algorithmic_choices.ukbb_lss_dgp" 
	"r2f.r2f_algorithmic_choices.ukbb_poly_int_dgp"
	"r2f.sparsity_sims.german_linear_dgp"  
	"r2f.sparsity_sims.german_lss_dgp"
	"r2f.sparsity_sims.german_poly_int_dgp"
	"r2f.sparsity_sims.ukbb_linear_dgp"  
	"r2f.sparsity_sims.ukbb_lss_dgp" 
	"r2f.sparsity_sims.ukbb_poly_int_dgp"
	"r2f.stability_sims.algorithmic_perturbation_experiment.normal_block_cor_linear_dgp"
	"r2f.stability_sims.data_resampling_experiment.normal_block_cor_linear_dgp"
	"r2f.stability_sims.data_resampling_experiment.normal_block_cor_lss_dgp"
	"r2f.stability_sims.data_resampling_experiment.normal_block_cor_poly_int_dgp")

omitted_var_sims=("r2f.german_linear_dgp"  
	"r2f.german_lss_dgp"
	"r2f.german_poly_int_dgp"
	"r2f.ukbb_linear_dgp"  
	"r2f.ukbb_lss_dgp" 
	"r2f.ukbb_poly_int_dgp")

for sim in "${sims[@]}"
do
	sbatch --job-name=${sim} submit_simulation_job.sh ${sim}
done


for sim in "${omitted_var_sims[@]}"
do
	sbatch --job-name=${sim} submit_simulation_omitted_var_job.sh ${sim}
done

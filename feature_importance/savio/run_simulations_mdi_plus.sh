sims=(
  # Regression: ccle_rnaseq
  "mdi_plus.regression_sims.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "mdi_plus.regression_sims.ccle_rnaseq_linear_lss_3m_2r_dgp"
  "mdi_plus.regression_sims.ccle_rnaseq_linear_dgp"
  "mdi_plus.regression_sims.ccle_rnaseq_lss_3m_2r_dgp"
  # Regression: enhancer
  "mdi_plus.regression_sims.enhancer_hier_poly_3m_2r_dgp"
  "mdi_plus.regression_sims.enhancer_linear_lss_3m_2r_dgp"
  "mdi_plus.regression_sims.enhancer_linear_dgp"
  "mdi_plus.regression_sims.enhancer_lss_3m_2r_dgp"
  # Regression: juvenile
  "mdi_plus.regression_sims.juvenile_hier_poly_3m_2r_dgp"
  "mdi_plus.regression_sims.juvenile_linear_lss_3m_2r_dgp"
  "mdi_plus.regression_sims.juvenile_linear_dgp"
  "mdi_plus.regression_sims.juvenile_lss_3m_2r_dgp"
  # Regression: splicing
  "mdi_plus.regression_sims.splicing_hier_poly_3m_2r_dgp"
  "mdi_plus.regression_sims.splicing_linear_lss_3m_2r_dgp"
  "mdi_plus.regression_sims.splicing_linear_dgp"
  "mdi_plus.regression_sims.splicing_lss_3m_2r_dgp"
  
  # Classification: ccle_rnaseq
  "mdi_plus.classification_sims.ccle_rnaseq_lss_3m_2r_logistic_dgp"
  "mdi_plus.classification_sims.ccle_rnaseq_logistic_dgp"
  "mdi_plus.classification_sims.ccle_rnaseq_linear_lss_3m_2r_logistic_dgp"
  "mdi_plus.classification_sims.ccle_rnaseq_hier_poly_3m_2r_logistic_dgp"
  # Classification: enhancer
  "mdi_plus.classification_sims.enhancer_lss_3m_2r_logistic_dgp"
  "mdi_plus.classification_sims.enhancer_logistic_dgp"
  "mdi_plus.classification_sims.enhancer_linear_lss_3m_2r_logistic_dgp"
  "mdi_plus.classification_sims.enhancer_hier_poly_3m_2r_logistic_dgp"
  # Classification: juvenile
  "mdi_plus.classification_sims.juvenile_lss_3m_2r_logistic_dgp"
  "mdi_plus.classification_sims.juvenile_logistic_dgp"
  "mdi_plus.classification_sims.juvenile_linear_lss_3m_2r_logistic_dgp"
  "mdi_plus.classification_sims.juvenile_hier_poly_3m_2r_logistic_dgp"
  # Classification: splicing
  "mdi_plus.classification_sims.splicing_lss_3m_2r_logistic_dgp"
  "mdi_plus.classification_sims.splicing_logistic_dgp"
  "mdi_plus.classification_sims.splicing_linear_lss_3m_2r_logistic_dgp"
  "mdi_plus.classification_sims.splicing_hier_poly_3m_2r_logistic_dgp"
  
  # Robust: enhancer
  "mdi_plus.robust_sims.enhancer_linear_10MS_robust_dgp"
  "mdi_plus.robust_sims.enhancer_linear_25MS_robust_dgp"
  "mdi_plus.robust_sims.enhancer_lss_3m_2r_10MS_robust_dgp"
  "mdi_plus.robust_sims.enhancer_lss_3m_2r_25MS_robust_dgp"
  "mdi_plus.robust_sims.enhancer_linear_lss_3m_2r_10MS_robust_dgp"
  "mdi_plus.robust_sims.enhancer_linear_lss_3m_2r_25MS_robust_dgp"
  "mdi_plus.robust_sims.enhancer_hier_poly_3m_2r_10MS_robust_dgp"
  "mdi_plus.robust_sims.enhancer_hier_poly_3m_2r_25MS_robust_dgp"
  # Robust: ccle_rnaseq
  "mdi_plus.robust_sims.ccle_rnaseq_linear_10MS_robust_dgp"
  "mdi_plus.robust_sims.ccle_rnaseq_linear_25MS_robust_dgp"
  "mdi_plus.robust_sims.ccle_rnaseq_lss_3m_2r_10MS_robust_dgp"
  "mdi_plus.robust_sims.ccle_rnaseq_lss_3m_2r_25MS_robust_dgp"
  "mdi_plus.robust_sims.ccle_rnaseq_linear_lss_3m_2r_10MS_robust_dgp"
  "mdi_plus.robust_sims.ccle_rnaseq_linear_lss_3m_2r_25MS_robust_dgp"
  "mdi_plus.robust_sims.ccle_rnaseq_hier_poly_3m_2r_10MS_robust_dgp"
  "mdi_plus.robust_sims.ccle_rnaseq_hier_poly_3m_2r_25MS_robust_dgp"
  
  # MDI bias simulations
  "mdi_plus.mdi_bias_sims.correlation_sims.normal_block_cor_partial_linear_lss_dgp"
  "mdi_plus.mdi_bias_sims.entropy_sims.linear_dgp"
  "mdi_plus.mdi_bias_sims.entropy_sims.logistic_dgp"
  
  # Regression (varying number of features): ccle_rnaseq
  "mdi_plus.other_regression_sims.varying_p.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "mdi_plus.other_regression_sims.varying_p.ccle_rnaseq_linear_lss_3m_2r_dgp"
  "mdi_plus.other_regression_sims.varying_p.ccle_rnaseq_linear_dgp"
  "mdi_plus.other_regression_sims.varying_p.ccle_rnaseq_lss_3m_2r_dgp"
  # Regression (varying sparsity level): juvenile
  "mdi_plus.other_regression_sims.varying_sparsity.juvenile_hier_poly_3m_2r_dgp"
  "mdi_plus.other_regression_sims.varying_sparsity.juvenile_linear_lss_3m_2r_dgp"
  "mdi_plus.other_regression_sims.varying_sparsity.juvenile_linear_dgp"
  "mdi_plus.other_regression_sims.varying_sparsity.juvenile_lss_3m_2r_dgp"
  # Regression (varying sparsity level): splicing
  "mdi_plus.other_regression_sims.varying_sparsity.splicing_hier_poly_3m_2r_dgp"
  "mdi_plus.other_regression_sims.varying_sparsity.splicing_linear_lss_3m_2r_dgp"
  "mdi_plus.other_regression_sims.varying_sparsity.splicing_linear_dgp"
  "mdi_plus.other_regression_sims.varying_sparsity.splicing_lss_3m_2r_dgp"
  
  # MDI+ Modeling Choices: Enhancer (min_samples_per_leaf = 5)
  "mdi_plus.other_regression_sims.modeling_choices_min_samples5.enhancer_hier_poly_3m_2r_dgp"
  "mdi_plus.other_regression_sims.modeling_choices_min_samples5.enhancer_linear_dgp"
  # MDI+ Modeling Choices: CCLE (min_samples_per_leaf = 5)
  "mdi_plus.other_regression_sims.modeling_choices_min_samples5.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "mdi_plus.other_regression_sims.modeling_choices_min_samples5.ccle_rnaseq_linear_dgp"
  # MDI+ Modeling Choices: Enhancer (min_samples_per_leaf = 1)
  "mdi_plus.other_regression_sims.modeling_choices_min_samples1.enhancer_hier_poly_3m_2r_dgp"
  "mdi_plus.other_regression_sims.modeling_choices_min_samples1.enhancer_linear_dgp"
  # MDI+ Modeling Choices: CCLE (min_samples_per_leaf = 1)
  "mdi_plus.other_regression_sims.modeling_choices_min_samples1.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "mdi_plus.other_regression_sims.modeling_choices_min_samples1.ccle_rnaseq_linear_dgp"
  
  # MDI+ GLM and Metric Choices: Regression
  "mdi_plus.glm_metric_choices_sims.regression_sims.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "mdi_plus.glm_metric_choices_sims.regression_sims.ccle_rnaseq_linear_dgp"
  "mdi_plus.glm_metric_choices_sims.regression_sims.enhancer_hier_poly_3m_2r_dgp"
  "mdi_plus.glm_metric_choices_sims.regression_sims.enhancer_linear_dgp"
  # MDI+ GLM and Metric Choices: Classification
  "mdi_plus.glm_metric_choices_sims.classification_sims.juvenile_logistic_dgp"
  "mdi_plus.glm_metric_choices_sims.classification_sims.juvenile_hier_poly_3m_2r_logistic_dgp"
  "mdi_plus.glm_metric_choices_sims.classification_sims.splicing_logistic_dgp"
  "mdi_plus.glm_metric_choices_sims.classification_sims.splicing_hier_poly_3m_2r_logistic_dgp"
)

for sim in "${sims[@]}"
do
  sbatch --job-name=${sim} submit_simulation_job.sh ${sim}
done


## Misspecified model simulations
misspecifiedsims=(
  # Misspecified Regression: ccle_rnaseq
  "mdi_plus.regression_sims.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "mdi_plus.regression_sims.ccle_rnaseq_linear_lss_3m_2r_dgp"
  "mdi_plus.regression_sims.ccle_rnaseq_linear_dgp"
  "mdi_plus.regression_sims.ccle_rnaseq_lss_3m_2r_dgp"
  # Misspecified Regression: enhancer
  "mdi_plus.regression_sims.enhancer_hier_poly_3m_2r_dgp"
  "mdi_plus.regression_sims.enhancer_linear_lss_3m_2r_dgp"
  "mdi_plus.regression_sims.enhancer_linear_dgp"
  "mdi_plus.regression_sims.enhancer_lss_3m_2r_dgp"
  # Misspecified Regression: juvenile
  "mdi_plus.regression_sims.juvenile_hier_poly_3m_2r_dgp"
  "mdi_plus.regression_sims.juvenile_linear_lss_3m_2r_dgp"
  "mdi_plus.regression_sims.juvenile_linear_dgp"
  "mdi_plus.regression_sims.juvenile_lss_3m_2r_dgp"
  # Misspecified Regression: splicing
  "mdi_plus.regression_sims.splicing_hier_poly_3m_2r_dgp"
  "mdi_plus.regression_sims.splicing_linear_lss_3m_2r_dgp"
  "mdi_plus.regression_sims.splicing_linear_dgp"
  "mdi_plus.regression_sims.splicing_lss_3m_2r_dgp"
)

for sim in "${misspecifiedsims[@]}"
do
  sbatch --job-name=${sim}_omitted_vars submit_simulation_job_omitted_vars.sh ${sim}
done


## Real-data Case Study
# CCLE RNASeq
drugs=(
  "17-AAG"
  "AEW541"
  "AZD0530"
  "AZD6244"
  "Erlotinib"
  "Irinotecan"
  "L-685458"
  "LBW242"
  "Lapatinib"
  "Nilotinib"
  "Nutlin-3"
  "PD-0325901"
  "PD-0332991"
  "PF2341066"
  "PHA-665752"
  "PLX4720"
  "Paclitaxel"
  "Panobinostat"
  "RAF265"
  "Sorafenib"
  "TAE684"
  "TKI258"
  "Topotecan"
  "ZD-6474"
)

sim="mdi_plus.real_data_case_study.ccle_rnaseq_regression-"
for drug in "${drugs[@]}"
do
  sbatch --job-name=${sim}_${drug} submit_simulation_job_real_data_multitask.sh ${sim} "regression" ${drug}
done

sim="mdi_plus.real_data_case_study_no_data_split.ccle_rnaseq_regression-"
for drug in "${drugs[@]}"
do
  sbatch --job-name=${sim}_${drug} submit_simulation_job_real_data_multitask.sh ${sim} "regression" ${drug}
done

# TCGA BRCA
sim="mdi_plus.real_data_case_study.tcga_brca_classification-"
sbatch --job-name=${sim} submit_simulation_job_real_data.sh ${sim} "multiclass_classification"

sim="mdi_plus.real_data_case_study_no_data_split.tcga_brca_classification-"
sbatch --job-name=${sim} submit_simulation_job_real_data.sh ${sim} "multiclass_classification"


## Prediction Simulations

# Real Data: Binary classification
sims=(
  "mdi_plus.prediction_sims.enhancer_classification-"
  "mdi_plus.prediction_sims.juvenile_classification-"
  "mdi_plus.prediction_sims.splicing_classification-"
)

for sim in "${sims[@]}"
do
  sbatch --job-name=${sim} submit_simulation_job_real_data_prediction.sh ${sim} "binary_classification"
done

# Real Data: Multi-class classification
sim="mdi_plus.prediction_sims.tcga_brca_classification-"
sbatch --job-name=${sim} submit_simulation_job_real_data_prediction.sh ${sim} "multiclass_classification"

# Real Data: Regression
sim="mdi_plus.prediction_sims.ccle_rnaseq_regression-"
for drug in "${drugs[@]}"
do
  sbatch --job-name=${sim}_${drug} submit_simulation_job_real_data_prediction_multitask.sh ${sim} ${drug} "regression"
done

# MDI+ GLM and Metric Choices: Regression
sims=(
  "mdi_plus.glm_metric_choices_sims.regression_prediction_sims.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "mdi_plus.glm_metric_choices_sims.regression_prediction_sims.ccle_rnaseq_linear_dgp"
  "mdi_plus.glm_metric_choices_sims.regression_prediction_sims.enhancer_hier_poly_3m_2r_dgp"
  "mdi_plus.glm_metric_choices_sims.regression_prediction_sims.enhancer_linear_dgp"
)

for sim in "${sims[@]}"
do
  sbatch --job-name=${sim} submit_simulation_job_prediction.sh ${sim} "regression"
done
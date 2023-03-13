sims=(
  # Regression: ccle_rnaseq
  "gmdi.regression_sims.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "gmdi.regression_sims.ccle_rnaseq_linear_lss_3m_2r_dgp"
  "gmdi.regression_sims.ccle_rnaseq_linear_dgp"
  "gmdi.regression_sims.ccle_rnaseq_lss_3m_2r_dgp"
  # Regression: enhancer
  "gmdi.regression_sims.enhancer_hier_poly_3m_2r_dgp"
  "gmdi.regression_sims.enhancer_linear_lss_3m_2r_dgp"
  "gmdi.regression_sims.enhancer_linear_dgp"
  "gmdi.regression_sims.enhancer_lss_3m_2r_dgp"
  # Regression: juvenile
  "gmdi.regression_sims.juvenile_hier_poly_3m_2r_dgp"
  "gmdi.regression_sims.juvenile_linear_lss_3m_2r_dgp"
  "gmdi.regression_sims.juvenile_linear_dgp"
  "gmdi.regression_sims.juvenile_lss_3m_2r_dgp"
  # Regression: splicing
  "gmdi.regression_sims.splicing_hier_poly_3m_2r_dgp"
  "gmdi.regression_sims.splicing_linear_lss_3m_2r_dgp"
  "gmdi.regression_sims.splicing_linear_dgp"
  "gmdi.regression_sims.splicing_lss_3m_2r_dgp"
  
  # Classification: ccle_rnaseq
  "gmdi.classification_sims.ccle_rnaseq_lss_3m_2r_logistic_dgp"
  "gmdi.classification_sims.ccle_rnaseq_logistic_dgp"
  "gmdi.classification_sims.ccle_rnaseq_linear_lss_3m_2r_logistic_dgp"
  "gmdi.classification_sims.ccle_rnaseq_hier_poly_3m_2r_logistic_dgp"
  # Classification: enhancer
  "gmdi.classification_sims.enhancer_lss_3m_2r_logistic_dgp"
  "gmdi.classification_sims.enhancer_logistic_dgp"
  "gmdi.classification_sims.enhancer_linear_lss_3m_2r_logistic_dgp"
  "gmdi.classification_sims.enhancer_hier_poly_3m_2r_logistic_dgp"
  # Classification: juvenile
  "gmdi.classification_sims.juvenile_lss_3m_2r_logistic_dgp"
  "gmdi.classification_sims.juvenile_logistic_dgp"
  "gmdi.classification_sims.juvenile_linear_lss_3m_2r_logistic_dgp"
  "gmdi.classification_sims.juvenile_hier_poly_3m_2r_logistic_dgp"
  # Classification: splicing
  "gmdi.classification_sims.splicing_lss_3m_2r_logistic_dgp"
  "gmdi.classification_sims.splicing_logistic_dgp"
  "gmdi.classification_sims.splicing_linear_lss_3m_2r_logistic_dgp"
  "gmdi.classification_sims.splicing_hier_poly_3m_2r_logistic_dgp"
  
  # Robust: enhancer
  "gmdi.robust_sims.enhancer_linear_10MS_robust_dgp"
  "gmdi.robust_sims.enhancer_linear_25MS_robust_dgp"
  "gmdi.robust_sims.enhancer_lss_3m_2r_10MS_robust_dgp"
  "gmdi.robust_sims.enhancer_lss_3m_2r_25MS_robust_dgp"
  "gmdi.robust_sims.enhancer_linear_lss_3m_2r_10MS_robust_dgp"
  "gmdi.robust_sims.enhancer_linear_lss_3m_2r_25MS_robust_dgp"
  "gmdi.robust_sims.enhancer_hier_poly_3m_2r_10MS_robust_dgp"
  "gmdi.robust_sims.enhancer_hier_poly_3m_2r_25MS_robust_dgp"
  # Robust: ccle_rnaseq
  "gmdi.robust_sims.ccle_rnaseq_linear_10MS_robust_dgp"
  "gmdi.robust_sims.ccle_rnaseq_linear_25MS_robust_dgp"
  "gmdi.robust_sims.ccle_rnaseq_lss_3m_2r_10MS_robust_dgp"
  "gmdi.robust_sims.ccle_rnaseq_lss_3m_2r_25MS_robust_dgp"
  "gmdi.robust_sims.ccle_rnaseq_linear_lss_3m_2r_10MS_robust_dgp"
  "gmdi.robust_sims.ccle_rnaseq_linear_lss_3m_2r_25MS_robust_dgp"
  "gmdi.robust_sims.ccle_rnaseq_hier_poly_3m_2r_10MS_robust_dgp"
  "gmdi.robust_sims.ccle_rnaseq_hier_poly_3m_2r_25MS_robust_dgp"
  
  # MDI bias simulations
  "gmdi.mdi_bias_sims.correlation_sims.normal_block_cor_partial_linear_lss_dgp"
  "gmdi.mdi_bias_sims.entropy_sims.linear_dgp"
  "gmdi.mdi_bias_sims.entropy_sims.logistic_dgp"
  
  # Regression (varying number of features): ccle_rnaseq
  "gmdi.other_regression_sims.varying_p.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "gmdi.other_regression_sims.varying_p.ccle_rnaseq_linear_lss_3m_2r_dgp"
  "gmdi.other_regression_sims.varying_p.ccle_rnaseq_linear_dgp"
  "gmdi.other_regression_sims.varying_p.ccle_rnaseq_lss_3m_2r_dgp"
  # Regression (varying sparsity level): juvenile
  "gmdi.other_regression_sims.varying_sparsity.juvenile_hier_poly_3m_2r_dgp"
  "gmdi.other_regression_sims.varying_sparsity.juvenile_linear_lss_3m_2r_dgp"
  "gmdi.other_regression_sims.varying_sparsity.juvenile_linear_dgp"
  "gmdi.other_regression_sims.varying_sparsity.juvenile_lss_3m_2r_dgp"
  # Regression (varying sparsity level): splicing
  "gmdi.other_regression_sims.varying_sparsity.splicing_hier_poly_3m_2r_dgp"
  "gmdi.other_regression_sims.varying_sparsity.splicing_linear_lss_3m_2r_dgp"
  "gmdi.other_regression_sims.varying_sparsity.splicing_linear_dgp"
  "gmdi.other_regression_sims.varying_sparsity.splicing_lss_3m_2r_dgp"
  
  # GMDI Modeling Choices: Enhancer (min_samples_per_leaf = 5)
  "gmdi.other_regression_sims.modeling_choices_min_samples5.enhancer_hier_poly_3m_2r_dgp"
  "gmdi.other_regression_sims.modeling_choices_min_samples5.enhancer_linear_dgp"
  # GMDI Modeling Choices: CCLE (min_samples_per_leaf = 5)
  "gmdi.other_regression_sims.modeling_choices_min_samples5.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "gmdi.other_regression_sims.modeling_choices_min_samples5.ccle_rnaseq_linear_dgp"
  # GMDI Modeling Choices: Enhancer (min_samples_per_leaf = 1)
  "gmdi.other_regression_sims.modeling_choices_min_samples1.enhancer_hier_poly_3m_2r_dgp"
  "gmdi.other_regression_sims.modeling_choices_min_samples1.enhancer_linear_dgp"
  # GMDI Modeling Choices: CCLE (min_samples_per_leaf = 1)
  "gmdi.other_regression_sims.modeling_choices_min_samples1.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "gmdi.other_regression_sims.modeling_choices_min_samples1.ccle_rnaseq_linear_dgp"
  
  # GMDI GLM and Metric Choices: Regression
  "gmdi.glm_metric_choices_sims.regression_sims.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "gmdi.glm_metric_choices_sims.regression_sims.ccle_rnaseq_linear_dgp"
  "gmdi.glm_metric_choices_sims.regression_sims.enhancer_hier_poly_3m_2r_dgp"
  "gmdi.glm_metric_choices_sims.regression_sims.enhancer_linear_dgp"
  # GMDI GLM and Metric Choices: Classification
  "gmdi.glm_metric_choices_sims.classification_sims.juvenile_logistic_dgp"
  "gmdi.glm_metric_choices_sims.classification_sims.juvenile_hier_poly_3m_2r_logistic_dgp"
  "gmdi.glm_metric_choices_sims.classification_sims.splicing_logistic_dgp"
  "gmdi.glm_metric_choices_sims.classification_sims.splicing_hier_poly_3m_2r_logistic_dgp"
)

for sim in "${sims[@]}"
do
  sbatch --job-name=${sim} submit_simulation_job.sh ${sim}
done


## Misspecified model simulations
misspecifiedsims=(
  # Misspecified Regression: ccle_rnaseq
  "gmdi.regression_sims.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "gmdi.regression_sims.ccle_rnaseq_linear_lss_3m_2r_dgp"
  "gmdi.regression_sims.ccle_rnaseq_linear_dgp"
  "gmdi.regression_sims.ccle_rnaseq_lss_3m_2r_dgp"
  # Misspecified Regression: enhancer
  "gmdi.regression_sims.enhancer_hier_poly_3m_2r_dgp"
  "gmdi.regression_sims.enhancer_linear_lss_3m_2r_dgp"
  "gmdi.regression_sims.enhancer_linear_dgp"
  "gmdi.regression_sims.enhancer_lss_3m_2r_dgp"
  # Misspecified Regression: juvenile
  "gmdi.regression_sims.juvenile_hier_poly_3m_2r_dgp"
  "gmdi.regression_sims.juvenile_linear_lss_3m_2r_dgp"
  "gmdi.regression_sims.juvenile_linear_dgp"
  "gmdi.regression_sims.juvenile_lss_3m_2r_dgp"
  # Misspecified Regression: splicing
  "gmdi.regression_sims.splicing_hier_poly_3m_2r_dgp"
  "gmdi.regression_sims.splicing_linear_lss_3m_2r_dgp"
  "gmdi.regression_sims.splicing_linear_dgp"
  "gmdi.regression_sims.splicing_lss_3m_2r_dgp"
)

for sim in "${misspecifiedsims[@]}"
do
  sbatch --job-name=${sim}_omitted_vars submit_simulation_job_omitted_vars.sh ${sim}
done


## Real-data Case Study
# CCLE RNASeq
sim="gmdi.real_data_case_study.ccle_rnaseq_regression-"
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

for drug in "${drugs[@]}"
do
  sbatch --job-name=${sim}_${drug} submit_simulation_job_real_data_multitask.sh ${sim} ${drug}
done

# TCGA BRCA
sim="gmdi.real_data_case_study.tcga_brca_classification-"
sbatch --job-name=${sim} submit_simulation_job_real_data.sh ${sim}


## Prediction Simulations

# Real Data: Binary classification
sims=(
  "gmdi.prediction_sims.enhancer_classification-"
  "gmdi.prediction_sims.juvenile_classification-"
  "gmdi.prediction_sims.splicing_classification-"
)

for sim in "${sims[@]}"
do
  sbatch --job-name=${sim} submit_simulation_job_real_data_prediction.sh ${sim} "binary_classification"
done

# Real Data: Multi-class classification
sim="gmdi.prediction_sims.tcga_brca_classification-"
sbatch --job-name=${sim} submit_simulation_job_real_data_prediction.sh ${sim} "multiclass_classification"

# Real Data: Regression
sim="gmdi.prediction_sims.ccle_rnaseq_regression-"
for drug in "${drugs[@]}"
do
  sbatch --job-name=${sim}_${drug} submit_simulation_job_real_data_prediction_multitask.sh ${sim} ${drug} "regression"
done

# GMDI GLM and Metric Choices: Regression
sims=(
  "gmdi.glm_metric_choices_sims.regression_prediction_sims.ccle_rnaseq_hier_poly_3m_2r_dgp"
  "gmdi.glm_metric_choices_sims.regression_prediction_sims.ccle_rnaseq_linear_dgp"
  "gmdi.glm_metric_choices_sims.regression_prediction_sims.enhancer_hier_poly_3m_2r_dgp"
  "gmdi.glm_metric_choices_sims.regression_prediction_sims.enhancer_linear_dgp"
)

for sim in "${sims[@]}"
do
  sbatch --job-name=${sim} submit_simulation_job_prediction.sh ${sim} "regression"
done
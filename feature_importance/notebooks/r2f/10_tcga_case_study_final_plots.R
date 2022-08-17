library(magrittr)

results_dir <- "/Users/Tiffany/My Documents/Research/Yu/imodels-experiments/feature_importance/results"

stability_df <- data.table::fread(file.path(results_dir, "tcga_brca_stability_results.csv"))

ranks <- c(1, 5, 10)
# keep_methods <- c("gjmdi_loocv", "mda", "mdi", "permutation", "shap")
# color_palette <- c("black", "orange", "#71beb7", "#218a1e", "#cc3399")

keep_methods <- c("gjmdi_loocv", "mda", "mdi", "shap")
color_palette <- c("black", "orange", "#71beb7", "#cc3399")

plt_df <- stability_df %>%
  dplyr::filter(
    rank %in% ranks,
    top_r_stability > 0,
    method %in% keep_methods
  ) %>%
  dplyr::mutate(
    method = forcats::fct_recode(method,
                                 "gMDI" = "gjmdi_loocv",
                                 "MDA" = "mda",
                                 "MDI" = "mdi",
                                 "Permutation" = "permutation",
                                 "TreeSHAP" = "shap"),
    rank = factor(sprintf("R = %s", rank),
                  sprintf("R = %s", sort(unique(rank))))
  ) %>%
  droplevels()

ggplot2::ggplot(plt_df) +
  ggplot2::aes(
    x = reorder(feature, -top_r_stability), y = top_r_stability, fill = method
  ) +
  ggplot2::facet_wrap(~ rank, scales = "free_x", nrow = 3, ncol = 1) +
  ggplot2::geom_bar(
    stat = "identity", 
    position = ggplot2::position_dodge(preserve = "single")
  ) +
  ggplot2::labs(
    x = "Gene", y = "Proportion of times in top R", fill = "Method"
  ) +
  ggplot2::scale_fill_manual(values = color_palette) +
  ggplot2::coord_cartesian(ylim = c(0, 1)) +
  # vthemes::scale_fill_vmodern(discrete = TRUE) +
  vthemes::theme_vmodern(x_text_angle = TRUE)


data_dir <- "/Users/Tiffany/My Documents/Research/Yu/imodels-experiments/feature_importance/data"

X <- data.table::fread(
  file.path(data_dir, "X_tcga_var_filtered_log_transformed.csv")
)
keep_features <- c("ESR1", "TPX2", "FOXA1", "FOXM1", "GATA3", "MYBL2", "PLK1",
                   "FOXC1", "AGR2", "MLPH")
vdocs::plot_cor_heatmap(
  X %>% dplyr::select(tidyselect::all_of(keep_features)), 
  text_size = 4.5, size_preset = "large"
) +
  ggplot2::labs(x = "Gene", y = "Gene", fill = "Correlation")

keep_features <- unique(plt_df$feature)
length(keep_features)
vdocs::plot_cor_heatmap(
  X %>% dplyr::select(tidyselect::all_of(keep_features)), 
  text_size = 3
) +
  ggplot2::labs(x = "Gene", y = "Gene", fill = "Correlation")

keep_features <- plt_df %>%
  dplyr::filter(method == "gjMDI (LOOCV)", rank == "R = 10") %>%
  dplyr::arrange(-top_r_stability) %>%
  dplyr::slice_head(n = 10) %>%
  dplyr::pull(feature)
vdocs::plot_cor_heatmap(
  X %>% dplyr::select(tidyselect::all_of(keep_features)), 
  text_size = 4.5, size_preset = "large"
) +
  ggplot2::labs(x = "Gene", y = "Gene", fill = "Correlation")


plt_df %>%
  dplyr::group_by(method, rank) %>%
  dplyr::summarise(n = dplyr::n()) %>%
  tidyr::pivot_wider(id_cols = method, names_from = rank, values_from = n) %>%
  dplyr::rename("Method" = "method") %>% #,
                # "Total # of features in top 1 across 10 replicates" = "R = 1",
                # "Total # of features in top 5 across 10 replicates" = "R = 5",
                # "Total # of features in top 10 across 10 replicates" = "R = 10") %>%
  vthemes::pretty_DT(rownames = FALSE, 
                     options = list(dom = "t", ordering = FALSE))



plt_df %>%
  dplyr::filter(rank == "R = 1") %>%
  ggplot2::ggplot() +
  ggplot2::aes(
    x = reorder(feature, -top_r_stability), y = top_r_stability, fill = method
  ) +
  # ggplot2::facet_wrap(~ rank, scales = "free_x", nrow = 3, ncol = 1) +
  ggplot2::geom_bar(
    stat = "identity", 
    position = ggplot2::position_dodge(preserve = "single")
  ) +
  ggplot2::labs(
    x = "Gene", y = "Proportion of times in top 1", fill = "Method"
  ) +
  ggplot2::scale_fill_manual(values = color_palette) +
  ggplot2::scale_y_continuous(breaks = seq(0, 1, by = 0.2), limits = c(0, 1)) +
  # vthemes::scale_fill_vmodern(discrete = TRUE) +
  vthemes::theme_vmodern(x_text_angle = TRUE, size_preset = "large")


plt_df %>%
  dplyr::filter(rank == "R = 5") %>%
  ggplot2::ggplot() +
  ggplot2::aes(
    x = reorder(feature, -top_r_stability), y = top_r_stability, fill = method
  ) +
  # ggplot2::facet_wrap(~ rank, scales = "free_x", nrow = 3, ncol = 1) +
  ggplot2::geom_bar(
    stat = "identity", 
    position = ggplot2::position_dodge(preserve = "single")
  ) +
  ggplot2::labs(
    x = "Gene", y = "Proportion of times in top 5", fill = "Method"
  ) +
  ggplot2::scale_fill_manual(values = color_palette) +
  ggplot2::scale_y_continuous(breaks = seq(0, 1, by = 0.2), limits = c(0, 1)) +
  ggplot2::coord_cartesian(xlim = c(1, 11.9)) +
  # vthemes::scale_fill_vmodern(discrete = TRUE) +
  vthemes::theme_vmodern(x_text_angle = TRUE, size_preset = "large")


plt_df %>%
  dplyr::filter(rank == "R = 10") %>%
  ggplot2::ggplot() +
  ggplot2::aes(
    x = reorder(feature, -top_r_stability), y = top_r_stability, fill = method
  ) +
  # ggplot2::facet_wrap(~ rank, scales = "free_x", nrow = 3, ncol = 1) +
  ggplot2::geom_bar(
    stat = "identity", 
    position = ggplot2::position_dodge(preserve = "single")
  ) +
  ggplot2::labs(
    x = "Gene", y = "Proportion of times in top 10", fill = "Method"
  ) +
  ggplot2::scale_fill_manual(values = color_palette) +
  ggplot2::scale_y_continuous(breaks = seq(0, 1, by = 0.2), limits = c(0, 1)) +
  ggplot2::coord_cartesian(xlim = c(1, 11.9)) +
  # vthemes::scale_fill_vmodern(discrete = TRUE) +
  vthemes::theme_vmodern(x_text_angle = TRUE, size_preset = "large")


plt_df %>%
  dplyr::mutate(
    method = ifelse(as.character(method) == "gjMDI (LOOCV)", 
                    "gMDI", as.character(method))
  ) %>%
  dplyr::group_by(method, rank) %>%
  dplyr::summarise(n = dplyr::n()) %>%
  tidyr::pivot_wider(id_cols = method, names_from = rank, values_from = n) %>%
  dplyr::rename("Method" = "method",
                "Total # of features in top 1 across 10 replicates" = "R = 1",
                "Total # of features in top 5 across 10 replicates" = "R = 5",
                "Total # of features in top 10 across 10 replicates" = "R = 10") %>%
  dplyr::select(Method, `Total # of features in top 10 across 10 replicates`) %>%
  vthemes::pretty_DT(rownames = FALSE, 
                     options = list(dom = "t", ordering = FALSE))

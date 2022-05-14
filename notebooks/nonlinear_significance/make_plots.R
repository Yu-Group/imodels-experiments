rm(list = ls())
library(magrittr)
library(patchwork)
params <- list()

#### Set Up ####
manual_color_palette <- c("#ff9902", "#6caa52", "#4a86e8", "#0f459f", "#a64d79")
results_dir <- "/global/scratch/users/tiffanytang/nonlinear-significance/results/"
plots_dir <- "plots"
fig_height <- 5
fig_width <- 10
metrics <- c("rocauc")
metrics_name <- "AUROC"
# metrics <- c("prauc")
# metrics_name <- "AUPRC"
metrics_all <- c("rocauc", "prauc", "r2_rocauc", "r2_prauc")

keep_x_models <- c("german", "ukbb")
keep_y_models <- c("linear", "lss", "hier_poly")

seed <- 12345
n_models <- list(
  ccle_prot = list(low = 100, high = 370),
  ukbb = list(low = 250, high = 1000),
  german = list(low = 250, high = 1000)
)
p_models <- list(
  ccle_prot = 214,
  ukbb = 500,
  german = 20
)

r2f <- "r2f"
keep_methods <- c(r2f, "Permutation_RF", "MDI_RF", "MDI-oob_RF", "TreeSHAP_RF")
legend_labels <- c(bquote(~r^2*"f"), "Permutation", "MDI", "MDI-oob", "TreeSHAP")

my_theme <- vthemes::theme_vmodern(
  size_preset = "medium", bg_color = "white", grid_color = "white",
  axis.title = ggplot2::element_text(size = 12, face = "plain"),
  legend.title = ggplot2::element_blank(),
  legend.text = ggplot2::element_text(size = 9),
  legend.position = c(0.8, 0.3),
  legend.background = ggplot2::element_rect(
    color = "slategray", size = .1
  ),
  plot.title = ggplot2::element_blank()
  # plot.title = ggplot2::element_text(size = 12, face = "plain", hjust = 0.5)
)

#### functions ####
reformat_results <- function(results) {
  if (!("support") %in% colnames(results)) {
    results <- results %>%
      dplyr::mutate(support = NA)
  }
  if ("r2_x" %in% colnames(results)) {
    results <- results %>%
      dplyr::select(-r2_y) %>%
      dplyr::rename(r2 = r2_x)
  }
  if ("n_components_x" %in% colnames(results)) {
    results <- results %>%
      dplyr::select(-n_components_y) %>%
      dplyr::rename(n_components = n_components_x)
  }
  if ("n_stumps_x" %in% colnames(results)) {
    results <- results %>%
      dplyr::select(-n_stumps_y) %>%
      dplyr::rename(n_stumps = n_stumps_x)
  }
  results_grouped <- results %>%
    dplyr::group_by(index) %>%
    tidyr::nest(fi_scores = var:(tidyselect::last_col())) %>%
    dplyr::ungroup() %>%
    dplyr::select(-index) %>%
    # join fi+model to get method column
    tidyr::unite(col = "method", fi, model, na.rm = TRUE, remove = FALSE) %>%
    dplyr::mutate(
      method = ifelse(stringr::str_detect(method, "^r2f.*RF$"),
                      stringr::str_remove(method, "\\_RF$"), method)
    )
  return(results_grouped)
}

plot_metrics <- function(results, metrics, x_str, facet_str, low, high,
                         keep_methods, point_size = 1, line_size = 1,
                         alpha = 0.35, errbar_width = 0, x_categorical = FALSE,
                         remove_axes = TRUE, show_errbars = FALSE) {
  if (x_categorical) {
    results <- results %>%
      dplyr::mutate(dplyr::across(
        tidyselect::all_of(x_str), ~factor(.x, levels = sort(unique(.x)))
      ))
  }
  
  plt_df <- results %>%
    dplyr::mutate(
      rocauc = ifelse(stringr::str_detect(method, "r2f"), r2_rocauc, rocauc),
      prauc = ifelse(stringr::str_detect(method, "r2f"), r2_prauc, prauc)
    ) %>%
    dplyr::select(rep, method,
                  tidyselect::all_of(c(metrics, x_str, facet_str))) %>%
    tidyr::pivot_longer(
      cols = tidyselect::all_of(metrics), names_to = "metric"
    )  %>%
    dplyr::group_by(
      method, metric, dplyr::across(tidyselect::all_of(c(x_str, facet_str)))
    ) %>%
    dplyr::summarise(mean = mean(value), sd = sd(value), .groups = "keep") %>%
    dplyr::filter(method %in% keep_methods) %>%
    dplyr::mutate(
      method = forcats::fct_recode(
        factor(method, levels = keep_methods), 
        R2F = !!r2f, Permutation = "Permutation_RF", 
        MDI = "MDI_RF", `MDI-oob` = "MDI-oob_RF", TreeSHAP = "TreeSHAP_RF"
      ),
      metric = forcats::fct_recode(
        factor(metric, levels = c("rocauc", "prauc")),
        AUROC = "rocauc", AUPRC = "prauc"
      )
    )
  
  if (!is.null(low)) {
    plt_df1 <- plt_df %>%
      dplyr::filter(.data[[facet_str]] == !!low)
  } else {
    plt_df1 <- plt_df
  }
  low_data_plt <- ggplot2::ggplot(plt_df1) +
    ggplot2::aes(x = .data[[x_str]], y = mean, 
                 color = method, alpha = method, group = method) +
    ggplot2::geom_point(size = point_size) +
    ggplot2::geom_line(size = line_size) +
    ggplot2::scale_color_manual(
      values = manual_color_palette, labels = legend_labels
    ) +
    ggplot2::scale_alpha_manual(
      values = c(1, rep(alpha, 4)), labels = legend_labels
    ) +
    my_theme
  
  if (!is.null(high)) {
    plt_df2 <- plt_df %>%
      dplyr::filter(.data[[facet_str]] == !!high)
  } else {
    plt_df2 <- plt_df
  }
  high_data_plt <- ggplot2::ggplot(plt_df2) +
    ggplot2::aes(x = .data[[x_str]], y = mean,
                 color = method, alpha = method, group = method) +
    ggplot2::geom_point(size = point_size) +
    ggplot2::geom_line(size = line_size) +
    ggplot2::scale_color_manual(values = manual_color_palette) +
    ggplot2::scale_alpha_manual(values = c(1, rep(alpha, 4))) +
    my_theme
  
  if (show_errbars) {
    low_data_plt <- low_data_plt + 
      ggplot2::geom_errorbar(
        ggplot2::aes(x = .data[[x_str]], ymin = mean - sd, ymax = mean + sd),
        width = errbar_width
      )
    high_data_plt <- high_data_plt + 
      ggplot2::geom_errorbar(
        ggplot2::aes(x = .data[[x_str]], ymin = mean - sd, ymax = mean + sd),
        width = errbar_width
      )
  }
  
  if (remove_axes) {
    if (x_model != "german") {
      low_data_plt <- low_data_plt +
        ggplot2::theme(
          axis.title = ggplot2::element_blank()
        )
      high_data_plt <- high_data_plt +
        ggplot2::theme(
          axis.title.y = ggplot2::element_blank()
        )
    } else {
      low_data_plt <- low_data_plt +
        ggplot2::theme(
          axis.title.x = ggplot2::element_blank()
        )
    }
  }
  return(list(low = low_data_plt, high = high_data_plt))
}

#### Real X, Artificial Y Simulations ####
y_models <- c("linear", "lss", "linear_lss", "hier_poly")
x_models <- c("ccle_prot", "ukbb", "german")
vary_param_name <- "heritability_sample_row_n"

keep_ns <- list(
  ccle_prot = c(100, 250, 370),
  ukbb = c(100, 250, 500, 1000),
  german = c(100, 250, 500, 1000)
)

# read in data
results_ls <- list()
for (y_model in y_models) {
  results_ls[[y_model]] <- list()
  for (x_model in x_models) {
    sim_name <- sprintf("%s_%s_dgp", x_model, y_model)
    fname <- file.path(results_dir, sim_name, paste0("varying_", vary_param_name), 
                       paste0("seed", seed), "results.csv")
    if (!file.exists(fname)) {
      warning(sprintf("Results cannot be found for %s", sim_name))
    }
    results_ls[[y_model]][[x_model]] <- data.table::fread(fname) %>%
      reformat_results()
  }
}

# saveRDS(results_ls, file.path(results_dir, "results_ls.rds"))

point_size <- 2
line_size <- 1
plt_ls <- list()
for (heritability in c(0.05, 0.1, 0.2, 0.4, 0.8)) {
  plt_ls[[as.character(heritability)]] <- list()
  for (x_model in x_models) {
    plt_ls[[as.character(heritability)]][[x_model]] <- list()
    for (y_model in y_models) {
      sim_name <- sprintf("%s_%s_dgp", x_model, y_model)
      sim_title <- dplyr::case_when(
        x_model == "ccle_prot" ~ "CCLE Protein",
        x_model == "ukbb" ~ "UK Biobank",
        x_model == "german" ~ "German Credit"
      )
      out <- plot_metrics(
        results_ls[[y_model]][[x_model]] %>%
          dplyr::filter(sample_row_n %in% keep_ns[[x_model]]), 
        metrics = metrics, 
        x_str = "sample_row_n", 
        facet_str = "heritability", 
        low = heritability, high = NULL,
        keep_methods = keep_methods, 
        point_size = point_size, line_size = line_size,
        x_categorical = FALSE,
        remove_axes = FALSE
      )
      plt <- out$low +
        ggplot2::labs(
          x = "Sample Size", y = metrics_name,
          color = "Method", alpha = "Method",
          title = sprintf("%s (p = %s)", sim_title, p_models[[x_model]])
        )
      if (heritability != 0.1) {
        plt <- plt +
          ggplot2::theme(axis.title.y = ggplot2::element_blank())
      }
      if (x_model != keep_x_models[length(keep_x_models)]) {
        plt <- plt +
          ggplot2::theme(axis.title.x = ggplot2::element_blank())
      }
      if ((heritability != 0.8) | (x_model != keep_x_models[1])) {
        plt <- plt +
          ggplot2::guides(color = "none", alpha = "none")
      }
      plt_ls[[as.character(heritability)]][[x_model]][[y_model]] <- plt
    }
  }
}

keep_heritabilities <- c(0.1, 0.2, 0.4, 0.8)
plt <- list()
for (y_model in keep_y_models) {
  for (x_model in keep_x_models) {
    for (heritability in keep_heritabilities) {
      if (identical(plt, list())) {
        plt <- plt_ls[[as.character(heritability)]][[x_model]][[y_model]]
      } else {
        plt <- plt + plt_ls[[as.character(heritability)]][[x_model]][[y_model]]
      }
    }
  }
}

plt <- plt +
  plot_layout(ncol = length(keep_heritabilities), nrow = 6)
ggplot2::ggsave(
  file.path(plots_dir, sprintf("real_data_artificial_y_sims_%s.pdf", metrics)), 
  plot = plt, units = "in", width = 12, height = 20
)

keep_heritabilities <- c(0.1, 0.2, 0.4, 0.8)
for (y_model in keep_y_models) {
  for (x_model in keep_x_models) {
    plt <- list()
    for (heritability in keep_heritabilities) {
      cur_plt <- plt_ls[[as.character(heritability)]][[x_model]][[y_model]]
      if (identical(plt, list())) {
        plt <- cur_plt
      } else {
        plt <- plt + cur_plt
      }
    }
    plt <- plt +
      plot_layout(ncol = length(keep_heritabilities), nrow = 1)
    ggplot2::ggsave(
      file.path(plots_dir, sprintf("real_data_artificial_y_sims_%s_%s_%s.pdf", 
                                   y_model, x_model, metrics)), 
      plot = plt, units = "in", width = 12, height = 3
    )
  }
}

point_size <- 2
line_size <- 1
errbar_width <- 0
plt_ls <- list()
for (heritability in c(0.05, 0.1, 0.2, 0.4, 0.8)) {
  plt_ls[[as.character(heritability)]] <- list()
  for (x_model in x_models) {
    plt_ls[[as.character(heritability)]][[x_model]] <- list()
    for (y_model in y_models) {
      sim_name <- sprintf("%s_%s_dgp", x_model, y_model)
      sim_title <- dplyr::case_when(
        x_model == "ccle_prot" ~ "CCLE Protein",
        x_model == "ukbb" ~ "UK Biobank",
        x_model == "german" ~ "German Credit"
      )
      out <- plot_metrics(
        results_ls[[y_model]][[x_model]] %>%
          dplyr::filter(sample_row_n %in% keep_ns[[x_model]]), 
        metrics = metrics, 
        x_str = "sample_row_n", 
        facet_str = "heritability", 
        low = heritability, high = NULL,
        keep_methods = keep_methods, 
        point_size = point_size, line_size = line_size, 
        errbar_width = errbar_width,
        x_categorical = FALSE,
        remove_axes = FALSE,
        show_errbars = TRUE
      )
      plt <- out$low +
        ggplot2::labs(
          x = "Sample Size", y = metrics_name,
          color = "Method", alpha = "Method",
          title = sprintf("%s (p = %s)", sim_title, p_models[[x_model]])
        )
      if (heritability != 0.1) {
        plt <- plt +
          ggplot2::theme(axis.title.y = ggplot2::element_blank())
      }
      if (x_model != keep_x_models[length(keep_x_models)]) {
        plt <- plt +
          ggplot2::theme(axis.title.x = ggplot2::element_blank())
      }
      if ((heritability != 0.8) | (x_model != keep_x_models[1])) {
        plt <- plt +
          ggplot2::guides(color = "none", alpha = "none")
      }
      plt_ls[[as.character(heritability)]][[x_model]][[y_model]] <- plt
    }
  }
}

keep_heritabilities <- c(0.1, 0.2, 0.4, 0.8)
plt <- list()
for (y_model in keep_y_models) {
  for (x_model in keep_x_models) {
    for (heritability in keep_heritabilities) {
      if (identical(plt, list())) {
        plt <- plt_ls[[as.character(heritability)]][[x_model]][[y_model]]
      } else {
        plt <- plt + plt_ls[[as.character(heritability)]][[x_model]][[y_model]]
      }
    }
  }
}

plt <- plt +
  plot_layout(ncol = length(keep_heritabilities), nrow = 6)
ggplot2::ggsave(
  file.path(plots_dir, sprintf("real_data_artificial_y_sims_%s_errbars.pdf", metrics)), 
  plot = plt, units = "in", width = 12, height = 20
)

keep_heritabilities <- c(0.1, 0.2, 0.4, 0.8)
for (y_model in keep_y_models) {
  for (x_model in keep_x_models) {
    plt <- list()
    for (heritability in keep_heritabilities) {
      cur_plt <- plt_ls[[as.character(heritability)]][[x_model]][[y_model]]
      if (identical(plt, list())) {
        plt <- cur_plt
      } else {
        plt <- plt + cur_plt
      }
    }
    plt <- plt +
      plot_layout(ncol = length(keep_heritabilities), nrow = 1)
    ggplot2::ggsave(
      file.path(plots_dir, sprintf("real_data_artificial_y_sims_%s_%s_%s_errbars.pdf", 
                                   y_model, x_model, metrics)), 
      plot = plt, units = "in", width = 12, height = 3
    )
  }
}

#### Real X, Real Y Simulations ####
fname <- file.path(results_dir, "fmri_augmented_dgp", 
                   "varying_sample_col_n_sample_row_n",
                   paste0("seed", seed), "results.csv")
results <- data.table::fread(fname) %>%
  reformat_results()
ps <- c(100, 250, 500, 1000)

plt_df <- results %>%
  dplyr::mutate(
    rocauc = ifelse(stringr::str_detect(method, "R2F"), r2_rocauc, rocauc),
    prauc = ifelse(stringr::str_detect(method, "R2F"), r2_prauc, prauc)
  ) %>%
  dplyr::select(rep, method, sample_row_n, sample_col_n,
                tidyselect::all_of(metrics)) %>%
  tidyr::pivot_longer(
    cols = tidyselect::all_of(metrics), names_to = "metric"
  )  %>%
  dplyr::group_by(
    method, sample_row_n, sample_col_n, metric
  ) %>%
  dplyr::summarise(mean = mean(value), sd = sd(value), .groups = "keep") %>%
  dplyr::filter(method %in% keep_methods) %>%
  dplyr::mutate(
    method = forcats::fct_recode(
      factor(method, levels = keep_methods), 
      R2F = !!r2f, Permutation = "Permutation_RF", 
      MDI = "MDI_RF", `MDI-oob` = "MDI-oob_RF", TreeSHAP = "TreeSHAP_RF"
    ),
    metric = forcats::fct_recode(
      factor(metric, levels = c("rocauc", "prauc")),
      AUROC = "rocauc", AUPRC = "prauc"
    )
  )

plt <- plt_df %>%
  dplyr::filter(sample_col_n %in% ps) %>%
  dplyr::mutate(
    sample_col_n = factor(paste0("p = ", sample_col_n), 
                          levels = paste0("p = ", ps))
  ) %>%
  ggplot2::ggplot() +
  ggplot2::aes(x = sample_row_n, y = mean, color = method) +
  ggplot2::facet_grid(~ sample_col_n) +
  ggplot2::geom_point() +
  ggplot2::geom_line() +
  ggplot2::labs(
    x = "Number of Samples", y = metrics_name, color = "Method", 
    title = "fMRI"
  ) +
  ggplot2::scale_color_manual(values = manual_color_palette) +
  vthemes::theme_vmodern(size_preset = "medium")

#### Experimental Results ####
manual_color_palette2 <- append(manual_color_palette, "#002e78", after = 1)
sim_names <- c(
  "ccle_prot_linear_dgp", "ccle_prot_lss_dgp",
  "ccle_prot_linear_lss_dgp", "ccle_prot_cart_dgp",
  "tcga_linear_dgp", "tcga_lss_dgp", "tcga_linear_lss_dgp",
  "enhancer_linear_dgp", "enhancer_lss_dgp", "enhancer_linear_lss_dgp"
)
vary_param_names <- c(
  "heritability; sample_row_n", "heritability; sample_row_n", 
  "heritability; sample_row_n", "heritability; n",
  "heritability; sample_row_n", "heritability; sample_row_n", "heritability; sample_row_n",
  "heritability; sample_row_n", "heritability; sample_row_n", "heritability; sample_row_n"
)
keep_methods <- c("R2F_05", "R2F_max", "Permutation_RF", "MDI_RF")
seed <- 331

for (i in 1:length(sim_names)) {
  sim_name <- sim_names[i]
  vary_param_name <- vary_param_names[i]
  vary_param_name_vec <- stringr::str_split(vary_param_name, "; ")[[1]]
  vary_param_name <- paste(vary_param_name_vec, collapse = "_")
  
  fname <- file.path(results_dir, sim_name, paste0("varying_", vary_param_name), 
                     paste0("seed", seed), "results.csv")
  if (!file.exists(fname)) {
    warning(sprintf("Results cannot be found for %s", sim_name))
  }
  results <- data.table::fread(fname) %>%
    reformat_results()
  
  if (!dir.exists(file.path(plots_dir, "experiments_0505"))) {
    dir.create(file.path(plots_dir, "experiments_0505"), recursive = TRUE)
  }
  
  plt <- plot_metrics(results, vary_param_name, vary_param_name_vec, 
                      show_errbars = FALSE, free_y = FALSE, 
                      keep_methods = keep_methods, 
                      `R2F (n/2)` = "R2F_05", `R2F (max)` = "R2F_max") +
    ggplot2::labs(x = "Number of Samples") +
    ggplot2::scale_color_manual(values = manual_color_palette2) +
    vthemes::theme_vmodern(size_preset = "medium")
  ggplot2::ggsave(filename = file.path(plots_dir, "experiments_0505", paste(sim_name, vary_param_name, ".png", sep = "_")), 
                  plot = plt, unit = "in", width = fig_width, height = fig_height)
  
  plt <- plot_metrics(results, vary_param_name, vary_param_name_vec, 
                      show_errbars = FALSE, free_y = TRUE, 
                      keep_methods = keep_methods, 
                      `R2F (n/2)` = "R2F_05", `R2F (max)` = "R2F_max") +
    ggplot2::labs(x = "Number of Samples") +
    ggplot2::scale_color_manual(values = manual_color_palette2) +
    vthemes::theme_vmodern(size_preset = "medium")
  ggplot2::ggsave(filename = file.path(plots_dir, "experiments_0505", paste(sim_name, vary_param_name, "free.png", sep = "_")), 
                  plot = plt, unit = "in", width = fig_width, height = fig_height)
}

#### Experimental Results ####
manual_color_palette3 <- append(manual_color_palette, c("#1650AF", "#002e78"), after = 1)
manual_color_palette3 <- c("#95BEFF", manual_color_palette3)
sim_names <- c(
  "ccle_prot_linear_dgp", "ccle_prot_lss_dgp",
  "ccle_prot_linear_lss_dgp", "ccle_prot_cart_dgp",
  "tcga_linear_dgp", "tcga_lss_dgp", "tcga_linear_lss_dgp",
  "enhancer_linear_dgp", "enhancer_lss_dgp", "enhancer_linear_lss_dgp"
)
vary_param_names <- c(
  "heritability; sample_row_n", "heritability; sample_row_n", 
  "heritability; sample_row_n", "heritability; n",
  "heritability; sample_row_n", "heritability; sample_row_n", "heritability; sample_row_n",
  "heritability; sample_row_n", "heritability; sample_row_n", "heritability; sample_row_n"
)
keep_methods <- c("R2F_05", "sswR2F_05_RF", "R2F_max", "sswR2F_max_RF", "Permutation_RF", "MDI_RF", "T-Test")
seed <- 331

for (i in 1:length(sim_names)) {
  sim_name <- sim_names[i]
  vary_param_name <- vary_param_names[i]
  vary_param_name_vec <- stringr::str_split(vary_param_name, "; ")[[1]]
  vary_param_name <- paste(vary_param_name_vec, collapse = "_")
  
  fname <- file.path(results_dir, sim_name, paste0("varying_", vary_param_name), 
                     paste0("seed", seed), "results.csv")
  if (!file.exists(fname)) {
    warning(sprintf("Results cannot be found for %s", sim_name))
  }
  results <- data.table::fread(fname) %>%
    reformat_results()
  
  if (!dir.exists(file.path(plots_dir, "experiments_0505b"))) {
    dir.create(file.path(plots_dir, "experiments_0505b"), recursive = TRUE)
  }
  
  plt <- plot_metrics(results, vary_param_name, vary_param_name_vec, 
                      show_errbars = FALSE, free_y = FALSE, 
                      keep_methods = keep_methods, 
                      `R2F (n/2)` = "R2F_05", `R2F (max)` = "R2F_max",
                      `R2F (n/2; ssw)` = "sswR2F_05_RF", 
                      `R2F (max; ssw)` = "sswR2F_max_RF") +
    ggplot2::labs(x = "Number of Samples") +
    ggplot2::scale_color_manual(values = manual_color_palette3) +
    vthemes::theme_vmodern(size_preset = "medium")
  ggplot2::ggsave(filename = file.path(plots_dir, "experiments_0505b", paste(sim_name, vary_param_name, ".png", sep = "_")), 
                  plot = plt, unit = "in", width = fig_width, height = fig_height)
  
  plt <- plot_metrics(results, vary_param_name, vary_param_name_vec, 
                      show_errbars = FALSE, free_y = TRUE, 
                      keep_methods = keep_methods, 
                      `R2F (n/2)` = "R2F_05", `R2F (max)` = "R2F_max",
                      `R2F (n/2; ssw)` = "sswR2F_05_RF", 
                      `R2F (max; ssw)` = "sswR2F_max_RF") +
    ggplot2::labs(x = "Number of Samples") +
    ggplot2::scale_color_manual(values = manual_color_palette3) +
    vthemes::theme_vmodern(size_preset = "medium")
  ggplot2::ggsave(filename = file.path(plots_dir, "experiments_0505b", paste(sim_name, vary_param_name, "free.png", sep = "_")), 
                  plot = plt, unit = "in", width = fig_width, height = fig_height)
}

#### Enhancer Static Heatmaps ####
sim_name <- "enhancer_static_linear_dgp"
vary_param_name <- "heritability; sample_row_n"
keep_methods <- c("R2F_05", "R2F_max", "Permutation_RF", "MDI_RF", "T-Test_OLS")
seed <- 331

vary_param_name_vec <- stringr::str_split(vary_param_name, "; ")[[1]]
vary_param_name <- paste(vary_param_name_vec, collapse = "_")

fname <- file.path(results_dir, sim_name, paste0("varying_", vary_param_name), 
                   paste0("seed", seed), "results.csv")
results <- data.table::fread(fname) %>%
  reformat_results()

metrics_plt <- plot_metrics(results, vary_param_name, vary_param_name_vec, 
                            keep_methods = keep_methods, show_errbars = FALSE,
                            `R2F (n/2)` = "R2F_05", `R2F (max)` = "R2F_max",
                            `R2F (n/2; ssw)` = "sswR2F_05_RF", 
                            `R2F (max; ssw)` = "sswR2F_max_RF")

plt_ls <- plot_ranking_heatmap(results, vary_param_name, vary_param_name_vec, 
                               keep_vars = 0:100, keep_methods = keep_methods, 
                               `R2F (n/2)` = "R2F_05", `R2F (max)` = "R2F_max")
plt <- plt_ls$`0.4`$`500` +
  ggplot2::labs(title = NULL)
if (!dir.exists(file.path(plots_dir, "other_explorations_0505"))) {
  dir.create(file.path(plots_dir, "other_explorations_0505"), recursive = TRUE)
}
ggplot2::ggsave(filename = file.path(plots_dir, "other_explorations_0505", paste(sim_name, vary_param_name, "heatmap.png", sep = "_")), 
                plot = plt + vthemes::theme_vmodern(size_preset = "large"),
                unit = "in", width = 12, height = 10)

signal_features <- c('wt_ZLD', 'gt2', 'hb1', 'kr1', 'twi1')
X <- data.table::fread("data/X_enhancer_uncorrelated.csv") %>%
  dplyr::relocate(tidyselect::all_of(signal_features), before = 1) %>%
  tibble::as_tibble()

rep <- 1
# method <- "R2F_max"
method <- "R2F_05"
heritability <- 0.8
sample_row_n <- 500

results_rep <- results %>%
  dplyr::filter(heritability == !!heritability, sample_row_n == !!sample_row_n, 
                rep == !!rep, method == !!method) %>%
  tidyr::unnest(fi_scores) %>%
  dplyr::select(rep, sample_row_n, heritability, method, var, importance, r2, support) %>%
  dplyr::arrange(-r2)

for (i in 1:nrow(results_rep)) {
  if (all(0:4 %in% results_rep$var[1:i])) {
    top_vars <- results_rep$var[1:i]
    print(i)
    break
  }
}

vdocs::plot_cor_heatmap(X[, top_vars + 1],
                        xytext_colors = ifelse(colnames(X[, top_vars + 1]) %in% signal_features, "signal", "non-signal"),
                        x_text_angle = TRUE) +
  ggplot2::labs(x = "Feature", y = "Feature", fill = "Correlation", color = "", title = method)
vdocs::plot_cor_heatmap(as.data.frame(X), clust = FALSE, 
                        xytext_colors = ifelse(1:ncol(X) %in% (top_vars + 1), "Top", "Bottom"),
                        x_text_angle = TRUE) +
  ggplot2::labs(x = "Feature", y = "Feature", fill = "Correlation", color = "", title = method)
vdocs::plot_cor_heatmap(as.data.frame(X), clust = TRUE, 
                        xytext_colors = ifelse(1:ncol(X) %in% (top_vars + 1), "Top", "Bottom"),
                        x_text_angle = TRUE) +
  ggplot2::labs(x = "Feature", y = "Feature", fill = "Correlation", color = "", title = method)





#### Hyperparameter Stability Results ####
sim_name <- "ccle_prot_linear_hyperparams_stability_dgp"
vary_param_name <- "heritability; sample_row_n"
keep_methods <- c("R2F_05", "R2F_max", "Permutation_RF", "MDI_RF", "T-Test_OLS")
seed <- 331

vary_param_name_vec <- stringr::str_split(vary_param_name, "; ")[[1]]
vary_param_name <- paste(vary_param_name_vec, collapse = "_")

fname <- file.path(results_dir, sim_name, paste0("varying_", vary_param_name), 
                   paste0("seed", seed), "results.csv")
results <- data.table::fread(fname) %>%
  reformat_results()

# method_orig <- "R2F+_05_RF"
# method_new <- "R2F_05"
# method_orig <- "R2F+_max_RF"
# method_new <- "R2F_max"
method_orig <- "MDI_RF"
method_new <- "MDI_RF"

r2f_results <- results %>%
  dplyr::filter(sample_row_n == 250, heritability == 0.2,
                startsWith(method, method_orig)) %>%
  dplyr::mutate(
    method = method_new,
    min_samples_leaf = as.factor(min_samples_leaf),
    fi_scores = mapply(name = fi, scores_df = fi_scores,
                       function(name, scores_df) {
                         if (stringr::str_detect(name, "R2F")) {
                           scores_df <- scores_df %>% 
                             dplyr::mutate(ranking = rank(-r2),
                                           importance = r2)
                         } else if (name %in% c("T-Test")) {
                           scores_df <- scores_df %>% 
                             dplyr::mutate(ranking = rank(importance))
                         } else {
                           scores_df <- scores_df %>%
                             dplyr::mutate(ranking = rank(-importance))
                         }
                         return(scores_df)
                       }, SIMPLIFY = FALSE)
  ) %>%
  tidyr::unnest(fi_scores) %>%
  dplyr::filter(var <= 50) %>%
  dplyr::mutate(ranking = ranking %/% 50 * 50)

ggplot2::ggplot(r2f_results) +
  ggplot2::aes(x = var, y = min_samples_leaf, fill = ranking) +
  ggplot2::facet_wrap(~ rep) +
  ggplot2::geom_tile() +
  vthemes::scale_fill_vmodern() +
  vthemes::theme_vmodern()

ggplot2::ggplot(r2f_results) +
  ggplot2::aes(x = var, y = min_samples_leaf, fill = r2) +
  ggplot2::facet_wrap(~ rep) +
  ggplot2::geom_tile() +
  vthemes::scale_fill_vmodern() +
  vthemes::theme_vmodern()


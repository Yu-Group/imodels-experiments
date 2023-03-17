library(magrittr)


# reformat results
reformat_results <- function(results, prediction = FALSE) {
  if (!prediction) {
    results_grouped <- results %>%
      dplyr::group_by(index) %>%
      tidyr::nest(fi_scores = var:(tidyselect::last_col())) %>%
      dplyr::ungroup() %>%
      dplyr::select(-index) %>%
      # join fi+model to get method column
      tidyr::unite(col = "method", fi, model, na.rm = TRUE, remove = FALSE) %>%
      dplyr::mutate(
        # get rid of duplicate RF in r2f method name
        method = ifelse(stringr::str_detect(method, "^r2f.*RF$"),
                        stringr::str_remove(method, "\\_RF$"), method),
        # unnest columns
        prediction_score = purrr::map_dbl(
          fi_scores,
          function(x) {
            ifelse("prediction_score" %in% colnames(x), x$prediction_score[[1]], NA)
          }
        ),
        tauAP = purrr::map_dbl(
          fi_scores,
          function(x) {
            ifelse("tauAP" %in% colnames(x), x$tauAP[[1]], NA)
          }
        ),
        RBO = purrr::map_dbl(
          fi_scores,
          function(x) {
            ifelse("RBO" %in% colnames(x), x$RBO[[1]], NA)
          }
        )
      )
  } else {
    results_grouped <- results %>%
      dplyr::group_by(index) %>%
      tidyr::nest(predictions = sample_id:(tidyselect::last_col())) %>%
      dplyr::ungroup() %>%
      dplyr::select(-index) %>%
      dplyr::mutate(
        # get rid of duplicate RF in r2f method name
        method = ifelse(stringr::str_detect(model, "^r2f.*RF$"),
                        stringr::str_remove(model, "\\_RF$"), model)
      )
  }
  return(results_grouped)
}

# plot metrics (mean value across repetitions with error bars)
plot_metrics <- function(results, 
                         metric = c("rocauc", "prauc"),# "tpr",
                                    #"median_signal_rank", "max_signal_rank"), 
                         x_str, facet_str, linetype_str = NULL,
                         point_size = 1, line_size = 1, errbar_width = 0,
                         alpha = 0.5, inside_legend = FALSE,
                         manual_color_palette = NULL,
                         show_methods = NULL,
                         method_labels = ggplot2::waiver(),
                         alpha_values = NULL,
                         legend_position = NULL,
                         custom_theme = vthemes::theme_vmodern(size_preset = "medium")) {
  if (is.null(show_methods)) {
    show_methods <- sort(unique(results$method))
  }
  metric_names <- metric
  plt_df <- results %>%
    dplyr::select(rep, method, 
                  tidyselect::all_of(c(metric, x_str, facet_str, linetype_str))) %>%
    tidyr::pivot_longer(
      cols = tidyselect::all_of(metric), names_to = "metric"
    )  %>%
    dplyr::group_by(
      method, metric, dplyr::across(tidyselect::all_of(c(x_str, facet_str, linetype_str)))
    ) %>%
    dplyr::summarise(mean = mean(value), 
                     sd = sd(value) / sqrt(dplyr::n()), 
                     .groups = "keep") %>%
    dplyr::filter(method %in% show_methods) %>%
    dplyr::mutate(
      method = factor(method, levels = show_methods),
      metric = forcats::fct_recode(
        factor(metric, levels = metric_names),
        AUROC = "rocauc", AUPRC = "prauc"
      )
    )
  
  if (is.null(linetype_str)) {
    plt <- ggplot2::ggplot(plt_df) +
      ggplot2::geom_point(
        ggplot2::aes(x = .data[[x_str]], y = mean, 
                     color = method, alpha = method, group = method),
        size = point_size
      ) +
      ggplot2::geom_line(
        ggplot2::aes(x = .data[[x_str]], y = mean, 
                     color = method, alpha = method, group = method),
        size = line_size
      ) +
      ggplot2::geom_errorbar(
        ggplot2::aes(x = .data[[x_str]], ymin = mean - sd, ymax = mean + sd,
                     color = method, alpha = method, group = method),
        width = errbar_width, show_guide = FALSE
      )
  } else {
    plt <- ggplot2::ggplot(plt_df) +
      ggplot2::geom_point(
        ggplot2::aes(x = .data[[x_str]], y = mean, 
                     color = method, alpha = method, group = interaction(method, !!rlang::sym(linetype_str))),
        size = point_size
      ) +
      ggplot2::geom_line(
        ggplot2::aes(x = .data[[x_str]], y = mean, 
                     color = method, alpha = method, group = interaction(method, !!rlang::sym(linetype_str)),
                     linetype = !!rlang::sym(linetype_str)),
        size = line_size
      ) +
      ggplot2::geom_errorbar(
        ggplot2::aes(x = .data[[x_str]], ymin = mean - sd, ymax = mean + sd,
                     color = method, alpha = method, group = interaction(method, !!rlang::sym(linetype_str))),
        width = errbar_width, show_guide = FALSE
      ) 
  }
  if (!is.null(manual_color_palette)) {
    if (is.null(alpha_values)) {
      alpha_values <- c(1, rep(alpha, length(method_labels) - 1))
    }
    plt <- plt + 
      ggplot2::scale_color_manual(
        values = manual_color_palette, labels = method_labels
      ) +
      ggplot2::scale_alpha_manual(
        values = alpha_values, labels = method_labels
      )
  }
  if (!is.null(custom_theme)) {
    plt <- plt + custom_theme
  }
  
  if (!is.null(facet_str)) {
    plt <- plt +
      ggplot2::facet_grid(reformulate(facet_str, "metric"), scales = "free")
  } else if (length(metric) > 1) {
    plt <- plt +
      ggplot2::facet_wrap(~ metric, scales = "free")
  }
  
  if (inside_legend) {
    if (is.null(legend_position)) {
      legend_position <- c(0.75, 0.3)
    }
    plt <- plt +
      ggplot2::theme(
        legend.title = ggplot2::element_blank(),
        legend.position = legend_position,
        legend.background = ggplot2::element_rect(
          color = "slategray", size = 0.1
        )
      )
  }
  
  return(plt)
}

# plot restricted metrics (mean value across repetitions with error bars)
plot_restricted_metrics <- function(results, metric = c("rocauc", "prauc"), 
                                    x_str, facet_str, 
                                    quantiles = c(0.1, 0.2, 0.3, 0.4),
                                    point_size = 1, line_size = 1, errbar_width = 0,
                                    alpha = 0.5, inside_legend = FALSE,
                                    manual_color_palette = NULL,
                                    show_methods = NULL,
                                    method_labels = ggplot2::waiver(),
                                    custom_theme = vthemes::theme_vmodern(size_preset = "medium")) {
  if (is.null(show_methods)) {
    show_methods <- sort(unique(results$method))
  }
  results <- results %>%
    dplyr::select(rep, method, fi_scores,
                  tidyselect::all_of(c(x_str, facet_str))) %>%
    dplyr::mutate(
      vars_ordered = purrr::map(
        fi_scores, 
        function(fi_df) {
          fi_df %>% 
            dplyr::filter(!is.na(cor_with_signal)) %>%
            dplyr::arrange(-cor_with_signal) %>%
            dplyr::pull(var)
        }
      )
    )
  
  plt_df_ls <- list()
  for (q in quantiles) {
    plt_df_ls[[as.character(q)]] <- results %>%
      dplyr::mutate(
        restricted_metrics = purrr::map2_dfr(
          fi_scores, vars_ordered,
          function(fi_df, ignore_vars) {
            ignore_vars <- ignore_vars[1:round(q * length(ignore_vars))]
            auroc_r <- fi_df %>%
              dplyr::filter(!(var %in% ignore_vars)) %>%
              yardstick::roc_auc(
                truth = factor(true_support, levels = c("1", "0")), importance,
                event_level = "first"
              ) %>%
              dplyr::pull(.estimate)
            auprc_r <- fi_df %>%
              dplyr::filter(!(var %in% ignore_vars)) %>%
              yardstick::pr_auc(
                truth = factor(true_support, levels = c("1", "0")), importance,
                event_level = "first"
              ) %>%
              dplyr::pull(.estimate)
            return(data.frame(restricted_auroc = auroc_r,
                              restricted_auprc = auprc_r))
          }
        )
      ) %>%
      tidyr::unnest(restricted_metrics) %>%
      tidyr::pivot_longer(
        cols = c(restricted_auroc, restricted_auprc), names_to = "metric"
      ) %>%
      dplyr::group_by(
        method, metric, dplyr::across(tidyselect::all_of(c(x_str, facet_str)))
      ) %>%
      dplyr::summarise(mean = mean(value), 
                       sd = sd(value) / sqrt(dplyr::n()), 
                       .groups = "keep") %>%
      dplyr::ungroup() %>%
      dplyr::filter(method %in% show_methods) %>%
      dplyr::mutate(
        method = factor(method, levels = show_methods),
        metric = forcats::fct_recode(
          factor(metric, levels = c("restricted_auroc", "restricted_auprc")),
          `Restricted AUROC` = "restricted_auroc", 
          `Restricted AUPRC` = "restricted_auprc"
        )
      )
  }
  
  plt_df <- purrr::map_dfr(plt_df_ls, ~.x, .id = ".threshold") %>%
    dplyr::mutate(.threshold = as.numeric(.threshold))
  
  plt <- ggplot2::ggplot(plt_df) +
    ggplot2::geom_point(
      ggplot2::aes(x = .data[[x_str]], y = mean, 
                   color = method, alpha = method, group = method),
      size = point_size
    ) +
    ggplot2::geom_line(
      ggplot2::aes(x = .data[[x_str]], y = mean, 
                   color = method, alpha = method, group = method),
      size = line_size
    ) +
    ggplot2::geom_errorbar(
      ggplot2::aes(x = .data[[x_str]], ymin = mean - sd, ymax = mean + sd,
                   color = method, alpha = method, group = method),
      width = errbar_width, show_guide = FALSE
    ) 
  if (!is.null(manual_color_palette)) {
    plt <- plt + 
      ggplot2::scale_color_manual(
        values = manual_color_palette, labels = method_labels
      ) +
      ggplot2::scale_alpha_manual(
        values = c(1, rep(alpha, length(method_labels) - 1)), 
        labels = method_labels
      )
  }
  if (!is.null(custom_theme)) {
    plt <- plt + custom_theme
  }
  
  if (!is.null(facet_str)) {
    formula <- sprintf("metric + .threshold ~ %s", paste0(facet_str, collapse = " + "))
    plt <- plt +
      ggplot2::facet_grid(as.formula(formula))
  } else {
    plt <- plt +
      ggplot2::facet_wrap(.threshold ~ metric, scales = "free")
  }
  
  if (inside_legend) {
    plt <- plt +
      ggplot2::theme(
        legend.title = ggplot2::element_blank(),
        legend.position = c(0.75, 0.3),
        legend.background = ggplot2::element_rect(
          color = "slategray", size = 0.1
        )
      )
  }
  
  return(plt)
}

# plot true positive rate across # positives
plot_tpr <- function(results, facet_vars, point_size = 0.85,
                     manual_color_palette = NULL,
                     show_methods = NULL,
                     method_labels = ggplot2::waiver(),
                     custom_theme = vthemes::theme_vmodern(size_preset = "medium")) {
  if (is.null(results)) {
    return(NULL)
  }
  
  if (is.null(show_methods)) {
    show_methods <- sort(unique(results$method))
  }
  
  facet_names <- names(facet_vars)
  names(facet_vars) <- NULL
  
  plt_df <- results %>%
    dplyr::mutate(
      fi_scores = mapply(name = fi, scores_df = fi_scores,
                         function(name, scores_df) {
                           scores_df <- scores_df %>%
                             dplyr::mutate(
                               ranking = rank(-importance, 
                                              ties.method = "random")
                             ) %>%
                             dplyr::arrange(ranking) %>%
                             dplyr::mutate(
                               .tp = cumsum(true_support) / sum(true_support)
                             )
                           return(scores_df)
                         }, SIMPLIFY = FALSE)
    ) %>%
    tidyr::unnest(fi_scores) %>%
    dplyr::select(tidyselect::all_of(facet_vars), rep, method, ranking, .tp) %>%
    dplyr::group_by(
      dplyr::across(tidyselect::all_of(facet_vars)), method, ranking
    ) %>%
    dplyr::summarise(.tp = mean(.tp), .groups = "keep") %>%
    dplyr::mutate(method = factor(method, levels = show_methods)) %>%
    dplyr::ungroup()
  
  if (!is.null(facet_names)) {
    for (i in 1:length(facet_vars)) {
      facet_var <- facet_vars[i]
      facet_name <- facet_names[i]
      if (facet_name != "") {
        plt_df <- plt_df %>%
          dplyr::mutate(dplyr::across(
            tidyselect::all_of(facet_var),
            ~factor(sprintf("%s = %s", facet_name, .x), 
                    levels = sprintf("%s = %s", facet_name, sort(unique(.x))))
          ))
      }
    }
  }
  
  if (length(facet_vars) == 1) {
    plt <- ggplot2::ggplot(plt_df) +
      ggplot2::aes(x = ranking, y = .tp, color = method) +
      ggplot2::geom_line(size = point_size) +
      ggplot2::facet_wrap(reformulate(facet_vars))
  } else {
    plt <- ggplot2::ggplot(plt_df) +
      ggplot2::aes(x = ranking, y = .tp, color = method) +
      ggplot2::geom_line(size = point_size) +
      ggplot2::facet_grid(reformulate(facet_vars[1], facet_vars[2]))
  }
  plt <- plt +
    ggplot2::labs(x = "Top n", y = "True Positive Rate",
                  fill = "Method", color = "Method")
  if (!is.null(manual_color_palette)) {
    plt <- plt +
      ggplot2::scale_color_manual(
        values = manual_color_palette, labels = method_labels
      )
  }
  if (!is.null(custom_theme)) {
    plt <- plt + custom_theme
  }
  
  return(plt)
}

# plot stability results
plot_perturbation_stability <- function(results,
                                        facet_rows = "heritability_name",
                                        facet_cols = "rho_name",
                                        param_name = NULL,
                                        facet_row_names = "Avg Rank (PVE = %s)",
                                        group_fun = NULL,
                                        descending_methods = NULL,
                                        manual_color_palette = NULL,
                                        show_methods = NULL,
                                        method_labels = ggplot2::waiver(),
                                        plot_types = c("boxplot", "errbar"),
                                        save_dir = ".",
                                        save_filename = NULL,
                                        fig_height = 11,
                                        fig_width = 11,
                                        ...) {
  plot_types <- match.arg(plot_types, several.ok = TRUE)
  
  my_theme <- vthemes::theme_vmodern(
    size_preset = "medium", bg_color = "white", grid_color = "white",
    axis.title = ggplot2::element_text(size = 12, face = "plain"),
    legend.title = ggplot2::element_blank(),
    legend.text = ggplot2::element_text(size = 9),
    legend.text.align = 0,
    plot.title = ggplot2::element_blank()
  )
  
  if (is.null(group_fun)) {
    group_fun <- function(var, sig_ids, cnsig_ids) {
      dplyr::case_when(
        var %in% sig_ids ~ "Sig",
        var %in% cnsig_ids ~ "C-NSig",
        TRUE ~ "NSig"
      ) %>%
        factor(levels = c("Sig", "C-NSig", "NSig"))
    }
  }
  
  if (!is.null(show_methods)) {
    results <- results %>%
      dplyr::filter(fi %in% show_methods)
    if (!identical(method_labels, ggplot2::waiver())) {
      method_names <- show_methods
      names(method_names) <- method_labels
      results$fi <- do.call(forcats::fct_recode, 
                            args = c(list(results$fi), as.list(method_names)))
      results$fi <- factor(results$fi, levels = method_labels)
      method_labels <- ggplot2::waiver()
    }
  }
  
  if (!is.null(descending_methods)) {
    results <- results %>%
      dplyr::mutate(
        importance = ifelse(fi %in% descending_methods, -importance, importance)
      )
  }
  
  rankings <- results %>%
    dplyr::group_by(
      rep, fi,
      dplyr::across(tidyselect::all_of(c(facet_rows, facet_cols, param_name))),
    ) %>%
    dplyr::mutate(
      rank = rank(-importance),
      group = group_fun(var, ...)
    ) %>%
    dplyr::ungroup()
  
  agg_rankings <- rankings %>%
    dplyr::group_by(
      rep, fi, group, 
      dplyr::across(tidyselect::all_of(c(facet_rows, facet_cols, param_name)))
    ) %>%
    dplyr::summarise(
      avgrank = mean(rank),
      .groups = "keep"
    ) %>%
    dplyr::ungroup()
  
  ymin <- min(agg_rankings$avgrank)
  ymax <- max(agg_rankings$avgrank)
  
  for (type in plot_types) {
    plt_ls <- list()
    for (val in unique(agg_rankings[[facet_rows]])) {
      if (identical(type, "boxplot")) {
        plt <- agg_rankings %>%
          dplyr::filter(.data[[facet_rows]] == val) %>%
          ggplot2::ggplot() +
          ggplot2::aes(x = group, y = avgrank, color = fi) +
          ggplot2::geom_boxplot()
      } else if (identical(type, "errbar")) {
        plt <- agg_rankings %>%
          dplyr::filter(.data[[facet_rows]] == val) %>%
          dplyr::group_by(
            fi, group, dplyr::across(tidyselect::all_of(facet_cols))
          ) %>%
          dplyr::summarise(
            .mean = mean(avgrank),
            .sd = sd(avgrank),
            .groups = "keep"
          ) %>%
          dplyr::ungroup() %>%
          ggplot2::ggplot() +
          # ggplot2::geom_point(
          #   ggplot2::aes(x = group, y = .mean, color = fi, group = fi),
          #   position = ggplot2::position_dodge2(width = 0.8, padding = 0.8)
          # ) +
          ggplot2::geom_errorbar(
            ggplot2::aes(x = group, ymin = .mean - .sd, ymax = .mean + .sd, 
                         color = fi, group = fi),
            position = ggplot2::position_dodge2(width = 0, padding = 0.5), 
            width = 0.5, show_guide = FALSE
          )
      }
      plt <- plt +
        ggplot2::coord_cartesian(ylim = c(ymin, ymax)) +
        ggplot2::facet_grid(~ .data[[facet_cols]], labeller = ggplot2::label_parsed) +
        my_theme +
        ggplot2::theme(
          panel.grid.major = ggplot2::element_line(colour = "#d9d9d9"),
          panel.grid.major.x = ggplot2::element_blank(),
          panel.grid.minor.x = ggplot2::element_blank(),
          axis.line.y = ggplot2::element_blank(),
          axis.ticks.y = ggplot2::element_blank(),
          legend.position = "right"
        ) +
        ggplot2::labs(x = "Feature Groups", y = sprintf(facet_row_names, val))
      if (!is.null(manual_color_palette)) {
        plt <- plt +
          ggplot2::scale_color_manual(values = manual_color_palette,
                                      labels = method_labels)
      }
      if (length(plt_ls) != 0) {
        plt <- plt + 
          ggplot2::theme(strip.text = ggplot2::element_blank())
      }
      plt_ls[[as.character(val)]] <- plt
    }
    agg_plt <- patchwork::wrap_plots(plt_ls) +
      patchwork::plot_layout(ncol = 1, guides = "collect")
    if (!is.null(save_filename)) {
      ggplot2::ggsave(
        filename = file.path(save_dir, 
                             sprintf("%s_%s_aggregated.pdf", save_filename, type)), 
        plot = agg_plt, units = "in", width = fig_width, height = fig_height
      )
    }
  }
  
  unagg_plt <- NULL
  if (!is.null(param_name)) {
    plt_ls <- list()
    for (val in unique(agg_rankings[[facet_rows]])) {
      plt <- agg_rankings %>%
        dplyr::filter(.data[[facet_rows]] == val) %>%
        ggplot2::ggplot() +
        ggplot2::aes(x = .data[[param_name]], y = avgrank, color = fi) +
        ggplot2::geom_boxplot() +
        ggplot2::coord_cartesian(ylim = c(ymin, ymax)) +
        ggplot2::facet_grid(
          reformulate(c(facet_cols, "group"), "fi"), 
          labeller = ggplot2::label_parsed
        ) +
        my_theme +
        ggplot2::theme(
          panel.grid.major = ggplot2::element_line(colour = "#d9d9d9"),
          panel.grid.major.x = ggplot2::element_blank(),
          panel.grid.minor.x = ggplot2::element_blank(),
          axis.line.y = ggplot2::element_blank(),
          axis.ticks.y = ggplot2::element_blank(),
          legend.position = "none"
        ) +
        ggplot2::labs(x = param_name, y = sprintf(facet_row_names, val))
      if (!is.null(manual_color_palette)) {
        plt <- plt +
          ggplot2::scale_color_manual(values = manual_color_palette,
                                      labels = method_labels)
      }
      plt_ls[[as.character(val)]] <- plt
    }
    unagg_plt <- patchwork::wrap_plots(plt_ls) +
      patchwork::plot_layout(guides = "collect")
    
    if (!is.null(save_filename)) {
      ggplot2::ggsave(
        filename = file.path(save_dir,
                             sprintf("%s_unaggregated.pdf", save_filename)), 
        plot = unagg_plt, units = "in", width = fig_width, height = fig_height
      )
    }
  }
  return(list(agg = agg_plt, unagg = unagg_plt))
}


plot_top_stability <- function(results,
                               group_id = NULL,
                               top_r = 10,
                               show_max_features = 5,
                               varnames = NULL,
                               base_method = "MDI+_ridge",
                               return_df = FALSE,
                               descending_methods = NULL,
                               manual_color_palette = NULL,
                               show_methods = NULL,
                               method_labels = ggplot2::waiver(),
                               ...) {
  
  plt_ls <- list()
  if (!is.null(show_methods)) {
    results <- results %>%
      dplyr::filter(fi %in% show_methods)
    if (!identical(method_labels, ggplot2::waiver())) {
      method_names <- show_methods
      names(method_names) <- method_labels
      results$fi <- do.call(forcats::fct_recode, 
                            args = c(list(results$fi), as.list(method_names)))
      results$fi <- factor(results$fi, levels = method_labels)
      method_labels <- ggplot2::waiver()
    }
  }
  
  if (!is.null(descending_methods)) {
    results <- results %>%
      dplyr::mutate(
        importance = ifelse(fi %in% descending_methods, -importance, importance)
      )
  }
  
  if (is.null(varnames)) {
    varnames <- 0:(length(unique(results$var)) - 1)
  }
  varnames_df <- data.frame(var = 0:(length(varnames) - 1), Feature = varnames)
  
  rankings <- results %>%
    dplyr::group_by(
      rep, fi, dplyr::across(tidyselect::all_of(group_id))
    ) %>%
    dplyr::mutate(
      rank = rank(-importance),
      in_top_r = importance >= sort(importance, decreasing = TRUE)[top_r]
    ) %>%
    dplyr::ungroup()
  
  stability_df <- rankings %>%
    dplyr::group_by(
      fi, var, dplyr::across(tidyselect::all_of(group_id))
    ) %>%
    dplyr::summarise(
      stability_score = mean(in_top_r),
      .groups = "keep"
    ) %>%
    dplyr::ungroup()
  
  n_nonzero_stability_df <- stability_df %>%
    dplyr::group_by(fi, dplyr::across(tidyselect::all_of(group_id))) %>%
    dplyr::summarise(
      n_features = sum(stability_score > 0),
      .groups = "keep"
    ) %>%
    dplyr::ungroup()
  
  if (!is.null(group_id)) {
    order_groups <- n_nonzero_stability_df %>%
      dplyr::group_by(dplyr::across(tidyselect::all_of(group_id))) %>%
      dplyr::summarise(
        order = min(n_features)
      ) %>%
      dplyr::arrange(order) %>%
      dplyr::pull(tidyselect::all_of(group_id)) %>%
      unique()
    
    n_nonzero_stability_df <- n_nonzero_stability_df %>%
      dplyr::mutate(
        dplyr::across(
          tidyselect::all_of(group_id), ~factor(.x, levels = order_groups)
        )
      )
    
    ytext_label_colors <- n_nonzero_stability_df %>%
      dplyr::group_by(dplyr::across(tidyselect::all_of(group_id))) %>%
      dplyr::summarise(
        is_best_method = n_features[fi == base_method] == min(n_features)
      ) %>%
      dplyr::mutate(
        color = ifelse(is_best_method, "black", "#4a86e8")
      ) %>%
      dplyr::arrange(tidyselect::all_of(group_id)) %>%
      dplyr::pull(color)
    
    plt <- vdocs::plot_horizontal_dotplot(
      n_nonzero_stability_df, 
      x_str = "n_features", y_str = group_id, color_str = "fi", 
      theme_options = list(size_preset = "xlarge")
    ) +
      ggplot2::labs(
        x = sprintf("Number of Distinct Features in Top 10 Across %s RF Fits", 
                    length(unique(results$rep))),
        color = "Method"
      ) +
      ggplot2::scale_y_discrete(limits = rev) +
      ggplot2::theme(
        axis.text.y = ggplot2::element_text(color = rev(ytext_label_colors))
      )
    
    if (!is.null(manual_color_palette)) {
      plt <- plt +
        ggplot2::scale_color_manual(values = manual_color_palette,
                                    labels = method_labels,
                                    guide = ggplot2::guide_legend(reverse = TRUE))
    }
    
    plt_ls[["Summary"]] <- plt
    
  } else {
    plt <- stability_df %>%
      dplyr::left_join(y = varnames_df, by = "var") %>%
      dplyr::filter(stability_score > 0) %>%
      dplyr::left_join(y = n_nonzero_stability_df, by = "fi") %>%
      dplyr::mutate(fi = sprintf("%s (# features = %s)", fi, n_features)) %>%
      ggplot2::ggplot() +
      ggplot2::aes(
        x = reorder(Feature, -stability_score), y = stability_score
      ) +
      ggplot2::facet_wrap(~ fi, scales = "free_x", ncol = 1) +
      ggplot2::geom_bar(stat = "identity") +
      ggplot2::labs(
        x = "Gene", y = sprintf("Proportion of RF Fits in Top %s", top_r)
      ) +
      vthemes::theme_vmodern(x_text_angle = TRUE, size_preset = "medium")
    
    if (!is.null(manual_color_palette)) {
      plt <- plt +
        ggplot2::scale_color_manual(values = manual_color_palette,
                                    labels = method_labels,
                                    guide = ggplot2::guide_legend(reverse = TRUE))
    }
    plt_ls[["Non-zero Stability Scores Per Method"]] <- plt
  }
  
  if (is.null(group_id)) {
    rankings <- rankings %>%
      dplyr::mutate(
        .group = "Stability of Top Features"
      )
    group_id <- ".group"
  }
  
  for (group in unique(rankings[[group_id]])) {
    plt <- rankings %>%
      dplyr::filter(.data[[group_id]] == group) %>%
      dplyr::group_by(fi, var) %>%
      dplyr::summarise(
        mean_rank = mean(rank),
        median_rank = median(rank),
        `SD(Feature Rank)` = sd(rank),
        .groups = "keep"
      ) %>%
      dplyr::group_by(fi) %>%
      dplyr::mutate(
        agg_feature_rank = rank(mean_rank, ties.method = "random")
      ) %>%
      dplyr::filter(agg_feature_rank <= show_max_features) %>%
      dplyr::mutate(`Average Feature Rank` = as.factor(agg_feature_rank)) %>%
      dplyr::left_join(y = varnames_df, by = "var") %>%
      dplyr::rename("Method" = "fi") %>%
      dplyr::ungroup() %>%
      ggplot2::ggplot() +
      ggplot2::aes(
        x = `Average Feature Rank`, y = `SD(Feature Rank)`, fill = Method, label = Feature
      ) +
      ggplot2::geom_bar(
        stat = "identity", position = "dodge"
      ) +
      ggplot2::labs(title = group) +
      vthemes::scale_fill_vmodern(discrete = TRUE) +
      vthemes::theme_vmodern()
    
    if (!is.null(manual_color_palette)) {
      plt <- plt +
        ggplot2::scale_fill_manual(values = manual_color_palette,
                                   labels = method_labels)
    }
    plt_ls[[group]] <- plt
  }
  
  if (return_df) {
    return(list(plot_ls = plt_ls, rankings = rankings, stability = stability_df))
  } else {
    return(plt_ls)
  }
}



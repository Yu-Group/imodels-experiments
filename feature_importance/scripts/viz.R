library(magrittr)


# reformat results
reformat_results <- function(results) {
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
                      stringr::str_remove(method, "\\_RF$"), method)
    )
  return(results_grouped)
}

# plot metrics (mean value across repetitions with error bars)
plot_metrics <- function(results, metric = c("rocauc", "prauc"), 
                         x_str, facet_str,
                         point_size = 1, line_size = 1, errbar_width = 0,
                         alpha = 0.5, inside_legend = FALSE,
                         manual_color_palette = NULL,
                         show_methods = NULL,
                         method_labels = ggplot2::waiver(),
                         custom_theme = vthemes::theme_vmodern(size_preset = "medium")) {
  if (is.null(show_methods)) {
    show_methods <- sort(unique(results$method))
  }
  plt_df <- results %>%
    dplyr::select(rep, method, 
                  tidyselect::all_of(c(metric, x_str, facet_str))) %>%
    tidyr::pivot_longer(
      cols = tidyselect::all_of(metric), names_to = "metric"
    )  %>%
    dplyr::group_by(
      method, metric, dplyr::across(tidyselect::all_of(c(x_str, facet_str)))
    ) %>%
    dplyr::summarise(mean = mean(value), 
                     sd = sd(value) / sqrt(dplyr::n()), 
                     .groups = "keep") %>%
    dplyr::filter(method %in% show_methods) %>%
    dplyr::mutate(
      method = factor(method, levels = show_methods),
      metric = forcats::fct_recode(
        factor(metric, levels = c("rocauc", "prauc")),
        AUROC = "rocauc", AUPRC = "prauc"
      )
    )
  
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
      width = errbar_width
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
    plt <- plt +
      ggplot2::facet_grid(reformulate(facet_str, "metric"), scales = "free_x")
  } else if (length(metric) > 1) {
    plt <- plt +
      ggplot2::facet_wrap(~ metric, scales = "free")
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
  plt_df_ls <- list()
  for (q in quantiles) {
    plt_df_ls[[as.character(q)]] <- results %>%
      dplyr::select(rep, method, fi_scores,
                    tidyselect::all_of(c(x_str, facet_str))) %>%
      dplyr::mutate(
        restricted_metrics = purrr::map_dfr(
          fi_scores, 
          function(fi_df) {
            ignore_vars <- fi_df %>% 
              dplyr::filter(!is.na(cor_with_signal)) %>%
              dplyr::arrange(-cor_with_signal) %>%
              dplyr::slice_head(prop = !!q) %>%
              dplyr::pull(var)
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
      width = errbar_width
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
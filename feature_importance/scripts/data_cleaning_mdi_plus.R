#### Setup ####
library(magrittr)
if (!require("vdocs")) devtools::install_github("Yu-Group/vdocs")

data_dir <- file.path("..", "data")

### Remove duplicate TFs in Enhancer ####
load(file.path(data_dir, "enhancer.Rdata"))

keep_vars <- varnames.all %>%
  dplyr::group_by(Predictor_collapsed) %>%
  dplyr::mutate(id = 1:dplyr::n()) %>%
  dplyr::filter(id == 1)
write.csv(
  X[, keep_vars$Predictor],
  file.path(data_dir, "X_enhancer.csv"),
  row.names = FALSE
)

#### Clean covariate matrices ####
X_paths <- c("X_juvenile.csv",
             "X_splicing.csv",
             "X_ccle_rnaseq.csv",
             "X_enhancer.csv")
log_transform <- c("X_splicing.csv",
                   "X_ccle_rnaseq.csv",
                   "X_enhancer.csv")

for (X_path in X_paths) {
  X_orig <- data.table::fread(file.path(data_dir, X_path)) %>%
    tibble::as_tibble()

  # dim(X_orig)
  # sum(is.na(X_orig))

  X <- X_orig %>%
    vdocs::remove_constant_cols(verbose = 2) %>%
    vdocs::remove_duplicate_cols(verbose = 2)

  # hist(as.matrix(X))
  if (X_path %in% log_transform) {
    X <- log(X + 1)
    # hist(as.matrix(X))
  }

  # dim(X)

  write.csv(
    X,
    file.path(data_dir, "%s_cleaned.csv", fs::path_ext_remove(X_path)),
    row.names = FALSE
  )
}

#### Filter number of features for real data case study ####
X_orig <- data.table::fread(file.path(data_dir, "X_ccle_rnaseq_cleaned.csv"))

X <- X_orig %>%
  vdocs::filter_cols_by_var(max_p = 5000)

write.csv(
  X,
  file.path(data_dir, "X_ccle_rnaseq_cleaned_filtered5000.csv"),
  row.names = FALSE
)

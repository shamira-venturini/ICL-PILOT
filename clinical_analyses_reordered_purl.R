## -----------------------------------------------------------------------------
# Packages
#install.packages(c("dplyr", "lme4", "lmerTest", "purrr", "broom.mixed","rlang"))


## -----------------------------------------------------------------------------
library(dplyr)
library(lme4)
library(lmerTest)
library(purrr)
library(broom.mixed)
library(rvest)
library(ggplot2)
library(emmeans)
library(tidyr)


## -----------------------------------------------------------------------------
# Read data
real <- read.csv(
  "/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/generated/measures/severity_features_raw_real.csv",
  check.names = FALSE
)

syn <- read.csv(
  "/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/generated/measures/severity_features_raw_syn.csv",
  check.names = FALSE
)

roster <- read.csv(
  "/Users/shamiraventurini/PycharmProjects/ICL-PILOT/configs/generation/master_frozen_roster_4_to_7_year_old.csv",
  check.names = FALSE
)

# -----------------------------
# Standardize synthetic pool first
# -----------------------------
syn <- syn %>%
  mutate(
    File_ID = as.character(File_ID),
    pair_id = as.character(pair_id),
    round_id = as.character(round_id),
    Age_Month = as.numeric(`Age(Month)`),
    source = "synthetic",
    clinical_group = case_when(
      comparison_target_group == "TD"  ~ "TD",
      comparison_target_group == "SLI" ~ "SLI",
      TRUE ~ NA_character_
    ),
    age_cohort = case_when(
      Age_Month >= 48 & Age_Month < 60 ~ "4_year_old",
      Age_Month >= 60 & Age_Month < 72 ~ "5_year_old",
      Age_Month >= 72 & Age_Month < 84 ~ "6_year_old",
      Age_Month >= 84 & Age_Month < 96 ~ "7_year_old",
      TRUE ~ NA_character_
    ),
    round_batch = interaction(age_cohort, round_id, drop = TRUE)
  ) %>%
  filter(
    !is.na(clinical_group),
    !is.na(age_cohort),
    Age_Month >= 48, Age_Month < 96
  )

# -----------------------------
# Prompt children used in each round
# Source of truth = synthetic metadata
# -----------------------------
prompt_map <- syn %>%
  transmute(
    round_id = as.character(round_id),
    age_cohort = as.character(age_cohort),
    round_batch = interaction(age_cohort, round_id, drop = TRUE),
    prompt_td_child_id = as.character(prompt_td_child_id),
    prompt_sli_child_id = as.character(prompt_sli_child_id)
  ) %>%
  distinct() %>%
  pivot_longer(
    cols = c(prompt_td_child_id, prompt_sli_child_id),
    names_to = "prompt_group",
    values_to = "child_id"
  ) %>%
  filter(!is.na(child_id), child_id != "") %>%
  mutate(
    clinical_group = case_when(
      prompt_group == "prompt_td_child_id"  ~ "TD",
      prompt_group == "prompt_sli_child_id" ~ "SLI"
    )
  ) %>%
  select(round_id, age_cohort, round_batch, clinical_group, child_id) %>%
  distinct()

# -----------------------------
# Standardize real pool
# -----------------------------
real <- real %>%
  mutate(
    File_ID = as.character(File_ID),
    Age_Month = as.numeric(`Age(Month)`),
    source = "real",
    pair_id = NA_character_,
    round_id = NA_character_,
    clinical_group = case_when(
      Group == "TD"  ~ "TD",
      Group == "SLI" ~ "SLI",
      TRUE ~ NA_character_
    ),
    comparison_target_group = clinical_group,
    age_cohort = case_when(
      Age_Month >= 48 & Age_Month < 60 ~ "4_year_old",
      Age_Month >= 60 & Age_Month < 72 ~ "5_year_old",
      Age_Month >= 72 & Age_Month < 84 ~ "6_year_old",
      Age_Month >= 84 & Age_Month < 96 ~ "7_year_old",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(
    !is.na(clinical_group),
    !is.na(age_cohort),
    Age_Month >= 48, Age_Month < 96
  )

# -----------------------------
# Build fold-stacked held-out evaluation dataframe
# For each age-specific round:
# - real children are included only if they were NOT prompts in that round
# - real children are restricted to the same age cohort
# - synthetic children come from that same age-specific round
# -----------------------------
round_batches <- sort(unique(syn$round_batch))

df_all <- bind_rows(lapply(round_batches, function(rb) {

  syn_r <- syn %>%
    filter(round_batch == rb)

  cohort_r <- unique(syn_r$age_cohort)
  round_id_r <- unique(syn_r$round_id)

  if (length(cohort_r) != 1 || length(round_id_r) != 1) {
    stop(paste("Problem: round_batch", rb, "does not map uniquely to one cohort/round_id"))
  }

  prompt_r <- prompt_map %>%
    filter(round_batch == rb)

  real_r <- real %>%
    filter(age_cohort == cohort_r) %>%
    mutate(
      round_id = round_id_r,
      round_batch = rb
    ) %>%
    anti_join(
      prompt_r,
      by = c("File_ID" = "child_id", "clinical_group")
    )

  bind_rows(real_r, syn_r)
})) %>%
  filter(!is.na(Age_Month)) %>%
  mutate(
    source = factor(source, levels = c("real", "synthetic")),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI")),
    age_cohort = factor(
      age_cohort,
      levels = c("4_year_old", "5_year_old", "6_year_old", "7_year_old")
    ),
    population = factor(
      paste(source, clinical_group, sep = "_"),
      levels = c("real_TD", "synthetic_TD", "real_SLI", "synthetic_SLI")
    ),
    source_is_synthetic = if_else(source == "synthetic", 1L, 0L),

    # IMPORTANT: make synthetic cluster_id unique across age cohorts / rounds
    cluster_id = if_else(
      source == "synthetic",
      paste0("synthetic_", age_cohort, "_", round_id, "_", pair_id),
      paste0("real_", File_ID)
    ),

    unit_id = cluster_id,
    Age_c = Age_Month - mean(Age_Month, na.rm = TRUE)
  )

# Optional subsets
df_td  <- df_all %>% filter(clinical_group == "TD")
df_sli <- df_all %>% filter(clinical_group == "SLI")

# Quick checks
table(df_all$source, df_all$clinical_group)
table(df_all$age_cohort, df_all$source, df_all$clinical_group)
table(df_all$round_batch, df_all$population)
summary(df_all$Age_Month)

## -----------------------------------------------------------------------------
library(dplyr)

df_all <- df_all %>%
  mutate(
    age_cohort = case_when(
      Age_Month >= 48 & Age_Month < 60 ~ "4_year_old",
      Age_Month >= 60 & Age_Month < 72 ~ "5_year_old",
      Age_Month >= 72 & Age_Month < 84 ~ "6_year_old",
      Age_Month >= 84 & Age_Month < 96 ~ "7_year_old",
      TRUE ~ NA_character_
    )
  )

df_all <- df_all %>%
  mutate(
    round_batch = interaction(age_cohort, round_id, drop = TRUE)
  )


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(tibble)

# -----------------------------
# 1) Build an independent transcript-level dataset
# -----------------------------
real_unique <- real %>%
  filter(!is.na(clinical_group), Age_Month >= 48, Age_Month < 96) %>%
  distinct(File_ID, .keep_all = TRUE)

syn_unique <- syn %>%
  filter(!is.na(clinical_group), Age_Month >= 48, Age_Month < 96) %>%
  distinct(File_ID, .keep_all = TRUE)

pca_base <- bind_rows(real_unique, syn_unique) %>%
  mutate(
    source = factor(source, levels = c("real", "synthetic")),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI")),
    source_is_synthetic = if_else(source == "synthetic", 1L, 0L),
    population = factor(
      paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  )

table(pca_base$population)

# -----------------------------
# 2) Exclude metadata / IDs / obvious non-features
# -----------------------------
exclude_vars <- c(
  "bundle_id", "pair_id", "round_id", "round_batch", "replicate_id",
  "source", "stage", "age_cohort",
  "comparison_target_group", "evaluation_pool_policy", "prompt_exclusion_policy",
  "source_child_id", "prompt_td_child_id", "prompt_sli_child_id",
  "td_child_id", "sli_child_id", "sli_age_generated_from",
  "sli_severity_band_generated_from", "sli_profile_label_generated_from",
  "story_slots", "n_stories",
  "File_ID", "File_kideval", "Group", "Sex",
  "Age(Month)", "Age_Month", "Age_c",
  "source_is_synthetic", "clinical_group", "population",
  "cluster_id", "unit_id",
  "#utterances", "#words",
  "DSS_tbl_warning_50",
  "retracing_per_utt",
  "past_overgeneralization_errors",
  "past_verb_opportunities"
  # add "repetition_per_utt" here too if you decide it was not extracted symmetrically
)

feature_df <- pca_base %>%
  select(-any_of(exclude_vars)) %>%
  select(where(is.numeric))

cat("Numeric candidate features:", ncol(feature_df), "\n")

# -----------------------------
# 3) Remove features with too much missingness
# -----------------------------
na_prop <- colMeans(is.na(feature_df))
keep_na <- names(na_prop[na_prop <= 0.20])
feature_df <- feature_df %>% select(all_of(keep_na))

cat("Features after missingness filter:", ncol(feature_df), "\n")

# -----------------------------
# 4) Remove near-zero variance features
# -----------------------------
is_nzv <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(TRUE)
  if (sd(x) < 1e-8) return(TRUE)
  if (length(unique(x)) <= 2 && max(table(x)) / length(x) > 0.95) return(TRUE)
  FALSE
}

nzv_flags <- sapply(feature_df, is_nzv)
feature_df <- feature_df[, !nzv_flags, drop = FALSE]

cat("Features after near-zero variance filter:", ncol(feature_df), "\n")

# -----------------------------
# 5) Remove highly correlated features
# -----------------------------
remove_high_corr <- function(df, cutoff = 0.90) {
  if (ncol(df) < 2) return(df)

  keep <- colnames(df)

  repeat {
    cor_mat <- cor(df[, keep, drop = FALSE], use = "pairwise.complete.obs")
    diag(cor_mat) <- 0

    max_cor <- suppressWarnings(max(abs(cor_mat), na.rm = TRUE))
    if (!is.finite(max_cor) || max_cor < cutoff) break

    idx <- which(abs(cor_mat) == max_cor, arr.ind = TRUE)[1, ]
    v1 <- colnames(cor_mat)[idx[1]]
    v2 <- colnames(cor_mat)[idx[2]]

    mean1 <- mean(abs(cor_mat[v1, ]), na.rm = TRUE)
    mean2 <- mean(abs(cor_mat[v2, ]), na.rm = TRUE)

    drop_var <- if (mean1 >= mean2) v1 else v2
    keep <- setdiff(keep, drop_var)
  }

  df[, keep, drop = FALSE]
}

feature_df_pruned <- remove_high_corr(feature_df, cutoff = 0.90)

cat("Features after correlation pruning:", ncol(feature_df_pruned), "\n")
print(colnames(feature_df_pruned))

# -----------------------------
# 6) Keep complete cases
# -----------------------------
pca_complete <- bind_cols(
  pca_base %>% select(population, source, clinical_group, Age_Month),
  feature_df_pruned
) %>%
  filter(complete.cases(across(all_of(colnames(feature_df_pruned)))))

cat("Rows entering PCA:", nrow(pca_complete), "\n")
table(pca_complete$population)

# -----------------------------
# 7) Run PCA
# -----------------------------
pca_fit <- prcomp(
  pca_complete %>% select(all_of(colnames(feature_df_pruned))),
  center = TRUE,
  scale. = TRUE
)

var_explained <- (pca_fit$sdev^2) / sum(pca_fit$sdev^2)

pca_var_df <- tibble(
  PC = paste0("PC", seq_along(var_explained)),
  variance_explained = var_explained,
  cumulative = cumsum(var_explained)
)

print(pca_var_df)


## -----------------------------------------------------------------------------
# scores
pca_scores <- as.data.frame(pca_fit$x) %>%
  bind_cols(pca_complete %>% select(population, source, clinical_group, Age_Month))

# -----------------------------
# 8) Scree plot
# -----------------------------
ggplot(pca_var_df, aes(x = seq_along(variance_explained), y = variance_explained)) +
  geom_col(width = 0.7) +
  geom_line(aes(y = cumulative), group = 1, linewidth = 0.8) +
  geom_point(aes(y = cumulative), size = 2) +
  scale_x_continuous(breaks = seq_len(nrow(pca_var_df)), labels = pca_var_df$PC) +
  labs(
    x = "Principal component",
    y = "Variance explained",
    title = "Scree plot"
  ) +
  theme_classic()

## -----------------------------------------------------------------------------
library(psych)

fa.parallel(
  pca_complete %>% select(all_of(colnames(feature_df_pruned))),
  fa = "pc",
  n.iter = 100,
  show.legend = TRUE,
  main = "Parallel analysis"
)


## -----------------------------------------------------------------------------
# loadings
pca_loadings <- as_tibble(pca_fit$rotation, rownames = "feature")

pca_loadings %>%
  select(feature, PC1, PC2, PC3, PC4) %>%
  mutate(across(c(PC1, PC2, PC3, PC4), ~ round(.x, 3))) %>%
  arrange(desc(abs(PC1))) %>%
  print(n = Inf)

## -----------------------------------------------------------------------------
library(dplyr)

# PC1
pca_loadings %>%
  select(feature, PC1) %>%
  mutate(PC1 = round(PC1, 3)) %>%
  arrange(desc(abs(PC1))) %>%
  print(n = Inf)

# PC2
pca_loadings %>%
  select(feature, PC2) %>%
  mutate(PC2 = round(PC2, 3)) %>%
  arrange(desc(abs(PC2))) %>%
  print(n = Inf)

# PC3
pca_loadings %>%
  select(feature, PC3) %>%
  mutate(PC3 = round(PC3, 3)) %>%
  arrange(desc(abs(PC3))) %>%
  print(n = Inf)

# PC4
pca_loadings %>%
  select(feature, PC4) %>%
  mutate(PC4 = round(PC4, 3)) %>%
  arrange(desc(abs(PC4))) %>%
  print(n = Inf)


## -----------------------------------------------------------------------------
library(dplyr)

pca_scores_named <- pca_scores %>%
  rename(
    Syntactic_Complexity = PC1,
    Grammatical_Maturity = PC2,
    Error_Disfluency = PC3,
    Residual_Morphosyntactic_Control = PC4
  )


## -----------------------------------------------------------------------------
library(dplyr)
library(GGally)

pca_scores_plotly <- as.data.frame(pca_fit$x) %>%
  mutate(
    population = pca_complete$population,
    PC3 = -PC3
  ) %>%
  rename(
    Syntax  = PC1,
    Grammar = PC2,
    Errors  = PC3,
    Control = PC4
  )

GGally::ggpairs(
  pca_scores_plotly,
  columns = c(
    "Syntax",
    "Grammar",
    "Errors",
    "Control"
  ),
  aes(color = population, alpha = 0.7)
)


## -----------------------------------------------------------------------------
library(ggplot2)

plot_pc <- function(df, xpc, ypc, var_explained) {
  ggplot(df, aes_string(x = xpc, y = ypc, color = "population")) +
    geom_point(size = 2.2, alpha = 0.75) +
    stat_ellipse(linewidth = 0.8, alpha = 0.7) +
    labs(
      x = paste0(xpc, " (", round(100 * var_explained[as.numeric(sub("PC", "", xpc))], 1), "%)"),
      y = paste0(ypc, " (", round(100 * var_explained[as.numeric(sub("PC", "", ypc))], 1), "%)"),
      color = NULL
    ) +
    theme_classic()
}

plot_pc(pca_scores_plotly, "Syntax", "Grammar", var_explained)


## -----------------------------------------------------------------------------
library(dplyr)
library(plotly)

pca_scores_plotly <- as.data.frame(pca_fit$x) %>%
  mutate(
    population = pca_complete$population,
    PC3 = -PC3
  ) %>%
  rename(
    Syntax  = PC1,
    Grammar = PC2,
    Errors  = PC3,
    Control = PC4
  )

plot_ly(
  pca_scores_plotly,
  x = ~Syntax, y = ~Grammar, z = ~Errors,
  color = ~population,
  colors = c(
    "Real TD" = "#E76F51",
    "Synthetic TD" = "#7CAE00",
    "Real SLI" = "#00BFC4",
    "Synthetic SLI" = "#C77CFF"
  ),
  type = "scatter3d",
  mode = "markers",
  marker = list(size = 4, opacity = 0.75)
) %>%
  layout(
    scene = list(
      xaxis = list(title = "Syntax"),
      yaxis = list(title = "Grammar"),
      zaxis = list(title = "Errors")
    ),
    legend = list(title = list(text = "Population"))
  )


## -----------------------------------------------------------------------------
library(dplyr)
library(purrr)
library(tibble)

# -----------------------------
# Independent transcript-level dataset: ages 4–7
# one row per transcript
# -----------------------------
real_indep <- real %>%
  filter(!is.na(clinical_group), Age_Month >= 48, Age_Month < 96) %>%
  distinct(File_ID, .keep_all = TRUE)

syn_indep <- syn %>%
  filter(!is.na(clinical_group), Age_Month >= 48, Age_Month < 96) %>%
  distinct(File_ID, .keep_all = TRUE)

profile_df <- bind_rows(real_indep, syn_indep) %>%
  mutate(
    source = factor(source, levels = c("real", "synthetic")),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI")),
    population = factor(
      paste(ifelse(source == "synthetic", "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  )

table(profile_df$population)

# Reuse the PCA feature set for distributional comparisons.
main_features <- colnames(feature_df_pruned)


## -----------------------------------------------------------------------------
wasserstein_1d <- function(x, y, probs = seq(0, 1, length.out = 200)) {
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  if (length(x) < 2 || length(y) < 2) return(NA_real_)
  qx <- quantile(x, probs = probs, na.rm = TRUE, type = 8)
  qy <- quantile(y, probs = probs, na.rm = TRUE, type = 8)
  mean(abs(qx - qy))
}



compare_distribution <- function(df, feature, group_name) {
  real_vals <- df %>%
    filter(source == "real", clinical_group == group_name) %>%
    pull(all_of(feature))

  syn_vals <- df %>%
    filter(source == "synthetic", clinical_group == group_name) %>%
    pull(all_of(feature))

  real_vals <- real_vals[is.finite(real_vals)]
  syn_vals  <- syn_vals[is.finite(syn_vals)]

  if (length(real_vals) < 2 || length(syn_vals) < 2) {
    return(tibble(
      feature = feature,
      group = group_name,
      n_real = length(real_vals),
      n_syn = length(syn_vals),
      real_mean = NA_real_,
      syn_mean = NA_real_,
      mean_diff = NA_real_,
      wasserstein = NA_real_,
      ks_D = NA_real_,
      ks_p = NA_real_
    ))
  }

  ks_out <- suppressWarnings(ks.test(real_vals, syn_vals))

  tibble(
    feature = feature,
    group = group_name,
    n_real = length(real_vals),
    n_syn = length(syn_vals),
    real_mean = mean(real_vals, na.rm = TRUE),
    syn_mean = mean(syn_vals, na.rm = TRUE),
    mean_diff = mean(syn_vals, na.rm = TRUE) - mean(real_vals, na.rm = TRUE),
    wasserstein = wasserstein_1d(real_vals, syn_vals),
    ks_D = unname(ks_out$statistic),
    ks_p = ks_out$p.value
  )
}

dist_results <- map_dfr(main_features, ~ compare_distribution(profile_df, .x, "TD")) %>%
  bind_rows(
    map_dfr(main_features, ~ compare_distribution(profile_df, .x, "SLI"))
  ) %>%
  mutate(
    ks_p_adj = p.adjust(ks_p, method = "BH")
  )

dist_results %>%
  arrange(group, desc(wasserstein))

## -----------------------------------------------------------------------------
library(dplyr)
library(purrr)
library(tibble)

# standardize each feature using mean/sd from the REAL data of that clinical group
compare_distribution_std <- function(df, feature, group_name) {
  real_vals <- df %>%
    filter(source == "real", clinical_group == group_name) %>%
    pull(all_of(feature))

  syn_vals <- df %>%
    filter(source == "synthetic", clinical_group == group_name) %>%
    pull(all_of(feature))

  real_vals <- real_vals[is.finite(real_vals)]
  syn_vals  <- syn_vals[is.finite(syn_vals)]

  if (length(real_vals) < 2 || length(syn_vals) < 2) {
    return(tibble(
      feature = feature,
      group = group_name,
      wasserstein_std = NA_real_
    ))
  }

  mu <- mean(real_vals, na.rm = TRUE)
  sd0 <- sd(real_vals, na.rm = TRUE)

  if (!is.finite(sd0) || sd0 == 0) {
    return(tibble(
      feature = feature,
      group = group_name,
      wasserstein_std = NA_real_
    ))
  }

  real_z <- (real_vals - mu) / sd0
  syn_z  <- (syn_vals  - mu) / sd0

  tibble(
    feature = feature,
    group = group_name,
    wasserstein_std = wasserstein_1d(real_z, syn_z)
  )
}

dist_std_results <- map_dfr(main_features, ~ compare_distribution_std(profile_df, .x, "TD")) %>%
  bind_rows(
    map_dfr(main_features, ~ compare_distribution_std(profile_df, .x, "SLI"))
  )

dist_std_results %>%
  arrange(group, desc(wasserstein_std))


## -----------------------------------------------------------------------------
plot_df <- dist_std_results %>%
  group_by(feature) %>%
  mutate(max_w = max(wasserstein_std, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(
    feature = reorder(feature, max_w)
  )

ggplot(plot_df, aes(x = feature, y = wasserstein_std, fill = group)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  coord_flip() +
  labs(
    x = NULL,
    y = "Standardized Wasserstein distance",
    title = "Distributional mismatch: synthetic vs real"
  ) +
  theme_classic()


## -----------------------------------------------------------------------------
library(dplyr)
library(tibble)
library(tidyr)
library(ggplot2)

# -----------------------------
# Use the same feature set that entered the PCA
# -----------------------------
broad_features <- colnames(feature_df_pruned)

# If needed, drop known problematic/asymmetric features here too
broad_features <- setdiff(
  broad_features,
  c("retracing_per_utt")
)
# add "repetition_per_utt" too if you decide it should be excluded:
# broad_features <- setdiff(broad_features, c("retracing_per_utt", "repetition_per_utt"))

# -----------------------------
# Use the independent transcript-level 4–7 dataset
# same one used for PCA
# -----------------------------
corr_base <- pca_complete %>%
  select(population, source, clinical_group, all_of(broad_features))

table(corr_base$population)

# -----------------------------
# Helper functions
# -----------------------------
upper_tri_vec <- function(mat) {
  mat[upper.tri(mat)]
}

get_corr_mat <- function(df, group_name, feature_names) {
  tmp <- df %>%
    filter(clinical_group == group_name) %>%
    select(source, all_of(feature_names))

  real_df <- tmp %>%
    filter(source == "real") %>%
    select(-source)

  syn_df <- tmp %>%
    filter(source == "synthetic") %>%
    select(-source)

  # keep only features with non-zero variance in BOTH real and synthetic
  keep_vars <- intersect(
    names(real_df)[sapply(real_df, function(x) sd(x, na.rm = TRUE) > 0)],
    names(syn_df)[sapply(syn_df, function(x) sd(x, na.rm = TRUE) > 0)]
  )

  real_df <- real_df %>% select(all_of(keep_vars))
  syn_df  <- syn_df  %>% select(all_of(keep_vars))

  list(
    keep_vars = keep_vars,
    real = cor(real_df, use = "pairwise.complete.obs"),
    synthetic = cor(syn_df, use = "pairwise.complete.obs")
  )
}

corr_similarity <- function(cor_real, cor_syn) {
  v1 <- upper_tri_vec(cor_real)
  v2 <- upper_tri_vec(cor_syn)

  tibble(
    corr_of_correlations = cor(v1, v2, use = "complete.obs"),
    mean_abs_diff = mean(abs(v1 - v2), na.rm = TRUE),
    max_abs_diff = max(abs(v1 - v2), na.rm = TRUE),
    frobenius = sqrt(sum((cor_real - cor_syn)^2, na.rm = TRUE))
  )
}


## -----------------------------------------------------------------------------
# -----------------------------
# Compute TD and SLI correlation matrices
# -----------------------------
corr_td_broad  <- get_corr_mat(corr_base, "TD", broad_features)
corr_sli_broad <- get_corr_mat(corr_base, "SLI", broad_features)

length(corr_td_broad$keep_vars)
length(corr_sli_broad$keep_vars)

setdiff(broad_features, corr_td_broad$keep_vars)
setdiff(broad_features, corr_sli_broad$keep_vars)

td_similarity_broad  <- corr_similarity(corr_td_broad$real,  corr_td_broad$synthetic)
sli_similarity_broad <- corr_similarity(corr_sli_broad$real, corr_sli_broad$synthetic)

td_similarity_broad

## -----------------------------------------------------------------------------
sli_similarity_broad


## -----------------------------------------------------------------------------
plot_corr_preservation <- function(cor_real, cor_syn, title_text) {
  df_plot <- tibble(
    real = upper_tri_vec(cor_real),
    synthetic = upper_tri_vec(cor_syn)
  )

  ggplot(df_plot, aes(x = real, y = synthetic)) +
    geom_point(alpha = 0.65, size = 2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(
      x = "Real correlations",
      y = "Synthetic correlations",
      title = title_text
    ) +
    theme_classic()
}

plot_corr_preservation(
  corr_td_broad$real,
  corr_td_broad$synthetic,
  "TD: correlation preservation"
)

plot_corr_preservation(
  corr_sli_broad$real,
  corr_sli_broad$synthetic,
  "SLI: correlation preservation"
)

## -----------------------------------------------------------------------------
plot_corr_diff <- function(cor_real, cor_syn, title_text) {
  diff_mat <- cor_syn - cor_real

  as.data.frame(diff_mat) %>%
    rownames_to_column("feature1") %>%
    pivot_longer(-feature1, names_to = "feature2", values_to = "diff") %>%
    ggplot(aes(x = feature1, y = feature2, fill = diff)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "#2166ac",
      mid = "white",
      high = "#b2182b",
      midpoint = 0
    ) +
    coord_equal() +
    labs(
      x = NULL,
      y = NULL,
      fill = "Syn - Real r",
      title = title_text
    ) +
    theme_classic() +
    theme(
      axis.text.x = element_text(angle = 60, hjust = 1, size = 8),
      axis.text.y = element_text(size = 8)
    )
}

plot_corr_diff(
  corr_td_broad$real,
  corr_td_broad$synthetic,
  "TD: correlation differences"
)

plot_corr_diff(
  corr_sli_broad$real,
  corr_sli_broad$synthetic,
  "SLI: correlation differences"
)


## -----------------------------------------------------------------------------
corr_summary_plot <- bind_rows(
  td_similarity_broad  %>% mutate(group = "TD"),
  sli_similarity_broad %>% mutate(group = "SLI")
) %>%
  pivot_longer(
    cols = c(corr_of_correlations, mean_abs_diff, max_abs_diff, frobenius),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = factor(
      metric,
      levels = c("corr_of_correlations", "mean_abs_diff", "max_abs_diff", "frobenius"),
      labels = c(
        "Correlation of correlations",
        "Mean absolute difference",
        "Max absolute difference",
        "Frobenius distance"
      )
    )
  )

ggplot(corr_summary_plot, aes(x = metric, y = value, fill = group)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  coord_flip() +
  labs(
    x = NULL,
    y = "Value",
    title = "Correlation-structure preservation summary"
  ) +
  theme_classic()


## -----------------------------------------------------------------------------
top_corr_diffs <- function(cor_real, cor_syn, top_n = 15) {
  diff_mat <- cor_syn - cor_real
  abs_diff_mat <- abs(diff_mat)

  as.data.frame(abs_diff_mat) %>%
    rownames_to_column("feature1") %>%
    pivot_longer(-feature1, names_to = "feature2", values_to = "abs_diff") %>%
    mutate(signed_diff = as.vector(diff_mat)) %>%
    filter(feature1 != feature2) %>%
    rowwise() %>%
    mutate(pair = paste(sort(c(feature1, feature2)), collapse = " -- ")) %>%
    ungroup() %>%
    distinct(pair, .keep_all = TRUE) %>%
    arrange(desc(abs_diff)) %>%
    slice_head(n = top_n) %>%
    select(pair, abs_diff, signed_diff)
}

top_td_broad  <- top_corr_diffs(corr_td_broad$real,  corr_td_broad$synthetic, top_n = 15)
top_sli_broad <- top_corr_diffs(corr_sli_broad$real, corr_sli_broad$synthetic, top_n = 15)

top_td_broad


## -----------------------------------------------------------------------------
top_sli_broad


## -----------------------------------------------------------------------------
m_fixed <- lmer(
  MLU_Morphemes ~ source_is_synthetic * clinical_group +
    Age_c + factor(round_batch) + (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_random <- lmer(
  MLU_Morphemes ~ source_is_synthetic * clinical_group +
    Age_c + (1 | round_batch) + (1 | cluster_id),
  data = df_all, REML = FALSE
)

AIC(m_fixed, m_random)
BIC(m_fixed, m_random)
summary(m_fixed)
summary(m_random)
VarCorr(m_random)


## -----------------------------------------------------------------------------
m_lin <- lmer(
  MLU_Morphemes ~ source_is_synthetic * clinical_group +
    Age_c + (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_quad <- lmer(
  MLU_Morphemes ~ source_is_synthetic * clinical_group +
    Age_c + I(Age_c^2) + (1 | cluster_id),
  data = df_all, REML = FALSE
)

anova(m_lin, m_quad)
AIC(m_lin, m_quad)
BIC(m_lin, m_quad)

## -----------------------------------------------------------------------------
library(ggplot2)

ggplot(df_all, aes(x = Age_Month, y = MLU_Morphemes, color = interaction(source, clinical_group))) +
  geom_point(alpha = 0.2) +
  geom_smooth(se = FALSE, method = "loess") +
  theme_classic()

## -----------------------------------------------------------------------------
library(splines)
library(lme4)

m_lin_int <- lmer(
  MLU_Morphemes ~ source_is_synthetic * clinical_group * Age_c +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_spline_int <- lmer(
  MLU_Morphemes ~ source_is_synthetic * clinical_group +
    source_is_synthetic:ns(Age_Month, df = 3) +
    clinical_group:ns(Age_Month, df = 3) +
    ns(Age_Month, df = 3) +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

AIC(m_lin_int, m_spline_int)
BIC(m_lin_int, m_spline_int)


## -----------------------------------------------------------------------------
m0_mlu <- lmer(
  `MLU_Morphemes` ~ source_is_synthetic * clinical_group + (1 | cluster_id),
  data = df_all, REML = FALSE
) # do real vs syn and td vs sli differ? does the difference depend from clinical group?

m1_mlu <- lmer(
  `MLU_Morphemes` ~ source_is_synthetic * clinical_group + Age_c + (1 | cluster_id),
  data = df_all, REML = FALSE
) # do real vs syn and td vs sli differ? does the difference depend from clinical group? does mlu change with age overall?

m2_mlu <- lmer(
  `MLU_Morphemes` ~ source_is_synthetic * clinical_group + Age_c +
    source_is_synthetic:Age_c + clinical_group:Age_c +
    (1 | cluster_id),
  data = df_all, REML = FALSE
) # do real vs syn and td vs sli differ? does the difference depend from clinical group? does mlu change with age overall? does age slope differ by source?

m3_mlu <- lmer(
  `MLU_Morphemes` ~ source_is_synthetic * clinical_group * Age_c + (1 | cluster_id),
  data = df_all, REML = FALSE
) # # do real vs syn and td vs sli differ? does the difference depend from clinical group? does mlu change with age overall? does age slope differ by source? does age slope differ jointly across both source and clinical group?

anova(m0_mlu, m1_mlu, m2_mlu, m3_mlu)



## -----------------------------------------------------------------------------
summary(m2_mlu)


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

# If not already present
df_plot <- df_all %>%
  mutate(
    population = factor(
      paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  ) %>%
  filter(!is.na(MLU_Morphemes), !is.na(population))

# Estimated marginal means from the model
emm <- emmeans(m2_mlu, ~ source_is_synthetic * clinical_group)
emm_df <- as.data.frame(emm) %>%
  mutate(
    population = factor(
      paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  )

ggplot(df_plot, aes(x = population, y = MLU_Morphemes, color = population)) +
  geom_boxplot(width = 0.45, alpha = 0.15, outlier.shape = NA) +
  geom_jitter(width = 0.10, alpha = 0.28, size = 1.8) +
  geom_pointrange(
    data = emm_df,
    aes(x = population, y = emmean, ymin = lower.CL, ymax = upper.CL),
    inherit.aes = FALSE,
    linewidth = 0.9
  ) +
  labs(
    x = NULL,
    y = "MLU (Morphemes)"
  ) +
  theme_classic() +
  theme(legend.position = "none")


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

# grid over observed age range (in centered age units)
age_grid <- seq(
  min(df_all$Age_c, na.rm = TRUE),
  max(df_all$Age_c, na.rm = TRUE),
  length.out = 100
)

# model-based predictions
emm_mlu <- emmeans(
  m2_mlu,
  ~ source_is_synthetic * clinical_group * Age_c,
  at = list(Age_c = age_grid)
)

emm_mlu_df <- as.data.frame(emm_mlu) %>%
  mutate(
    Age_Month_plot = Age_c + mean(df_all$Age_Month, na.rm = TRUE),
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

# raw data
plot_df <- df_all %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

ggplot(plot_df, aes(x = Age_Month, y = MLU_Morphemes, color = source_label)) +
  geom_jitter(width = 0.25, height = 0, alpha = 0.18, size = 1.4) +
  geom_ribbon(
    data = emm_mlu_df,
    aes(
      x = Age_Month_plot,
      y = emmean,
      ymin = lower.CL,
      ymax = upper.CL,
      fill = source_label
    ),
    alpha = 0.15,
    colour = NA,
    inherit.aes = FALSE
  ) +
  geom_line(
    data = emm_mlu_df,
    aes(
      x = Age_Month_plot,
      y = emmean,
      color = source_label
    ),
    linewidth = 1.2,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ clinical_group) +
  labs(
    x = "Age (months)",
    y = "MLU (morphemes)",
    color = "Source",
    fill = "Source",
    title = "MLU by age, source, and clinical group"
  ) +
  theme_classic()


## -----------------------------------------------------------------------------
library(splines)
library(lme4)

m_lin_int <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group * Age_c +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_spline_int <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group +
    source_is_synthetic:ns(Age_Month, df = 3) +
    clinical_group:ns(Age_Month, df = 3) +
    ns(Age_Month, df = 3) +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

AIC(m_lin_int, m_spline_int)
BIC(m_lin_int, m_spline_int)


## -----------------------------------------------------------------------------
m0_vocd <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group  + (1 | cluster_id),
  data = df_all, REML = FALSE
)

m1_vocd <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group + Age_c  + (1 | cluster_id),
  data = df_all, REML = FALSE
)

m2_vocd <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group + Age_c +
    source_is_synthetic:Age_c + clinical_group:Age_c +
     + (1 | cluster_id),
  data = df_all, REML = FALSE
) # exploratory model: do real vs syn and td vs sli differ? does the difference depend from clinical group? does mlu change with age overall? does age slope differ by source?

m3_vocd <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group * Age_c +
     + (1 | cluster_id),
  data = df_all, REML = FALSE
)

anova(m0_vocd, m1_vocd, m2_vocd, m3_vocd)

## -----------------------------------------------------------------------------
summary(m1_vocd)  # main model - do real vs syn and td vs sli differ? does the difference depend from clinical group? does mlu change with age overall?


## -----------------------------------------------------------------------------
library(emmeans)

# Estimated marginal means at mean age, averaged over round
emm_vocd <- emmeans(
  m1_vocd,
  ~ source_is_synthetic * clinical_group,
  at = list(source_is_synthetic = c(0, 1))
)

emm_vocd

## -----------------------------------------------------------------------------
pairs(
  emmeans(
    m1_vocd,
    ~ source_is_synthetic | clinical_group,
    at = list(source_is_synthetic = c(0, 1))
  ),
  infer = c(TRUE, TRUE),
  adjust = "none"
)


## -----------------------------------------------------------------------------
pairs(
  emmeans(
    m1_vocd,
    ~ clinical_group | source_is_synthetic,
    at = list(source_is_synthetic = c(0, 1))
  ),
  infer = c(TRUE, TRUE),
  adjust = "none"
)


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

emm_vocd_df <- as.data.frame(emm_vocd) %>%
  mutate(
    population = factor(
      paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  )

df_plot <- df_all %>%
  mutate(
    population = factor(
      paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  ) %>%
  filter(!is.na(VOCD_D_optimum_average))

ggplot(df_plot, aes(x = population, y = VOCD_D_optimum_average, color = population)) +
  geom_boxplot(width = 0.45, alpha = 0.15, outlier.shape = NA) +
  geom_jitter(width = 0.10, alpha = 0.20, size = 1.7) +
  geom_pointrange(
    data = emm_vocd_df,
    aes(y = emmean, ymin = lower.CL, ymax = upper.CL),
    inherit.aes = TRUE,
    linewidth = 0.9,
    color = "black"
  ) +
  labs(
    x = NULL,
    y = "VOCD-D"
  ) +
  theme_classic() +
  theme(legend.position = "none")


## -----------------------------------------------------------------------------
df_all %>%
  mutate(
    population = paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group)
  ) %>%
  group_by(population) %>%
  summarise(
    raw_mean = mean(VOCD_D_optimum_average, na.rm = TRUE),
    raw_sd = sd(VOCD_D_optimum_average, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )


## -----------------------------------------------------------------------------
summary(m2_vocd)


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

# Mean age used to convert centered age back to months for plotting
age_mean <- mean(df_all$Age_Month, na.rm = TRUE)

# Raw data for plotting
df_plot <- df_all %>%
  mutate(
    Age_Month_plot = Age_c + age_mean,
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  ) %>%
  filter(
    !is.na(VOCD_D_optimum_average),
    !is.na(Age_c),
    !is.na(Age_Month_plot)
  )

# Age grid for predictions
age_grid <- seq(
  min(df_plot$Age_c, na.rm = TRUE),
  max(df_plot$Age_c, na.rm = TRUE),
  length.out = 100
)

# Model-based predictions from m2_vocd
emm_age <- emmeans(
  m2_vocd,
  ~ source_is_synthetic * clinical_group * Age_c,
  at = list(
    source_is_synthetic = c(0, 1),
    Age_c = age_grid
  )
)

pred_df <- as.data.frame(emm_age) %>%
  mutate(
    Age_Month_plot = Age_c + age_mean,
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

# Plot
ggplot(df_plot, aes(x = Age_Month_plot, y = VOCD_D_optimum_average, color = source_label)) +
  geom_jitter(width = 0.15, height = 0, alpha = 0.18, size = 1.6) +
  geom_line(
    data = pred_df,
    aes(x = Age_Month_plot, y = emmean, color = source_label),
    linewidth = 1.1,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ clinical_group) +
  labs(
    x = "Age (months)",
    y = "VOCD-D",
    color = "Source"
  ) +
  theme_classic()


## -----------------------------------------------------------------------------
library(splines)
library(lme4)

m_lin_int <- lmer(
  DSS_tbl_score ~ source_is_synthetic * clinical_group * Age_c +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_spline_int <- lmer(
  DSS_tbl_score ~ source_is_synthetic * clinical_group +
    source_is_synthetic:ns(Age_Month, df = 3) +
    clinical_group:ns(Age_Month, df = 3) +
    ns(Age_Month, df = 3) +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

AIC(m_lin_int, m_spline_int)
BIC(m_lin_int, m_spline_int)


## -----------------------------------------------------------------------------
m0_dss <- lmer(
  DSS_tbl_score ~ source_is_synthetic * clinical_group  + (1 | cluster_id),
  data = df_all, REML = FALSE
)

m1_dss <- lmer(
  DSS_tbl_score ~ source_is_synthetic * clinical_group + Age_c  + (1 | cluster_id),
  data = df_all, REML = FALSE
)

m2_dss <- lmer(
  DSS_tbl_score ~ source_is_synthetic * clinical_group + Age_c +
    source_is_synthetic:Age_c + clinical_group:Age_c +
     + (1 | cluster_id),
  data = df_all, REML = FALSE
)

m3_dss <- lmer(
  DSS_tbl_score ~ source_is_synthetic * clinical_group * Age_c  + (1 | cluster_id),
  data = df_all, REML = FALSE
)

anova(m0_dss, m1_dss, m2_dss, m3_dss)


## -----------------------------------------------------------------------------
summary(m1_dss)


## -----------------------------------------------------------------------------
# Adjusted means from the retained model
emm_dss <- emmeans(
  m1_dss,
  ~ source_is_synthetic * clinical_group,
  at = list(source_is_synthetic = c(0, 1))
)

emm_dss_df <- as.data.frame(emm_dss) %>%
  mutate(
    population = factor(
      paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  )

# Raw data for plotting
df_plot <- df_all %>%
  mutate(
    population = factor(
      paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  ) %>%
  filter(!is.na(DSS_tbl_score), !is.na(population))

# Plot
ggplot(df_plot, aes(x = population, y = DSS_tbl_score, color = population)) +
  geom_boxplot(width = 0.45, alpha = 0.15, outlier.shape = NA) +
  geom_jitter(width = 0.10, alpha = 0.22, size = 1.7) +
  geom_pointrange(
    data = emm_dss_df,
    aes(x = population, y = emmean, ymin = lower.CL, ymax = upper.CL),
    inherit.aes = FALSE,
    linewidth = 0.9,
    color = "black"
  ) +
  labs(
    x = NULL,
    y = "DSS"
  ) +
  theme_classic() +
  theme(legend.position = "none")


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

age_mean <- mean(df_all$Age_Month, na.rm = TRUE)

df_age_plot <- df_all %>%
  mutate(
    Age_Month_plot = Age_c + age_mean,
    population = factor(
      paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  ) %>%
  filter(!is.na(DSS_tbl_score), !is.na(Age_c), !is.na(Age_Month_plot))

age_grid <- seq(
  min(df_age_plot$Age_c, na.rm = TRUE),
  max(df_age_plot$Age_c, na.rm = TRUE),
  length.out = 100
)

emm_age <- emmeans(
  m1_dss,
  ~ source_is_synthetic * clinical_group * Age_c,
  at = list(
    source_is_synthetic = c(0, 1),
    Age_c = age_grid
  )
)

pred_df <- as.data.frame(emm_age) %>%
  mutate(
    Age_Month_plot = Age_c + age_mean,
    population = factor(
      paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  )

ggplot(df_age_plot, aes(x = Age_Month_plot, y = DSS_tbl_score, color = population)) +
  geom_jitter(width = 0.15, height = 0, alpha = 0.18, size = 1.6) +
  geom_line(
    data = pred_df,
    aes(x = Age_Month_plot, y = emmean, color = population),
    linewidth = 1.1
  ) +
  labs(
    x = "Age (months)",
    y = "DSS",
    color = NULL
  ) +
  theme_classic()


## -----------------------------------------------------------------------------
# -----------------------------
# Read DEV data
# -----------------------------
real_dev <- read.csv(
  "/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/ENNI/measures/dev_measures.csv",
  check.names = FALSE
)

syn_dev <- read.csv(
  "/Users/shamiraventurini/PycharmProjects/ICL-PILOT/phase2/generated/measures/dev_measures.csv",
  check.names = FALSE
)

# -----------------------------
# Build prompt map from synthetic metadata
# If 'syn' already exists in memory from the main feature pipeline, use it.
# Otherwise use syn_dev if it contains the prompt columns.
# -----------------------------
prompt_source <- if (exists("syn")) syn else syn_dev

prompt_source <- prompt_source %>%
  mutate(
    round_id = as.character(round_id),
    Age_Month = as.numeric(`Age(Month)`),
    age_cohort = case_when(
      Age_Month >= 48 & Age_Month < 60 ~ "4_year_old",
      Age_Month >= 60 & Age_Month < 72 ~ "5_year_old",
      Age_Month >= 72 & Age_Month < 84 ~ "6_year_old",
      Age_Month >= 84 & Age_Month < 96 ~ "7_year_old",
      TRUE ~ NA_character_
    ),
    round_batch = interaction(age_cohort, round_id, drop = TRUE)
  )

prompt_map <- prompt_source %>%
  transmute(
    round_id = as.character(round_id),
    age_cohort = as.character(age_cohort),
    round_batch = interaction(age_cohort, round_id, drop = TRUE),
    prompt_td_child_id = as.character(prompt_td_child_id),
    prompt_sli_child_id = as.character(prompt_sli_child_id)
  ) %>%
  distinct() %>%
  pivot_longer(
    cols = c(prompt_td_child_id, prompt_sli_child_id),
    names_to = "prompt_group",
    values_to = "child_id"
  ) %>%
  filter(!is.na(child_id), child_id != "") %>%
  mutate(
    clinical_group = case_when(
      prompt_group == "prompt_td_child_id"  ~ "TD",
      prompt_group == "prompt_sli_child_id" ~ "SLI"
    )
  ) %>%
  select(round_id, age_cohort, round_batch, clinical_group, child_id) %>%
  distinct()

# -----------------------------
# Real DEV pool: ages 4–7
# -----------------------------
real_dev_4to7 <- real_dev %>%
  mutate(
    File_ID = as.character(File_ID),
    Age_Month = as.numeric(`Age(Month)`),
    source = "real",
    pair_id = NA_character_,
    round_id = NA_character_,
    clinical_group = case_when(
      Group == "TD"  ~ "TD",
      Group == "SLI" ~ "SLI",
      TRUE ~ NA_character_
    ),
    age_cohort = case_when(
      Age_Month >= 48 & Age_Month < 60 ~ "4_year_old",
      Age_Month >= 60 & Age_Month < 72 ~ "5_year_old",
      Age_Month >= 72 & Age_Month < 84 ~ "6_year_old",
      Age_Month >= 84 & Age_Month < 96 ~ "7_year_old",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(
    !is.na(clinical_group),
    !is.na(age_cohort),
    Age_Month >= 48, Age_Month < 96
  )

# -----------------------------
# Synthetic DEV pool: ages 4–7
# -----------------------------
syn_dev_4to7 <- syn_dev %>%
  mutate(
    File_ID = as.character(File_ID),
    pair_id = as.character(pair_id),
    round_id = as.character(round_id),
    Age_Month = as.numeric(`Age(Month)`),
    source = "synthetic",
    clinical_group = case_when(
      comparison_target_group == "TD"  ~ "TD",
      comparison_target_group == "SLI" ~ "SLI",
      TRUE ~ NA_character_
    ),
    age_cohort = case_when(
      Age_Month >= 48 & Age_Month < 60 ~ "4_year_old",
      Age_Month >= 60 & Age_Month < 72 ~ "5_year_old",
      Age_Month >= 72 & Age_Month < 84 ~ "6_year_old",
      Age_Month >= 84 & Age_Month < 96 ~ "7_year_old",
      TRUE ~ NA_character_
    ),
    round_batch = interaction(age_cohort, round_id, drop = TRUE)
  ) %>%
  filter(
    !is.na(clinical_group),
    !is.na(age_cohort),
    Age_Month >= 48, Age_Month < 96
  )

# -----------------------------
# Build fold-stacked held-out DEV dataframe
# Stack by age-specific round_batch
# -----------------------------
round_batches <- sort(unique(syn_dev_4to7$round_batch))


df_dev_all <- bind_rows(lapply(round_batches, function(rb) {

  syn_r <- syn_dev_4to7 %>%
    filter(round_batch == rb)

  cohort_r <- unique(syn_r$age_cohort)
  round_id_r <- unique(syn_r$round_id)

  if (length(cohort_r) != 1 || length(round_id_r) != 1) {
    stop(paste("Problem: round_batch", rb, "does not map uniquely to one cohort/round_id"))
  }

  prompt_r <- prompt_map %>%
    filter(round_batch == rb)

  real_r <- real_dev_4to7 %>%
    filter(age_cohort == cohort_r) %>%
    mutate(
      round_id = round_id_r,
      round_batch = rb
    ) %>%
    anti_join(
      prompt_r,
      by = c("File_ID" = "child_id", "clinical_group")
    )

  bind_rows(real_r, syn_r)
})) %>%
  mutate(
    source = factor(source, levels = c("real", "synthetic")),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI")),
    source_is_synthetic = if_else(source == "synthetic", 1L, 0L),
    age_cohort = factor(
      age_cohort,
      levels = c("4_year_old", "5_year_old", "6_year_old", "7_year_old")
    ),
    cluster_id = if_else(
      source == "synthetic",
      paste0("synthetic_", age_cohort, "_", round_id, "_", pair_id),
      paste0("real_", File_ID)
    ),
    Age_c = Age_Month - mean(Age_Month, na.rm = TRUE)
  ) %>%
  filter(
    !is.na(past_overgeneralization_errors),
    !is.na(past_verb_opportunities),
    past_verb_opportunities > 0
  )

df_dev_all_clean <- df_dev_all %>%
  filter(
    past_verb_opportunities > 0,
    past_overgeneralization_errors >= 0,
    past_overgeneralization_errors <= past_verb_opportunities
  )
df_dev_all_clean <- df_dev_all_clean %>%
  mutate(Age_z = as.numeric(scale(Age_Month)))

# Quick checks
table(df_dev_all_clean$source, df_dev_all_clean$clinical_group)
table(df_dev_all_clean$age_cohort, df_dev_all_clean$source, df_dev_all_clean$clinical_group)
summary(df_dev_all_clean$past_overgeneralization_errors)
summary(df_dev_all_clean$past_verb_opportunities)



## -----------------------------------------------------------------------------
library(lme4)

m_age0_overgen <- glmer(
  cbind(
    past_overgeneralization_errors,
    past_verb_opportunities - past_overgeneralization_errors
  ) ~
    source_is_synthetic * clinical_group +
    (1 | cluster_id),
  data = df_dev_all_clean,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

m_age1_overgen <- glmer(
  cbind(
    past_overgeneralization_errors,
    past_verb_opportunities - past_overgeneralization_errors
  ) ~
    source_is_synthetic * clinical_group +
    Age_c +
    (1 | cluster_id),
  data = df_dev_all_clean,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

m_age2_overgen <- glmer(
  cbind(
    past_overgeneralization_errors,
    past_verb_opportunities - past_overgeneralization_errors
  ) ~
    source_is_synthetic * clinical_group +
    Age_c + I(Age_c^2) +
    (1 | cluster_id),
  data = df_dev_all_clean,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

anova(m_age0_overgen, m_age1_overgen, m_age2_overgen, test = "Chisq")
AIC(m_age0_overgen, m_age1_overgen, m_age2_overgen)
BIC(m_age0_overgen, m_age1_overgen, m_age2_overgen)


## -----------------------------------------------------------------------------
m0_overgen <- glmer(
  cbind(past_overgeneralization_errors,
        past_verb_opportunities - past_overgeneralization_errors) ~
    source_is_synthetic * clinical_group  + (1 | cluster_id),
  data = df_dev_all_clean,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
) # Does the probability of past overgeneralization differ by source (real vs synthetic), by clinical group (TD vs SLI), and by the source-by-group combination, after controlling for round and repeated observations?

m1_overgen <- glmer(
  cbind(past_overgeneralization_errors,
        past_verb_opportunities - past_overgeneralization_errors) ~
    source_is_synthetic * clinical_group + Age_c  + (1 | cluster_id),
  data = df_dev_all_clean,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
) # Does the probability of past overgeneralization change with age overall?

m2_overgen <- glmer(
  cbind(past_overgeneralization_errors,
        past_verb_opportunities - past_overgeneralization_errors) ~
    source_is_synthetic * clinical_group + Age_c +
    source_is_synthetic:Age_c + clinical_group:Age_c +
     + (1 | cluster_id),
  data = df_dev_all_clean,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
) # Do real and synthetic data show different age slopes, or do TD and SLI show different age slopes?

m3_overgen <- glmer(
  cbind(past_overgeneralization_errors,
        past_verb_opportunities - past_overgeneralization_errors) ~
    source_is_synthetic * clinical_group * Age_c +
     + (1 | cluster_id),
  data = df_dev_all_clean,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
) # Is the developmental pattern of past overgeneralization across age and clinical group reproduced differently in synthetic versus real data?

anova(m0_overgen, m1_overgen, m2_overgen, m3_overgen, test = "Chisq")

## -----------------------------------------------------------------------------
summary(m3_overgen) # Does the developmental pattern across age differ jointly by source and clinical group?


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)

# -----------------------------
# Cleaner plot for m3_overgen
# - points = age-binned observed means
# - lines/ribbons = model-predicted probabilities
# -----------------------------

# 1) Add observed rate
plot_df <- df_dev_all_clean %>%
  mutate(
    obs_rate = past_overgeneralization_errors / past_verb_opportunities,
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

# 2) Create 6-month age bins
plot_df <- plot_df %>%
  mutate(
    age_bin = cut(
      Age_Month,
      breaks = c(48, 54, 60, 66, 72, 78, 84, 90, 96),
      right = FALSE,
      include.lowest = TRUE
    )
  )

# 3) Compute binned observed means
#    Weighted by past opportunities, so rows with more opportunities count more
binned_df <- plot_df %>%
  group_by(clinical_group, source_label, age_bin) %>%
  summarise(
    Age_Month_plot = weighted.mean(Age_Month, w = past_verb_opportunities, na.rm = TRUE),
    obs_rate = sum(past_overgeneralization_errors, na.rm = TRUE) /
      sum(past_verb_opportunities, na.rm = TRUE),
    total_opp = sum(past_verb_opportunities, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

# 4) Prediction grid from m3_overgen
age_grid <- seq(
  min(df_dev_all_clean$Age_c, na.rm = TRUE),
  max(df_dev_all_clean$Age_c, na.rm = TRUE),
  length.out = 150
)

newdat <- expand.grid(
  Age_c = age_grid,
  source_is_synthetic = c(0, 1),
  clinical_group = c("TD", "SLI")
) %>%
  mutate(
    clinical_group = factor(clinical_group, levels = levels(df_dev_all_clean$clinical_group)),
    source_is_synthetic = as.integer(source_is_synthetic)
  )

X <- model.matrix(
  ~ source_is_synthetic * clinical_group * Age_c,
  data = newdat
)

beta <- lme4::fixef(m3_overgen)
V <- as.matrix(vcov(m3_overgen))

eta <- as.numeric(X %*% beta)
se_eta <- sqrt(diag(X %*% V %*% t(X)))

newdat <- newdat %>%
  mutate(
    Age_Month_plot = Age_c + mean(df_dev_all_clean$Age_Month, na.rm = TRUE),
    fit_link = eta,
    lower_link = eta - 1.96 * se_eta,
    upper_link = eta + 1.96 * se_eta,
    fit = plogis(fit_link),
    lower = plogis(lower_link),
    upper = plogis(upper_link),
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

# 5) Plot
ggplot() +
  geom_ribbon(
    data = newdat,
    aes(
      x = Age_Month_plot,
      ymin = lower,
      ymax = upper,
      fill = source_label
    ),
    alpha = 0.14,
    color = NA
  ) +
  geom_line(
    data = newdat,
    aes(
      x = Age_Month_plot,
      y = fit,
      color = source_label
    ),
    linewidth = 1.2
  ) +
  geom_point(
    data = binned_df,
    aes(
      x = Age_Month_plot,
      y = obs_rate,
      color = source_label,
      size = total_opp
    ),
    alpha = 0.9
  ) +
  facet_wrap(~ clinical_group, nrow = 1) +
  scale_size_area(
    max_size = 8,
    breaks = c(25, 50, 100, 200),
    labels = c("25", "50", "100", "200")
  ) +
  labs(
    x = "Age (months)",
    y = "Past overgeneralization rate",
    color = "Source",
    fill = "Source",
    size = "Past opportunities\n(within bin)",
    title = "Past overgeneralization across age",
    subtitle = "Points are weighted age-binned observed rates; lines are model-predicted probabilities"
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_classic(base_size = 13) +
  theme(
    legend.position = "right",
    strip.background = element_rect(fill = "white", color = "black"),
    strip.text = element_text(face = "bold")
  )


## -----------------------------------------------------------------------------
library(splines)
library(lme4)

m_lin_int <- lmer(
  bare_marking_errors_per_word ~ source_is_synthetic * clinical_group * Age_c +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_spline_int <- lmer(
  bare_marking_errors_per_word ~ source_is_synthetic * clinical_group +
    source_is_synthetic:ns(Age_Month, df = 3) +
    clinical_group:ns(Age_Month, df = 3) +
    ns(Age_Month, df = 3) +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

AIC(m_lin_int, m_spline_int)
BIC(m_lin_int, m_spline_int)


## -----------------------------------------------------------------------------
library(lme4)
library(lmerTest)

m0_bare <- lmer(
  bare_marking_errors_per_word ~ source_is_synthetic * clinical_group +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
) # Does nonfiniteness differ by source, by clinical group, and by their interaction?

m1_bare <- lmer(
  bare_marking_errors_per_word ~ source_is_synthetic * clinical_group +
    Age_c  + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m2_bare <- lmer(
  bare_marking_errors_per_word ~ source_is_synthetic * clinical_group +
    Age_c +
    source_is_synthetic:Age_c +
    clinical_group:Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m3_bare <- lmer(
  bare_marking_errors_per_word ~ source_is_synthetic * clinical_group * Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

anova(m0_bare, m1_bare, m2_bare, m3_bare)

## -----------------------------------------------------------------------------
summary(m3_bare)


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

# -----------------------------
# Predicted trajectories from m3_bare
# -----------------------------
age_grid <- seq(
  min(df_all$Age_c, na.rm = TRUE),
  max(df_all$Age_c, na.rm = TRUE),
  length.out = 150
)

emm_bare <- emmeans(
  m3_bare,
  ~ source_is_synthetic * clinical_group * Age_c,
  at = list(Age_c = age_grid)
)

emm_bare_df <- as.data.frame(emm_bare) %>%
  mutate(
    Age_Month_plot = Age_c + mean(df_all$Age_Month, na.rm = TRUE),
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

# -----------------------------
# Observed data
# -----------------------------
plot_df <- df_all %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

# -----------------------------
# Plot
# -----------------------------
ggplot(plot_df, aes(x = Age_Month, y = bare_marking_errors_per_word, color = source_label)) +
  geom_jitter(width = 0.35, height = 0, alpha = 0.18, size = 1.3) +
  geom_ribbon(
    data = emm_bare_df,
    aes(
      x = Age_Month_plot,
      y = emmean,
      ymin = lower.CL,
      ymax = upper.CL,
      fill = source_label
    ),
    alpha = 0.14,
    colour = NA,
    inherit.aes = FALSE
  ) +
  geom_line(
    data = emm_bare_df,
    aes(
      x = Age_Month_plot,
      y = emmean,
      color = source_label
    ),
    linewidth = 1.2,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ clinical_group, nrow = 1) +
  labs(
    x = "Age (months)",
    y = "Bare marking errors per word",
    color = "Source",
    fill = "Source",
    title = "Bare marking errors across age",
    subtitle = "Points are observed transcript-level values; lines are model-estimated trajectories"
  ) +
  theme_classic(base_size = 13) +
  theme(
    legend.position = "right",
    strip.background = element_rect(fill = "white", color = "black"),
    strip.text = element_text(face = "bold")
  )


## -----------------------------------------------------------------------------
library(splines)
library(lme4)

m_lin_int <- lmer(
  agreement_errors_per_finite_verb ~ source_is_synthetic * clinical_group * Age_c +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_spline_int <- lmer(
  agreement_errors_per_finite_verb ~ source_is_synthetic * clinical_group +
    source_is_synthetic:ns(Age_Month, df = 3) +
    clinical_group:ns(Age_Month, df = 3) +
    ns(Age_Month, df = 3) +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

AIC(m_lin_int, m_spline_int)
BIC(m_lin_int, m_spline_int)


## -----------------------------------------------------------------------------
m0_agr <- lmer(
  agreement_errors_per_finite_verb ~ source_is_synthetic * clinical_group +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m1_agr <- lmer(
  agreement_errors_per_finite_verb ~ source_is_synthetic * clinical_group +
    Age_c  + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m2_agr <- lmer(
  agreement_errors_per_finite_verb ~ source_is_synthetic * clinical_group +
    Age_c +
    source_is_synthetic:Age_c +
    clinical_group:Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m3_agr <- lmer(
  agreement_errors_per_finite_verb ~ source_is_synthetic * clinical_group * Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

anova(m0_agr, m1_agr, m2_agr, m3_agr)


## -----------------------------------------------------------------------------
summary(m2_agr)


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

# Adjusted means from the retained model
emm_agr <- emmeans(
  m0_agr,
  ~ source_is_synthetic * clinical_group,
  at = list(source_is_synthetic = c(0, 1))
)

emm_agr_df <- as.data.frame(emm_agr) %>%
  mutate(
    population = factor(
      paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  )

# Raw data
df_plot <- df_all %>%
  mutate(
    population = factor(
      paste(ifelse(source_is_synthetic == 1, "Synthetic", "Real"), clinical_group),
      levels = c("Real TD", "Synthetic TD", "Real SLI", "Synthetic SLI")
    )
  ) %>%
  filter(!is.na(agreement_errors_per_finite_verb), !is.na(population))

# Plot
ggplot(df_plot, aes(x = population, y = agreement_errors_per_finite_verb, color = population)) +
  geom_boxplot(width = 0.45, alpha = 0.15, outlier.shape = NA) +
  geom_jitter(width = 0.10, alpha = 0.22, size = 1.7) +
  geom_pointrange(
    data = emm_agr_df,
    aes(x = population, y = emmean, ymin = lower.CL, ymax = upper.CL),
    inherit.aes = FALSE,
    linewidth = 0.9,
    color = "black"
  ) +
  labs(
    x = NULL,
    y = "Agreement errors per finite verb"
  ) +
  theme_classic() +
  theme(legend.position = "none")


## -----------------------------------------------------------------------------
library(splines)
library(lme4)

m_lin_int <- lmer(
  Syntax_AuxCop_Utt_Prop ~ source_is_synthetic * clinical_group * Age_c +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_spline_int <- lmer(
  Syntax_AuxCop_Utt_Prop ~ source_is_synthetic * clinical_group +
    source_is_synthetic:ns(Age_Month, df = 3) +
    clinical_group:ns(Age_Month, df = 3) +
    ns(Age_Month, df = 3) +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

AIC(m_lin_int, m_spline_int)
BIC(m_lin_int, m_spline_int)


## -----------------------------------------------------------------------------
m0_auxcop <- lmer(
  Syntax_AuxCop_Utt_Prop ~ source_is_synthetic * clinical_group +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m1_auxcop <- lmer(
  Syntax_AuxCop_Utt_Prop ~ source_is_synthetic * clinical_group +
    Age_c  + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m2_auxcop <- lmer(
  Syntax_AuxCop_Utt_Prop ~ source_is_synthetic * clinical_group +
    Age_c +
    source_is_synthetic:Age_c +
    clinical_group:Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m3_auxcop <- lmer(
  Syntax_AuxCop_Utt_Prop ~ source_is_synthetic * clinical_group * Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

anova(m0_auxcop, m1_auxcop, m2_auxcop, m3_auxcop)

## -----------------------------------------------------------------------------
summary(m2_auxcop)


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

# Adjusted means from the primary model
emm_auxcop <- emmeans(
  m2_auxcop,
  ~ source_is_synthetic * clinical_group,
  at = list(source_is_synthetic = c(0, 1))
)

emm_auxcop_df <- as.data.frame(emm_auxcop) %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

# Raw data
df_plot <- df_all %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  ) %>%
  filter(!is.na(Syntax_AuxCop_Utt_Prop))

# Plot
ggplot(df_plot, aes(x = source_label, y = Syntax_AuxCop_Utt_Prop, color = source_label)) +
  geom_boxplot(width = 0.45, alpha = 0.15, outlier.shape = NA) +
  geom_jitter(width = 0.10, alpha = 0.22, size = 1.7) +
  geom_pointrange(
    data = emm_auxcop_df,
    aes(x = source_label, y = emmean, ymin = lower.CL, ymax = upper.CL),
    inherit.aes = FALSE,
    linewidth = 0.9,
    color = "black"
  ) +
  facet_wrap(~ clinical_group) +
  labs(
    x = "Source",
    y = "Aux/Cop utterance proportion"
  ) +
  theme_classic() +
  theme(legend.position = "none")


## -----------------------------------------------------------------------------

age_mean <- mean(df_all$Age_Month, na.rm = TRUE)

df_age_plot <- df_all %>%
  mutate(
    Age_Month_plot = Age_c + age_mean,
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  ) %>%
  filter(
    !is.na(Syntax_AuxCop_Utt_Prop),
    !is.na(Age_c),
    !is.na(Age_Month_plot)
  )

age_grid <- seq(
  min(df_age_plot$Age_c, na.rm = TRUE),
  max(df_age_plot$Age_c, na.rm = TRUE),
  length.out = 100
)

emm_age <- emmeans(
  m2_auxcop,
  ~ source_is_synthetic * clinical_group * Age_c,
  at = list(
    source_is_synthetic = c(0, 1),
    Age_c = age_grid
  )
)

pred_df <- as.data.frame(emm_age) %>%
  mutate(
    Age_Month_plot = Age_c + age_mean,
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

ggplot(df_age_plot, aes(x = Age_Month_plot, y = Syntax_AuxCop_Utt_Prop, color = source_label)) +
  geom_jitter(width = 0.15, height = 0, alpha = 0.18, size = 1.6) +
  geom_line(
    data = pred_df,
    aes(x = Age_Month_plot, y = emmean, color = source_label),
    linewidth = 1.1,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ clinical_group) +
  labs(
    x = "Age (months)",
    y = "Aux/Cop utterance proportion",
    color = "Source"
  ) +
  theme_classic()


## -----------------------------------------------------------------------------
df_dev_all_clean <- df_dev_all_clean %>%
  mutate(
    det_art_per_utt = `det:art` / `#utterances`
  )


## -----------------------------------------------------------------------------
library(splines)
library(lme4)

m_lin_int <- lmer(
  det_art_per_utt ~ source_is_synthetic * clinical_group * Age_c +
    (1 | cluster_id),
  data = df_dev_all_clean, REML = FALSE
)

m_spline_int <- lmer(
  det_art_per_utt ~ source_is_synthetic * clinical_group +
    source_is_synthetic:ns(Age_Month, df = 3) +
    clinical_group:ns(Age_Month, df = 3) +
    ns(Age_Month, df = 3) +
    (1 | cluster_id),
  data = df_dev_all_clean, REML = FALSE
)

AIC(m_lin_int, m_spline_int)
BIC(m_lin_int, m_spline_int)


## -----------------------------------------------------------------------------
m0_detuse <- lmer(
  det_art_per_utt ~ source_is_synthetic * clinical_group +
     + (1 | cluster_id),
  data = df_dev_all_clean,
  REML = FALSE
)

m1_detuse <- lmer(
  det_art_per_utt ~ source_is_synthetic * clinical_group +
    Age_c  + (1 | cluster_id),
  data = df_dev_all_clean,
  REML = FALSE
)

m2_detuse <- lmer(
  det_art_per_utt ~ source_is_synthetic * clinical_group +
    Age_c +
    source_is_synthetic:Age_c +
    clinical_group:Age_c +
     + (1 | cluster_id),
  data = df_dev_all_clean,
  REML = FALSE
)

m3_detuse <- lmer(
  det_art_per_utt ~ source_is_synthetic * clinical_group * Age_c +
     + (1 | cluster_id),
  data = df_dev_all_clean,
  REML = FALSE
)

anova(m0_detuse, m1_detuse, m2_detuse, m3_detuse)

## -----------------------------------------------------------------------------
summary(m2_detuse)


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

# Create emmeans object from the retained model
emm_detuse <- emmeans(
  m2_detuse,
  ~ source_is_synthetic * clinical_group,
  at = list(source_is_synthetic = c(0, 1))
)

emm_detuse_df <- as.data.frame(emm_detuse) %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

df_plot <- df_dev_all_clean %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  ) %>%
  filter(!is.na(det_art_per_utt))

ggplot(df_plot, aes(x = source_label, y = det_art_per_utt, color = source_label)) +
  geom_boxplot(width = 0.45, alpha = 0.15, outlier.shape = NA) +
  geom_jitter(width = 0.10, alpha = 0.22, size = 1.7) +
  geom_pointrange(
    data = emm_detuse_df,
    aes(x = source_label, y = emmean, ymin = lower.CL, ymax = upper.CL),
    inherit.aes = FALSE,
    linewidth = 0.9,
    color = "black"
  ) +
  facet_wrap(~ clinical_group) +
  labs(
    x = "Source",
    y = "Determiner tokens per utterance"
  ) +
  theme_classic() +
  theme(legend.position = "none")


## -----------------------------------------------------------------------------
library(emmeans)

age_mean <- mean(df_dev_all_clean$Age_Month, na.rm = TRUE)

df_age_plot <- df_dev_all_clean %>%
  mutate(
    Age_Month_plot = Age_c + age_mean,
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  ) %>%
  filter(
    !is.na(det_art_per_utt),
    !is.na(Age_c),
    !is.na(Age_Month_plot)
  )

age_grid <- seq(
  min(df_age_plot$Age_c, na.rm = TRUE),
  max(df_age_plot$Age_c, na.rm = TRUE),
  length.out = 100
)

emm_age <- emmeans(
  m2_detuse,
  ~ source_is_synthetic * clinical_group * Age_c,
  at = list(
    source_is_synthetic = c(0, 1),
    Age_c = age_grid
  )
)

pred_df <- as.data.frame(emm_age) %>%
  mutate(
    Age_Month_plot = Age_c + age_mean,
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

ggplot(df_age_plot, aes(x = Age_Month_plot, y = det_art_per_utt, color = source_label)) +
  geom_jitter(width = 0.15, height = 0, alpha = 0.18, size = 1.6) +
  geom_line(
    data = pred_df,
    aes(x = Age_Month_plot, y = emmean, color = source_label),
    linewidth = 1.1,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ clinical_group) +
  labs(
    x = "Age (months)",
    y = "Determiner tokens per utterance",
    color = "Source"
  ) #+
  theme_classic()


## -----------------------------------------------------------------------------
library(splines)
library(lme4)

m_lin_int <- lmer(
  Syntax_Mean_TreeDepth ~ source_is_synthetic * clinical_group * Age_c +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_spline_int <- lmer(
  Syntax_Mean_TreeDepth ~ source_is_synthetic * clinical_group +
    source_is_synthetic:ns(Age_Month, df = 3) +
    clinical_group:ns(Age_Month, df = 3) +
    ns(Age_Month, df = 3) +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

AIC(m_lin_int, m_spline_int)
BIC(m_lin_int, m_spline_int)


## -----------------------------------------------------------------------------
m0_mtd <- lmer(
  Syntax_Mean_TreeDepth ~ source_is_synthetic * clinical_group +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m1_mtd <- lmer(
  Syntax_Mean_TreeDepth ~ source_is_synthetic * clinical_group +
    Age_c  + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m2_mtd <- lmer(
  Syntax_Mean_TreeDepth ~ source_is_synthetic * clinical_group +
    Age_c +
    source_is_synthetic:Age_c +
    clinical_group:Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m3_mtd <- lmer(
  Syntax_Mean_TreeDepth ~ source_is_synthetic * clinical_group * Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

anova(m0_mtd, m1_mtd, m2_mtd, m3_mtd)



## -----------------------------------------------------------------------------
summary(m2_mtd)


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

# Adjusted means from the retained model
emm_mtd <- emmeans(
  m2_mtd,
  ~ source_is_synthetic * clinical_group,
  at = list(source_is_synthetic = c(0, 1))
)

emm_mtd_df <- as.data.frame(emm_mtd) %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

# Raw data
df_plot <- df_all %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  ) %>%
  filter(!is.na(Syntax_Mean_TreeDepth))

# Plot
ggplot(df_plot, aes(x = source_label, y = Syntax_Mean_TreeDepth, color = source_label)) +
  geom_boxplot(width = 0.45, alpha = 0.15, outlier.shape = NA) +
  geom_jitter(width = 0.10, alpha = 0.22, size = 1.7) +
  geom_pointrange(
    data = emm_mtd_df,
    aes(x = source_label, y = emmean, ymin = lower.CL, ymax = upper.CL),
    inherit.aes = FALSE,
    linewidth = 0.9,
    color = "black"
  ) +
  facet_wrap(~ clinical_group) +
  labs(
    x = "Source",
    y = "Mean tree depth"
  ) +
  theme_classic() +
  theme(legend.position = "none")


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

age_mean <- mean(df_all$Age_Month, na.rm = TRUE)

df_age_plot <- df_all %>%
  mutate(
    Age_Month_plot = Age_c + age_mean,
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  ) %>%
  filter(
    !is.na(Syntax_Mean_TreeDepth),
    !is.na(Age_c),
    !is.na(Age_Month_plot)
  )

age_grid <- seq(
  min(df_age_plot$Age_c, na.rm = TRUE),
  max(df_age_plot$Age_c, na.rm = TRUE),
  length.out = 100
)

emm_age <- emmeans(
  m2_mtd,
  ~ source_is_synthetic * clinical_group * Age_c,
  at = list(
    source_is_synthetic = c(0, 1),
    Age_c = age_grid
  )
)

pred_df <- as.data.frame(emm_age) %>%
  mutate(
    Age_Month_plot = Age_c + age_mean,
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

ggplot(df_age_plot, aes(x = Age_Month_plot, y = Syntax_Mean_TreeDepth, color = source_label)) +
  geom_jitter(width = 0.15, height = 0, alpha = 0.18, size = 1.6) +
  geom_line(
    data = pred_df,
    aes(x = Age_Month_plot, y = emmean, color = source_label),
    linewidth = 1.1,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ clinical_group) +
  labs(
    x = "Age (months)",
    y = "Mean tree depth",
    color = "Source"
  ) +
  theme_classic()


## -----------------------------------------------------------------------------
library(splines)
library(lme4)

m_lin_int <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group * Age_c +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_spline_int <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group +
    source_is_synthetic:ns(Age_Month, df = 3) +
    clinical_group:ns(Age_Month, df = 3) +
    ns(Age_Month, df = 3) +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

AIC(m_lin_int, m_spline_int)
BIC(m_lin_int, m_spline_int)


## -----------------------------------------------------------------------------
m0_cnp <- lmer(
  Syntax_ComplexNP_Utt_Prop ~ source_is_synthetic * clinical_group +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m1_cnp <- lmer(
  Syntax_ComplexNP_Utt_Prop ~ source_is_synthetic * clinical_group +
    Age_c  + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m2_cnp <- lmer(
  Syntax_ComplexNP_Utt_Prop ~ source_is_synthetic * clinical_group +
    Age_c +
    source_is_synthetic:Age_c +
    clinical_group:Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m3_cnp <- lmer(
  Syntax_ComplexNP_Utt_Prop ~ source_is_synthetic * clinical_group * Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

anova(m0_cnp, m1_cnp, m2_cnp, m3_cnp)


## -----------------------------------------------------------------------------
summary(m0_cnp)


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

# Adjusted means from the retained model
emm_cnp <- emmeans(
  m0_cnp,
  ~ source_is_synthetic * clinical_group,
  at = list(source_is_synthetic = c(0, 1))
)

emm_cnp_df <- as.data.frame(emm_cnp) %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

# Raw data
df_plot <- df_all %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  ) %>%
  filter(!is.na(Syntax_ComplexNP_Utt_Prop))

# Plot
ggplot(df_plot, aes(x = source_label, y = Syntax_ComplexNP_Utt_Prop, color = source_label)) +
  geom_boxplot(width = 0.45, alpha = 0.15, outlier.shape = NA) +
  geom_jitter(width = 0.10, alpha = 0.22, size = 1.7) +
  geom_pointrange(
    data = emm_cnp_df,
    aes(x = source_label, y = emmean, ymin = lower.CL, ymax = upper.CL),
    inherit.aes = FALSE,
    linewidth = 0.9,
    color = "black"
  ) +
  facet_wrap(~ clinical_group) +
  labs(
    x = "Source",
    y = "Complex NP utterance proportion"
  ) +
  theme_classic() +
  theme(legend.position = "none")


## -----------------------------------------------------------------------------
library(splines)
library(lme4)

m_lin_int <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group * Age_c +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_spline_int <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group +
    source_is_synthetic:ns(Age_Month, df = 3) +
    clinical_group:ns(Age_Month, df = 3) +
    ns(Age_Month, df = 3) +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

AIC(m_lin_int, m_spline_int)
BIC(m_lin_int, m_spline_int)


## -----------------------------------------------------------------------------
m0_sub <- lmer(
  Syntax_Subordination_Utt_Prop ~ source_is_synthetic * clinical_group +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m1_sub <- lmer(
  Syntax_Subordination_Utt_Prop ~ source_is_synthetic * clinical_group +
    Age_c  + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m2_sub <- lmer(
  Syntax_Subordination_Utt_Prop ~ source_is_synthetic * clinical_group +
    Age_c +
    source_is_synthetic:Age_c +
    clinical_group:Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m3_sub <- lmer(
  Syntax_Subordination_Utt_Prop ~ source_is_synthetic * clinical_group * Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

anova(m0_sub, m1_sub, m2_sub, m3_sub)

## -----------------------------------------------------------------------------
summary(m1_sub)


## -----------------------------------------------------------------------------
library(dplyr)
library(ggplot2)
library(emmeans)

# Adjusted means from the retained model
emm_sub <- emmeans(
  m1_sub,
  ~ source_is_synthetic * clinical_group,
  at = list(source_is_synthetic = c(0, 1))
)

emm_sub_df <- as.data.frame(emm_sub) %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

# Raw data
df_plot <- df_all %>%
  mutate(
    source_label = factor(
      source_is_synthetic,
      levels = c(0, 1),
      labels = c("Real", "Synthetic")
    ),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  ) %>%
  filter(!is.na(Syntax_Subordination_Utt_Prop))

# Plot
ggplot(df_plot, aes(x = source_label, y = Syntax_Subordination_Utt_Prop, color = source_label)) +
  geom_boxplot(width = 0.45, alpha = 0.15, outlier.shape = NA) +
  geom_jitter(width = 0.10, alpha = 0.22, size = 1.7) +
  geom_pointrange(
    data = emm_sub_df,
    aes(x = source_label, y = emmean, ymin = lower.CL, ymax = upper.CL),
    inherit.aes = FALSE,
    linewidth = 0.9,
    color = "black"
  ) +
  facet_wrap(~ clinical_group) +
  labs(
    x = "Source",
    y = "Subordination utterance proportion"
  ) +
  theme_classic() +
  theme(legend.position = "none")

## -----------------------------------------------------------------------------
library(emmeans)

pairs(
  emmeans(
    m1_sub,
    ~ source_is_synthetic | clinical_group,
    at = list(source_is_synthetic = c(0, 1))
  ),
  infer = c(TRUE, TRUE),
  adjust = "none"
)


## -----------------------------------------------------------------------------
library(splines)
library(lme4)

m_lin_int <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group * Age_c +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

m_spline_int <- lmer(
  VOCD_D_optimum_average ~ source_is_synthetic * clinical_group +
    source_is_synthetic:ns(Age_Month, df = 3) +
    clinical_group:ns(Age_Month, df = 3) +
    ns(Age_Month, df = 3) +
    (1 | cluster_id),
  data = df_all, REML = FALSE
)

AIC(m_lin_int, m_spline_int)
BIC(m_lin_int, m_spline_int)


## -----------------------------------------------------------------------------
m0_subj <- lmer(
  Syntax_Subject_Utt_Prop ~ source_is_synthetic * clinical_group +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m1_subj <- lmer(
  Syntax_Subject_Utt_Prop ~ source_is_synthetic * clinical_group +
    Age_c + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m2_subj <- lmer(
  Syntax_Subject_Utt_Prop ~ source_is_synthetic * clinical_group +
    Age_c +
    source_is_synthetic:Age_c +
    clinical_group:Age_c +
     + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

m3_subj <- lmer(
  Syntax_Subject_Utt_Prop ~ source_is_synthetic * clinical_group * Age_c +
    factor(round_batch) + (1 | cluster_id),
  data = df_all,
  REML = FALSE
)

anova(m0_subj, m1_subj, m2_subj, m3_subj)


## -----------------------------------------------------------------------------
summary(m2_subj)


## -----------------------------------------------------------------------------
library(ggplot2)
library(dplyr)
library(emmeans)

# EMMs for m2_subj across observed age range
age_grid <- with(df_all, seq(min(Age_c, na.rm = TRUE), max(Age_c, na.rm = TRUE), length.out = 100))

emm_subj <- emmeans(
  m2_subj,
  ~ source_is_synthetic * clinical_group * Age_c,
  at = list(Age_c = age_grid)
)

emm_subj_df <- as.data.frame(emm_subj) %>%
  mutate(
    source = ifelse(source_is_synthetic == 1, "Synthetic", "Real"),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

# raw data with nicer labels
plot_df <- df_all %>%
  mutate(
    source = ifelse(source_is_synthetic == 1, "Synthetic", "Real"),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

ggplot(plot_df, aes(x = Age_Month, y = Syntax_Subject_Utt_Prop, color = source)) +
  geom_jitter(width = 0.25, height = 0, alpha = 0.22, size = 1.4) +
  geom_ribbon(
    data = emm_subj_df,
    aes(
      x = Age_c + mean(df_all$Age_Month, na.rm = TRUE),
      y = emmean,
      ymin = lower.CL,
      ymax = upper.CL,
      fill = source
    ),
    alpha = 0.18,
    colour = NA,
    inherit.aes = FALSE
  ) +
  geom_line(
    data = emm_subj_df,
    aes(
      x = Age_c + mean(df_all$Age_Month, na.rm = TRUE),
      y = emmean,
      color = source
    ),
    linewidth = 1.2,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ clinical_group) +
  scale_color_manual(values = c("Real" = "#1b9e77", "Synthetic" = "#d95f02")) +
  scale_fill_manual(values = c("Real" = "#1b9e77", "Synthetic" = "#d95f02")) +
  labs(
    x = "Age (months)",
    y = "Proportion of utterances with a subject",
    color = "Source",
    fill = "Source",
    title = "Subject expression by age, source, and clinical group"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

## -----------------------------------------------------------------------------
emm_subj_means <- emmeans(m2_subj, ~ source_is_synthetic * clinical_group)

emm_subj_means_df <- as.data.frame(emm_subj_means) %>%
  mutate(
    source = ifelse(source_is_synthetic == 1, "Synthetic", "Real"),
    clinical_group = factor(clinical_group, levels = c("TD", "SLI"))
  )

ggplot(plot_df, aes(x = source, y = Syntax_Subject_Utt_Prop, color = source)) +
  geom_jitter(width = 0.15, height = 0, alpha = 0.20, size = 1.4) +
  geom_pointrange(
    data = emm_subj_means_df,
    aes(x = source, y = emmean, ymin = lower.CL, ymax = upper.CL),
    colour = "black",
    size = 0.6,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ clinical_group) +
  labs(
    x = "Source",
    y = "Proportion of utterances with a subject",
    title = "Subject expression by source and clinical group"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank()
  )


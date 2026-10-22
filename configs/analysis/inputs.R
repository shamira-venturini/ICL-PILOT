# REVISED STUDY: INPUT PATHS PENDING
# Update these five entries when the revised files and their paths are known.
# Paths may be absolute, or relative to the repository root. No historical
# dataset or roster is selected automatically. See README.md in this folder.
analysis_input_paths <- list(
  real_features_csv = "",
  synthetic_features_csv = "",
  frozen_roster_csv = "",
  real_dev_csv = "",
  synthetic_dev_csv = ""
)

# Shared base-R input resolution for both R Markdown documents and the retained
# extracted R script. This checks paths only; it does not read or alter data.
resolve_analysis_inputs <- function(project_root = ".", paths = analysis_input_paths) {
  environment_names <- c(
    real_features_csv = "ICL_PILOT_REAL_CSV",
    synthetic_features_csv = "ICL_PILOT_SYN_CSV",
    frozen_roster_csv = "ICL_PILOT_ROSTER_CSV",
    real_dev_csv = "ICL_PILOT_REAL_DEV_CSV",
    synthetic_dev_csv = "ICL_PILOT_SYN_DEV_CSV"
  )
  required <- names(environment_names)
  if (!all(required %in% names(paths))) {
    stop("Analysis configuration must contain all five input entries in configs/analysis/inputs.R.", call. = FALSE)
  }
  values <- vapply(paths[required], function(value) {
    if (!is.character(value) || length(value) != 1L || is.na(value)) {
      stop("Each analysis input must be a single path string; leave pending paths as an empty string.", call. = FALSE)
    }
    value
  }, character(1))
  overrides <- Sys.getenv(unname(environment_names), unset = "")
  supplied <- nzchar(overrides)
  values[supplied] <- overrides[supplied]

  pending <- !nzchar(trimws(values))
  if (any(pending)) {
    stop(paste0(
      "REVISED STUDY: input paths are pending.\n",
      "Update configs/analysis/inputs.R when the revised files and paths are available.\n",
      "Missing entries: ", paste(names(values)[pending], collapse = ", "), ".\n",
      "No archived Phase 2 inputs are selected automatically."
    ), call. = FALSE)
  }

  resolved <- path.expand(values)
  absolute <- grepl("^(/|[A-Za-z]:[/\\\\]|\\\\\\\\)", resolved)
  resolved[!absolute] <- file.path(project_root, resolved[!absolute])
  invalid <- !file.exists(resolved) | dir.exists(resolved)
  if (any(invalid)) {
    stop(paste0(
      "Configured analysis inputs are missing or are directories:\n",
      paste(paste0("- ", names(values)[invalid], ": ", resolved[invalid]), collapse = "\n"),
      "\nUpdate configs/analysis/inputs.R or the corresponding ICL_PILOT_*_CSV overrides."
    ), call. = FALSE)
  }
  resolved <- normalizePath(resolved, winslash = "/", mustWork = TRUE)
  names(resolved) <- names(values)
  as.list(resolved)
}

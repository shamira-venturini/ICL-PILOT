# Base-R regression checks; run from the repository root:
# Rscript --vanilla tests/test_r_analysis_inputs.R
local({
  project_root <- normalizePath(".", winslash = "/", mustWork = TRUE)
  config <- new.env(parent = baseenv())
  source(file.path(project_root, "configs", "analysis", "inputs.R"), local = config)
  keys <- names(config$analysis_input_paths)
  variables <- c("ICL_PILOT_REAL_CSV", "ICL_PILOT_SYN_CSV", "ICL_PILOT_ROSTER_CSV",
                 "ICL_PILOT_REAL_DEV_CSV", "ICL_PILOT_SYN_DEV_CSV", "ICL_PILOT_PROJECT_ROOT")
  original <- Sys.getenv(variables, unset = NA_character_)
  on.exit({
    Sys.unsetenv(variables)
    restore <- !is.na(original)
    if (any(restore)) do.call(Sys.setenv, as.list(setNames(original[restore], variables[restore])))
  }, add = TRUE)
  Sys.unsetenv(variables)
  fixture <- tempfile("icl_r_inputs_")
  dir.create(fixture)
  on.exit(unlink(fixture, recursive = TRUE), add = TRUE)

  expect_error <- function(expr, pattern) {
    error <- tryCatch({ force(expr); NULL }, error = identity)
    stopifnot(inherits(error, "error"), grepl(pattern, conditionMessage(error), fixed = TRUE))
    invisible(error)
  }

  # A checked-in pending config cannot fall back to existing historical files.
  expect_error(config$resolve_analysis_inputs(project_root), "REVISED STUDY: input paths are pending")
  relative <- setNames(as.list(paste0("fixture ", seq_along(keys), ".csv")), keys)
  for (path in relative) writeLines("fixture", file.path(fixture, path))
  resolved <- config$resolve_analysis_inputs(fixture, relative)
  stopifnot(identical(names(resolved), keys), all(file.exists(unlist(resolved))))
  stopifnot(identical(resolved$real_features_csv,
                      normalizePath(file.path(fixture, relative$real_features_csv), winslash = "/")))
  stopifnot(identical(config$resolve_analysis_inputs(fixture, resolved), resolved))

  missing <- relative
  missing$synthetic_dev_csv <- "not available.csv"
  expect_error(config$resolve_analysis_inputs(fixture, missing), "synthetic_dev_csv: ")
  directory <- relative
  directory$real_features_csv <- fixture
  expect_error(config$resolve_analysis_inputs(fixture, directory), "missing or are directories")

  override <- file.path(fixture, "selected reference.csv")
  writeLines("selected fixture", override)
  Sys.setenv(ICL_PILOT_REAL_CSV = override)
  stopifnot(identical(config$resolve_analysis_inputs(fixture, relative)$real_features_csv,
                      normalizePath(override, winslash = "/")))
  expect_error(config$resolve_analysis_inputs(fixture), "synthetic_features_csv")
  Sys.unsetenv("ICL_PILOT_REAL_CSV")

  for (index in seq_along(keys)) {
    do.call(Sys.setenv, setNames(list(resolved[[keys[index]]]), variables[index]))
  }
  stopifnot(identical(config$resolve_analysis_inputs(fixture), resolved))
  Sys.unsetenv(variables)

  # The real entry point stops at the signpost before packages or data loading,
  # including when the checkout is elsewhere than the working directory.
  Sys.setenv(ICL_PILOT_PROJECT_ROOT = project_root)
  previous_wd <- getwd()
  on.exit(setwd(previous_wd), add = TRUE)
  setwd(fixture)
  entries <- "notebooks/clinical_analyses_reordered.Rmd"
  for (entry in entries) {
    lines <- readLines(file.path(project_root, entry), warn = FALSE)
    if (grepl("Rmd$", entry)) {
      start <- which(lines == "```{r analysis-inputs}")
      stopifnot(length(start) == 1L)
      end <- which(seq_along(lines) > start & lines == "```")[1]
      setup <- lines[seq.int(start + 1L, end - 1L)]
    } else {
      first_library <- grep("^library\\(", lines)[1]
      setup <- lines[seq_len(first_library - 1L)]
    }
    environment <- new.env(parent = baseenv())
    environment$read.csv <- function(...) stop("Data loading must not occur during the pending-input check")
    expect_error(eval(parse(text = setup), envir = environment), "REVISED STUDY: input paths are pending")
  }
  setwd(previous_wd)
  # Knitting from notebooks/ also finds the checkout without an override.
  Sys.unsetenv("ICL_PILOT_PROJECT_ROOT")
  setwd(file.path(project_root, "notebooks"))
  expect_error(eval(parse(text = setup), envir = new.env(parent = baseenv())), "REVISED STUDY: input paths are pending")
  setwd(previous_wd)
  cat("R input checks passed: pending paths, explicit files, relative/absolute paths, overrides, missing files and the maintained R entry point.\n")
})

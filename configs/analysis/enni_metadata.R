# Canonical ENNI CHI metadata, independent of CLAN's rounded age column.
# This changes input metadata only; it does not change statistical specifications.
reconcile_enni_metadata <- function(frame, manifest_csv) {
  canonical <- read.csv(manifest_csv, check.names = FALSE, stringsAsFactors = FALSE,
                        colClasses = c(File_ID = "character"))
  if (!all(c("File_ID", "Group") %in% names(frame)) ||
      !all(c("File_ID", "Group", "Age_Raw", "Age(Month)", "File_kideval", "Sex") %in% names(canonical))) {
    stop("ENNI features and canonical manifest require participant IDs and CHI metadata.", call. = FALSE)
  }
  frame$File_ID <- as.character(frame$File_ID)
  if (anyDuplicated(canonical$File_ID) || anyDuplicated(frame$File_ID)) {
    stop("Duplicate ENNI participant IDs; resolve source identities before analysis.", call. = FALSE)
  }
  index <- match(frame$File_ID, canonical$File_ID)
  bad <- is.na(index) | is.na(frame$Group) |
    as.character(frame$Group) != canonical$Group[index]
  bad[is.na(bad)] <- TRUE
  if (any(bad)) {
    stop(paste0("ENNI participant/group conflict: ", paste(frame$File_ID[bad], collapse = ", "),
                ". Rebuild features from verified source transcripts; do not relabel measurements from TD/B/523.cha."),
         call. = FALSE)
  }
  # When a source path is available, the same numeric ID is insufficient evidence.
  for (column in intersect(c("File_kideval", "transcript_path", "DSS_tbl_transcript_relpath"), names(frame))) {
    values <- gsub("\\\\", "/", as.character(frame[[column]]))
    expected <- canonical$File_kideval[index]
    populated <- !is.na(values) & nzchar(values)
    matches <- mapply(endsWith, values, expected)
    if (any(populated & !matches)) {
      stop(paste0("ENNI feature source path conflicts with the verified corpus: ", column,
                  ". Rebuild affected features before analysis."), call. = FALSE)
    }
  }
  old_age <- if ("Age(Month)" %in% names(frame)) as.numeric(frame[["Age(Month)"]]) else rep(NA_real_, nrow(frame))
  new_age <- canonical[["Age(Month)"]][index]
  changed <- is.na(old_age) | old_age != new_age
  attr(frame, "enni_age_corrections") <- data.frame(File_ID = frame$File_ID[changed],
       previous_age_months = old_age[changed], completed_age_months = new_age[changed])
  frame[["Age(Month)"]] <- new_age
  frame$Age_Raw <- canonical$Age_Raw[index]
  frame$Age <- canonical$Age_Raw[index]
  frame$Sex <- canonical$Sex[index]
  message("ENNI metadata verified against literal CHI headers; corrected age values: ", sum(changed))
  frame
}

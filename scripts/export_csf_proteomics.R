#!/usr/bin/env Rscript
# Export CSF MRM proteomics from .rdata to CSV
# Run once locally or on Hellbender before Python pipeline

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
  rdata_path <- args[1]
} else {
  # Default paths (try local first, then Hellbender)
  candidates <- c(
    file.path(Sys.getenv("DIGITALTWIN_ROOT", ""), "Data/ADNIMERGE/data/csfmrm.rdata"),
    "/Users/nathanbresette/Desktop/DigitalTwin/Data/ADNIMERGE/data/csfmrm.rdata",
    "/home/nbhtd/data/digitaltwin/ADNIMERGE/data/csfmrm.rdata"
  )
  rdata_path <- ""
  for (p in candidates) {
    if (file.exists(p)) { rdata_path <- p; break }
  }
  if (rdata_path == "") stop("Cannot find csfmrm.rdata")
}

cat(sprintf("Loading %s\n", rdata_path))
env <- new.env()
load(rdata_path, envir = env)
obj_name <- ls(env)[1]
df <- get(obj_name, envir = env)
cat(sprintf("  Object: %s (%d rows x %d cols)\n", obj_name, nrow(df), ncol(df)))

# Output to same directory as this script's parent results/
script_dir <- dirname(sys.frame(1)$ofile %||% ".")
out_dir <- file.path(dirname(script_dir), "results")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

out_path <- file.path(out_dir, "csfmrm_raw.csv")
write.csv(df, out_path, row.names = FALSE)
cat(sprintf("Saved -> %s\n", out_path))
cat(sprintf("  %d unique RIDs\n", length(unique(df$RID))))

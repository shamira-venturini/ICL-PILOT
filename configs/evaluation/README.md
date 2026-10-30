# Framework evaluation inputs and protocol

inputs.json requires two explicit files:

- text_records_csv: definitive ENNI reference stories and the selected three-model synthetic outputs.
- gillam_external_text_csv: independently selected external Gillam stories.

Unset inputs stop the notebook before result creation. Scientific evaluation must use new generation records with complete model, prompt and parent provenance.

The original Gillam archive has been validated and its external input is set to
data/inputs/gillam/external_story_records.csv. The sample contains 182 children
(91 TD, 91 SLI), aged 60--95 completed months, with TD counts matched to all eligible
SLI children within one-year age-cohort by sex strata. TD selection uses fixed
hash-ranking seed 20260827. The explicit child IDs a-1-68pre/a-1-68 identify
two visits by one child; matching uses the earliest eligible recording, a-1-68pre.
Session identity and the exclusion of the later visit are recorded separately.
Three SLI children have two available canonical
stories; the export retains all available stories: 543 narratives, 273 TD and
270 SLI. Gillam is balanced by child, not by story count. The raw archive is
preserved under data/raw/gillam/2026-10-28/. ENNI/generated combined text remains
pending. See docs/reproducibility/gillam.json for source and selection checks.

For ENNI text export, use configs/generation/inputs/enni_participants_2026-10-28.csv as the real metadata table and story_unit_narratives_enni_2026-10-28.csv in that directory as the story table. No linguistic feature table is needed to obtain reference ages/groups.

Provide generation CSVs explicitly to the exporter, together with the metadata, stories and a fresh output destination:

    PYTHONPATH=src python -m icl_pilot.evaluation.story_text_exports --help

protocol.json defines generator identities, common sampling, classifier budgets/seeds and conditional bootstrap settings. The notebook reads model pins from the generation catalogue. See [the framework guide](../../docs/guides/framework_evaluation.md).

The notebook also contains a separate optional synthetic:real ratio block. Its
prespecified default levels are 1:1 and 10:1, with a shared real-only baseline and
update-matched real-only controls at each level. Set RUN_RATIO_EXPERIMENT=True
in that block to execute it; False writes plans only. It uses the same prepared
splits and nested synthetic selections, saves its own configuration/results under
ratio_experiment/, and reuses only exactly matching verified completed fits.
Protocol schema 2 uses all 238 real children in five age-stratified, grouped
classifier folds independent of generation batches. Complete synthetic training
bundles are bootstrapped with replacement to preserve the real training fold's
age/group/story counts at every sampled ratio. Unweighted loss is retained.
Record unique narratives separately from sampled exposures; Gillam is unchanged.

Results are written to fresh directories under results/framework_based_evaluation/. Keep per-generator outputs, shared controls, predictions, split/count audits, text normalization and parent hashes. Missing or conflicting clinical identities are rejected; age is derived from literal CHAT age rather than rounded measurement columns.

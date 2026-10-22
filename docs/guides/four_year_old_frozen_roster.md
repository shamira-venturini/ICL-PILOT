# Four-year-old generation roster

The portable generator selects `cohort == "4-year-old"` from
`configs/generation/master_frozen_roster_4_to_7_year_old.csv`.
These eleven child pairs and their order come from the original generation CSV,
with age/KIDEVAL metadata joined by child ID. It does not optimize new pairs.

Two recorded pairs differ from the archived older roster:

| SLI child | TD child actually used | TD child's raw CHAT age |
| --- | --- | --- |
| 474 | 459 | 4;08.04 |
| 475 | 417 | 4;07.26 |

The new configuration preserves these identities and their recorded rounds.
The author confirmed that age four used the same generation settings as ages
five through seven, so all ages share `configs/generation/experiment.json`.

The old standalone roster and its profile labels are preserved in the verified
external generation backup. The later four-cohort inputs and obsolete combined
master are backed up under
`/Volumes/TranscendBU/ICL-PILOT_78a5cadad659_20261014T123617Z/GenerationRoster/`.
Existing narratives, CHAT files and analysis inputs were not rewritten.
R file references now use the single master; current R exclusions still use
actual synthetic ancestry and pass the membership audit.

The age-only pairing builder remains available for a new design or comparison.
Choose a fresh output path; it refuses to overwrite an existing frozen roster.

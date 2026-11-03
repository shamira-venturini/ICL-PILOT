# Real–synthetic annotation comparison, 2 November 2026

## Assessment

The pilot supports continuing with the common automated annotation pipeline.
Many ordinary clauses have plausible analyses in both real and synthetic data,
and clear noun/verb or finite-verb mistakes occur in both. Synthetic Stage 2
contains more structurally ambiguous fragments in the inspected sample, so its
dependency-derived measures need particular care.

This is a preliminary AI-assisted linguistic spot-check. It establishes neither
gold-standard accuracy nor equal error rates across conditions. It provides no
reason to interpret automated labels as verified clinical observations.

## Material and procedure

The real pilot contains six four-year-old ENNI children, three TD and three SLI,
with 36 canonical narratives in six CHAT files. The synthetic pilot contains 24
Qwen3-8B narratives: 12 per stage, generated from one four-year-old prompt pair
with two reversed exemplar assignments. These are processing pilots, not study
results or representative accuracy benchmarks.

UD model/catalog checksums and package versions match between runs. TalkTag
configuration matches. Real boundaries are retained from ENNI; generated text
has predicted Batchalign boundaries. This comparison therefore includes the
effects of segmentation on parsing and cannot isolate them experimentally.

Coverage checks include all 848 CHI utterances: 550 real and 298 synthetic.
Three real nonlexical utterances have no UD analysis; all 845 tagged utterances
have matching morphology/dependency node counts when clitic segments joined by
`~` are expanded correctly. Original texts remain preserved.

The linguistic inspection uses a fixed sample selected without searching for
errors: two tagged CHI utterances per real child/story and three per synthetic
story/replicate, ranked by SHA256 with seed label 42. This gives 36 utterances
per condition, 144 total, drawn from canonical A1–B3 story sections. Thirty-four
real CHI utterances outside those canonical sections were not sample candidates.

The AI assistant reviewed the selected text alongside its predicted morphology
and dependencies. Judgments were not blinded or independently adjudicated.
The inspection concentrates on coarse noun/verb distinctions, finite verbs and
salient dependency problems. It is not exhaustive word-level or attachment
annotation. Unclear analyses are recorded as uncertain rather than forced into
correct/incorrect categories.

## Inspection findings

| Condition | Inspected utterances | Clear POS/finite-verb issues | Clear dependency-relation issue only | Uncertain cases | No obvious issue in this limited check |
|---|---:|---:|---:|---:|---:|
| Real TD | 36 | 3 | 0 | 3 | 30 |
| Real DLD | 36 | 0 | 0 | 3 | 33 |
| Synthetic Stage 1 | 36 | 0 | 1 | 1 | 34 |
| Synthetic Stage 2 | 36 | 1 | 0 | 9 | 26 |

These counts describe the assistant's spot-check judgments. A zero means no
clear issue was identified under this restricted inspection, not perfect
annotation. They are not accuracy percentages or evidence that one condition
is statistically better. Utterances are clustered by child, story, prompt pair
and replicate, and some synthetic texts repeat across stages.

### Examples and consequences for measures

1. Real TD, child 410, A2: **“the elephant's running and slipping.”** The parser
   treats `'s` as possessive and running/slipping as nouns. In narrative context,
   contracted *is* with progressive verbs is the natural interpretation. The
   resulting finite-auxiliary count is zero instead of one.
2. Real TD, child 419, A1: **“elephant looked happy to giraffe.”** *Giraffe* is
   classified as an infinitive and receives an ADVCL relation. The animal-name
   noun reading is clear in context. The erroneous relation can create a false
   indication of subordination even though the finite-verb count is unchanged.
3. Real TD, child 450, B3: **“the guy's mad.”** Possessive `'s` plus noun *mad*
   replaces the natural contracted *is* plus adjective analysis, again missing
   one finite auxiliary.
4. Synthetic Stage 2, replicate 2, A1: **“ball falls in rabbit dives in rabbit
   gets ball.”** *Falls* and *dives* are tagged as plural nouns. In story context
   the utterance describes three actions, giving one detected finite lexical
   verb instead of three. The concatenated clauses also complicate dependency
   interpretation.
5. Synthetic Stage 1, replicate 2, A1: **“the ball is floating.”** The graph
   labels *is* as COP rather than AUX, while the morphology treats *floating* as
   a participial verb. This specific distinction does not change a combined
   AUX/COP-presence measure, illustrating that the impact depends on the feature.

Previously noticed targeted examples such as *bear* becoming a verb in “fox
and bear happy again” remain useful diagnostics, but they are not added to the
fixed sample's counts. Fragmentary event lists can have several possible
analyses; their trees should not be treated as uniquely established structures.

## TalkTag

The synthetic pilot has four saved error annotations, including the preposition
error excluded from the requested analysed counts. The remaining three markers
provide little evidence about error-detection sensitivity: “a airplane” supports
an article-form error, while the subject/verb analyses in fragmented or repeated
Stage 2 strings are less certain. Low annotation counts alone cannot establish
either accurate error detection or clinical fidelity.

Real examples such as *falled* and subject *him* receive recognizable error
labels. This does not establish their correctness in every context or quantify
missed errors. Three real words restored after TalkTag changes remain explicitly
unassessed for error labels; no synthetic words have that recorded coverage gap.

Some new annotations include replacement forms, such as `falled [: fell]`.
TalkBank's documented convention allows morphology to reflect a replacement
form. Produced text, error codes and UD morphology therefore answer different
questions; a normalized morphological analysis must not be interpreted as proof
that the child produced a correct inflection. See the
[CLAN manual, section 9.1](https://talkbank.org/0info/manuals/CLAN.html).

## Feature-reader compatibility finding

A separate reproducible extraction issue was found in the existing
`src/icl_pilot/features/syntax_features.py` reader. Its whitespace-based
morphology-token count does not expand `~` clitics to match dependency nodes.
Running that reader on these actual files would skip 100 correctly aligned
utterances:

| Condition | Tagged utterances skipped as token mismatches |
|---|---:|
| Real TD | 51 |
| Real DLD | 30 |
| Synthetic Stage 1 | 12 |
| Synthetic Stage 2 | 7 |

The comparison reader expands clitics and finds zero such mismatches. This is
a feature-extraction compatibility problem, not evidence of invalid UD output.
It needs correction when constructing the fresh feature tables; those tables
and the R analyses remain pending. No production feature code was changed by
this annotation-comparison task.

Subsequent implementation on 2 November corrected the syntax reader to expand
`~` components in dependency order. All 30 saved pilot CHAT files were rechecked:
all 845 tagged utterances now enter the syntax summary, with zero token
mismatches; the three nonlexical real utterances still lack UD tiers. Original
file hashes and annotation labels are unchanged. Twelve regression checks pass.
This corrects extraction coverage without establishing linguistic tag accuracy;
fresh study feature tables and R analyses remain pending. See the separate
[verification record](syntax_reader_clitics_20261102.json).

## Recommended use

- Retain the common annotation pipeline and source-preserving outputs. This
  comparison alone does not justify a change of annotation model.
- Treat POS-, morphology- and dependency-derived measures as automated
  estimates. Review the flagged examples before selecting strong interpretations
  of finite-verb, subordination, tree-depth or dependency-length differences.
- Give Stage 2 dependency findings particular scrutiny because fragmentation
  and predicted boundaries can make several structures plausible.
- Use the corrected syntax reader's clitic handling when deriving fresh feature
  tables. Word counts, surface-word lexical diversity and text classifiers do
  not depend on these UD labels; utterance-based measures still depend on
  segmentation.
- Preserve unavailable TalkTag error labels as missing, and exclude preposition
  error codes as requested. This audit does not validate TalkTag sensitivity.

No annotations, texts, prompts, selection rules or analysis specifications were
changed during the comparison. No stories were excluded. Independent human
adjudication would be required to report formal annotation accuracy or to claim
equivalent accuracy across real and synthetic conditions. These findings apply
to this four-year-old Qwen pilot; other ages and generators were not assessed.

## Review artifacts

The directory
`data/generated/smoke_diagnostics/2026-11-02/annotation_comparison/` contains:

- `reviewed_sample.csv`: all 144 selected cases, predicted tiers, judgments and reasons.
- `review_judgments.json`: assessment scope and explicit case-level decisions.
- `comparison_summary.json`: counts, source provenance and limitations.
- `legacy_reader_qc.json`: actual output from the existing syntax reader.
- `sample.json`, `all_utterances.json` and local analysis scripts for traceability.

Source archives and recovery records remain unchanged. The saved older real
annotations are also automatic Stanza output and were not used as a human gold
standard.

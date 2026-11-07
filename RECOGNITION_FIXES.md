# Flowchart recognition improvements

This update improves recognition of filled-color flowcharts without adding rules for particular labels or words.

## Changes

- Expanded arrowhead evidence checks to include small rasterized arrowheads.
- Added a constrained, warning-tagged direction fallback for connectors with exactly two node contacts where the input/decision/output roles provide a clear transition. Ambiguous connectors remain unresolved rather than being connected by reading order.
- Improved OCR cleanup for stray punctuation immediately after `If` and for a likely missing stroke in an equality operator. The original OCR reading remains available as a review alternative, and a bare single `=` is not rewritten without additional OCR-noise evidence.
- Improved branch-label inference for diagrams with a vertical branch and a sideways branch; inferred labels are marked with a warning for review.
- Accepted descriptive input labels such as `Input variable: Number` in the input notation parser.
- Added a regression fixture and tests for the supplied modulo/even-odd flowchart.

## Verification

- Unit/regression suite: `python -m pytest -q` — 50 passed, 14 skipped.
- Supplied flowchart integration test: `FLOWCHART_RUN_PIPELINE=1 python -m pytest tests/test_recognition_improvements.py::test_modulo_even_odd_flowchart_recognizes_nodes_edges_and_decision_text -q` — passed.

The supplied diagram does not draw a connector from the `Number is odd` output to `End`. The recognizer intentionally does not invent that missing edge. It reports the resulting incomplete path so the diagram can be corrected or reviewed.

---

# Recognition and graph-construction overhaul (samples 1, 7, 8, 10)

## Baseline and root causes

Scored structurally against `tests/fixtures/samples_ground_truth.json` (node
detection, shape type, OCR text, directed edges, branch labels), the original
pipeline passed samples 2-6 and 9 and failed 1, 7, 8 and 10:

| Sample | Root causes found |
|---|---|
| 1 | Bold coloured "True"/"False" labels detected as filled nodes; contacts were *proximity* based (any stroke pixel within 5% of page width), so the long loop-back line "touched" four boxes and End; arrowhead test examined windows longer than short connectors and flagged both ends as heads; loop-back lines merging into one arrowhead rejected as ambiguous; line removal for label OCR erased the bars of bold T/F glyphs. |
| 7 | Arrowheads not detected (filled heads erased by node cleanup); the "True"/"False" labels were never read (a connector runs through them, and arrowhead-filter dropped words touching a stroke), so geometric label inference guessed the branches the wrong way round, giving an empty IF branch. |
| 8 | Fill-detection used a fixed 9x9 opening; this chart's ~9px saturated connectors survived it and a 15x15 close welded every node and line into one page-sized "node". Off-white page made the label-OCR image blank (`lines > 0` on an un-thresholded inverted image). "Stop" not recognised as End. |
| 10 | Fixed morphology rounded filled rectangles/parallelograms into "terminals"; arrowheads overlapping filled nodes were absorbed by the node mask; a screenshot border line skewed stroke-width estimates; "Yes"/"No" labels sitting *on* lines (and one at an elbow corner) split connectors. |

## What changed

* **Shape detection** (`vision_service.filled_region_contours`): morphology is
  sized from the measured stroke width; each region is rebuilt from the raw fill
  mask so corners and slants survive; text blobs are rejected by solidity. A
  side-profile parallelogram test corrects slight-slant I/O shapes that the
  angle rules called "process". Node outlines of any colour are removed by
  bounded growth through ink (`grow_through_ink`).
* **OCR**: crops are masked outside the node's own outline (diamond edges were
  read as `~~`); border-debris tokens are stripped as candidates; process
  readings are ranked by the notation grammar (assignment or call); single-glyph
  names l/I/O/o get a small ranking penalty. A single Tesseract crash costs one
  candidate instead of failing the analysis.
* **Branch labels**: a targeted pass OCRs text blobs next to connectors one at a
  time (raw and line-removed), keeps branch vocabulary or confident words, and
  uses per-glyph boxes. Page-level OCR now thresholds against the page
  background. Labels are erased from the connector mask only when their
  components lie wholly inside the label box and do not touch a node, so lines
  and arrowheads fused to a label survive; interrupted lines are bridged
  straight through or around an elbow corner. Yes/No words are assigned among
  decision branches first.
* **Connectors** (`connector_geometry.py`, `GraphService`): contacts must be
  genuine stroke *ends* (close approach, or a single exit from the contact
  band) - lines running past a node no longer attach. Arrowheads use a
  localised, stroke-width-relative width profile on raw ink (recovering heads
  cut by node cleanup) and must be wider than the shaft, bounded, symmetric,
  of plausible length, start at the end itself, and lie within half the
  connector; corners, T-junctions, crossings, tails of short connectors and
  noise specks are rejected. Several tails joining one arrowhead become one
  edge per tail (merge). One port per node per component; a stroke end claimed
  by two nodes goes to the closer. Redundant direction-ambiguous fragments
  beside an already-traced connector are reported as warnings, not errors.
* **Interpretation** (`notation.py`): `terminal_role` accepts Begin/Stop/Exit/
  Finish/Halt; `normalize_condition` strips question wrappers ("is a > b?") and
  reads a lone `=` in a condition as equality, only when that yields a valid
  expression. Recognised node text is never rewritten.
* New setting: `CONNECTOR_TOUCH_PERMILLE` (default 22).

## Results

* Samples: **10/10 structurally correct** (baseline 6/10). Samples 2-7 and 9
  reach pseudocode; 2-6 and 9 generate Python. 1, 8 and 10 are natural-language
  charts: their graphs are exact, and validation correctly reports
  `UNSUPPORTED_STATEMENT` for text outside the executable notation.
* 20 generated unseen-style charts (`tests/generate_unseen_flowcharts.py`):
  **20/20 exact graphs**, 17/20 fully correct; the original code scored 2/20.
* Synthetic fixtures: `decision.png` now completes (was ambiguous); `loop.png`
  graph is now exact; no fixture regressed.
* Tests: `python -m pytest -q` 103 passed; with `FLOWCHART_RUN_PIPELINE=1`
  128 passed, 1 failed - `test_countdown_loop_preserves_both_loop_directions_and_exit_branch`
  needs `samples/countdown.png`, which is not in the repository (fails
  identically on the original code).

## Known limitations

* Sample 7 draws `print "Enter a number"` but never inputs `num`; code
  generation correctly refuses the undefined variable. No step is invented.
* Tesseract reads DejaVu-style "i = i + 1" as `i=it+l` (also `loop.png`).
* Loops drawn with a decision diamond (samples 1, 10, `loop.png`) are traced
  correctly but the notation still requires a WHILE/FOR hexagon to emit code.
* `nested_branches.png` (three-way decision with labels glued to arrowheads)
  still loses two branches; it is malformed for the notation in any case.
* "ρ" is read as "p" (English traineddata only).
* Label OCR adds roughly 0-4 s per chart; total time stays within ~1.4x of
  the original on the slowest sample measured.
* Not implemented: deskewing beyond the ~1.5 degree rotation that was tested,
  curved-connector crossing disambiguation by tangent continuity, hand-drawn
  photographs.

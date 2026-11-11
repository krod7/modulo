# Accepted flowcharts

Flowcharts verified to convert **end to end** with the current pipeline:
analysis reaches `ready`, the pseudocode matches the drawn logic, Python is
generated without errors, and the code does what the chart describes. Files
here are copies; the originals stay in `samples/` and `test_flowcharts/`, which
the ground-truth tests use unchanged.

`rotation.json` lists the ten charts used by **Try A Sample**, in display order
(simplest first). The `/api/samples` endpoint serves exactly that list; if the
manifest is missing it falls back to every file in `samples/`.

## Contents

| File | Source | Constructs | Overall confidence |
|---|---|---|---|
| `samples/sample2.png` | `samples/` | Sequence: assignment, output | 97 Excellent |
| `samples/sample3.png` | `samples/` | `WHILE` hexagon loop | 97 Excellent |
| `samples/sample4.png` | `samples/` | If/else, typed input `Input score:int` | 98 Excellent |
| `samples/sample5.png` | `samples/` | If/else, descriptive input, `=` read as equality | 89 Good |
| `samples/sample6.png` | `samples/` | If/else with `If` prefix, quoted outputs | 94 Excellent |
| `samples/sample7.png` | `samples/` | Two consecutive ifs without else | 80 Good |
| `samples/sample8.jpg` | `samples/` | Nested decisions, question-form conditions, multi-input, `Declare`, `Stop` | 84 Good |
| `samples/sample9.jpg` | `samples/` | Error-path decision, `Calculate factorial` | 76 Moderate |
| `test_flowcharts/{png,jpg,jpeg}/05_while_loop_count.*` | `test_flowcharts/` | Pre-test loop drawn as a diamond with a back edge | 97-98 Excellent (PNG/JPG) |
| `test_flowcharts/{png,jpg,jpeg}/07_do_while_guess.*` | `test_flowcharts/` | Post-test (do-while) loop, input inside the loop | 98 Excellent (PNG/JPG) |

Scores are from the scale in README Section 12.1 (PNG copies of 05 and 07 are
the ones in the rotation). The SVG and PDF copies of 05 and 07 were not
included because they could not be re-verified in the environment used for
this check (PyMuPDF unavailable); add them after confirming locally.

## Why the other test charts are not here

Recorded run of all 15 `test_flowcharts/` charts (PNG and JPG):

| Chart | Result | Cause |
|---|---|---|
| 02, 03, 04 | Now convert (not yet added here) | Unquoted single-word outputs (`Output Adult`) are now printed as text with a review warning. |
| 11, 14 | 11 now converts (not yet added here); 14 still misread | Low-confidence nodes are now re-read at 2x, which fixes `i = i + 1` in 11; 14 still reads `i=itl`. |
| 01 | Blocked | One shape is classified as unsupported (`UNSUPPORTED_SHAPE`). |
| 06, 08, 12 | Blocked | A connector's arrow direction is ambiguous or a node becomes unreachable. |
| 09 | PNG now converts (not yet added here); JPG blocked | JPG: a statement is misread. |
| 10, 13, 15 | Blocked: unstructured flow | The loop has a second exit from its body (straight to End). |

The invalid and stress charts are expected to be rejected and are not candidates.

## Re-verifying

```bash
FLOWCHART_RUN_PIPELINE=1 python -m pytest tests/test_sample_rotation.py -q
```

This checks that every rotation chart analyzes to `ready`, scores at least
Moderate, and generates Python that compiles.

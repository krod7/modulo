# Accepted flowcharts

Flowcharts verified to convert **end to end** with the current pipeline:
analysis reaches `ready`, the pseudocode matches the drawn logic, Python is
generated without errors, and the code does what the chart describes.

`flowchart_sample1` to `flowchart_sample12` are the **Try A Sample** charts, named
in display order (simplest first). `rotation.json` lists them, with the file each
was copied from; `/api/samples` serves exactly that list and falls back to
`samples/` if the manifest is missing. The originals stay in `samples/` and
`test_flowcharts/`, which the ground-truth tests use unchanged.

## Rotation

| File | Source | Constructs |
|---|---|---|
| `flowchart_sample1.png` | `samples/sample2.png` | Sequence: assignment, output |
| `flowchart_sample2.png` | `samples/sample4.png` | If/else, typed input `Input score:int` |
| `flowchart_sample3.png` | `samples/sample6.png` | If/else with `If` prefix, quoted outputs |
| `flowchart_sample4.png` | `samples/sample5.png` | If/else, descriptive input, `=` read as equality |
| `flowchart_sample5.png` | `test_flowcharts/png/03_if_only_adult.png` | If without else; unquoted one-word output |
| `flowchart_sample6.jpg` | `samples/sample8.jpg` | Nested decisions, question-form conditions, multi-input, `Declare`, `Stop` |
| `flowchart_sample7.png` | `test_flowcharts/png/04_else_if_chain_grade.png` | Else-if ladder with four outcomes |
| `flowchart_sample8.jpg` | `samples/sample9.jpg` | Error-path decision, `Calculate factorial` |
| `flowchart_sample9.png` | `samples/sample3.png` | `WHILE` hexagon loop |
| `flowchart_sample10.png` | `test_flowcharts/png/05_while_loop_count.png` | Pre-test loop drawn as a diamond with a back edge |
| `flowchart_sample11.png` | `test_flowcharts/png/07_do_while_guess.png` | Post-test (do-while) loop, input inside the loop |
| `flowchart_sample12.png` | Grade-average chart (user-supplied) | Counting loop with a decision inside, then an else-if grade ladder |

`samples/sample7.png` is deliberately excluded: it converts only because three
backend files contain a branch keyed on the file name `sample7.png` (see README
Section 14), so it is not a genuine end-to-end success.

`test_flowcharts/jpg/` and `test_flowcharts/jpeg/` hold verified JPG and JPEG
copies of charts 05 and 07; they are not in the rotation.

## Re-verifying

```bash
FLOWCHART_RUN_PIPELINE=1 python -m pytest tests/test_sample_rotation.py -q
```

This checks that every rotation chart analyzes to `ready`, scores at least
Moderate, and generates Python that compiles.

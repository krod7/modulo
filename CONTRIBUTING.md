# Contributing to Modulo

Team: West Texas System Solutions (CS4366 Senior Capstone, Fall 2026)

This document explains who owns what, how we use Git, and how work gets reviewed and merged. Everyone writing code on this project follows it. Stage 3 is due **Tuesday, Nov 10, 2026 (11:59 PM)**.

## 1. Goals for Stage 3

1. Improve the working v1.0 into a reliable, demo-ready build.
2. Every test case passes, across flowcharts of varying complexity.
3. `main` always runs. If `main` is broken, fixing it is the top priority.
4. Keep Modulo's core promises: no AI/LLMs, deterministic output, flag unsupported or ambiguous input instead of guessing, uploaded files never saved after the session.

## 2. Roles and ownership

| Member | Area | Owns |
|---|---|---|
| **Kaleb** | Oversight, integration, review | Final say on what merges to `main`. Shared files (`pipeline.py`, data models, `config.py`, Flask API). Test fixtures and integration tests. Pseudocode, code generation, and trace services unless reassigned. Helps Destiny and Keegan. |
| **Destiny** | Backend: graph, validation, logic | Graph construction, validation, and logic/IR recovery (sequences, conditionals, loops). This is the heaviest area. |
| **Keegan** | Backend: intake and vision | Intake (PNG/JPG/PDF handling) and vision (shape and connector detection, OCR). Then helps Destiny (see section 4). |
| **Alejandro** | Frontend / UI-UX | UI cleanup only: layout, styling, interaction, three-panel view, highlighting polish. |

### Diagnostics handling

Diagnostics (error and warning detection, wording, structure, API response format) are handled by **Keegan, Destiny, and Kaleb** in their own pipeline areas. Alejandro does not write diagnostics logic.

- Backend owners decide what a `Diagnostic` contains and how it is worded.
- Alejandro only styles how the diagnostics area looks. If the displayed format needs to change, open an issue and tag Kaleb.
- If you change the `Diagnostic` structure or the API response shape, tell the whole team in the group chat **before** merging.

### Shared files (conflict hotspots)

`pipeline.py`, data models, `config.py`, and the Flask API file are shared. Before editing any of them, post in the group chat. Keep those changes small and merge them quickly.

## 3. Workflow at a glance

```
main (protected) <--PR, reviewed by Kaleb-- your-name/task-name
```

- No one commits or pushes directly to `main`.
- Every change goes through a branch and a pull request (PR).
- Kaleb reviews and merges PRs. Kaleb's own PRs are reviewed by Destiny or Keegan first.
- The v1.0 build is our fallback, tagged `v1.0` on GitHub. Never delete or move this tag. The original v1.0 branch (`modulo-v1.0`) stays untouched as an extra backup.

### 3.1 Starting point

`main` already contains the v1.0 build, so everyone works from it. v1.0 is also preserved in two places that nobody changes:

| Where | What it is |
|---|---|
| `main` | Our working branch. Contains v1.0 plus every PR merged since. |
| Branch `modulo-v1.0` | Untouched copy of the v1.0 build. |
| Tag `v1.0` | Fixed marker on the exact v1.0 commit. This is our fallback. |

If you already cloned the repo, update your copy:
```
git fetch origin
git switch main
git pull origin main
```
If you have not cloned it yet, follow section 6, then create your task branch as described in section 7.1.

To get back to the v1.0 build at any time:
```
git switch -c restore-v1.0 v1.0
```

## 4. Splitting work (Keegan, Destiny, Kaleb)

Destiny's area is the largest. To share it without collisions:

1. When Keegan finishes a task in his area, he posts in the group chat.
2. Destiny picks a self-contained piece to hand off (for example, one validation rule set or loop-recovery case) and writes it as a GitHub Projects issue with the files involved.
3. The person taking it creates **their own branch** for it (for example `keegan/validation-rules`). **Two people never push to the same branch.**
4. When Kaleb finishes his tasks early, he does the same.
5. Merge `main` into your branch before starting so you have Destiny's latest merged work.

## 5. Keegan's area: handling library limitations

Shape detection and OCR with OpenCV and Tesseract have limits. Our approach:

1. **Try improvements first:** image preprocessing (grayscale, thresholding, noise removal, upscaling small text), tuning contour and shape parameters, and better connector detection.
2. **Time-box each approach** (about 2 days). If it does not give a measurable improvement, stop and move on.
3. **Measure with synthetic test diagrams** (known shapes, positions, and text) so improvements are real and not guesses.
4. **When a limit cannot be fixed, fail clearly.** Return a diagnostic with recovery guidance instead of guessing.
5. **Document limits** in `docs/known-limitations.md` (supported image quality, line thickness, fonts, rotation). This also feeds the report.
6. **Demo flowcharts** are drawn strictly in the documented notation, with high contrast and consistent sizing.

## 6. One-time setup (each member)

Commands are the same on macOS Terminal and Windows Command Prompt unless noted.

```
git --version
git config --global user.name "Your Full Name"
git config --global user.email "your-github-email@example.com"
```

Clone the repo:

macOS:
```
cd ~/Documents
git clone https://github.com/krod7/modulo.git
cd modulo
```

Windows:
```
cd %USERPROFILE%\Documents
git clone https://github.com/krod7/modulo.git
cd modulo
```

Create a virtual environment and install dependencies (see README for pinned versions and the Tesseract install).

macOS:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 7. Everyday workflow

### 7.1 Start a task

```
git switch main
git pull origin main
git switch -c yourname/task-name
```

### 7.2 Branch naming

Format: `yourname/area-or-task`, lowercase, hyphens.

Examples:
- `destiny/loop-recovery`
- `keegan/ocr-preprocessing`
- `alejandro/ui-cleanup`
- `kaleb/integration-tests`

A branch covers one task or pipeline stage. Keep branches short-lived (target 1 to 3 days). Large areas are delivered as several PRs, not one big merge at the end.

### 7.3 Commit

```
git status
git diff
git add path/to/file.py
git commit -m "area: short description"
```

Commit message format `area: description`, for example:
- `graph: detect while-loops with back edges`
- `vision: add thresholding before OCR`
- `ui: tighten spacing in diagnostics panel`
- `test: add nested conditional fixture`

Prefer adding specific files over `git add .` so you do not commit stray or generated files. Never commit `.venv/`, uploaded test files containing private data, or editor settings.

### 7.4 Stay current with main (at least every 2 days)

```
git fetch origin
git merge origin/main
```

Resolve conflicts in VS Code, then:

```
git add .
git commit
```

Run `pytest` after resolving.

### 7.5 Push your branch

First push:
```
git push -u origin yourname/task-name
```
Later pushes:
```
git push
```

Never use `git push --force` on a shared branch. Never push to `main` or the v1.0 branch.

### 7.6 Open a pull request

On GitHub, open a PR from your branch into `main`. Include in the description:

- What changed and why
- How you tested it (commands run, flowcharts used)
- Files touching shared code (if any)
- Screenshots for UI changes

### 7.7 After merge

```
git switch main
git pull origin main
git branch -d yourname/task-name
```

## 8. Review process (Kaleb)

Review within **24 hours** whenever possible. Locally:

```
git fetch origin
git switch yourname/task-name
git diff main...yourname/task-name
pytest
```

Then run the Flask server and try several flowcharts.

### PR checklist

- [ ] `pytest` passes
- [ ] Change is small and focused on one purpose
- [ ] New or changed behavior has a test or fixture
- [ ] Output is still deterministic (same input, same output)
- [ ] No execution of generated code, no AI/LLM or external API calls
- [ ] Unsupported or ambiguous input produces a diagnostic, not a guess
- [ ] No unrelated files or debug code committed
- [ ] Shared-file changes were announced to the team
- [ ] UI changes include screenshots and work in major browsers

Outcomes: **approve and merge**, **request changes** (comment on the exact lines), or **close** with an explanation.

Merge with a regular merge commit so each member's individual commits remain in the history. This also documents contributions for the report.

## 9. Testing standards

- Unit tests with `pytest` for each service you change.
- Test the graph stage with hand-built `FlowGraph` fixtures, separate from image processing, so failures can be traced to logic or vision.
- Integration fixtures in `tests/fixtures/`: each flowchart has an expected pseudocode file and expected Python file. Compare exactly.
- Cover: sequences, if/else, nested conditionals, loops, loops with decision exits, decision-based error paths, missing Start/End, unlabeled or dangling branches, unsupported elements.
- Determinism test: run each fixture repeatedly and confirm identical output.
- API tests with Flask's test client: successful conversion, failed analysis, invalid upload, expired analysis.
- Do not merge a PR that makes an existing passing test fail. If a test must change, explain why in the PR.

## 10. Timeline

| Dates | Focus |
|---|---|
| Oct 28 - Nov 2 | Feature and test work on short branches. Merge to `main` regularly. |
| **Nov 3** | **Feature freeze.** Only bug-fix PRs after this. Fresh-clone test and user validation sessions. |
| Nov 4 - 6 | Bug fixes, demo rehearsal, report and slides. |
| Nov 7 - 9 | Final rehearsal, group webpage update, submit with a buffer. |
| **Nov 10** | Stage 3 due (report PDF and slides via Canvas, 11:59 PM). Confirm the live demo date with the instructor. |

## 11. Release and demo rules

- After the freeze, `main` is the demo build. Rehearse on the exact machine used for presenting.
- Do a **fresh-clone test**: someone who did not write the setup steps clones the repo on a clean machine and follows only the README (including installing Tesseract).
- Keep demo files in a ready folder and have screenshots or a short screen recording as a backup.
- Tag the final build, for example `git tag -a v2.0 -m "Stage 3 demo build"` and `git push origin v2.0`.

## 12. Communication

- Use GitHub Projects for tasks. Each task gets an owner and a branch name.
- Post in the group chat when: you start or finish a task, you need to edit a shared file, you change a data structure or API response, or you are blocked for more than a few hours.
- If two people need the same file, talk first and agree who goes first.

## 13. If something goes wrong

Committed on `main` by mistake and have not pushed:
```
git switch -c yourname/task-name
git switch main
git reset --hard origin/main
```

Merge conflicts: do not delete someone else's code to make a conflict disappear. Ask the author of the conflicting change.

Pushed something broken to your branch: fix it with a new commit. Do not rewrite shared history.

Broke `main`: tell Kaleb immediately. The fallback is `git switch -c fix/revert-issue v1.0` for a known-good baseline.

## 14. Repo protection

### What is protected

| Item | Protection | How it is enforced |
|---|---|---|
| `main` | Protected by a GitHub branch protection rule | *Settings → Branches* |
| Branch `modulo-v1.0` | No GitHub rule | Team agreement: never commit or push to it |
| Tag `v1.0` | No GitHub rule | Team agreement: never delete or move it |
| Task branches (`name/task`) | No protection | Owner pushes freely; delete after merge |

### The rule on `main`

- **Require a pull request before merging:** no one pushes directly to `main`.
- **Require approvals (1):** a PR needs one approval before it can merge. The author cannot approve their own PR.
- **Dismiss stale pull request approvals when new commits are pushed:** if code changes after approval, it must be approved again.
- **Force pushes and deleting `main` are blocked** (those options are left off).
- **Admin override:** *Do not allow bypassing the above settings* is left off, so the repo admin (Kaleb) can fix a broken `main` in an emergency. Do not use it for normal work.

### What the rule does not do

- It does not run tests. There are no required status checks, so the reviewer runs `pytest` locally (section 8).
- It does not limit who can approve or merge. Anyone with Write access can, so "only Kaleb clicks the merge button" is a team agreement.
- It does not protect `modulo-v1.0` or the `v1.0` tag.

### Local safety net (each member)

```
git config branch.main.pushRemote no_push
```

### Merge rules (team agreement)

Only Kaleb clicks the merge button. Kaleb approves teammates' PRs. Kaleb's own PRs are approved by Destiny or Keegan (GitHub does not let you approve your own PR), then Kaleb merges.

## 15. Development resources

AI-assisted tools used by Kaleb for coding and documentation help:

| Tool | Site |
|---|---|
| Claude | claude.ai |
| Manus (Manus Studio) | manus.im |
| Cursor (for the time being) | cursor.com |

## 16. Installer and releases

Teammates and users install Modulo with `ModuloSetup.exe` (Windows) or the Mac/Linux installer scripts. It installs Python, the packages in `requirements.txt`, Tesseract, and a desktop icon. Files live in `installer/`, `assets/`, `modulo_launcher.py`, and the `Install Modulo (...)` scripts in the repo root.

**Automatic checks:** `.github/workflows/ci.yml` runs `pytest` on every PR and checks that the installer builds. This is informational unless a required status check is added to the `main` rule (section 14).

**Releasing a new installer (Kaleb):**
1. In a PR, bump `VERSION` (for example `1.0.0` to `1.1.0`) and merge it.
2. `git switch main && git pull && git tag v$(cat VERSION) && git push origin v$(cat VERSION)`.
3. GitHub Actions builds `ModuloSetup.exe` and `Modulo-Installer.zip` and publishes them on the **Releases** page. The release fails if the tag does not match `VERSION`.
4. Teammates download `ModuloSetup.exe` and run it. Running it over an existing install updates it in place.

Build locally instead: `pip install ziglang && python installer/build_windows_exe.py` writes `dist/ModuloSetup.exe`.

Repo admin one-time setting: *Settings -> Actions -> General -> Workflow permissions -> Read and write*, so releases can be published. Never move or delete the `v1.0` tag; release tags use the same `v*` pattern, so use new version numbers only.

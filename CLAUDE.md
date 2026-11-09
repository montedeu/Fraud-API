# fraud-api

## Review

- Reviewer subagents run with `isolation: "worktree"`, in their own checkout, so the working tree and branch stay where the implementer left them.
- Reviewers apply `CODING_STANDARDS.md`.

## Where numbers live

Results come from `model/metadata.json`. Besides the README's generated blocks, they appear by hand in `notebooks/eda.ipynb` markdown, `docs/adr/`, and README prose; a change to results updates all of them (`CODING_STANDARDS.md`, "One source per number").

## Agent skills

### Issue tracker

Issues are tracked in this repo's GitHub Issues via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

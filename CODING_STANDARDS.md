# Coding standards

Rules for the reviewer. Each one is a judgement call no automated check can make. Mechanical rules live in tooling instead: ruff (`pyproject.toml`), the hooks in `.claude/settings.json`, the commit-msg hook in `.githooks/`, and the CI jobs in `.github/workflows/ci.yml`. A finding that tooling already reports is not a review finding.

## Docs state only what the developer decided

Every reason, motive or judgement the README, an ADR, the EDA notebook or a PR body attributes to the developer traces to something the developer said: an issue, a comment, a commit by them, a message in the session. A reason the agent inferred is either cut, or written as the agent's observation with its evidence (a number, a cell, a test). For example, the 8 November revisit in ADR 0003 records that age stays a feature with no motive attached, because the developer gave none.

## One source per number

Results live in `model/metadata.json`. The README's generated blocks are rebuilt from it by `fraud-api readme`, and a test fails when they drift. Hand-written text does not repeat those numbers; it points at the generated block or the table. When a PR changes results (a retrain, a new metric, a new experiment), every hand-written mention is updated in the same PR:

- `notebooks/eda.ipynb` markdown cells, especially each section's closing decision;
- `docs/adr/*.md`;
- README prose outside `<!-- generated:… -->` blocks, including the request rules (`null`, 422) and the threat model.

The reviewer lists each of these places the diff touches or leaves stale.

## Glossary terms in prose

Prose, names, test names and error messages use the terms in `GLOSSARY.md` and none of the terms each entry marks _Avoid_. A concept with no glossary term is a gap to raise, not a synonym to invent.

## ADRs are surfaced, not overridden

A change that contradicts an ADR in `docs/adr/` says so in the PR body and either updates the ADR or explains why the ADR still holds.

## Comments say why

A comment explains a reason the code cannot show (a dataset quirk, a GCP limit, a past bug). It does not narrate what the next line does.

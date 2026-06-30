# Contributing

## Branch naming

Branch off `master` using a `type/short-description` name in kebab-case:

| Prefix      | Use for                                   |
| ----------- | ----------------------------------------- |
| `feat/`     | new features                              |
| `fix/`      | bug fixes                                 |
| `chore/`    | maintenance, dependencies, configuration  |
| `docs/`     | documentation only                        |
| `refactor/` | code changes with no behaviour change     |

Example: `feat/tournament-selection`.

## Commit messages — Conventional Commits

Commits **must** follow [Conventional Commits](https://www.conventionalcommits.org/),
because releases are automated from the commit history (see below). Format:

```
type(scope): short description

Longer body explaining the why, not just the what.
```

The type determines the next version bump:

| Commit type                              | Release effect          |
| ---------------------------------------- | ----------------------- |
| `fix:`                                   | patch (1.0.0 → 1.0.1)   |
| `feat:`                                  | minor (1.0.0 → 1.1.0)   |
| `feat!:` or a `BREAKING CHANGE:` footer  | major (1.0.0 → 2.0.0)   |
| `chore:` / `docs:` / `refactor:` / etc.  | no release              |

Always explain the **why** in the body — the title says what changed, the body
says why and what it unblocks.

## Before opening a pull request

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

CI runs these same checks on every pull request.

## Releases

Releases are fully automated by
[python-semantic-release](https://python-semantic-release.readthedocs.io/).
When commits land on `master`, the release workflow analyses them, bumps the
version in `pyproject.toml`, updates `CHANGELOG.md`, tags the commit, and
publishes a GitHub Release. No manual version bumps or tags are needed —
correct commit types are what make this work.

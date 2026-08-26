# Contributing to AgileRL

### Thank you for taking the time to contribute to AgileRL! 🤖🎉

Contributions are really valuable to AgileRL, and we are grateful to the community for your support. We are happy to have you here! AgileRL powers real-world AI systems, and your contributions directly impact the tools that researchers and engineers depend on every day.

In this guide you will get an overview of the contribution workflow from opening an issue, creating a PR, reviewing, and merging the PR.

If you have any questions, please raise them on our [Discord](https://discord.gg/eB8HyTA2ux).

When contributing to AgileRL, we ask that you:

- Let us know what you plan in the GitHub Issue tracker so we can provide feedback.

- Provide tests and documentation whenever possible. When fixing a bug, provide a failing test case that your patch solves.

- Open a GitHub Pull Request with your patches and we will review your contribution and respond as quickly as possible. Keep in mind that this is an open source project, and it may take us some time to get back to you. Your patience is very much appreciated.

There are many ways to contribute, from writing tutorials or blog posts, improving the documentation, submitting bug reports and feature requests or writing code which can be incorporated into AgileRL itself.

All members of the AgileRL community are expected to follow the [Code of Conduct](https://github.com/AgileRL/AgileRL/blob/main/CODE_OF_CONDUCT.md).

## Your First Contribution

There is a lot of scope to contribute on what you find interesting, and your
contributions can have a real impact on the framework.

We always appreciate new algorithms, features, and improvements for RL and LLM training. Whether you want to implement a cutting-edge paper, optimize existing code, or improve our docs - there's room for it all. Feel free to suggest new ideas and check the GitHub Issue tracker for bugs, improvements, and feature requests.

Contributing to an open source project for the first time? You can learn how from [this great resource](https://www.firsttimersonly.com/).

## Getting started
### How to contribute
If you've noticed a bug or have a feature request, [make one](https://github.com/agilerl/agilerl/issues/new)! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

To make a code contribution:

1. Create your own fork of the code.
2. Create a new branch from the `main` branch for your changes.
3. Do the changes in your fork.
4. Run the tests with `pytest tests`
5. Make the test pass.
6. Run the type checker with `just typecheck` (or `uv run ty check agilerl agilerl/arena`) and fix any errors it reports.
7. Commit your changes. Please use an appropriate commit prefix. If your pull request fixes an issue specify it in the commit message.
8. Push to your fork and submit a pull request **to the `main` branch**. Please provide us with some explanation of why you made the changes you made. For new features make sure to explain a standard use case to us.

Keep your pull request open — do not merge it yourself. If we ask for changes, push them here (rebase onto `main` if you are behind). A maintainer adds the `hub-sync-import` label when the change is ready for internal CI; we merge this PR after that passes.

#### Type checking
AgileRL is type-checked with [`ty`](https://docs.astral.sh/ty/) and ships `py.typed` markers, so the public API's annotations are part of the released package. The configuration lives in `pyproject.toml` under `[tool.ty]`: the whole `agilerl` package is checked strictly. The CI workflow's ty job blocks PRs on new type errors, so run `just typecheck` before pushing.

Prefer fixing types at the source — tighter annotations, `@overload`s, or a small typed wrapper — over narrowing with `cast()` / `assert isinstance(...)`, or suppressing with `# ty: ignore`. Reach for a suppression only for genuine third-party or stub gaps, and justify it with an inline comment.

#### Naming conventions
Avoid leading-underscore "private" names unless the privacy is genuinely meaningful (e.g. a helper defined *inside* another function). Module-level classes, functions, type aliases, and `TypeVar`s should use plain public names even when they are only referenced within their own module — a leading underscore reads as "deliberately hidden", so reserve it for cases where that is actually true.

#### Pre-commit hooks
Checks will be run automatically by the CI on code pushed to the AgileRL repository. These checks can also be run locally with the following steps:

1) [Install `pre-commit`](https://pre-commit.com/#install).
2) Install the Git hooks by running `pre-commit install`.

Once these steps are done, the hooks will be run automatically at every new commit.

## How to report a bug
If you find a security vulnerability, do NOT open an issue. Email dev@agilerl.com instead.

When filing an issue, make sure to answer these five questions:

- What version of AgileRL are you using?
- What operating system and processor architecture are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

General questions should go to the [AgileRL Discord](https://discord.gg/eB8HyTA2ux) instead of the issue tracker.

## How to suggest a feature or enhancement
The AgileRL philosophy is to provide streamlined tooling for reinforcement learning development, to make it easier and faster to create incredible models.

If you find yourself wishing for a feature that doesn't exist in AgileRL, you are probably not alone. There are bound to be others out there with similar needs. Many of the features that AgileRL has today have been added because our users saw the need. Open an issue on our issues list on GitHub which describes the feature you would like to see, why you need it, and how it should work.

## Community
You can chat with the core team on the [AgileRL Discord](https://discord.gg/eB8HyTA2ux). We try to respond as quickly as possible.

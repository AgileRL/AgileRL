# Contributing to AgileRL

### Thank you for taking the time to contribute to AgileRL! 🤖🎉

Contributions are really valuable to AgileRL, and we are grateful to the community for your support. We are happy to have you here!<br>
In this guide you will get an overview of the contribution workflow from opening an issue, creating a PR, reviewing, and merging the PR.

If you have any questions, please raise them on our [Discord](https://discord.gg/eB8HyTA2ux).

When contributing to AgileRL, we ask that you:

- Let us know what you plan in the GitHub Issue tracker so we can provide feedback.

- Provide tests and documentation whenever possible. When fixing a bug, provide a failing test case that your patch solves.

- Open a GitHub Pull Request with your patches and we will review your contribution and respond as quickly as possible. Keep in mind that this is an open source project, and it may take us some time to get back to you. Your patience is very much appreciated.

There are many ways to contribute, from writing tutorials or blog posts, improving the documentation, submitting bug reports and feature requests or writing code which can be incorporated into AgileRL itself.

All members of the AgileRL community are expected to follow the [Code of Conduct](https://github.com/AgileRL/AgileRL/blob/main/CODE_OF_CONDUCT.md).

## Your First Contribution
AgileRL is nascent, and so there is a lot of scope to contribute on what you find interesting. You can have a big impact on the framework now and get involved in our exciting early development.
Please feel free to suggest new, crazy ideas and have a go at the coolest RL implementations you can think of.
We expect the GitHub Issue tracker to fill up with bugs, improvements and feature requests, so please take a look there too.

Some early suggestions for contributions we think would add value to AgileRL include adding more RL algorithms and evolvable network types, parallelising and benchmarking training, writing documentation, and general improvements to current implementations.

Contributing to an open source project for the first time? You can learn how from this great resource, https://www.firsttimersonly.com/.

## Getting started
### How to contribute
If you've noticed a bug or have a feature request, [make one](https://github.com/agilerl/agilerl/issues/new)! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

For something that is bigger than a one or two line fix:

1. Create your own fork of the code.
2. Do the changes in your fork.
3. Run the linter.
````
cd AgileRL
ruff --format=github --target-version=py37 --ignore E501 ./
````
4. Run the tests. - **(TESTS UNDER CONSTRUCTION)**
````
pytest
````
5. Make the test pass.
6. Commit your changes. Please use an appropriate commit prefix. If your pull request fixes an issue specify it in the commit message.
7. Push to your fork and submit a pull request. Please provide us with some explanation of why you made the changes you made. For new features make sure to explain a standard use case to us.

Small contributions such as fixing spelling errors, where the content is small enough to not be considered intellectual property, can be submitted by a contributor as a patch. <br>
As a rule of thumb, changes are obvious fixes if they do not introduce any new functionality or creative thinking. As long as the change does not affect functionality, some likely examples include the following:

- Spelling / grammar fixes
- Typo correction, white space and formatting changes
- Comment clean up
- Bug fixes that change default return values or error codes stored in constants
- Adding logging messages or debugging output
- Changes to ‘metadata’ files like Gemfile, .gitignore, build scripts, etc.
- Moving source files from one directory or package to another

## How to report a bug
If you find a security vulnerability, do NOT open an issue. Email dev@agilerl.com instead.

When filing an issue, make sure to answer these five questions:

- What version of AgileRL are you using?
- What operating system and processor architecture are you using?
- What did you do?
- What did you expect to see?
- What did you see instead? General questions should go to the [AgileRL Discord](https://discord.gg/eB8HyTA2ux) instead of the issue tracker.

## How to suggest a feature or enhancement
The AgileRL philosophy is to provide streamlined tooling for reinforcement learning development, to make it easier and faster to create incredible models.

If you find yourself wishing for a feature that doesn't exist in AgileRL, you are probably not alone. There are bound to be others out there with similar needs. Many of the features that AgileRL has today have been added because our users saw the need. Open an issue on our issues list on GitHub which describes the feature you would like to see, why you need it, and how it should work.

## Community
You can chat with the core team on the [AgileRL Discord](https://discord.gg/eB8HyTA2ux). We try to respond as quickly as possible.

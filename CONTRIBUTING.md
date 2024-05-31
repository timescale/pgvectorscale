# Contributing to pgvectorscale

We appreciate any help the community can provide to make pgvectorscale better!

You can help in different ways:

* Open an [issue](https://github.com/timescale/pgvectorscale/issues) with a
  bug report, build issue, feature request, suggestion, etc.

* Fork this repository and submit a pull request

For any particular improvement you want to make, it can be beneficial to
begin discussion on the GitHub issues page. This is the best place to
discuss your proposed improvement (and its implementation) with the core
development team.

Before we accept any code contributions, pgvectorscale contributors need to
sign the [Contributor License Agreement](https://cla-assistant.io/timescale/pgvectorscale) (CLA). By signing a CLA, we can
ensure that the community is free and confident in its ability to use your
contributions.

## Development

Please follow our DEVELOPMENT doc for [instructions how to develop and test](https://github.com/timescale/pgvectorscale/blob/main/DEVELOPMENT.md).

## Code review workflow

* Sign the [Contributor License Agreement](https://cla-assistant.io/timescale/pgvectorscale) (CLA) if you're a new contributor.

* Develop on your local branch:

    * Fork the repository and create a local feature branch to do work on,
      ideally on one thing at a time.  Don't mix bug fixes with unrelated
      feature enhancements or stylistical changes.

    * Hack away. Add tests for non-trivial changes.

    * Run the [test suite](#testing) and make sure everything passes.

    * When committing, be sure to write good commit messages according to [these
      seven rules](https://chris.beams.io/posts/git-commit/#seven-rules). Doing 
      `git commit` prints a message if any of the rules is violated. 
      Stylistically,
      we use commit message titles in the imperative tense, e.g., `Add
      merge-append query optimization for time aggregate`.  In the case of
      non-trivial changes, include a longer description in the commit message
      body explaining and detailing the changes.  That is, a commit message
      should have a short title, followed by a empty line, and then
      followed by the longer description.

    * When committing, link which GitHub issue of [this 
      repository](https://github.com/timescale/pgvectorscale/issues) is fixed or 
      closed by the commit with a [linking keyword recognised by 
      GitHub](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword). 
      For example, if the commit fixes bug 123, add a line at the end of the 
      commit message with  `Fixes #123`, if the commit implements feature 
      request 321, add a line at the end of the commit message `Closes #321`.
      This will be recognized by GitHub. It will close the corresponding issue 
      and place a hyperlink under the number.

* Push your changes to an upstream branch:

    * Make sure that each commit in the pull request will represent a
      logical change to the code, will compile, and will pass tests.

    * Make sure that the pull request message contains all important 
      information from the commit messages including which issues are
      fixed and closed. If a pull request contains one commit only, then
      repeating the commit message is preferred, which is done automatically
      by GitHub when it creates the pull request.

    * Rebase your local feature branch against main (`git fetch origin`,
      then `git rebase origin/main`) to make sure you're
      submitting your changes on top of the newest version of our code.

    * When finalizing your PR (i.e., it has been approved for merging),
      aim for the fewest number of commits that
      make sense. That is, squash any "fix up" commits into the commit they
      fix rather than keep them separate. Each commit should represent a
      clean, logical change and include a descriptive commit message.

    * Push your commit to your upstream feature branch: `git push -u <yourfork> my-feature-branch`

* Create and manage pull request:

    * [Create a pull request using GitHub](https://help.github.com/articles/creating-a-pull-request).
      If you know a core developer well suited to reviewing your pull
      request, either mention them (preferably by GitHub name) in the PR's
      body or [assign them as a reviewer](https://help.github.com/articles/assigning-issues-and-pull-requests-to-other-github-users/).

    * Address feedback by amending your commit(s). If your change contains
      multiple commits, address each piece of feedback by amending that
      commit to which the particular feedback is aimed.

    * The PR is marked as accepted when the reviewer thinks it's ready to be
      merged.  Most new contributors aren't allowed to merge themselves; in
      that case, we'll do it for you.

## Testing

Every non-trivial change to the code base should be accompanied by a
relevant addition to or modification of the test suite.

Please check that the full test suite (including your test additions
or changes) passes successfully on your local machine **before you
open a pull request**.

See our [testing](https://github.com/timescale/pgvectorscale/blob/main/DEVELOPMENT.md#testing)
instructions for help with how to test.

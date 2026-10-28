# Contributing

Useful contributions include clearer explanations, reproducibility fixes, and an explicitly scoped comparison using the same graph for both algorithms. Read the [README](README.md) before interpreting the saved results.

## Report a problem

Open a [GitHub issue](https://github.com/kris0516/project2-community-detection/issues) with the commit, Python/package versions, affected cell or command, expected result, and observed result. Use a small public example where possible. Remove personal paths, credentials, and unrelated local output from new reports.

## Submit a focused PR

- Use a branch or fork and explain what changes for readers or reproducibility.
- Preserve source acknowledgements and distinguish saved historical results from newly executed results.
- State whether the notebook was executed. For execution, record the graph used, seed, package versions, and affected output/figure changes. For documentation-only edits, check links and consistency with the code; execution is not required.
- Keep Louvain's current full-graph results distinct from Girvan–Newman's largest-component results. A new benchmark should define a shared graph, algorithm settings, and timing scope before comparing scores.
- Include only output and figure changes needed for the contribution; avoid unrelated notebook metadata churn.

Before submitting, run `git diff --check` and review the diff for unintended files or private output. The [MIT License](LICENSE) covers the original code within the [README's license scope](README.md#license); third-party data, data-derived outputs, and other excluded material retain their applicable terms. Keep the existing dataset archive and its notices intact; discuss new third-party data or licensing changes with the maintainer before including them.

# Publish a reviewed manual

The site builds and its artifact is uploaded by CI on every push. It has not been
published: that remains an explicit, reviewed step.

## Keep one source of truth

The Julia repository owns the manual, docstrings and executable examples. The
shared LM15 website is the cross-language entry point. It can link to the generated
Julia site or host a versioned copy of its static artifact under an agreed path.
Do not manually copy the same Julia manual into Astro/Starlight or teach two renderers
to maintain separate interpretations of Julia docstrings.

The first delivery should be a preview artifact. The documentation workflow
produces one on every push and pull request; it never publishes. A separate navigation/search experience is acceptable until
the shared-site handoff has been reviewed. Avoid a custom rendering bridge merely
to make the first preview look identical to the hub.

## Publication checklist

- Run the documented isolated build and inspect its results.
- Review the examples, built-in help, links, status wording and mobile/keyboard use.
- Confirm the artifact receipt matches the current source.
- Choose and review the final hosting path and canonical URL. Set
  `LM15_DOCS_CANONICAL` for the build only after the address is known.
- Label unreleased work `dev`. Use `stable` only for an actual reviewed release.
- Give publishing credentials only to trusted deployment code. Never expose them
  to a contributor's pull-request code or to example execution.

## Optional project-pages entry point

`docs/deploy.jl` is a separate, explicit Documenter deployment entry point for this
repository's project pages. It is not called by `make.jl` or the preview workflow.
After a reviewed build, a trusted deployment environment must provide appropriate
GitHub authentication and explicitly set both consent flags:

```sh
LM15_DOCS_PUBLISH=yes LM15_DOCS_REVIEWED=yes \
  julia --startup-file=no --project=docs docs/deploy.jl
```

These flags assert a human decision; they do not perform the review themselves.
The script checks the source-matching build receipt and uses `main` as the development
branch. Configure the repository's Pages settings and release triggers deliberately;
this documentation does not claim they already exist.

For shared-website hosting, consume the same reviewed `docs/build` artifact instead
of invoking project-pages publication. Carry its source digest, Julia/package
versions and explicit draft/release state with it. Test links at the real path.

## Roll back without rewriting evidence

Keep the last known-good artifact and its receipt. If a new publication is wrong,
restore that artifact or revert the source and rebuild through the same checks.
Do not rewrite old verification logs or relabel a failing draft as a tested release.

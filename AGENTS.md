# AGENTS.md

This file governs the whole `vLLM 2080 Ti Definitive Edition` repository.

## Project Identity And Credit

This repository is a hardware-focused fork for dual RTX 2080 Ti / SM75 vLLM
serving. It builds on upstream vLLM and preserves the local runtime work,
profiles, documentation, and benchmark evidence needed to reproduce the 2080 Ti
stack.

If you publish, redistribute, repackage, benchmark, or build a derivative from
this repository, keep clear credit to:

- Upstream vLLM and its original license.
- `vLLM 2080 Ti Definitive Edition`.
- The repository author: `github.com/weicj`.

Do not remove existing attribution, license notices, benchmark provenance, or
project identity text. If you maintain a public derivative, state that it is
based on this project unless the code has been independently replaced.

## Upstream Compatibility

This project remains a fork of upstream vLLM. When changing source files that
come from upstream vLLM:

- Preserve upstream license and copyright notices.
- Prefer small, reviewable patches over broad rewrites.
- Keep SM75/Turing-specific behavior guarded or clearly documented.
- Do not present fork-specific behavior as upstream vLLM behavior.
- If an upstream `AGENTS.md` or contribution instruction applies in a copied
  upstream subtree, follow it as well.

## Runtime And Profile Rules

This repository is organized around validated runtime routes, not generic
benchmark guesses.

- Do not invent context-size, throughput, or support claims without evidence.
- Keep profile files focused on route parameters only. Do not store global
  service settings such as GPU selection, port, chat template, or reasoning
  defaults inside route profiles.
- Use the hardware-specific README files under `profiles/` as the source of
  truth for shipped profiles.
- If adding or promoting a profile, include capacity evidence and throughput
  evidence using the repository's documented benchmark口径.
- Do not keep tiny smoke-only profiles as recommended deployment presets.

## Validation Before Publishing

Before committing or publishing changes, run the relevant subset of:

```bash
bash -n build.sh launcher.sh tools/validate_profiles.sh
bash tools/validate_profiles.sh
python3 -m py_compile <changed-python-files>
git diff --check
```

For launcher/profile changes, also verify `launcher.sh --print-config` for the
affected route and mode. For runtime kernel or graph-policy changes, include a
real benchmark or smoke result that proves the changed path still works.

## Documentation Discipline

- Keep English and Simplified Chinese documentation consistent when both exist.
- Keep benchmark numbers tied to the exact model, KV precision, MTP setting,
  context, and benchmark method.
- Restore or update linked assets when moving documentation. Broken benchmark
  figures are treated as documentation regressions.
- Avoid overstating support. Use precise wording such as `validated`,
  `supported`, `experimental`, or `not promoted` according to the evidence.

## Repository Hygiene

- Do not commit local caches, model weights, logs, temporary workspace state,
  run outputs, or generated native build artifacts.
- Keep `README.md`, `README.zh-CN.md`, `CHANGELOG.md`, `VERSION`, and
  `PROJECT_RELEASE.env` aligned for releases.
- Release tags and GitHub Releases are separate. Pushing a tag is not enough to
  update the GitHub Release page.

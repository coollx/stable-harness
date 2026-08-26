---
description: Compile the paper shell from paper/overleaf/ and report pages + citation and bibliography warnings
allowed-tools: Bash
---

Run `bash paper/compile.sh` from the repo root and relay its one-line report verbatim.
If it fails, show the log tail it printed and stop — do not modify any source files.

The same script runs with zero model inference via `! bash paper/compile.sh` or any terminal; use this command only when the result should be interpreted or acted on.

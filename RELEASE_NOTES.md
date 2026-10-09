# GitHub packaging revision 2026-10-09

This revision prepares the existing reviewer reproducibility archive for repository upload. It is a packaging revision, not a new scientific experiment or a new set of results.

- Every pre-existing Python analysis script, fixed protocol, numerical result, dependency specification and figure is preserved byte-for-byte.
- README files now describe the final manuscript title, the deterministic application and binning commands, repository upload, review access and provenance limits.
- Added `.gitignore`; the previous README mentioned ignore protection, but the supplied archive did not actually contain that file.
- Added `.gitattributes` to prevent automatic line-ending changes to checksummed files.
- Added a standard-library-only, read-only `verify_manifest.py` command and Chinese upload instructions.
- Updated the integrity manifest and recorded this packaging step separately in PROVENANCE.json.

The bundled `verification.json` and analysis-specific verification reports remain historical records. This packaging step does not rerun simulations, numerical replays, bootstrap samples or clinical models. The packaging checks only verify archive integrity, preserved analysis bytes, file inventory, Python syntax and Git ignore/attribute behavior.

No public repository URL, DOI, software license or new ethics approval is asserted here. A public GitHub repository is not itself a license grant for all included or externally downloaded material. Patient-level source datasets, predictions and splits are excluded; see DATA_SOURCES.txt.

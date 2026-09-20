# Bundled large inputs

The five large CSV/MAT inputs are distributed as gzip archives in `data/archives/`. The manifest lists each input's destination and its archive parts in reading order. The raw tracker file is supplied separately; request it from max.o.b.townsend@gmail.com when needed.

Each input has one gzip stream. Smaller streams use a single `.gz` file. Larger streams are split into parts of at most 40 MiB, named `.gz.part001`, `.gz.part002`, and so on. Read these parts together in manifest order; individual parts are not separate datasets.

## Restoring data

From the repository root, using Python 3.13:

```powershell
py -3.13 -m clamp_analysis restore-data
```

This command unpacks missing inputs and leaves existing files in place. It uses only Python's standard library, so it can run before installing the scientific dependencies.

Analysis commands restore their required inputs automatically. `preprocess` restores raw trials, `literature` restores the three large MATLAB files, and the remaining commands restore the formatted dataset. Restored files are ignored by Git.

## Storage

Keep all numbered archive parts together with the manifest. No separate archiving software or Git LFS installation is needed. Local disk usage increases when inputs are unpacked because both archives and usable files remain available.

The separately supplied `data/raw/trackers.csv` is needed only to rebuild movement-bout extraction. The bundled tracker-derived caches support the standard analyses and plots.

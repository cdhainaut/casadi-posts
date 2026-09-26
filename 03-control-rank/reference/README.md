# Reference received from Joris Gillis

Files extracted **without modification** from the received archive
`dhainaut-20260914T072903Z-1-001.zip`:

- `sparse_operator.py`: original example;
- `results.json`: original reference results;
- `sparse_operator.pdf`: “Lifting the operator, not the solve”.

The script and results are byte-for-byte identical to the references used in
our J0/J1 campaign. Hashes of these three files are in `sha256sums.txt`.

```bash
cd reference
sha256sum -c sha256sums.txt
```

These documents are attributed to Joris Gillis and preserved as received.
No additional license is presumed or applied to these third-party documents.
The complete received archive is retained locally outside Git; it is not a
dependency of the shared reproducer.

The original script writes `results.json` in its current working directory:
**do not run it from this directory**, as that would overwrite the reference
values. To reproduce the experiments, use the driver in the parent directory;
it creates a fresh output directory and stops at the first failure.

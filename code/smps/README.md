# SMPS instances

Test problems of two-stage stochastic linear programming, in the SMPS format, used by
`LShaped_scaling.jl`, `LShaped_smps_validation.jl` and the usage notebook. They are kept here so
that the benchmarks can be rerun without a network; `LShaped_instances.jl` and
`LShaped_smps_validation.jl` download them again if they are missing.

- `data-20`, `data-gbd`, `data-LandS`, `data-ssn`, `data-storm`: the instances of Linderoth,
  Shapiro and Wright, "The empirical behavior of sampling methods for stochastic programming",
  Annals of Operations Research 142 (2006), from
  https://pages.cs.wisc.edu/~swright/stochastic/sampling/, with their solutions (`.sol`).
- `slptestset`: Andy Felt's test set of stochastic linear programs, from
  https://www4.uwsp.edu/math/afelt/slptestset/download.html, with its documentation (`doc/`) and
  the published solutions. Left out: the multistage `bonds` instances (12 MB, which `read_smps`
  refuses), the PostScript documentation `doc.ps`, and the CVS bookkeeping.

`LShaped_smps_validation_results.md`, in the directory above, says which published values the
reader reproduces, and explains the differences.

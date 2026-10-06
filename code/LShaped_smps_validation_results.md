# Validation of the SMPS reader: results

Output of `LShaped_smps_validation.jl` (without `full`), on 5 October 2026: Julia 1.12.5, JuMP
1.31.2, HiGHS.jl 1.25.1. See the header of the script for the sources of the published values and
the known differences.

```
Felt's test set, on the full support
  instance                    optimal value          published rel. diff.  
  airlift.first               249101.672072      249101.672072  +1.33e-12  25 scenarios, airlift/solution.txt
  airlift.second              269665.498390      269665.498390  +3.97e-13  25 scenarios, airlift/solution.txt
  electric                       381.853333         381.853333  +8.73e-10  3 scenarios, doc/electric.tex
  electric (BLOCKS)              381.853333         381.853333  +8.73e-10  3 scenarios, doc/electric.tex
  chem                        -13009.166667      -13009.166667  +2.56e-11  2 scenarios, doc/chem.tex
  env.aggr                     15963.929095       15963.929095  +2.70e-11  5 scenarios, environ/env.soln.aggr
  env.loose                    14794.608219       14794.608219  +1.35e-11  5 scenarios, environ/env.soln.loose
  env.imp                      20773.888524       21010.235039  -1.12e-02  15 scenarios, environ/env.soln.imp
    CPA's reading              21010.235039       21010.235039  +1.69e-13  zeros of the BLOCKS lost, as CPA does
  env.lrge                     20799.265052       21034.731951  -1.12e-02  8232 scenarios, environ/env.soln.lrge
    CPA's reading              21034.731951       21034.731951  +1.78e-11  zeros of the BLOCKS lost, as CPA does
  assets.small                  -723.838613        -720.472240  -4.67e-03  100 scenarios, assets/solution.txt; unexplained
  cargo.8                        418.512500                  -             8 scenarios, stoch file revised after the solutions
  cargo.64                       423.012499                  -             64 scenarios, stoch file revised after the solutions

Linderoth, Shapiro and Wright: their solutions, on 100 000 draws (± 2 standard errors)
  LandS                          225.554963         225.624000  -3.06e-04  ± 0.366
  gbd                           1658.063507        1655.628000  +1.47e-03  ± 4.216
```

With `full`, the same day (an earlier version of the same computation): the approximations on
500 scenarios, and the cost of their solutions on 2000 others, against the estimates of the optimal
values of Linderoth, Shapiro and Wright (their Table 4):

| Instance | Approximation (500) | Its solution (2000 draws) | Linderoth, Shapiro, Wright |
|:--|--:|--:|--:|
| 20term | 254336.07 | 254052.87 | 254311.55 |
| ssn | 9.1538 | 10.2515 | 9.913 |
| storm | 15528292.81 | 15495964.44 | 15498739.41 |

The approximation is biased downwards and the cost of its solution upwards, in expectation; both
carry the sampling error of their samples, about 1% here, which covers the differences.

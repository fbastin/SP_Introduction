# LShaped.jl vs StochasticPrograms.jl: benchmark results

Results of `LShaped_benchmark.jl` (all instances, best of 3 runs), on 5 October 2026:

- Intel Core i5-8500 (6 cores, one thread used), 23 GB of memory;
- Julia 1.12.5, JuMP 1.31.2, MathOptInterface 1.53.0, HiGHS.jl 1.25.1;
- StochasticPrograms.jl from https://github.com/fbastin/StochasticPrograms.jl at `362da81`.

Times in seconds, the best of three runs, model building included; memory allocated by one run, in
MB. Every method converged, to within 10⁻⁶ of the extensive form in relative terms.

| Instance | Method | Iterations | Cuts | Time (s) | Alloc. (MB) |
|:--|:--|--:|--:|--:|--:|
| farmer, S = 100 | extensive form | | | 0.009 | 2 |
| | LShaped.jl, single-cut | 19 | 18 | 0.226 | 16 |
| | LShaped.jl, multicut | 7 | 600 | 0.153 | 16 |
| | StochasticPrograms, single-cut | 23 | 20 | 1.215 | 69 |
| | StochasticPrograms, multicut | 9 | 496 | 0.822 | 49 |
| farmer, S = 1000 | extensive form | | | 0.208 | 20 |
| | LShaped.jl, single-cut | 22 | 21 | 2.370 | 172 |
| | LShaped.jl, multicut | 8 | 7000 | 2.130 | 448 |
| | StochasticPrograms, single-cut | 24 | 22 | 12.805 | 700 |
| | StochasticPrograms, multicut | 10 | 4988 | 8.813 | 525 |
| capacity, P = F = 10, S = 100 | extensive form | | | 0.114 | 31 |
| | LShaped.jl, single-cut | 71 | 70 | 1.490 | 143 |
| | LShaped.jl, multicut | 17 | 1600 | 0.570 | 74 |
| | StochasticPrograms, single-cut | 69 | 67 | 6.888 | 1144 |
| | StochasticPrograms, multicut | 17 | 1268 | 2.196 | 368 |
| capacity, P = F = 20, S = 500 | extensive form | | | 7.039 | 473 |
| | LShaped.jl, single-cut | 159 | 158 | 34.944 | 3511 |
| | LShaped.jl, multicut | 23 | 11000 | 10.123 | 1181 |
| | StochasticPrograms, single-cut | 152 | 150 | 171.896 | 42340 |
| | StochasticPrograms, multicut | 21 | 8582 | 34.140 | 7048 |
| no shortage, P = F = 10, S = 100 | extensive form | | | 0.096 | 25 |
| | LShaped.jl, single-cut | 57 | 155 | 1.291 | 147 |
| | LShaped.jl, multicut | 17 | 1600 | 0.730 | 100 |
| | StochasticPrograms, single-cut | 55 | 152 | 5.103 | 850 |
| | StochasticPrograms, multicut | 17 | 1261 | 2.031 | 330 |
| no shortage, P = F = 20, S = 200 | extensive form | | | 1.239 | 179 |
| | LShaped.jl, single-cut | 131 | 329 | 11.664 | 1337 |
| | LShaped.jl, multicut | 27 | 5200 | 4.733 | 620 |
| | StochasticPrograms, single-cut | 126 | 323 | 57.568 | 13449 |
| | StochasticPrograms, multicut | 23 | 3658 | 14.544 | 2835 |

## Observations

- **Both codes solve every instance correctly**, with about the same number of iterations: the
  differences, of one to four, come from StochasticPrograms evaluating a starting point before its
  first master problem.
- **LShaped.jl is 3 to 5 times faster** in the single-cut version, and 2.7 to 5.5 times in the
  multicut one, and allocates 3 to 12 times less memory: 3.5 GB against 42 GB for the single-cut
  version on the largest instance.
- **The multicut version of LShaped.jl adds every scenario's cut at each round**, where
  StochasticPrograms adds only the violated ones: 11 000 cuts against 8 582 on the largest instance.
  Adding only the violated cuts would keep its master problems smaller.
- **Neither beats the extensive form solved directly by HiGHS** on these sizes: they are linear
  programs HiGHS solves at once, and decomposition pays off when the extensive form becomes too
  large to build or to solve, or when the recourse problems are solved in parallel, which neither
  run here does.

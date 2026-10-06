# LShaped.jl vs StochasticPrograms.jl: benchmark results

The raw measures are in `LShaped_benchmark_results.csv`.

Results of `LShaped_benchmark.jl` (all instances, best of 3 runs), started with 6 threads
(`julia -t 6`), on 5 October 2026:

- Intel Core i5-8500 (6 cores), 23 GB of memory;
- Julia 1.12.5, JuMP 1.31.2, MathOptInterface 1.53.0, HiGHS.jl 1.25.1;
- StochasticPrograms.jl from https://github.com/fbastin/StochasticPrograms.jl at `362da81`, which
  does not use Julia's threads.

Times in seconds, the best of three runs, model building included; memory allocated by one run, in
MB. Every method converged, to within 10⁻⁶ of the extensive form in relative terms. The variants of
LShaped.jl: dropping the optimality cuts slack at 5 consecutive iterates (`drop_inactive = 5`),
starting at the mean-value decision (`x0 = :mean_value`), a trust region
(`regularization = :trust_region`), 10 clusters of scenarios (`cuts = 10`), and solving the
recourse problems on the 6 threads (`threads = true`). Regularized decomposition is measured
apart, below.

| Instance | Method | Iterations | Cuts | Time (s) | Alloc. (MB) |
|:--|:--|--:|--:|--:|--:|
| farmer, S = 100 | extensive form |  |  | 0.010 | 2 |
|  | LShaped.jl, single-cut | 19 | 18 | 0.163 | 9 |
|  | LShaped.jl, single-cut, drop 5 | 19 | 18 | 0.166 | 10 |
|  | LShaped.jl, single-cut, from x̄ | 18 | 17 | 0.158 | 9 |
|  | LShaped.jl, single-cut, trust region | 32 | 21 | 0.240 | 14 |
|  | LShaped.jl, 10 clusters | 11 | 86 | 0.110 | 6 |
|  | LShaped.jl, multicut | 7 | 496 | 0.091 | 6 |
|  | LShaped.jl, multicut, trust region | 17 | 396 | 0.148 | 19 |
|  | LShaped.jl, multicut, 6 threads | 7 | 496 | 0.044 | 6 |
|  | StochasticPrograms, single-cut | 23 | 20 | 1.292 | 69 |
|  | StochasticPrograms, multicut | 9 | 496 | 0.870 | 49 |
| farmer, S = 1000 | extensive form |  |  | 0.211 | 20 |
|  | LShaped.jl, single-cut | 22 | 21 | 1.735 | 107 |
|  | LShaped.jl, single-cut, drop 5 | 22 | 21 | 1.708 | 107 |
|  | LShaped.jl, single-cut, from x̄ | 21 | 20 | 1.673 | 103 |
|  | LShaped.jl, single-cut, trust region | 29 | 19 | 1.981 | 129 |
|  | LShaped.jl, 10 clusters | 19 | 163 | 1.509 | 94 |
|  | LShaped.jl, multicut | 8 | 4988 | 1.287 | 103 |
|  | LShaped.jl, multicut, trust region | 17 | 3959 | 1.826 | 782 |
|  | LShaped.jl, multicut, 6 threads | 8 | 4988 | 0.748 | 103 |
|  | StochasticPrograms, single-cut | 24 | 22 | 13.372 | 699 |
|  | StochasticPrograms, multicut | 10 | 4988 | 8.916 | 525 |
| capacity, P = F = 10, S = 100 | extensive form |  |  | 0.118 | 31 |
|  | LShaped.jl, single-cut | 71 | 70 | 1.196 | 65 |
|  | LShaped.jl, single-cut, drop 5 | 66 | 65 | 1.150 | 64 |
|  | LShaped.jl, single-cut, from x̄ | 72 | 71 | 1.289 | 67 |
|  | LShaped.jl, single-cut, trust region | 74 | 68 | 1.257 | 67 |
|  | LShaped.jl, 10 clusters | 26 | 235 | 0.578 | 34 |
|  | LShaped.jl, multicut | 17 | 1249 | 0.458 | 31 |
|  | LShaped.jl, multicut, trust region | 21 | 1153 | 0.447 | 48 |
|  | LShaped.jl, multicut, 6 threads | 17 | 1249 | 0.185 | 31 |
|  | StochasticPrograms, single-cut | 69 | 67 | 6.645 | 1144 |
|  | StochasticPrograms, multicut | 17 | 1268 | 2.140 | 368 |
| capacity, P = F = 20, S = 500 | extensive form |  |  | 6.888 | 468 |
|  | LShaped.jl, single-cut | 159 | 158 | 27.598 | 1142 |
|  | LShaped.jl, single-cut, drop 5 | 188 | 187 | 32.404 | 1322 |
|  | LShaped.jl, single-cut, from x̄ | 150 | 149 | 26.940 | 1091 |
|  | LShaped.jl, single-cut, trust region | 140 | 134 | 23.766 | 1025 |
|  | LShaped.jl, 10 clusters | 68 | 631 | 14.318 | 609 |
|  | LShaped.jl, multicut | 23 | 9789 | 8.860 | 403 |
|  | LShaped.jl, multicut, trust region | 25 | 8077 | 7.144 | 677 |
|  | LShaped.jl, multicut, 6 threads | 23 | 9789 | 4.514 | 404 |
|  | StochasticPrograms, single-cut | 152 | 150 | 179.455 | 42340 |
|  | StochasticPrograms, multicut | 21 | 8582 | 35.160 | 7048 |
| no shortage, P = F = 10, S = 100 | extensive form |  |  | 0.102 | 25 |
|  | LShaped.jl, single-cut | 57 | 155 | 1.056 | 70 |
|  | LShaped.jl, single-cut, drop 5 | 57 | 155 | 1.087 | 72 |
|  | LShaped.jl, single-cut, from x̄ | 57 | 106 | 1.071 | 62 |
|  | LShaped.jl, single-cut, trust region | 45 | 143 | 0.896 | 61 |
|  | LShaped.jl, 10 clusters | 24 | 295 | 0.637 | 46 |
|  | LShaped.jl, multicut | 17 | 1303 | 0.580 | 45 |
|  | LShaped.jl, multicut, trust region | 15 | 1065 | 0.496 | 51 |
|  | LShaped.jl, multicut, 6 threads | 17 | 1303 | 0.222 | 45 |
|  | StochasticPrograms, single-cut | 51 | 148 | 4.893 | 793 |
|  | StochasticPrograms, multicut | 17 | 1249 | 2.047 | 330 |
| no shortage, P = F = 20, S = 200 | extensive form |  |  | 1.211 | 177 |
|  | LShaped.jl, single-cut | 131 | 329 | 9.518 | 473 |
|  | LShaped.jl, single-cut, drop 5 | 146 | 344 | 10.212 | 517 |
|  | LShaped.jl, single-cut, from x̄ | 131 | 230 | 9.491 | 433 |
|  | LShaped.jl, single-cut, trust region | 119 | 317 | 8.584 | 444 |
|  | LShaped.jl, 10 clusters | 59 | 726 | 5.510 | 305 |
|  | LShaped.jl, multicut | 27 | 4578 | 4.022 | 252 |
|  | LShaped.jl, multicut, trust region | 22 | 3773 | 3.270 | 274 |
|  | LShaped.jl, multicut, 6 threads | 27 | 4578 | 1.577 | 252 |
|  | StochasticPrograms, single-cut | 132 | 329 | 60.602 | 14073 |
|  | StochasticPrograms, multicut | 23 | 3801 | 13.514 | 2839 |

## Observations

- **Both codes solve every instance correctly**, with about the same number of iterations: the
  differences, of one to four, come from StochasticPrograms evaluating a starting point before its
  first master problem.
- **LShaped.jl is 4.6 to 7.9 times faster in its single-cut version, 3.4 to 9.6 times in its
  multicut version, on one thread**, and allocates 5 to 37 times less memory: 1.1 GB against 42 GB
  for the single-cut version on the largest instance.
- **Both multicut versions add only the violated cuts**, hence comparable cut counts. In LShaped.jl,
  this removed 10 to 30% of the cuts it added before (9 789 instead of 11 000 on the largest
  instance, 4 578 instead of 5 200 on the largest one without shortage).
- **Solving the recourse problems on 6 threads** makes the multicut version 1.7 to 2.6 times faster,
  with the same iterates and cuts. On the largest instance, it beats the extensive form solved
  directly by HiGHS: 4.5 s against 6.9 s.
- **Dropping the inactive cuts does not pay off here.** With a window of 5 iterations, the long
  single-cut runs take more iterations, because the dropped cuts must be generated again: 188
  instead of 159 on the largest instance, 146 instead of 131 on the largest one without shortage.
  It is off by default.
- **The code optimizations of LShaped.jl** (direct models for problems given as data, constraints
  read and written by typed groups, constant data converted once, the master objective rebuilt only
  when it changes) cut its times by 14 to 47% and its memory by 55 to 77% with respect to the
  previous version, the most on many small recourse problems: 2.34 s to 1.59 s and 163 MB to 101 MB
  for the single-cut version on the farmer with 1000 scenarios.
- **Ten clusters of scenarios are the compromise of deck 04's hybrid approaches**: two to three
  times fewer iterations than the single-cut version (68 instead of 159 on the largest instance),
  ten to fifteen times fewer cuts than the multicut one (631 instead of 9 789), and a time in
  between, nearer the multicut one: 14.3 s, against 27.6 s and 8.9 s.
- **The trust region** (`regularization = :trust_region`, default radius) mostly helps the multicut
  version on the capacity instances: 8 to 18% fewer cuts and up to 19% less time (7.1 s instead
  of 8.9 s on the largest instance). On the farmer, whose optimum the plain method reaches in 7 or 8
  iterations, the box slows it down: 17 iterations instead of 7 or 8, and up to 8 times the memory.
  In the single-cut version, it saves at most 15% of the time.
- **Starting at the mean-value decision** (`x0 = :mean_value`) saves at most 9 iterations, and
  once costs one: the first cut, at a good point, does not change much on runs of 20 to 160
  iterations.
- **Regularized decomposition is measured apart** (`LShaped_benchmark.jl quick rd`), on the small
  instances, because its quadratic masters need Ipopt (HiGHS's QP solver returns wrong solutions
  on them, which `lshaped` detects), and because the fixed `rho = 1` of deck 04 takes small steps
  on decisions in the hundreds. On the farmer with 50 scenarios, it does what it is meant to do for
  the single-cut version, 13 iterations instead of 21, but the QP masters make it slower: 0.11 s
  instead of 0.09 s. On the small capacity instances, it takes 3 to 18 times more iterations than
  without regularization (78 to 182 against 10 to 28), and 0.8 to 2.0 s instead of 0.04 to 0.07 s. On the capacity instance with 500 scenarios, its multicut version had
  not converged after 40 iterations and 90 s; the trust region, whose master stays an LP, converged
  in 25 iterations and 8.8 s.
- **What remains is mostly the solver**: about 0.45 ms per recourse problem, of which 0.13 ms is a
  fixed cost per call that neither presolve, the dual simplex nor direct mode reduce.
- **On one thread, neither code beats the extensive form** solved directly by HiGHS: these are
  linear programs HiGHS solves at once, and decomposition pays off when the extensive form becomes
  too large, or, as above, when the recourse problems are solved in parallel.

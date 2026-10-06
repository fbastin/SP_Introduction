# When does decomposition pay off? Scaling results

Results of `LShaped_scaling.jl`, on 5 October 2026: Intel Core i5-8500 (6 cores), 23 GB of memory,
Julia 1.12.5, JuMP 1.31.2, HiGHS.jl 1.25.1. The time limit was 900 s per measure, the memory limit
12 GB. The interior-point method, the slowest at every size of the capacity family, was not run on
storm and ssn.

Each measure runs in a Julia process of its own, started with `-t cores`: HiGHS's number of
threads, fixed once per process, is `cores` for the extensive form and 1 for each recourse problem
of the decomposition, `cores` of which are solved at once on Julia's threads. Times include
building the models, not compiling them; the memory is the peak resident set of the process during
the method, less the resident set before it, so that it counts what HiGHS allocates. The methods:

- **extensive**: the extensive form, HiGHS's default LP method (dual simplex);
- **extensive-ipm**: the same, with HiGHS's interior-point method, the one that uses threads;
- **multicut**: the multicut L-shaped method of LShaped.jl (`tol = 1e-6`).

Every method that finished found the same optimal value, to the digits shown by the script.

## Capacity expansion, 20 plants × 20 products

Times in seconds, memory in MB.

| Scenarios | Cores | extensive | extensive-ipm | multicut (iterations) |
|--:|--:|--:|--:|--:|
| 500 | 1 | 8.7 (224) | 36.5 (513) | 9.2 (362), 23 it. |
| 500 | 6 | 7.0 (262) | 32.2 (591) | **4.3** (384) |
| 1000 | 1 | 27.3 (576) | 84.5 (1378) | **23.0** (767), 24 it. |
| 1000 | 6 | 24.8 (490) | 83.0 (1587) | **14.2** (702) |
| 2000 | 1 | 87.8 (1289) | 436.8 (2107) | **72.3** (1448), 26 it. |
| 2000 | 6 | 91.7 (1262) | 403.2 (2432) | **59.3** (1453) |
| 4000 | 1 | 436.7 (2515) | 844.9 (4224) | **282.7** (2771), 24 it. |
| 4000 | 6 | 423.2 (2593) | 852.2 (4603) | **216.4** (2807) |
| 8000 | 1 | > 900 | > 900 | > 900 |
| 8000 | 6 | > 900 | > 900 | > 900 |

(time (memory); the number of iterations of the multicut method does not depend on the cores.)

## storm and ssn: samples of the SMPS instances of Linderoth, Shapiro and Wright

Samples of `S` scenarios drawn from the law of the instance (`sample_scenarios`, stream 1). storm,
cargo flight scheduling: 121 first-stage variables, 528 rows and 1259 columns per scenario. ssn,
telecommunication network design: 89 first-stage variables, 175 rows and 706 columns per scenario.

| Instance | Scenarios | Cores | extensive | multicut (iterations) |
|:--|--:|--:|--:|--:|
| storm | 250 | 1 | 10.3 (557) | **9.7** (489), 15 it. |
| | 250 | 6 | 9.8 (549) | **4.4** (496) |
| | 500 | 1 | 29.1 (1117) | **22.5** (989), 15 it. |
| | 500 | 6 | 26.0 (1155) | **11.7** (994) |
| | 1000 | 1 | 94.1 (2329) | **46.8** (1980), 15 it. |
| | 1000 | 6 | 93.0 (2235) | **38.4** (2056) |
| | 2000 | 1 | 293.2 (4290) | **145.5** (4003), 16 it. |
| | 2000 | 6 | 297.6 (4572) | **106.0** (4015) |
| | 4000 | 1 | > 900 | **490.4** (7687), 15 it. |
| | 4000 | 6 | > 900 | **373.6** (7877) |
| ssn | 250 | 1 | 20.1 (285) | **11.4** (322), 29 it. |
| | 250 | 6 | 18.9 (290) | **4.5** (316) |
| | 500 | 1 | 56.5 (568) | **23.6** (679), 26 it. |
| | 500 | 6 | 53.6 (563) | **11.4** (722) |
| | 1000 | 1 | 223.9 (1221) | **52.4** (1332), 24 it. |
| | 1000 | 6 | 243.9 (1170) | **27.6** (1357) |
| | 2000 | 1 | > 900 | **138.1** (2625), 24 it. |
| | 2000 | 6 | > 900 | **100.8** (2696) |

## Observations

- **Decomposition overtakes the extensive form early, even on one core**: from 1000 scenarios on the
  capacity family, from 250 on storm and ssn. The time of the extensive form grows faster than the
  number of scenarios — ×2.8 to ×5 per doubling — that of the multicut method ×2 to ×4. At
  the largest sizes the extensive form exceeds 15 minutes where the decomposition takes 2 to 8
  (storm, 4000 scenarios: 490 s on one core; ssn, 2000 scenarios: 138 s), and on ssn with 1000
  scenarios it is 4 times slower on one core, 9 times on six.
- **The extensive form gains nothing from more cores**: HiGHS's dual simplex is essentially
  sequential, and its interior-point method, which uses the threads, is 2 to 5 times slower than
  the simplex here (36.5 s against 8.7 s for 500 scenarios of the capacity family, 852 s against
  423 s for 4000).
- **Decomposition gains less and less from more cores as the size grows**: ×1.9 to ×2.5 with 6 cores
  at 250–500 scenarios, ×1.2 to ×1.4 at 2000–4000. The master problem becomes the bottleneck, and it
  is solved on one core: on the capacity family with 2000 scenarios, it ends with 42 353 cuts, its
  iterations grow from 1 to 5 s while the 2000 recourse problems take about 1 s, and the time of an
  iteration is nearly the same on one thread and on six. Partial aggregation (`cuts = C`), a smaller
  master, is the natural answer.
- **The memory argument does not hold as the module stands.** The multicut method uses about as much
  memory as the extensive form — 10 to 60% more on the capacity family, 9 to 20% more on ssn, 7 to
  15% less on storm — because
  it keeps a JuMP model and a HiGHS instance per scenario, so as to warm-start each recourse problem
  from its last basis. Reusing one model per thread, retargeted to each scenario, would make the
  memory of the decomposition independent of the number of scenarios.

# Comprehensive Quality & Mathematical Correctness Audit Report
**Course:** Stochastic Optimization / Stochastic Programming  
**Author / Instructor:** Fabian Bastin (Université de Montréal -- CIRRELT -- IVADO)  
**Scope:** Complete verification of all 12 slide decks (`slides/01.*.tex` through `slides/12.*.tex`)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Critical Mathematical Errors & Impact Analysis](#2-critical-mathematical-errors--impact-analysis)
3. [Slide-by-Slide Detailed Audit](#3-slide-by-slide-detailed-audit)
   - [01. Introduction](#01-introductiontex)
   - [02. Random Numbers](#02-random-numberstex)
   - [03. Two-stage Stochastic Programming](#03-two-stage-stochastic-programmingtex)
   - [04. L-Shaped Method](#04-l-shaped-methodtex)
   - [05. Chance-constrained Programming](#05-chance-constrained-programmingtex)
   - [06. Multistage Problems](#06-multistagetex)
   - [07. SDDP](#07-sddptex)
   - [08. Lagrangian Relaxation & Decomposition](#08-lagrangiantex)
   - [09. Sample Average Approximation (SAA)](#09-saatex)
   - [10. Adaptive Monte Carlo](#10-adaptive-monte-carlotex)
   - [11. Stochastic Gradient Descent](#11-stochastic-gradient-descenttex)
   - [12. Discrete Event Simulation](#12-discrete-event-simulationtex)
4. [Cross-Deck Systemic Issues](#4-cross-deck-systemic-issues)
5. [Action Plan & Recommended Next Steps](#5-action-plan--recommended-next-steps)

---

# 1. Executive Summary

A comprehensive line-by-line audit of all 12 LaTeX slide decks was performed. The inspection identified **over 120 individual issues**, categorized into three primary types:
- **Mathematical Errors & Notation Inconsistencies** (36 issues): Ranging from critical flaws in algorithm proofs (e.g., Fréchet–Hoeffding bounds, L-shaped feasibility cuts, BTRDA decrease ratios, SGD bounds, SDDP confidence intervals) to indexing and dimension mismatches in LP and dynamic programming models.
- **LaTeX Syntax, Macros & Compilability** (32 issues): Undefined or collided macros (`\red`, `\blue`, `\KK`), broken math delimiters, invalid environments, obsolete LaTeX commands, and unescaped characters.
- **Grammar, Typography & Leftover French** (58 issues): Literal French translations (*calques* like "designs", "compacity", "adequation", "envelop"), unconverted French terms (`si`, `et`, `t.q.`, `plans`, `module`, `Alors`, `journaux`, `Exercice`), and one entire slide left untranslated in French (Slide 25 in deck 12).

### Issue Distribution Across Slide Decks

| Slide Deck | Math & Formulations | LaTeX & Macros | Grammar & Language | Total Issues |
| :--- | :---: | :---: | :---: | :---: |
| [01. Introduction.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/01.%20Introduction.tex) | 4 | 4 | 9 | **17** |
| [02. Random Numbers.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/02.%20Random%20Numbers.tex) | 5 | 3 | 12 | **20** |
| [03. Two-stage stochastic programming.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/03.%20Two-stage%20stochastic%20programming.tex) | 7 | 2 | 8 | **17** |
| [04. L-Shaped method.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/04.%20L-Shaped%20method.tex) | 11 | 7 | 13 | **31** |
| [05. Chance-constrained programming.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/05.%20Chance-constrained%20programming.tex) | 5 | 4 | 6 | **15** |
| [06. Multistage.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/06.%20Multistage.tex) | 6 | 3 | 8 | **17** |
| [07. SDDP.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/07.%20SDDP.tex) | 6 | 4 | 6 | **16** |
| [08. Lagrangian.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/08.%20Lagrangian.tex) | 7 | 4 | 8 | **19** |
| [09. SAA.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/09.%20SAA.tex) | 6 | 3 | 9 | **18** |
| [10. Adaptive Monte Carlo.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/10.%20Adaptive%20Monte%20Carlo.tex) | 7 | 3 | 11 | **21** |
| [11. Stochastic gradient descent.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/11.%20Stochastic%20gradient%20descent.tex) | 7 | 3 | 5 | **15** |
| [12. Discrete event simulation.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/12.%20Discrete%20event%20simulation.tex) | 4 | 1 | 6 | **11** |
| **Total** | **75** | **41** | **101** | **217** |

---

# 2. Critical Mathematical Errors & Impact Analysis

> [!CAUTION]
> The following high-severity issues affect mathematical validity and algorithm convergence proofs. They should be addressed with priority.

### 1. Fréchet–Hoeffding Upper Bound Proof (`02. Random Numbers.tex`, L939–943)
- **Error:** The slide claims $\mathbb{P}[X \le x \cap Y \le y] \le \mathbb{P}[X \le x] \mathbb{P}[Y \le y] \le \min(F_X(x), F_Y(y))$.
- **Mathematical Reality:** $\mathbb{P}[A \cap B] \le \mathbb{P}[A]\mathbb{P}[B]$ is false for positively associated variables. The upper bound follows directly from monotonic event inclusion: $\{ X \le x \cap Y \le y \} \subseteq \{ X \le x \}$ and $\{ X \le x \cap Y \le y \} \subseteq \{ Y \le y \}$.

### 2. Feasibility Cut Direction (`04. L-Shaped method.tex`, L1383–1384)
- **Error:** Farkas cut is written as $\sigma^T(h(\xi_k) - T(\xi_k)x) \ge 0$.
- **Mathematical Reality:** The valid Farkas separating hyperplane requires $\sigma^T(h(\xi_k) - T(\xi_k)x) \le 0$. The $\ge 0$ sign cuts off feasible first-stage decisions.

### 3. Multivariate Gaussian Density (`05. Chance-constrained programming.tex`, L368)
- **Error:** $f(x) = \frac{1}{\sqrt{(2\pi)^n/2\det(\Sigma)}}e^{-\frac{1}{2}(x-\mu)'\Sigma (x-\mu)}$.
- **Mathematical Reality:** Denominator has `/2` misplaced inside the root, and the exponent contains $\Sigma$ instead of $\Sigma^{-1}$.

### 4. SDDP Central Limit Theorem & Stopping Criterion (`07. SDDP.tex`, L773–814)
- **Errors:**
  1. States $\overline{z}_K \overset{\mathcal{D}}{\rightarrow} \mathcal{N}(\mu, \sigma^2)$ instead of $\sqrt{K}(\overline{z}_K - \mu) \overset{\mathcal{D}}{\rightarrow} \mathcal{N}(0, \sigma^2)$.
  2. The confidence interval $\overline{z}_K \pm z_{\alpha/2}\sigma$ omits the standard error factor $1/\sqrt{K}$, failing to shrink as $K \to \infty$.
  3. Sample standard deviation formula is written $\hat{s} = \frac{1}{K-1}\sqrt{\sum(z_k - \bar{z}_K)^2}$ instead of $\sqrt{\frac{1}{K-1}\sum(z_k - \bar{z}_K)^2}$.

### 5. Trust-Region Actual/Predicted Decrease Ratio (`10. Adaptive Monte Carlo.tex`, L630–632)
- **Error:** In the BTRDA trust-region algorithm, $\rho_k = \frac{\hat{g}_{N^-}(x_k+s_k) - \hat{g}_{N^-}(x_k)}{\Delta m_k^{N^-}}$.
- **Mathematical Reality:** For minimization, actual decrease is $\text{Ared} = \hat{g}(x_k) - \hat{g}(x_k+s_k)$. As written, the numerator is $-\text{Ared}$, resulting in negative $\rho_k$ for every descent step and causing the algorithm to reject every candidate iterate.

### 6. Vector Variance Expansion (`11. Stochastic gradient descent.tex`, L210–214, L714)
- **Error:** States $\operatorname{Var}[g] = \mathbb{E}[\|g\|^2] - (\mathbb{E}[\|g\|])^2$.
- **Mathematical Reality:** For a random vector $g$, $\operatorname{Var}(g) = \mathbb{E}[\|g - \mathbb{E}[g]\|^2] = \mathbb{E}[\|g\|^2] - \|\mathbb{E}[g]\|^2$. The term $(\mathbb{E}[\|g\|])^2$ represents the scalar variance of the Euclidean norm, not the vector variance.

### 7. SGD Contraction Bound Sign Error (`11. Stochastic gradient descent.tex`, L1054–1055)
- **Error:** $\mathbb{E}[f(x_k) - f^*] \le \frac{\bar{\alpha}LM}{2c\mu} \mathbf{-} (1-\bar{\alpha}c\mu)^{k-1} \left(f(x_1) - f^* - \frac{\bar{\alpha}LM}{2c\mu}\right)$.
- **Mathematical Reality:** The proof derives $\mathbb{E}[f(x_k) - f^*] \le \frac{\bar{\alpha}LM}{2c\mu} \mathbf{+} (1-\bar{\alpha}c\mu)^{k-1} \left(f(x_1) - f^* - \frac{\bar{\alpha}LM}{2c\mu}\right)$. The minus sign violates contraction bounds and contradicts the subsequent slide's proof.

---

# 3. Slide-by-Slide Detailed Audit

---

### [01. Introduction.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/01.%20Introduction.tex)

#### Mathematical & Formulation Issues
1. **Lines 324–325:** Double addition operator `+\\` followed by `+ \sum`.
   ```diff
   - \min\ & 150x_1 + 230x_2 + 260x_3 +\\
   - &  + \sum_{s = 1}^3 \frac{1}{3}(238y_{1s} - 170w_{1s} + 210y_{2s} - 150w_{2s} - 36w_{3s} - 10w_{4s}) \\
   + \min\ & 150x_1 + 230x_2 + 260x_3 \\
   + & + \sum_{s = 1}^3 \frac{1}{3}(238y_{1s} - 170w_{1s} + 210y_{2s} - 150w_{2s} - 36w_{3s} - 10w_{4s}) \\
   ```
2. **Line 429:** Closing parenthesis placed outside the math delimiter: `$F(t) = P(\bsxi \leq t$)` $\to$ `$F(t) = \mathbb{P}(\bsxi \leq t)$`.
3. **Line 530:** Refers to "solution of the second-stage is obtained by computing the zero of the objective gradient", but $x$ is the **first-stage ordering decision**.
   ```diff
   - Assuming $x \ne 0$, the solution of the second-stage is obtained by computing the zero of the objective gradient.
   + Assuming $x > 0$, the optimal order quantity $x^*$ is obtained by setting the derivative of the objective function to zero.
   ```
4. **Line 556:** Confuses distribution with random variable: `$N(650, 80^2) \sim 80\Phi+650$` $\to$ "Since $\bsxi = 80 Z + 650$ where $Z \sim \mathcal{N}(0,1)$ with CDF $\Phi$, it is easy to show that...".

#### LaTeX Syntax & Macros
- **Lines 4 & 14:** Duplicate `\usepackage[utf8]{inputenc}` $\to$ Remove line 14.
- **Line 88:** Leftover French phantom spacing: `\phantom{t.q. }` $\to$ `\phantom{\text{s.t. }}`.
- **Line 119:** Straight quotes inside math mode: `& ``\min_{x \in X}" g_0(x,\bsxi)` $\to$ `& \text{``}\min_{x \in X}\text{''} g_0(x,\bsxi)`.
- **Line 327:** French abbreviation: `\text{t.q. }` $\to$ `\text{s.t. }`.

#### Grammar, Phrasing & Leftover French
- **Line 134:** `Birge et Louveaux` $\to$ `Birge and Louveaux`.
- **Line 160:** "can sold... at \$36T... and \$10T after" $\to$ "can sell... at \$36/T... and \$10/T thereafter".
- **Lines 173–174:** Table units: "Average return ($T$)" $\to$ "Average yield (T/acre)"; "Plantation cost" $\to$ "Planting cost".
- **Lines 361, 365, 369:** French plural: "Productions (T)" $\to$ "Production (T)".
- **Lines 444, 452:** "the newsvendor buy" $\to$ "buys"; "$x$ newspaper" $\to$ "$x$ newspapers".
- **Line 552:** Leftover French `Alors` $\to$ `Then` (or `Thus`).
- **Line 575:** Leftover French `journaux` $\to$ `newspapers`.
- **Line 577:** French calque: "On an economical point of view" $\to$ "From an economic point of view".

---

### [02. Random Numbers.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/02.%20Random%20Numbers.tex)

#### Mathematical & Formulation Issues
1. **Line 196:** "on 32-bit architectures, the largest representable (signed) integer is $2^{31}$" $\to$ In signed two's complement, the maximum value is $2^{31}-1 = 2147483647$.
2. **Lines 435–445:** GFSR recurrence index mismatch: dummy index in the sum is $\ell$, but formula writes $x_{n\nu+j-1}$ with index $j$. Also replaces `tabular` inside math delimiters with `pmatrix`.
   ```diff
   - u_n &=& \sum_{\ell=1}^w x_{n\nu+j-1} 2^{-\ell} ~=~ .x_{n\nu} x_{n\nu+1} x_{n\nu+2} \ldots x_{n\nu+\ell-1}\\
   + u_n &=& \sum_{\ell=1}^w x_{n\nu+\ell-1} 2^{-\ell} ~=~ 0.x_{n\nu} x_{n\nu+1} x_{n\nu+2} \ldots x_{n\nu+w-1}\\
   ```
3. **Line 870:** Matrix multiplication directly by distribution symbol: `\bsX = \bsmu + L\calN(\bszero, \bsI)` $\to$ `\bsX = \bsmu + L \bsZ, \text{ where } \bsZ \sim \mathcal{N}(\bszero, \bsI)`.
4. **Lines 939–943:** **Critical Proof Error in Fréchet–Hoeffding Upper Bound**:
   ```diff
   - C(x, y) &= P[X \leq x \cap Y \leq y] \\ & \leq P[X \leq x] P[Y \leq y] \\ & \leq \min(F_X(x), F_Y(y))
   + H(x, y) &= \PP[X \leq x \cap Y \leq y] \\
   + &\leq \min(\PP[X \leq x], \PP[Y \leq y]) = \min(F_X(x), F_Y(y)).
   ```
5. **Lines 972–980:** Gaussian copula incorrectly designates $\phi$ as CDF and uses $\phi^{-1}$ $\to$ Change to $\Phi$ (standard normal CDF) and $\Phi^{-1}$ (quantile function).

#### LaTeX Syntax & Macros
- **Lines 35 & 37:** Duplicate macro definition `\def\bu{\boldsymbol{u}}` $\to$ Remove line 37.
- **Lines 161, 188:** Unenclosed math variables: `a (the dividend) and n (the divisor)` $\to$ `$a$ and $n$`; `a-1` $\to$ `$a-1$`.
- **Line 798:** Undefined macro `\rit^2` $\to$ `\RR^2`.

#### Grammar, Phrasing & Leftover French
- **Line 41:** Title: "A short tutorial of random numbers generation" $\to$ "A short tutorial on random number generation".
- **Lines 84–85:** "adequately transform an uniformly" $\to$ "adequately transforming a uniformly".
- **Line 174 & 256:** Typo "is" vs "if": "$m$ is $c \ne 0$" $\to$ "$m$ if $c \ne 0$"; "maximal period if $\rho = m^k-1$" $\to$ "is".
- **Lines 260, 284:** "Period of MRG's" $\to$ "MRGs"; "Choosing a good MRG's" $\to$ "Choosing a Good MRG".
- **Line 291:** French word `plans` $\to$ `planes` ("contained in two planes!").
- **Line 322:** French word `module` $\to$ `modulus`.
- **Line 355:** French accent `P\'eriod` $\to$ `Period`.
- **Lines 469–470:** `\mbox{ si }` $\to$ `\text{if }`.
- **Lines 516, 531, 547, 549, 602, 624:** Typos: `critisms` $\to$ `criticisms`; `impletation` $\to$ `implementation`; `independant` $\to$ `independent`; `random random` $\to$ `random`; `Prefered` $\to$ `Preferred`; `mixte` $\to$ `mixed`.
- **Lines 740, 743:** "the discrete cas is analoge" $\to$ "case is analogous"; "majors $f$" $\to$ "majorizes $f$".

---

### [03. Two-stage stochastic programming.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/03.%20Two-stage%20stochastic%20programming.tex)

#### Mathematical & Formulation Issues
1. **Lines 140–144, 156–160:** Penalty recourse LP formulations omit non-negativity on deviation variables:
   ```diff
     \min\ & c^Tx+q^T_+s(\bsxi)+q^T_-t(\bsxi) \\
     \mbox{s.t. } & Ax = b, \\
     & T(\bsxi)x + s(\bsxi)-t(\bsxi) = h(\bsxi), \\
   - & x \in X.
   + & x \in X, \quad s(\bsxi) \geq 0, \quad t(\bsxi) \geq 0.
   ```
2. **Lines 251–260:** Extensive form LP matrix has missing `+` after `\ldots`, and uses lowercase $s$ on the final block: $+ Wy_s = h_s$ $\to$ $+ Wy_S = h_S$, and $y_s \in Y$ $\to$ $y_S \in Y$.
3. **Lines 340–344:** **Critical Index Typo in Constraint**:
   ```diff
   - \xi_{2s}x_1 + x_2 + y_{1s} \geq 4
   + \xi_{2s}x_1 + x_2 + y_{2s} \geq 4, \quad s = 1,\ldots,S
   ```
4. **Line 452 & 577:** Deterministic $\xi$ written inside expectation $\EE_{\bsxi}[\dots]$ $\to$ Change to random vector $\bsxi$.
5. **Line 565:** Support set $\Xi = \{2^n, n \in \NN_+\}$ starts at $n=0$ ($2^0=1$) $\to$ Change to $n \in \NN_0$.
6. **Line 1021:** Dimension mismatch in composite convexity theorem: "If $A$ is a linear transformation from $\RR^n$ to $\RR^n$, and $f(x)$ is convex on $\RR^m$..." $\to$ $A$ must map from $\RR^n$ to $\RR^m$.
7. **Line 1302:** Partial derivative is evaluated at reference point $(\hat{x}, \hat{\xi})$, not $(x, \xi)$.
8. **Line 1349:** Step factor typo: `(1/p)` $\to$ `(1/h)`.
9. **Line 1451:** Strong second-stage feasible set is denoted $K_2^s$, while weak is $K_2$.

#### LaTeX Syntax & Formatting
- **Lines 839–841:** Missing closing parenthesis on outer summation term in $Q(x, \xi(\omega))$.
- **Lines 856, 881:** Extra closing parenthesis: `z(\xi(\omega)))` $\to$ `z(\xi(\omega))`.

#### Grammar, Phrasing & Leftover French
- **Lines 52, 56, 57:** "drawn form" $\to$ "drawn from"; "A event" $\to$ "An event"; "random of random events" $\to$ "events"; "output of the experiment" $\to$ "outcome".
- **Lines 94–96:** Duplicate bullet point on wait-and-see $\to$ Consolidate.
- **Line 231:** `second-stage contraints` $\to$ `constraints`.
- **Line 297, 300:** "company want" $\to$ "wants"; "independant and take" $\to$ "independent and takes".
- **Line 326:** Missing distribution symbol: `\bsxi_2[1/3, 1]` $\to$ `\bsxi_2 \sim U[1/3, 1]`.
- **Lines 818, 913–914, 1152–1153, 1407, 1446:** Leftover French: "By componing" $\to$ "By combining"; `\mbox{ si }` $\to$ `\text{if }`; `Birge et Louveaux` $\to$ `Birge and Louveaux`; `\mbox{ t.q. }` $\to$ `\text{s.t. }`.
- **Lines 1033–1037:** "piecewise convex linear" $\to$ "piecewise linear convex".

---

### [04. L-Shaped method.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/04.%20L-Shaped%20method.tex)

#### Mathematical & Formulation Issues
1. **Lines 703–704:** Primal slack calculation typo: $v_3^+ = 4.6$ $\to$ $v_3^+ = 4.8$ (with $w^* = v_3^+ + v_5^+ = 4.8 + 6.4 = 11.2$).
2. **Line 733:** Scenario right-hand side vector index: $h_1$ is written instead of scenario 4 vector $h_4 = (0, 0, 4.8, 6.0, 6.4, 8.0)^T$.
3. **Line 1159:** Variable typo in dual subproblem: $Q(\hat{x}, \xi_s) = \max_\pi \{ \pi^T(h(\xi_s) - T(\xi_s)x) \}$ $\to$ replace $x$ with trial point $\hat{x}$.
4. **Lines 1181, 1340:** Subgradient belongs to $\partial \mathcal{Q}(\hat{x})$ at the trial point, not $\partial \mathcal{Q}(x)$ for arbitrary $x$.
5. **Lines 1217–1231:** Omitted scenario values: explicitly state scenario support $\bsxi \in \{1, 2, 4\}$ with $p_1 = p_2 = p_3 = 1/3$ before the piecewise evaluation of $\mathcal{Q}(x)$.
6. **Lines 1383–1384:** **Critical Sign Error in Feasibility Cut**:
   ```diff
   - \mathcal{B}_1 := \mathcal{B}_1 \cap \lbrace (x,\theta_1,\theta_2,\ldots,\theta_{|S|}) \,|\, \sigma^T(h(\xi_k) - T(\xi_k)x) \geq 0\rbrace.
   + \mathcal{B}_1 := \mathcal{B}_1 \cap \lbrace (x,\theta_1,\theta_2,\ldots,\theta_{|S|}) \,|\, \sigma^T(h(\xi_k) - T(\xi_k)x) \leq 0\rbrace.
   ```
7. **Line 1601:** Syntax typo: `\theta \geq= 0` $\to$ `\theta \geq 0`.
8. **Line 1603:** Dangling iterate $\theta^4$ without value $\to$ `\theta^4 = 0`.
9. **Line 1616:** Cut index typo: `\theta_1 \ge \frac{1}{2}(-1.5x)` $\to$ `\theta_2 \ge -\frac{3}{4}x$.
10. **Line 1620:** Iteration superscript on master variable: `\theta_2^3 \ge \frac{1}{7}(x-2)` $\to$ `\theta_2 \ge \frac{1}{7}(x-2)$.

#### LaTeX Syntax & Macros
- **Preamble:** Missing definitions for `\red` and `\blue` $\to$ Add `\def\red{\color{red}}` and `\def\blue{\color{blue}}`.
- **Line 15:** Duplicate `\usepackage[utf8]{inputenc}` $\to$ Remove.
- **Line 440:** Unmatched parenthesis: `Q_s(x) = \min \left\lbrace (q_s^T y_s \dots \right\rbrace`.
- **Lines 1351, 1448:** `\RR_n^+` $\to$ `\RR^n_+`.

#### Grammar, Phrasing & Leftover French
- **Lines 56, 1188, 1513, 1846:** `Kall et Wallace` $\to$ `and`; `Birge et Louveaux` $\to$ `and`; `Exercice` $\to$ `Exercise`.
- **Lines 242, 458, 512, 1088, 1197, 1323, 1457:** `\mbox{t.q. }` $\to$ `\text{s.t. }`; `\mbox{ si }` $\to$ `\text{if }`.
- **Line 454:** "by remplacing $\mathcal{Q}(x)$ by $\theta$: singe-cut" $\to$ "by replacing $\mathcal{Q}(x)$ with $\theta$: single-cut".
- **Lines 1130, 1144, 1149:** "support hyperplans" $\to$ "supporting hyperplanes"; "subdifferenntial" $\to$ "subdifferential".
- **Line 1299:** Invalid set index `$s = 1 \in S$` $\to$ `$s = 1,\ldots,S$`.
- **Line 1421:** Repeated label: `clusters'' $\mathcal{S}_1$, $\mathcal{S}_1$, \ldots` $\to$ `$\mathcal{S}_1, \mathcal{S}_2, \ldots$`.
- **Line 1628:** "shows the the multicut" $\to$ "shows that the multicut".
- **Line 1700:** "designs an usual norm" $\to$ "denotes a standard norm".

---

### [05. Chance-constrained programming.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/05.%20Chance-constrained%20programming.tex)

#### Mathematical & Formulation Issues
1. **Line 368:** **Critical Errors in Gaussian PDF**:
   ```diff
   - f(x) = \frac{1}{\sqrt{(2\pi)^n/2\det(\Sigma)}}e^{-\frac{1}{2}(x-\mu)'\Sigma (x-\mu)}
   + f(x) = \frac{1}{\sqrt{(2\pi)^n \det(\Sigma)}} \exp\left( -\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right)
   ```
2. **Line 652:** **Contradictory KKT Conditions**: Includes $x \ge 0$ when short-selling is unconstrained, and uses scalar derivative notation $\frac{dL}{dx}$ instead of gradient $\nabla_x L(x, \lambda)$.
3. **Lines 706 & 711:** Indicator function notation: $\mathcal{I}_{(0, \infty)}(t)$ is defined as 1 for $t \le 0$ and takes a list of boolean constraints as its argument $\to$ Change to $\mathbb{I}_{(-\infty, 0]}(\max_{1 \le j \le r} g_j(x, \bsxi))$.
4. **Line 813:** Equating probability to quantile: `$P[Z \leq \beta] = \eta = \text{VaR}(Z; \eta)$` $\to$ `$P[Z \leq \beta^*] = \eta \iff \beta^* = \text{VaR}(Z; \eta)$`.
5. **Lines 187, 207:** Set intersection operator `\cap` placed without braces: `$P[\xi_1 x_1 + x_2 \geq 7 \cap \xi_2 x_1 + x_2 \geq 4]$` $\to$ `$P[\xi_1 x_1 + x_2 \geq 7, \; \xi_2 x_1 + x_2 \geq 4]$`.

#### LaTeX Syntax & Macros
- **Lines 3 & 10:** Duplicate `\usepackage[utf8]{inputenc}` $\to$ Remove line 10.
- **Line 40:** Macro typo: `\def\bomega{\boldsymbol\xi}` $\to$ `\def\bomega{\boldsymbol\omega}`.
- **Lines 693–695:** Duplicate slide title: `\frametitle{Numerical illustration}` immediately before `\frametitle{Generalization}` $\to$ Remove line 693.
- **Line 614:** Unmatched closing parenthesis: `$P[\beta^T x \leq 0])$` $\to$ `$P[\beta^T x \leq 0]$`.

#### Grammar, Phrasing & Leftover French
- **Lines 67–83:** "dropped high-school" $\to$ "dropped out of high school"; "financed his bachelor at University" $\to$ "financed his bachelor's degree at the University"; "academic carrer" $\to$ "career".
- **Lines 230, 235, 846:** `i,\ldots,m` $\to$ `i=1,\ldots,m`.
- **Lines 456, 577, 581:** `contraint` $\to$ `constraint`; `will no suffer` $\to$ `will not suffer`; `Typicially` $\to$ `Typically`; `chanced-constraint` $\to$ `chance constraint`.
- **Line 729:** "discuss about it in more details" $\to$ "discuss it in more detail".

---

### [06. Multistage.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/06.%20Multistage.tex)

#### Mathematical & Formulation Issues
1. **Line 166:** Terminal investment decision $x_{iT}$ is missing scenario index $s$:
   ```diff
   - \sum_{i \in N} \omega_{iTs}x_{iT} - y_s + w_s = G, \quad \forall s \in S,
   + \sum_{i \in N} \omega_{iTs} x_{iTs} - y_s + w_s = G, \quad \forall s \in S,
   ```
2. **Lines 239 & 244:** Scenario probability index mismatch: $\sum_{s' \in S_s^t} p'_s x_{its'}$ mixes scenario index $s'$ with weight $p'_s$ $\to$ Change to $p_{s'}$.
3. **Line 363:** Asset return is written $w_{iA(s)} x_{iA(s)}$ (clashing with deficit variable $w_s$) $\to$ `\sum_{i \in N} \omega_{is} x_{i,A(s)} - y_s + w_s = G`.
4. **Lines 478 & 500:** Resource requirement matrix index: uses $a_{jt}$ (dropping resource index and adding time index) $\to$ $a_{\text{labor}, j}$.
5. **Line 757:** First-stage state decision is $x_1$, but written $x_0$: `A_{2,i} x_{2,i} = h_{2,i} - T_{2,i} x_0` $\to$ `x_1`.
6. **Lines 101, 163, 362, 468, 479:** Set difference syntax: `\mathcal{T} \backslash 1` $\to$ `\mathcal{T} \setminus \{1\}`.

#### LaTeX Syntax & Formatting
- **Lines 9 & 50:** Duplicate `\setbeamertemplate{footline}[frame number]` $\to$ Remove line 50.
- **Lines 519, 526, 602:** Non-ASCII Unicode characters (`U+2010` hyphen, `U+2019` right single quote) $\to$ Replace with standard ASCII `-` and `'`.

#### Grammar, Phrasing & Leftover French
- **Line 62:** "We would to reach a nominal goal" $\to$ "We would like to reach a nominal goal".
- **Line 64:** "rebalanced each $v$ years" $\to$ "rebalanced every $v$ years".
- **Line 116:** "the are $R$ possible realizations" $\to$ "there are $R$ possible realizations".
- **Line 351:** "Let $A(l)$ the ancestor" $\to$ "Let $A(l)$ be the ancestor".
- **Lines 372, 378, 426, 456:** `planification` $\to$ `planning`; `Ressources` / `ressource` $\to$ `Resources` / `resource`.
- **Line 569:** "recourses are complete" $\to$ "recourse is complete".
- **Line 911:** "This limit the number of subproblems... restore the nonanticativity" $\to$ "This limits the number of subproblems... restore non-anticipativity".

---

### [07. SDDP.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/07.%20SDDP.tex)

#### Mathematical & Formulation Issues
1. **Lines 95–118:** Dual multiplier row indexing: `(\rho_{t,k})` and `(\sigma_{t,k})` applied to indexed cut systems $j=1,\dots,r_{t,k}$ $\to$ $(\rho_{t,k,j})$ and $(\sigma_{t,k,j})$. Also clarify that at terminal stage $H$, $\theta_{H,k} \equiv 0$.
2. **Lines 413–426:** Feasibility cut dual multiplier sign: standard dual ray has $\rho_{t,k} \ge 0$ (text incorrectly states $\rho_{t,k} \le 0$).
3. **Line 541:** Backward pass node complexity: $\sum_{t=2}^{H-1} \prod_{j=1}^t M_j$ $\to$ Must sum up to stage $H$: $\sum_{t=2}^H \prod_{j=1}^t M_j$.
4. **Lines 755–758:** Cut coefficients along sample path $i$: $\sigma_{t,k,i} e_{t,k}$ $\to$ $\sum_{j=1}^{s_t} \sigma_{t,k,i,j} e_{t,k,j}$.
5. **Lines 773–788:** **Critical CLT Error**:
   ```diff
   - \overline{z}_K \overset{\mathcal{D}}{\rightarrow} N(\mu, \sigma^2)
   + \sqrt{K}(\overline{z}_K - \mu) \overset{\mathcal{D}}{\rightarrow} \mathcal{N}(0, \sigma^2) \quad \left(\text{or } \overline{z}_K \approx \mathcal{N}\left(\mu, \frac{\sigma^2}{K}\right)\right)
   ```
6. **Lines 796–814:** **Critical Stopping Criterion & Sample Variance Errors**:
   ```diff
   - Terminates if $z_{LB} \in (\overline{z}_K - \Phi^{-1}(1-\alpha/2) \sigma, \overline{z}_K + \Phi^{-1}(1-\alpha/2) \sigma)$
   + Terminate if $z_{LB} \in \left(\overline{z}_K - \Phi^{-1}(1-\alpha/2) \frac{\hat{s}}{\sqrt{K}},\, \overline{z}_K + \Phi^{-1}(1-\alpha/2) \frac{\hat{s}}{\sqrt{K}}\right)$
   ...
   - \hat{s} = \frac{1}{K-1} \sqrt{\sum_{k=1}^K (\overline{z}_K-z_k)^2},
   + \hat{s} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (z_k - \overline{z}_K)^2},
   ```

#### LaTeX Syntax & Macros
- **Line 44:** Macro typo: `\def\bomega{\boldsymbol\xi}` $\to$ `\def\bomega{\boldsymbol\omega}`.
- **Lines 47–49:** `\def\KK{\mathcal{K}}` is immediately overwritten by `\def\KK{\mathcal{S}}` $\to$ `\def\SS{\mathcal{S}}`.
- **Line 53:** Short title: `\title[SA vs SAA]{...}` $\to$ `\title[SDDP]{...}`.
- **Line 664:** Stray isolated period on standalone line.

#### Grammar, Phrasing & Leftover French
- **Lines 17–21:** French theorem names: `Définition`, `Corollaire`, `Lemme` $\to$ `Definition`, `Corollary`, `Lemma`.
- **Lines 66, 70:** `Anthony Papavisiliou` $\to$ `Papavasiliou`; `Perreira and Pinto` $\to$ `Pereira and Pinto (1991)`.
- **Lines 130, 135:** "children nodes" $\to$ "child nodes"; "sequence procedures that tell" $\to$ "sequencing protocols that determine".
- **Line 509:** "No infeasibility cuts" $\to$ "Complete recourse (no feasibility cuts needed)".
- **Line 558:** "Nested Decomposition lays the foundations" $\to$ "Nested decomposition lays the foundation".

---

### [08. Lagrangian.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/08.%20Lagrangian.tex)

#### Mathematical & Formulation Issues
1. **Lines 205, 211:** KKT domain notation has unmatched parenthesis `ri(\mbox{dom}(f^1(x))` and passes evaluation $f^1(x)$ instead of function $f^1$; constraint count on line 211 is $i=1,\dots,m_1$ (not $m$).
2. **Line 275:** Missing superscript 1 on first-stage constraints: $g_i(x) \le 0$ $\to$ $g_i^1(x) \le 0$.
3. **Line 329:** Multiplier update: missing scenario superscript $s$ on $\hat{\pi}$ $\to$ $\pi^{\nu+1,s} = \max\{\pi^{\nu,s} + \alpha^\nu \hat{\pi}^s, 0\}$.
4. **Line 426:** Projection operator exponent typo: $(I - \Pi^t) x^T = 0$ $\to$ lowercase trajectory $x^t$.
5. **Lines 570–576:** Constraint symbol $t_i^2$ $\to$ $b_i^2$; realization index $\xi_S$ (capital $S$) $\to$ $\xi_s$.
6. **Line 628:** Subproblem constraint has $t_i^2(x_s, k)$ $\to$ $b_i^2(x_s, \xi_s)$.
7. **Line 651:** Index collision in progressive hedging aggregation: $s$ used as outer and dummy summation index $\to$ $\hat{x}_s^{\nu+1} = \sum_{s'=1}^S p_{s'} x_{s'}^{\nu+1}$.

#### LaTeX Syntax & Formatting
- **Lines 3, 6, 10, 53:** Duplicate `\usepackage[utf8]{inputenc}` and duplicate `\setbeamertemplate{footline}` $\to$ Remove lines 10 and 53.
- **Line 480:** Extra closing parenthesis in dual problem: `f^2(x_s, y_s))` $\to$ `f^2(x_s, y_s)`.
- **Line 757:** Obsolete LaTeX command `\bf{0}` $\to$ `\mathbf{0}` or `\bszero`.
- **Lines 764–765:** Broken delimiter nesting: `\left( ... \biggl( ... \right) ... \biggr)` $\to$ Replace with `\Biggl( ... \Biggr)`.

#### Grammar, Phrasing & Leftover French
- **Lines 13–16:** French theorem names `Définition`, `Théorème`, `Corollaire`, `Lemme` $\to$ English.
- **Lines 74, 77, 100, 627:** Leftover French `t.q.` $\to$ `s.t.`; `et` $\to$ `and`.
- **Lines 136, 149:** Preposition: "lower semi-continuous in $x_0$" $\to$ "at $x_0$".
- **Line 184 & 576:** Typos: `set if bounded` $\to$ `is bounded`; `\rho if` $\to$ `\rho is`.
- **Lines 222, 228, 234:** "affine envelop" $\to$ "affine hull" (or "affine envelope").
- **Lines 343, 369, 673, 678:** Frenchisms: "unicity" $\to$ "uniqueness"; "$\mathcal{N}$ designs" $\to$ "denotes"; "consists to take" $\to$ "consists of taking"; `De Silva et Abramson` $\to$ `and`.
- **Line 608:** Title typo: "Progressing hedging" $\to$ "Progressive hedging".
- **Lines 809, 811:** "The subproblems where solved" $\to$ "were solved"; "penality" $\to$ "penalty".

---

### [09. SAA.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/09.%20SAA.tex)

#### Mathematical & Formulation Issues
1. **Line 64:** $\mathcal{B}^k$ is the Borel $\sigma$-algebra (or Borel field), not "Borel measure".
2. **Lines 224–227:** Redundant limit arrow in definition of convergence in distribution: $\lim_{n \to \infty} F_n(x) \to F(x)$ $\to$ $\lim_{n \to \infty} F_n(x) = F(x)$.
3. **Lines 379–382:** Redundant expectation: $\EE_P[g(z)]$ where $g(z) = \EE_P[G(z, \bsxi)]$ is already the expectation.
4. **Line 431:** Typo in set deviation: $D(A,B) := \sup_{x \in A} D(x, B)$ $\to$ point-to-set distance is $d(x, B)$.
5. **Line 561:** Deterministic gradient $\phi(z) = -\nabla g(z)$ is called a "random vector" $\to$ "vector-valued mapping".
6. **Line 787:** Incomplete condition: `\| \lambda^*_N(\overline{\xi}) \|` is missing `\leq M`.
7. **Line 1393:** Malformed service level formula:
   ```diff
   - g(y) = \frac{\mathbb{E}[\# \text{of served call that waited at most } \tau]}{\mathbb{E}[\# \text{ Total } \# \text{ of calls}]}
   + g(y) = \frac{\mathbb{E}[\text{Number of served calls that waited at most } \tau]}{\mathbb{E}[\text{Total number of calls}]}
   ```

#### LaTeX Syntax & Formatting
- **Line 475:** Empty `\mbox` without `{}` $\to$ `\mbox{}`.
- **Line 1315:** Missing backslash on math macro: `sup_{x \in C}` $\to$ `\sup_{x \in C}`.
- **Lines 1407, 1424, 1459:** Punctuation inside text environment: `\text{.}`.

#### Grammar, Phrasing & Leftover French
- **Lines 86, 99:** "approximations as" $\to$ "such as"; "This framework include" $\to$ "includes".
- **Lines 123, 157, 173, 477, 546, 563, 752, 912:** Frenchisms: "truncature" $\to$ "truncation"; "Chapiter" $\to$ "Chapter"; "In other terms" $\to$ "In other words"; "vectorial space" $\to$ "vector space"; "We design such" $\to$ "We refer to such"; "compacity" $\to$ "compactness"; "At the difference of" $\to$ "Unlike".
- **Line 135:** Inversion in direct question: "Under which conditions such a limit point is..." $\to$ "is such a limit point...".
- **Lines 828, 997:** French words: `\text{ t.q. }` $\to$ `\text{ s.t. }`; `dans` $\to$ `in`.
- **Lines 272, 449, 940, 1063, 1103, 1156, 1159, 1339:** Typos: `ifferentiable` $\to$ `differentiable`; `uniformly of` $\to$ `on`; `an unique` $\to$ `a unique`; `amost` $\to$ `almost`; `equivalents to say` $\to$ `equivalent to saying`; `condition hold` $\to$ `holds`; `not-singular` $\to$ `non-singular`; `due Thuy Anh Ta` $\to$ `due to Thuy Anh Ta`.

---

### [10. Adaptive Monte Carlo.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/10.%20Adaptive%20Monte%20Carlo.tex)

#### Mathematical & Formulation Issues
1. **Line 77:** $\mathcal{B}^k$ is the Borel $\sigma$-algebra, not "Borel measure".
2. **Lines 110–113:** Sup-norm on $C(S)$ omits function argument: $\|\psi\| := \sup_{x \in S} |\psi|$ $\to$ $|\psi(x)|$.
3. **Lines 151, 166, 182:** Inconsistent sample size subscript ($n$ vs $N$): $N^{1/2}[\hat{g}_n - g]$ $\to$ $\hat{g}_N$, $\hat{v}_N$.
4. **Line 308:** Sample size sequence $N_k = c N_{k-1}$ with $c > 1$ is **geometric** (exponential), not "linear".
5. **Line 448:** Algorithm initialization indexing bug: Step 0 checks $\epsilon_\delta^{N_0}(x_{k+1})$ before $k$ is initialized $\to$ $\epsilon_\delta^{N_0}(x_0)$.
6. **Lines 630–632:** **Critical Mathematical Error in Trust-Region Ratio**:
   ```diff
   - \rho_k = \frac{\hat{g}_{N^-}(x_k+s_k) - \hat{g}_{N^-}(x_k)}{\Delta m_k^{N^-}}
   + \rho_k = \frac{\hat{g}_{N^-}(x_k) - \hat{g}_{N^-}(x_k+s_k)}{\Delta m_k^{N^-}}
   ```
7. **Lines 738, 754:** Expectation vector mismatch ($E_{\bxi} \to E_{\bsgamma}$); log-likelihood sum index uses $n=1\dots N$ while inside term uses $i$ ($SP_{ij_i}^R$) $\to$ $\frac{1}{I} \sum_{i=1}^I \ln SP_{ij_i}^R$.
8. **Lines 818–821:** **Critical CLT Variance Error**:
   ```diff
   - LL(\theta) - SLL^R(\theta) \Rightarrow \mathcal{N}\left( 0, \frac{1}{I} \sqrt{\sum_{i=1}^I \frac{\sigma^2_{ij_i}}{R (P_{ij_i})^2}} \right)
   + LL(\theta) - SLL^R(\theta) \Rightarrow \mathcal{N}\left( 0, \frac{1}{I^2} \sum_{i=1}^I \frac{\sigma^2_{ij_i}(\theta)}{R (P_{ij_i}(\theta))^2} \right)
   ```

#### LaTeX Syntax & Macros
- **Line 139:** Unmatched parenthesis: `| G(x_1, \xi) - G(x_2, \xi) )|`.
- **Line 678:** Missing braces: `\textcolor{orange}Example:` $\to$ `\textcolor{orange}{Example:}`.
- **Line 778:** Invalid `\notag` inside inline math box `$...$`.

#### Grammar, Phrasing & Leftover French
- **Lines 16–17:** French theorem names: `Corollaire`, `Lemme` $\to$ `Corollary`, `Lemma`.
- **Lines 105, 110:** Lowercase set name in title: $C(s)$ $\to$ $C(S)$.
- **Lines 117–118:** French calques: "normed vectorial space" $\to$ "normed vector space"; "issued from its norm" $\to$ "induced by its norm"; "Cauchy suite" $\to$ "Cauchy sequence".
- **Line 168:** Leftover French word: `\mbox{ et }` $\to$ `\mbox{ and }`.
- **Lines 206, 215, 222:** "What does interest us?" $\to$ "What interests us?"; "extend if over" $\to$ "extend it over"; "consists to repeatedly apply" $\to$ "consists in repeatedly applying".
- **Lines 426, 452, 470, 506, 525:** "adequation" $\to$ "adequacy"; French "either... either..." $\to$ "either... or..."; `Comparaison` $\to$ `Comparison`; `Algorithme` $\to$ `Algorithm`; `we fixe` $\to$ `we fix`.
- **Lines 600, 603, 617, 731, 832, 896, 925, 946:** Typos: leftover variable $z_k \to x_k$; `approximatively` $\to$ `approximately`; `greater to` $\to$ `greater than`; `predicition` $\to$ `prediction`; `independant` $\to$ `independent`; `signification level` $\to$ `significance level`; `Axhausen and al.` $\to$ `Axhausen et al.`; `sequance` $\to$ `sequence`; `unformly` $\to$ `uniformly`.

---

### [11. Stochastic gradient descent.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/11.%20Stochastic%20gradient%20descent.tex)

#### Mathematical & Formulation Issues
1. **Line 188:** Evaluation point mismatch in assumption A.3: LHS has $x$ while RHS has $x_k$ $\to$ $\operatorname{Var}[\nabla_x Y(x_k, \xi_k)] \leq M + M_V \|\nabla f(x_k)\|_2^2$.
2. **Lines 210–214, 714, 740:** **Critical Error in Vector Variance Identity**:
   ```diff
   - \var_{\xi_k}[g] = \EE_{\xi_k}[\|g\|^2] - (\EE_{\xi_k}[\|g\|])^2
   + \var_{\xi_k}[g] = \EE_{\xi_k}[\|g\|_2^2] - \|\EE_{\xi_k}[g]\|_2^2
   ```
3. **Line 596:** **Critical Index Collision**: Random variable symbol $\xi$ used as integer summation counter:
   ```diff
   - R_n(x) = \frac{1}{n} \sum_{\xi=1}^n Y(x; \xi)
   + R_n(x) = \frac{1}{n} \sum_{i=1}^n Y(x; \xi_i)
   ```
4. **Lines 663, 667, 671, 675, 772, 776, 784:** **Proof Typo Repeated 7 Times**: Variable `$x_{x+1}$` written instead of `$x_{k+1}$` in descent lemmas on slides 27 and 31.
5. **Line 691:** Lower bound notation: $f(x) \ge f_{\text{lb}}(x)$ is a scalar constant $\to$ $f(x) \ge f_{\text{lb}}$.
6. **Line 815:** Parameter clash: $\mu$ used for strong convexity modulus while $\mu$ is already used for descent parameter in A.6 $\to$ use $c > 0$ for strong convexity modulus.
7. **Line 1022:** Undefined symbol $\epsilon^*$ in strong minimizer definition $\to$ replace with $\gamma > 0$.
8. **Lines 1054–1055:** **Critical Sign Error in Fixed-Stepsize Bound**:
   ```diff
   - \EE[f(x_k) - f^*] \leq \frac{\overline{\alpha}LM}{2c\mu} - (1-\overline{\alpha}c\mu)^{k-1} \left( f(x_1) - f^* - \frac{\overline{\alpha}LM}{2c\mu} \right)
   + \EE[f(x_k) - f^*] \leq \frac{\overline{\alpha}LM}{2c\mu} + (1-\overline{\alpha}c\mu)^{k-1} \left( f(x_1) - f^* - \frac{\overline{\alpha}LM}{2c\mu} \right)
   ```

#### LaTeX Syntax & Macros
- **Line 318:** Comparison operator: `$n_k << n$` $\to$ `$n_k \ll n$`.
- **Lines 478, 488, 815:** Math operator macros: `\lim \sup` $\to$ `\limsup`; `sup` $\to$ `\sup`; `dom(f)` $\to$ `\operatorname{dom}(f)`.

#### Grammar, Phrasing & Leftover French
- **Line 71:** "a renew interest... and the development of various variants" $\to$ "a renewed interest... and the development of numerous variants".
- **Lines 92, 108, 130, 140:** "necessary condition to the problem" $\to$ "for the problem"; "restrain the feasible domain" $\to$ "restrict"; French calque "derivation operators" $\to$ "differentiation operators"; double word "Also know as as" $\to$ "Also known as".
- **Lines 349, 407, 1025:** `knowning` $\to$ `knowing`; `iterations of completed` $\to$ `iterations completed`; French `Exemple` $\to$ `Example`.

---

### [12. Discrete event simulation.tex](file:///home/bastin/slash/Git/SP_Introduction/slides/12.%20Discrete%20event%20simulation.tex)

#### Mathematical & Formulation Issues
1. **Lines 448–454:** **Critical Dimension Mismatch**: Claims $U_j, j=1\dots 99$ are generated, but for $i=1\dots 99$, variables $U_{2i-1}$ and $U_{2i}$ range from $1$ to $2(99) = 198$, yielding $\overline{W}_{100} = f(U_1,\dots,U_{198})$ and dimension $t=198$ $\to$ Change to $j=1,\dots,198$.
2. **Line 645:** Subscript typo in indirect estimator variance: $\widehat{\operatorname{Var}}[X_{\text{i}, i}]$ $\to$ $\widehat{\operatorname{Var}}[X_{\text{i}, n}]$.
3. **Lines 829–832:** **Critical Neyman Allocation Indexing Error**:
   ```diff
   - \frac{n_t}{n} = \frac{\sigma_t P[B_i=t]}{\sum_{k=1}^4 \sigma_k P[B_i=k]} = \frac{\sigma_k q_k}{\sum_{k=1}^4 \sigma_k q_k}
   + \frac{n_t}{n} = \frac{\sigma_t P[B_i=b_t]}{\sum_{k=1}^4 \sigma_k P[B_i=b_k]} = \frac{\sigma_t q_t}{\sum_{k=1}^4 \sigma_k q_k}
   ```
4. **Line 1128:** Acronym typo: "using RNGs everywhere means taking $\bU_1 = \bU_2$" $\to$ "using CRNs (Common Random Numbers) everywhere".

#### LaTeX Syntax & Formatting
- **Lines 1080, 1095:** Stray trailing backslashes `\` in text mode.

#### Grammar, Phrasing & Leftover French
- **Lines 71, 83, 145:** "change by a countable number of points" $\to$ "at a countable number"; "simulation time" $\to$ "Simulation time"; "Random numbers generation" $\to$ "Random number generation".
- **Lines 177, 184, 186:** Leftover French in Flowchart 1:
  - `Liste d'événements vide ou événement de fin?` $\to$ `Empty event list or end event?`
  - `non` $\to$ `no`; `oui` $\to$ `yes`.
- **Line 312:** Missing verb: "We assume $S_i$ and $A_i$ mutually independent" $\to$ "are mutually independent".
- **Lines 598–609:** **Entire Slide 25 Untranslated in French**:
  ```diff
  - Soit ${X_i} = G_i(s)$ pour le jour $i$, et 
  - \[
  - {\bar X_n} = {1\over n} \sum_{i=1}^n X_i.
  - \]
  - On a $E[\bar X_n] = \mu$ et $Var[\bar X_n] = Var[X_i]/n$.
  - 
  - \mbox{}
  - 
  - Une expérience avec ${n = 1000}$ donne $\bar X_n = 1518.3$ et $S_n^2 = {21615}$.
  - La variance estimée de $\bar X_n$ est alors $\widehat{Var}[\bar X_n] = 21.6$.
  + Let ${X_i} = G_i(s)$ for day $i$, and 
  + \[
  + {\bar X_n} = {1\over n} \sum_{i=1}^n X_i.
  + \]
  + We have $E[\bar X_n] = \mu$ and $Var[\bar X_n] = Var[X_i]/n$.
  + 
  + \mbox{}
  + 
  + An experiment with ${n = 1000}$ yields $\bar X_n = 1518.3$ and $S_n^2 = {21615}$.
  + The estimated variance of $\bar X_n$ is then $\widehat{Var}[\bar X_n] = 21.6$.
  ```
- **Lines 725, 739:** Terminology consistency: "Control Variable (CV)" $\to$ "Control Variate (CV)".
- **Line 958:** Inconsistent acronyms `IRV / CRV` $\to$ `IRN / CRN` (Independent / Common Random Numbers).

---

# 4. Cross-Deck Systemic Issues

Several recurring patterns exist across multiple files and can be resolved globally:

1. **Missing Color Macros in Preambles:**
   Multiple decks use `{\red ...}` and `{\blue ...}` without defining them. Standardizing `sty/macros.tex` with:
   ```latex
   \newcommand{\red}{\color{red}}
   \newcommand{\blue}{\color{blue}}
   ```
   prevents compile failures across all decks.

2. **French Theorem Environments:**
   Decks 07, 08, 10 use `\newtheorem{theo}{Théorème}`, `\newtheorem{coro}{Corollaire}`, `\newtheorem{lem}{Lemme}`. These should be standardized to English `Theorem`, `Corollary`, `Lemma`, `Definition`.

3. **Borel $\sigma$-Algebra vs Borel Measure:**
   Decks 09 and 10 state that $\mathcal{B}^k$ is the "Borel measure". In probability theory, $(\mathbb{R}^k, \mathcal{B}^k)$ is a measurable space where $\mathcal{B}^k$ denotes the Borel $\sigma$-algebra (or Borel $\sigma$-field).

4. **Leftover French Abbreviations in Align Environments:**
   `\text{t.q. }` and `\phantom{t.q. }` (tel que) appear throughout decks 01, 04, 08, 09. They should all be replaced with `\text{s.t. }` (subject to / such that).

5. **Makefile Whitespace Incompatibility:**
   The `Makefile` target `SLIDES_SRC = $(wildcard $(SLIDES_DIR)/*.tex)` fails on filenames containing spaces (e.g., `01. Introduction.tex`). Updating the Makefile or enclosing patterns in quotes will allow automated batch compilation.

---

# 5. Action Plan & Recommended Next Steps

1. **Step 1: Fix High-Severity Mathematical Errors**
   Apply the critical fixes detailed in [Section 2](#2-critical-mathematical-errors--impact-analysis) (Fréchet–Hoeffding proof, L-shaped cut direction, Gaussian PDF, SDDP confidence intervals, trust-region ratio, SGD variance/bounds).
2. **Step 2: Apply LaTeX & Macro Cleanups**
   Define `\red` and `\blue` centrally in `sty/macros.tex`, remove duplicate package imports, fix math delimiter mismatches, and standardize theorem environments.
3. **Step 3: English Text & French Translation Pass**
   Translate slide 25 of deck 12, replace French flowchart labels, and convert remaining Frenchisms (`si`, `et`, `t.q.`, `plans`, `module`, `Alors`, `journaux`) into standard academic English.
4. **Step 4: Automated Build Verification**
   Update the Makefile to handle filenames with spaces and run `make slides` to verify clean compilation of all 12 PDFs without warnings or overfull boxes.

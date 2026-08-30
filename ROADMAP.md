# Roadmap

## Slides — `slides/02. Random Numbers.tex`

### Diapo « RandomDataStreams.jl »

- [x] Remplacer le pseudo-code illustratif (`u = draw(stream = s, substream = r, position = i)`) par l'API réelle du package, une fois la librairie enregistrée (dans les prochains jours).
- [ ] Ajuster la description de la diapo si les fonctionnalités évoluent (streams / substreams MRG32k3a).
- [ ] Recompiler et vérifier la mise en page après mise à jour (`make slides` une fois le Makefile corrigé, sinon `pdflatex` direct).

Liens :
- Librairie : Officiellement enregistrée (ajoutable via `] add RandomDataStreams`)
- Documentation : <https://jlchartrand.github.io/RandomDataStreams.jl/dev/>

---

## Backlog — corrections de l'audit (`slides_audit_report.md`)

### Priorité 1 — Erreurs mathématiques critiques (section 2 de l'audit)

- [x] **Deck 02** — Preuve borne supérieure de Fréchet–Hoeffding : `P[A∩B] ≤ P[A]P[B]` faux en général → inclusion monotone. *(corrigé, vérifié à la compilation)*
- [x] **Deck 04** — Signe de la coupe de faisabilité (multicut, ~l.1384) : `σᵀ(h(ξₖ) − T(ξₖ)x) ≥ 0` → `≤ 0` (toutes les autres occurrences du deck étaient déjà correctes).
- [x] **Deck 05** — Densité gaussienne multivariée (~l.368) : `(2π)^{n/2}` mal placé et `Σ` → `Σ⁻¹` dans l'exposant.
- [x] **Deck 07** — TCL (~l.784) : `√K(z̄_K − μ) → N(0, σ²)` (avec `z̄_K ≈ N(μ, σ²/K)`) ; intervalle de confiance (~l.802) : facteur `ŝ/√K` ; formule de `ŝ` (~l.809) : `1/(K−1)` sous la racine ; typo « set but the user » → « set by the user ».
- [x] **Deck 10** — Ratio trust-region (~l.633) : numérateur inversé → `ĝ(xₖ) − ĝ(xₖ+sₖ)` (décroissance réelle, cohérent avec Δm = m(xₖ) − m(xₖ+sₖ) défini l.472).
- [x] **Deck 11** — Identité de variance vectorielle (~l.213, 714, 740) : `E‖g‖² − ‖E[g]‖²` (et non `(E‖g‖)²`) ; la dérivation l.740-744 reste valide via ‖E[g]‖ ≤ μ_G‖∇f‖.
- [x] **Deck 11** — Signe de la borne SGD à pas fixe (~l.1054) : `−` → `+`, cohérent avec la preuve l.1096-1103.

**P1 terminé** : les 7 erreurs critiques de l'audit sont corrigées ; decks 04, 05, 07, 10, 11 recompilés sans erreur.

### Priorité 2 — Infrastructure

- [x] **Makefile** — reconstruit ; `make slides`, `make background`, `make all` et les cibles individuelles (`make "pdf/07. SDDP.pdf"`) fonctionnent :
  - boucle shell sur les sources (les espaces dans les noms cassaient le wildcard) ;
  - compilation avec `-jobname` sans espaces (incompatibilité de `ifplatform`/`auto-pst-pdf` avec un `\jobname` contenant des espaces) ;
  - nouvelle cible `make graphs` : pré-conversion des 15 EPS d'`imgs/` en `-eps-converted-to.pdf` (l'`epstopdf` local ne résout pas TEXINPUTS et exige `--outfile` explicite).
- [x] Environnements de théorèmes en français → anglais — decks **07, 08, 10 et 11** (le deck 11 avait été raté par l'audit).
- [x] **Deck 10** — suppression des dessins PSTricks incompatibles avec pdflatex : `\psframebox`/`\psshadowbox` → `\colorbox`, arbre `pst-tree` (modes de transport) → TikZ ; correction au passage de `\textcolor{orange}Example` sans accolades (~l.678).

Vérifié : les 12 slides + 3 notes background compilent sans erreur (15 PDF régénérés).

### Priorité 3 — Langue et typos (~120 issues secondaires)

- [x] **Deck 12** — Diapo 25 entièrement en français (`~l.598-609`) + libellés du flowchart (`Liste d'événements…`, `oui`/`non`).
- [ ] Gallicismes récurrents tous decks : `t.q.` → `s.t.`, `si` → `if`, `et` → `and`, `plans` → `planes`, `module` → `modulus`, `Alors`, `journaux`, `Exercice`, `planification`, etc.
- [ ] Typos mathématiques secondaires : index et dimensions (deck 03 `y_{1s}`→`y_{2s}`, deck 06 `x_{iT}`/`p'_s`, deck 08 `t_i²`→`b_i²`, deck 12 allocation de Neyman), `x_{x+1}` ×7 (deck 11), etc.
- [ ] Syntaxe LaTeX : délimiteurs appariés, parenthèses en trop, macros dupliquées (`inputenc`, `\bu`, `\KK`).

> Note : l'item « macros `\red`/`\blue` manquantes » de l'audit est **réfuté** — `pstricks` les définit ; aucune action requise.

Référence complète : [`slides_audit_report.md`](slides_audit_report.md) (vérifié ligne à ligne en session : 7/7 erreurs critiques confirmées, ~60 revendications secondaires échantillonnées toutes confirmées).

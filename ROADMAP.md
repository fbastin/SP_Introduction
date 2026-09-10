# Roadmap

## Slides — `slides/02. Random Numbers.tex`

### Diapo « RandomDataStreams.jl »

- [x] Remplacer le pseudo-code illustratif (`u = draw(stream = s, substream = r, position = i)`) par l'API réelle du package, une fois la librairie enregistrée (dans les prochains jours).
- [x] Ajuster la description de la diapo si les fonctionnalités évoluent (streams / substreams MRG32k3a). API vérifiée contre la v0.1.0 : la diapo utilise bien `next_stream!` et `next_substream!` (`next_stream` sans `!` est déprécié et n'existe que pour `MRG32k3aGen`).
- [x] Recompiler et vérifier la mise en page après mise à jour.

Liens :
- Librairie : Officiellement enregistrée (ajoutable via `] add RandomDataStreams`)
- Documentation : <https://jlchartrand.github.io/RandomDataStreams.jl/dev/>

### Révision complète du deck

- [x] **Vérification et corrections effectuées** :
  - **Valeurs numériques contrôlées par calcul** : $m_1 = 2^{32}-209$, $m_2 = 2^{32}-22853$, les six coefficients $a_{i,j}$, le produit $m_1 m_2 = 18446645023178547541$ et surtout les trois coefficients du MRG équivalent, reconstruits indépendamment par restes chinois ($a_j \equiv a_{1,j} \bmod m_1$, $a_j \equiv a_{2,j} \bmod m_2$) — les trois entiers de 20 chiffres sont exacts. Le tableau LFSR ($k=4$, $x_n = x_{n-1} \oplus x_{n-4}$) et sa période 15 ont été validés par simulation ; les constantes Philox ($M_0$, $M_1$, $W_0$, $W_1$) et le câblage du round correspondent à l'implémentation de référence Random123 et à celle de `RandomDataStreams.jl` (`src/philox/philox.jl`).
  - **Erreur sur l'espacement de Tausworthe** : la diapo annonçait $\nu = 1$ alors que le découpage dessiné et les cinq valeurs $7/8, 5/8, 3/8, 1/8, 0$ correspondent à $\nu = 3$ (vérifié sur le flux réel `111101011001000` ; $\nu = 1$ donnerait $7/8, 7/8, 3/4, 5/8, \ldots$). Corrigé en $\nu = 3$ et transformé en illustration de la condition de coprimalité : $\gcd(3, 2^4-1) = 3$, d'où l'effondrement de la période de sortie à $15/3 = 5$, ce qui motive la condition $\gcd(\nu, 2^k-1) = 1$ de la diapo suivante.
  - **Copule gaussienne** : $\Phi_\Sigma$ renommé $\Phi_R$ avec mention explicite que $R$ doit être une matrice de *corrélation* (diagonale unitaire) et non une covariance quelconque — sinon les marges ne sont pas $\mathcal{N}(0,1)$ et $C$ n'est pas une copule. Lève aussi la collision de notation avec le $\Sigma$ (vraie covariance) de la diapo « Multivariate distributions: simple cases ».
  - **Période maximale des LCG** : l'énoncé « $m-1$ si $c = 0$ » n'est valable que pour $m$ premier ; précision ajoutée (pour $m = 2^k$ la période maximale tombe à $m/4$).
  - **Synchronisation avec `code/random.ipynb`** : le code `getlcg` de la diapo reprend maintenant la version du notebook (`Ref{typeof(seed)}` et `invm` au lieu de `state = seed` et `am_mil`), et la légende de `lcg.png` est corrigée — la figure montre les 5000 paires successives $(u_{2i-1}, u_{2i})$ issues de 10 000 nombres, et non « 10000 generated points ».
  - **Précisions** : adoption de MRG32k3a (disponible dans R sous `L'Ecuyer-CMRG`, défaut du paquet `parallel`, et `mrg32k3a` de MATLAB) au lieu du vague « used in R » ; affirmation « monotonicity is strictly required by CRN or antithetic variates » nuancée (la monotonie garantit le signe de la corrélation, elle n'est pas une condition d'existence) ; note sur Box-Muller rappelant que $U_1$ doit être dans $(0,1]$ alors que `rand()` renvoie dans $[0,1)$.
  - **Mise en page Beamer** : résorption des deux dépassements verticaux (`Overfull \vbox`) — la diapo MRG32k3a (25.16pt) est scindée en deux (« MRG32k3a » et « MRG32k3a: the equivalent MRG », les paramètres passant en tableau aligné), et les espacements de la diapo `RandomDataStreams.jl` (1.89pt) sont resserrés.
  - **Compilation** : vérifiée avec `pdflatex` (58 pages, 0 erreur, 0 `Overfull \vbox`, 0 `Overfull \hbox`), PDF synchronisé dans `pdf/02. Random Numbers.pdf`.

---

## Slides — `slides/04. L-Shaped method.tex`

- [x] **Révision complète effectuée** :
  - **Précision mathématique** : reformulation épigraphe du problème maître ($\min_{x, \theta} c^T x + \theta$ sous $\theta \ge f(x)$ au lieu d'un mélange erroné objectif/contrainte), dimensions $n+pS$ variables et $m_1+m_2 S$ contraintes, définition rigoureuse des ensembles réalisables $\mathcal{B}_0$ et $\mathcal{B}_1$ dans $\mathbb{R}^n_+ \times \mathbb{R}$ / $\mathbb{R}^{|S|}$, cohérence des variables d'itération et des points d'évaluation des sous-gradients ($x^\nu, \hat{x}$), coupures de faisabilité et d'optimalité (dimensions et vecteurs $D_\ell x \ge d_\ell$), cohérence multicut ($\theta_s$ bornant $p_s Q(x, \xi_s)$), problème maître des régularisations unifié en $\min_{x, \theta}$.
  - **Mise en page Beamer** : résorption de tous les avertissements de dépassement vertical (Overfull `\vbox`) sur les diapos 11, 23, 24, 27 et 39 grâce à l'ajustement de `\small`, des espacements verticaux et des formules hors-texte.
  - **Typographie et LaTeX** : `\overset{\mathrm{def}}{=}`, `\texorpdfstring` pour le titre PDF, environnement `theorem` Beamer standard, suppression des paquetages inutilisés (`listings`, `mathlist` désactivé évitant la dépendance manquante `eclbkbox.sty`).
  - **Anglais et syntaxe** : élimination systématique des calques (« the same than » $\to$ « the same as », « in other terms » $\to$ « in other words », « support hyperplan » $\to$ « supporting hyperplane », « unfeasible » $\to$ « infeasible »), index de scénario « $s = 1 \in S$ » corrigé en « $s = 1, \ldots, S$ », grappes dupliquées « $\mathcal{S}_1, \mathcal{S}_1$ » corrigées en « $\mathcal{S}_1, \mathcal{S}_2$ », coquille « Opimization » $\to$ « Optimization ».
  - **Compilation** : vérifiée avec `pdflatex` (87 pages sans erreur ni warning de dépassement), PDF synchronisé dans `pdf/04. L-Shaped method.pdf`.

---

## Slides — `slides/05. Chance-constrained programming.tex`

- [x] **Révision complète effectuée** :
  - **Précision mathématique** :
    - Élimination de la contradiction dans les conditions KKT du cas de vente à découvert (*short selling*) : retrait de la contrainte erronée $x \ge 0$ incompatible avec les positions courtes, et adoption de la notation de gradient multivarié $\nabla_x L(x, \lambda) = 0$ au lieu d'une dérivée scalaire $\frac{dL}{dx}$.
    - Correction majeure de la relation quantile/probabilité pour la VaR : clarification de $P[Z \le \beta^*] = \eta \iff \beta^* = \text{VaR}(Z; \eta)$ (l'ancienne version écrivait $\eta = \text{VaR}(Z; \eta)$, confondant probabilité et quantile).
    - Harmonisation de la fonction indicatrice dans la généralisation SAA : formulation rigoureuse via le maximum des contraintes $G(x, \boldsymbol{\xi}) = \max_j g_j(x, \boldsymbol{\xi})$ avec $\mathbb{I}_{(-\infty, 0]}(G(x, \boldsymbol{\xi})) \ge \alpha$ (indicateur de faisabilité) et $\mathcal{I}_{(0, \infty)}(G(x, \boldsymbol{\xi})) \le 1-\alpha$ (indicateur de violation), éliminant une notation contradictoire.
    - Correction de la terminologie : suppression de la mention trompeuse « (or integrated) » pour les contraintes jointes (l'appellation *integrated chance constraints* étant réservée aux espérances de dépassement $\mathbb{E}[\max_i (g_i(x, \boldsymbol{\xi}))_+] \le d$ présentées en fin de deck).
    - Remplacement des intersections brutes dans les probabilités par des délimiteurs virgules : $P[\xi_1 x_1 + x_2 \ge 7, \, \xi_2 x_1 + x_2 \ge 4] \ge \alpha$.
    - Précision sur les ensembles convexes pour la loi uniforme dans le cadre du théorème de Prékopa : support $S$ explicitement convexe avec densité $1/\operatorname{vol}(S)$.
    - Harmonisation des bornes d'index $i=1,\ldots,m$ (au lieu de $i,\ldots,m$) et des notations vectorielles ($\boldsymbol{x}, \boldsymbol{\xi}$, variables aléatoires $Z$, budget $W$ distinct du vecteur $x$).
  - **Mise en page Beamer** :
    - Résorption complète des 3 dépassements verticaux (Overfull `\vbox`) :
      - Diapo 20 (« Common log-concave distributions », 7.64pt) : compactage des densités usuelles et `\small`.
      - Diapo 27 (« Other solvable cases », 2.97pt) : remplacement des `\mbox{}` parasites par `\smallskip`, notation `\operatorname{diag}` et clarification $\alpha > 0.5$.
      - Diapo 31 (« Robust portfolio optimization », 10.00pt) : épuration des espacements `\mbox{}` et intégration harmonieuse du programme.
  - **Typographie et LaTeX** :
    - Suppression du double titre de diapo `\frametitle{Numerical illustration}` sur la diapo de généralisation.
    - Nettoyage du préambule : élimination du chargement dupliqué de `inputenc`, suppression des packages `listings` C++ inutilisés, correction de la macro `\bomega` ($\boldsymbol{\omega}$ au lieu de $\boldsymbol{\xi}$).
    - Titre Beamer mis à jour : `[Chance-constrained programming]` avec trait d'union au lieu du raccourci `[CP]`.
    - Correction de parenthèse non appariée dans le tableau numérique : `$P[\beta^T x \le 0])$` $\to$ `$P[\beta^T x \le 0]$`.
    - Suppression de tous les avertissements PDF Unicode d'hyperref sur les métadonnées auteur via `\texorpdfstring`.
  - **Compilation** : vérifiée avec MiKTeX `pdflatex` (45 pages, 0 erreur, 0 avertissement overfull `\vbox`/`\hbox`), PDF synchronisé dans `pdf/05. Chance-constrained programming.pdf`.

---

## Slides — `slides/07. SDDP.tex`

- [x] **Révision complète effectuée** :
  - **Figures invisibles (défaut principal)** : les dix figures PSTricks/`pst-tree` du deck n'apparaissaient dans *aucune* page du PDF livré. Deux causes cumulées : `auto-pst-pdf` construit sa compilation auxiliaire à partir de `\jobname` et échoue silencieusement quand le nom du fichier contient une espace (« Could not create 07.SDDP-pics.pdf », suivi de « pspicture No. 1 undefined » pour chaque figure) ; et, même sans espace, Ghostscript 10.02 avorte sur le PostScript produit (« Error: /typecheck in --div-- »), ne produisant qu'une page de figures. Les arbres de scénarios, le schéma de communication `NLSD` et les graphes recombinés ont donc été **redessinés en TikZ** (même parti pris que pour le deck 10), avec une macro `\nlsdtree` paramétrée par le style et le contenu des nœuds de chaque étage, qui produit les cinq diapos d'illustration à partir d'une seule définition. Plus de `pstricks`, de `pst-tree`, d'`auto-pst-pdf` ni de `--shell-escape` : `make pdf/"07. SDDP.pdf"` suffit.
  - **Précision mathématique** :
    - Nouvelle diapo « Problem setting » : le deck attaquait directement le sous-problème $NLSD(t,k)$ sans jamais énoncer le programme multi-étapes ni la récursion de programmation dynamique qu'il décompose. Ajout du programme en notation de Birge–Louveaux ($W_t$, $h_t$, $T_{t-1}$, horizon noté $H$ puisque $T$ désigne la matrice de technologie) et des fonctions $V_t(x_{t-1}, \xi_{[t]})$ et $\QQ_{t+1}(x_t, \xi_{[t]}) = \mathbb{E}[V_{t+1} \mid \xi_{[t]}]$, avec mention explicite que le conditionnement disparaît sous indépendance inter-étapes — ce qui prépare le partage de coupes introduit plus loin.
    - Nombre de nœuds contre nombre d'aléas : la macro `\KK` était définie deux fois (`\mathcal{K}` puis `\mathcal{S}`) et servait à désigner « the number of distinct scenarios at stage $t$ », c'est-à-dire un entier noté par un symbole d'ensemble. Introduction de $N_t$ (nombre de nœuds à l'étape $t$), distinct de $M_t$ (nombre de réalisations par étape), et ajout de la relation $N_t = \prod_{j=1}^t M_j$ sur la diapo de non-scalabilité, dont elle justifie directement les décomptes. Les diapos d'illustration parlaient de $k_2$ et $k_3$, harmonisés en $N_2$, $N_3$.
    - Décompte de la passe arrière corrigé : $\sum_{t=2}^{H-1} \prod_{j=1}^t M_j$ → $\sum_{t=2}^{H} \prod_{j=1}^t M_j$ (les sous-problèmes de la dernière étape sont bien résolus).
    - Nouvelle diapo « Optimality cuts: interpretation » : par dualité linéaire, $Q_{t,k}(\hat{x}_{t-1}) = \pi_{t,k}^T(h_{t,k} - T_{t-1,k}\hat{x}_{t-1}) + \rho_{t,k}^Td_{t,k} + \sigma_{t,k}^Te_{t,k}$, d'où $e - E\hat{x}_{t-1} = \QQ_t(\hat{x}_{t-1})$ et $-E \in \partial \QQ_t(\hat{x}_{t-1})$ : la coupe est l'hyperplan d'appui de la fonction de recours au point d'essai, valide partout et serré en $\hat{x}_{t-1}$. Le deck donnait les formules de $E$ et $e$ sans jamais dire ce qu'elles représentent.
    - Diapo « Optimality cuts » : l'énoncé « Solve $NLSD(t,k)$ for $j = 1,\ldots,\KK_{t-1}$ » mélangeait les indices $j$ et $k$ ; reformulé en « pour un nœud $j$ de l'étape $t-1$, résoudre $NLSD(t,k)$ pour tout descendant $k \in \DD_t(j)$ », et ajout de la coupe effectivement ajoutée, $E x_{t-1,j} + \theta_{t-1,j} \ge e$, absente de la diapo.
    - Passe arrière SDDP : la formule de $e_{t-1}$ omettait le terme de coupes de faisabilité $\rho^Td$ présent dans la formule générale trois diapos plus haut, et traitait $\sigma$ comme un scalaire ; harmonisée avec la diapo « Optimality cuts ».
    - Borne supérieure probabiliste : coût du scénario écrit $z_k = \sum_{t=1}^H c_k^Tx_k$ (indices d'étape manquants) → $z_i = \sum_{t=1}^H c_{t,i}^Tx_{t,i}$ ; horizon noté $T$ dans « solve $NLSD(t,k)$ for $t = 1,\ldots,T$ » → $H$ ; « for $t = 0$, the problem is deterministic » → $t = 1$ (il n'y a pas d'étape 0 dans le deck) ; indice d'échantillon $k$ renommé $i$ pour ne plus entrer en collision avec l'indice de réalisation ; ajout de la mise en garde correspondante : c'est l'espérance $\mu$ du coût de la politique courante qui majore la valeur optimale, $\overline{z}_K$ n'en est qu'une estimation et peut tomber sous l'optimum.
    - Critère d'arrêt : la taille d'échantillon était obtenue en posant « $\Phi^{-1}(1-\alpha/2) \approx 0.01 \overline{z}_K$ », qui compare un quantile à un coût ; rétabli en $\Phi^{-1}(1-\alpha/2)\hat{s}/\sqrt{K} \approx 0.01 \overline{z}_K$, cohérent avec la formule de $K$ donnée juste après. Ajout de deux remarques : le test est en pratique unilatéral ($z_{LB}$ minore), et il est optimiste — un $K$ faible donne un intervalle large dans lequel la borne inférieure entre bien avant que la politique ne soit optimale (comportement reproduit sur l'exemple numérique : arrêt à l'itération 3 avec 1,47 % d'écart réel).
    - Justification ajoutée du statut de borne inférieure de $NLSD(1,1)$ : les coupes sont une approximation extérieure de la fonction de recours, donc le problème de première étape est une relaxation.
    - Coquilles : « Pereira and Pinto (1991), 1991 », `NLDS(1,1)` pour `NLSD(1,1)`, « independant », « unfeasible ».
  - **Mise en page Beamer** : sur douze diapos la dernière ligne de texte venait chevaucher les symboles de navigation du coin inférieur droit (visible par exemple sur « Feasibility cuts »). Compactage ciblé (`\small` sur les listes de définitions, fusion de deux puces sur la diapo `NLSD(t,k)`, définitions de $D_{t-1,a(k)}$ et $d_{t-1,a(k)}$ passées en ligne, figures des diapos « non-scalable » et « Numerical illustration » réduites) : plus aucun chevauchement, et 0 avertissement `Overfull \vbox`/`\hbox` sur les 30 pages.
  - **Typographie et LaTeX** :
    - Titre court `[SA vs SAA]` (copié d'un autre deck) → `[SDDP]` ; suppression des avertissements Unicode d'hyperref sur le titre et l'auteur via `\texorpdfstring`.
    - Titres de diapos en casse de phrase (« The Nested L-Shaped Decomposition Subproblem », « Enumerating Versus Simulating », « Making Nested Decomposition Scalable ») ; les cinq diapos « Illustration » homonymes deviennent « Illustration: forward/backward pass, stage $t$ ».
    - Diapo « SDDP algorithm » : la liste de quatre puces vagues devient un pseudo-code `algpseudocode` avec passes avant et arrière, test d'arrêt et limite d'itérations — les paquets `algorithmicx`/`algpseudocode` étaient chargés sans être utilisés.
    - Préambule nettoyé : suppression de `pstricks`, `pst-tree`, `auto-pst-pdf`, `ulem`, `epstopdf`, des quatre `\newtheorem` inutilisés et des macros mortes (`\bxi`, `\bepsilon`, `\bomega`, `\aff`, `\tim`, `\KK` dupliqué) ; ajout de `tikz` et `graphicx`.
  - **Nouveau contenu** : deux diapos d'illustration numérique (convergence des bornes, et coupes contre fonction de recours exacte) produites par le notebook `code/SDDP_from_scratch.ipynb` — figures `imgs/sddp_convergence.pdf` et `imgs/sddp_value_function.pdf` — et une diapo de références (Pereira et Pinto 1991, Shapiro 2011, Birge et Louveaux 2011 ch. 7, Dowson et Kapelevich 2021, Papavasiliou).
  - **Compilation** : vérifiée avec `pdflatex` via `make` (30 pages, 0 erreur, 0 `Overfull \vbox`, 0 `Overfull \hbox`), les dix figures TikZ contrôlées page à page, PDF synchronisé dans `pdf/07. SDDP.pdf`.

---

## Notebook — `code/random.ipynb`

- [x] **Révision complète effectuée** (corrections vérifiées en exécutant le notebook avec Julia 1.12) :
  - **API `RandomDataStreams`** : remplacement de `next_stream` par `next_stream!` dans la section MRG32k3a (cellules « mrg_1 » et « mrg_2 »). `next_stream` est déprécié depuis la v0.1.0 (`Warning: next_stream(rng_gen::MRG32k3aGen) is deprecated`) et n'existe que pour `MRG32k3aGen`, ce qui rendait la section MRG incohérente avec la section Xoshiro qui utilisait déjà la forme avec `!`.
  - **Mesures de performance faussées** : `N = Normal()` était un global non `const`, si bien que `@btime` mesurait surtout l'instabilité de type et non le coût de l'inversion (120 ns / 7 allocations contre 41 ns / 0 allocation avec `const N`). Ajout de `const`, remplacement de `@btime X, Y = f()` par `@btime f()` (l'affectation à des globaux entrait dans la mesure), et ajout d'une cellule de conclusion : inversion et Box-Muller sont de coût comparable (≈ 37–41 ns pour deux variables normales), tous deux environ un ordre de grandeur plus lents que `randn()` (ziggurat, ≈ 3 ns) ; l'inversion reste préférable en simulation car monotone en $U$ et consommant exactement un uniforme par variable (variables aléatoires communes, quasi-Monte Carlo).
  - **Section TestU01 rendue exploitable** : `RNGTest.smallcrushJulia` n'imprime rien (`swrite_Basic` est mis à 0 par le package) — elle *retourne* les p-valeurs, ce qui rendait les deux cellules de test muettes. Ajout d'une fonction `report_smallcrush` qui étiquette les dix tests de SmallCrush et signale les p-valeurs hors de $[10^{-3}, 1-10^{-3}]$, et mention de `smallcrushTestU01` comme variante imprimant le rapport TestU01 standard. Résultats obtenus : le générateur minimal standard échoue à `BirthdaySpacings`, `Collision` et `MaxOft` (p-valeurs nulles), alors que MRG32k3a passe la batterie complète ; commentaires ajoutés pour relier ces échecs à la période $2^{31}-2$ et à la structure en réseau visible sur les nuages de points.
  - **Erreurs factuelles dans le texte** : la description du générateur par défaut de Julia confondait `Random.TaskLocalRNG` avec le `Xoshiro256ppGen` de `RandomDataStreams` (objets distincts, même famille) ; `show` imprime la graine du *prochain* flux et non « la graine courante » ; le commentaire « returns a Float64 in [0, 1) » ne correspondait pas au LCG utilisé, qui produit dans $(0,1)$ ($c = 0$, $m$ premier, l'état n'atteint jamais 0).
  - **Robustesse et lisibilité du code** : Box-Muller protégé contre `U[1] == 0.0` (`rand()` renvoie dans $[0,1)$, donc $\log 0 = -\infty$ est atteignable) via `1 - U[1]` ; `am_mil` renommé `invm` ; `α = quantile(N, 0.975)` renommé `z975` (α désigne le niveau de confiance ailleurs dans le cours) ; les deux appels `Pkg.add` dispersés regroupés en une cellule d'installation unique ; note sur le débordement entier ($a(m-1)+c$ doit rester sous `typemax(Int64)`).
  - **Anglais et titres** : « Random numbers generation » → « Random number generation », coquille « avaible » → « available », reformulation de l'introduction et de la phrase « equivalent to compute the quantile ».
  - **Validation** : notebook réexécuté de bout en bout (hors cellules `Pkg.add` et tracés `Plots`, inchangés) sans erreur ni avertissement de dépréciation.
  - **Nouvelle section « Why is `randn()` so fast? »** : explication du ziggurat (Marsaglia et Tsang, 2000) tel qu'implémenté dans `Random/src/normal.jl`, avec trois cellules de mesure — taux d'acceptation du chemin rapide, décomposition des coûts en boucle serrée, et comptage des uniformes consommés par variable via une réimplémentation de `randn` vérifiée identique à `Random.randn`. Suivie d'une section « Limitations » couvrant l'absence d'application fixe uniforme → variable (incompatibilité CRN / antithétiques / QMC), la non-inversibilité, le caractère spécifique des tables à chaque loi (`ki, wi, fi` pour la normale, `ke, we, fe` pour l'exponentielle), la dépendance du flux à l'algorithme et non à la seule graine, et la réutilisation des bits de poids faible (l'indice de couche est `rabs & 0xFF`, bits qui servent aussi à la position dans la couche).
  - Reformulation de la cellule de conclusion des benchmarks : l'écart « ordre de grandeur » vaut pour les `@btime` sur appel isolé ; en boucle serrée le rapport est plus proche de 5, ce que la nouvelle section mesure et explique.
  - **Installation d'une branche git** : nouvelle sous-section expliquant `Pkg.add(name = ..., rev = ...)` (branche, tag ou SHA), la forme `url = ... , rev = ...`, le retour au registre par `Pkg.free`, et le raccourci REPL `] add RandomDataStreams#Philox`. Deux pièges documentés parce que rencontrés à l'exécution : la forme `#` est réservée au REPL (`Pkg.add("RandomDataStreams#Philox")` échoue avec *is not a valid package name*), et libgit2 réclame des identifiants SSH en session non interactive, d'où `ENV["JULIA_PKG_USE_CLI_GIT"] = "true"` dans la cellule d'installation.
  - **Passage à la branche `Philox` (v0.2.0)**, appelée à être enregistrée fin automne 2026. Vérifié que toutes les cellules MRG32k3a et Xoshiro existantes fonctionnent inchangées sur cette version.
  - **Nouvelle sous-section « Counter-based generators »** : Philox4x32-10, Philox4x64-10, Threefry4x32/4x64, PCG64 et PCG64-DXSM, tous exercés via la même interface flux/sous-flux que MRG32k3a. Illustration du saut en avant (`advance_state!`) : sauter 999 997 tirages coûte ~0.5 µs contre ~6 ms pour les effectuer, et donne des valeurs bit à bit identiques — la propriété qui motive les CBRNG pour la simulation parallèle.
  - **SmallCrush sur Philox4x32-10** ajouté à la section TestU01 : 0 p-valeur suspecte sur 15 (21 s), à côté du LCG minimal (3 échecs) et de MRG32k3a (0).
  - **`code/philox.jl` supprimé** : la branche fournit `PhiloxRNG` et `PhiloxGen` sous les mêmes noms, dans un cadre CBRNG générique (`CBRNG{B,W,N,K}`) couvrant aussi Philox4x64 et Threefry. Vérifié avant suppression que les deux implémentations produisent des blocs identiques sur 1000 couples (compteur, clé) tirés au hasard, qu'aucun `include` ni `using` du fichier n'existait dans le dépôt (les diapos décrivent l'algorithme sans référencer le fichier), et que les vecteurs de référence Random123 qu'il contenait figurent dans la suite de tests du paquet (`test/runtests.jl`, testset « Philox reference values »), avec une couverture plus large. Le fichier reste récupérable par git (dernier état en `f76e347`).
  - `code/philox_test.jl` supprimé également : prototype antérieur du même algorithme, sous-typant `Random.AbstractRNG` directement, sans dépendance à `RandomDataStreams` et référencé nulle part. Même état récupérable en `f76e347`.
  - Suppression des scripts de mise au point `test_rng*.jl` à la racine, devenus sans objet une fois identifiée la cause du silence de SmallCrush.

---

## Notebook — `code/SDDP_from_scratch.ipynb`

- [x] **Nouveau notebook** : implémentation pédagogique de SDDP en Julia (JuMP + HiGHS), écrite à la notation du deck 07 et destinée à être lue à côté de lui — `code/SDDP_hydro.ipynb` n'utilise que l'API de `SDDP.jl`, qui masque l'algorithme.
  - Instance : ordonnancement hydro-thermique à un réservoir, $H = 6$ étapes, $M_t = 4$ apports indépendants d'étape en étape (1024 scénarios), première étape déterministe comme dans le deck. Recours relativement complet par construction, donc pas de coupe de faisabilité (le point est explicité, et repris en exercice).
  - Contenu : forme extensive comme solution de référence ; sous-problème d'étape unique par étape (partage de coupes justifié par l'indépendance inter-étapes), état entrant fixé par le membre de droite d'une contrainte dont le dual est le sous-gradient ; dérivation des coupes et correspondance avec la forme $E x + \theta \ge e$ des diapos ; passes avant et arrière ; bornes et critère d'arrêt statistique ; simulation de la politique.
  - Vérifications exécutées dans le notebook : le dual coïncide avec la dérivée par différences finies ; la borne inférieure atteint la valeur de la forme extensive (écart relatif $1{,}3 \times 10^{-14}$) en résolvant 3250 programmes linéaires à 6 variables au lieu d'un seul à 5460 ; les coupes sont validées numériquement contre la fonction de recours exacte, calculée en résolvant la forme extensive des étapes $t,\ldots,H$ depuis un niveau donné — approximation serrée aux états visités, écart jusqu'à 210 ailleurs ; le critère d'arrêt des diapos s'arrête à 1,47 % de l'optimum ; le cas $H = 2$ redonne la méthode L-shaped du deck 04, coupes dupliquées à l'appui (motivation de la sélection de coupes).
  - Se termine par ce qu'ajoute une implémentation de production (coupes de faisabilité, multicoupes, mesures de risque, graphes de politique cycliques, SDDiP) et cinq exercices.
  - Notebook livré sans sorties, comme `code/Farmer.ipynb` ; code exécuté de bout en bout avec Julia 1.12 avant livraison.

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

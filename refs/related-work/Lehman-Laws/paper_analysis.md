# Programs, Life Cycles, and Laws of Software Evolution

- **Original title**: Programs, Life Cycles, and Laws of Software Evolution
- **Author**: Meir M. Lehman (Senior Member, IEEE)
- **Affiliation**: Department of Computing, Imperial College of Science and Technology, 180 Queen's Gate, London SW7 2BZ, England
- **Venue / year**: *Proceedings of the IEEE*, vol. 68, no. 9, pp. 1060–1076, September 1980. Manuscript received 27 February 1980; revised 22 May 1980.
- **DOI**: [10.1109/PROC.1980.11805](https://doi.org/10.1109/PROC.1980.11805)
- **Source URL**: https://users.ece.utexas.edu/~perry/education/SE-Intro/lehman.pdf · local copy `paper.pdf` (17 scanned pages, pp. 1060–1076; six sections, five tables, six figure groups, 92 references). Not on arXiv.
- **OpenReview**: no OpenReview (1980 journal article). See section 7.
- **Code**: no public code (a 1980 journal article; the measured system is proprietary and is identified only as "system X").

**Classification: empirical, with a conceptual front half.** The durable contribution is a set of five measured regularities (Table I, p. 1068) grounded in release-by-release counts from one production operating system (Table II and Fig. 6, pp. 1069–1070) and applied to a real release plan (Tables III–V, pp. 1071–1073); the S/P/E program classification of Section II is the conceptual scaffolding that says *why* those regularities exist, not a method anyone reproduces. The empirical prompt is the backbone here, with the classification folded into sections 2a and 4a.

**Why this reference matters here**: our working hypothesis is that a self-editing harness accumulates disorder unless work is spent removing it, and that a dedicated cleaning process is needed. This paper is the oldest formal statement of exactly that, for evolving programs, with numbers attached. Its second law — complexity increases *unless work is done to maintain or reduce it* — is the hypothesis, written in 1980. Section 5f maps it law by law onto a harness and says which laws carry over, which do not, and what measurement design we can copy.

---

### Terminology, translated

The paper is written in 1980 software-engineering vocabulary. Translations used throughout this analysis:

| Paper's term | What it means | Nearest thing in our setting |
|---|---|---|
| program | a body of code in continuous productive use, maintained by an organization | the harness: the files around a frozen language model |
| module | the unit of partition of a program; the paper's counting unit for size and for change ("we consider only one aspect of partitioning using the term module for convenience", p. 1064) | one harness entry: a skill file, a memory entry, a tool description, a prompt clause |
| release | a batch of changes packaged, validated, and installed at user sites, numbered by release sequence number (R15, R16, …) | one committed harness edit, or one batch of edits |
| release sequence number | the index of the release in the series; the paper's discrete time axis | position in the task stream |
| maintenance | *all* changes made to a program after its first installation, not repair — the paper says so explicitly and objects to the word (p. 1061) | all harness editing after deployment |
| system age | calendar days since first installation; the paper's continuous time axis, distinct from the release index | wall-clock, as opposed to number of edits |
| incremental growth | net modules added in one release (modules added minus modules removed) | net growth in harness entries per edit |
| work rate (m/d) | modules changed in a release divided by the release interval in days | edit throughput |
| fraction of modules changed | changed modules divided by system size, per release; the paper's complexity proxy | fraction of the harness rewritten per edit |
| interconnectivity ratio | old modules changed divided by new modules added, for one planned feature — how much existing material a new feature disturbs | how much of the existing harness a new entry forces you to touch |
| metasystem | the program *plus* the organization, process, and users that change it — the closed loop, not the artifact | the harness plus the harness agent plus the scoring machinery |
| clean-up release | a release whose content is restructuring rather than new function | a harness edit that removes, merges, or promotes rather than adds |

---

## 1. Research Motivation and Questions

### a) Why was this study conducted?

Two economic facts open the paper. Total United States expenditure on programming in 1977 "is estimated to have exceeded $50 billion, and may have been as high as $100 billion", "more than 3 percent of the U.S. GNP for that year" (Section I-A, p. 1060). And of that, "some 70 percent was spent on program *maintenance* and only about 30 percent on program *development*" (Section I-C, p. 1060). The paper's target is that 70 percent. Its thesis is that the ratio is not a symptom of bad practice to be engineered away but a property of what programs are: "the need for continuing change is *intrinsic* to the nature of computer usage" (p. 1061). If change is intrinsic, then the question is not how to avoid it but how to control its cost, and the paper's answer is that the cost is governed by measurable dynamics.

The paper is equally explicit that the word "maintenance" is the wrong word and does damage: "for software the term *maintenance* is generally used to describe *all* changes made to a program after its first installation. It therefore differs significantly from the more general concept that describes the *restoration* of a system or system component to its *former* state … software does not deteriorate spontaneously or by interaction with its operational environment. Programs do *not* suffer from wear, tear, corrosion, or pollution. They do not change unless and until *people* change them" (p. 1061).

### b) What is missing in current understanding?

Three concrete gaps, in the paper's own framing.

First, the field had no principled criterion separating programs that must keep changing from programs that need not. Prior work leaned on a notion of *largeness*: "Program evolution dynamics … have always been associated with a concept of *largeness* … Great difficulty has, however, been experienced in defining these classes" (Section II-A, p. 1061). Size is a proxy with no causal content, and the paper replaces it.

Second, the evolution of software was assumed to be governed entirely by human decision, so nobody looked for regularity: "The resultant evolution of software appears to be driven and controlled by human decision, managerial edict, and programmer judgment. Yet as shown by extended studies … measures of its evolution display patterns, regularity and trends that suggest an underlying dynamics" (Section IV-B, p. 1067). The paper's surprise is that a process made entirely of free human choices produces statistically regular aggregate behavior.

Third, release planning had no empirical basis. Existing estimation techniques "are based on extrapolation of past experience and tend to produce results in the nature of self-fulfilling prophecies. In general, it has not yet proved possible to develop techniques that estimate project requirements on the basis of objective measurement of such attributes as application complexity and size and the work required to create a satisfactory system" (Section III-F, p. 1067). Section V is the paper's demonstration that the laws supply that basis.

### c) Research questions

The paper numbers no hypotheses. Stated as they function:

1. Which programs are condemned to never-ending change, and what distinguishes them from programs that can be finished? (Section II, the S/P/E classification.)
2. Does the release-by-release evolution of a real production system show statistical regularity, despite being driven by uncoordinated human decisions? (Sections IV–V.)
3. What are those regularities, stated as general laws? (Table I, p. 1068.)
4. Do the measured regularities have enough predictive force to overrule a management release plan before the release is built? (Section V-B, the R20 case.)

### d) Type of study

Observational measurement of one production system over its life, generalized into empirical laws, plus a worked forecasting exercise on a real plan. There is no manipulation, no control condition, and no counterfactual system: it is longitudinal industrial measurement in the style of an observational science, and the paper says so — "the laws are abstractions of observed behavior based on statistical models. They have no meaning until a system, a project and the organizational metasystem are well established" (p. 1068).

---

## 2. Study Design

### a) The S/P/E classification: the paper's explanatory variable

Section II replaces "large versus small" with a three-way classification by a program's relationship to the world it runs in. The classification rests on the observation "that, at the very least, any program is *a model of a model within a theory of a model of an abstraction of some portion of the world or of some universe of discourse*" (Section II-A, p. 1061).

**Class S** (Section II-B, p. 1061): programs "whose function is formally defined by and derivable from a specification". Examples given: lowest common multiple of two integers, eight queens, dining philosophers, the classical travelling salesman problem. Correctness is the only criterion and it is decidable against the specification: "since, by definition, the sole criterion of correctness of an S-program is the satisfaction of its specification, (correct) S-programs are always *provably* correct" (Section II-E, p. 1064). If the world changes, the specification changes, and that means "it has a *new program* for its solution" — a different program, not an evolved one. **Class S programs need not evolve.** The paper's conclusion states this as the paper's criterion: the classification "establishes the existence of a determining specification as the criterion for nonevolution" (Section VI, p. 1074).

**Class P** (Section II-C, p. 1062): the problem can be stated precisely but the solution must approximate a real world too large or too uncertain to compute exactly — a chess program, weather prediction, the travelling salesman problem as a working salesman actually faces it. The specification is exact; the *acceptability* of the solution is judged by the environment. The process of creating such programs "is modeled by Fig. 2 which shows the intrinsic feedback loop that is present in the P-situation. Despite the fact that the problem to be solved can be precisely defined, the acceptability of a solution is determined by the environment in which it is" used (p. 1062); the figure's comparison step, which has no counterpart in the class S figure, is the loop that never terminates. Because the comparison never terminates, "*P*-programs are very likely to undergo never-ending change or to become steadily less and less effective and cost effective" (p. 1062).

**Class E** (Section II-D, p. 1062): programs "that mechanize a human or societal activity" — operating systems, air-traffic control, stock control. Here the program does not merely model the world; installing it changes the world it models: "The program has become a part of the world it models, it is *embedded* in it" (Section II-D, p. 1063). Users "will modify their behavior to minimize effort or maximize effectiveness. Inevitably this leads to pressure for system change." Figs. 3 and 4 (p. 1063) add the extra loop that Fig. 2 lacks: the application in the real world contains the program, so the program's own output re-enters its requirements. The paper's summary sentence: "Change is built in. It is intrinsic to the nature of computing systems and the way they are developed and used" (p. 1063).

Class P and class E together are called **A-type programs** ("members of the union of the *P* and *E* classes", p. 1063); the laws in Table I apply to them and not to class S.

One structural claim matters later: the paper argues that an A-type program can always be *partitioned* into class S parts — "we suggest that it is *always* possible to continue the system partitioning process until *all* modules are implementable as S-programs" (Section II-F, p. 1064). Evolution pressure therefore lives at the seams and in the specification, not inside the leaves.

### b) Subjects of study

**One system is measured in this paper.** "System X is a general purpose batch operating system running on a range of machines. The eighteenth release (R18) of the system is operational in some tens of installations running a variety of work loads. The nineteenth release (R19) is about to be shipped" (Section V-B1, p. 1069). It is 4.3 years old at R19 and 4,800 modules / 1.3 million assembly statements in size (Table II, p. 1069). The vendor and product are not named.

The paper is explicit that the laws rest on more than this one system but that the supporting data is elsewhere: "Since the original observation [63], studies of program evolution have continued, based on measurements obtained from a variety of systems. Typical examples of the resultant models have been reported [69]–[72], [74], [76] including also one detailed example of their application to release planning [77]" (Section IV-B, p. 1068); and "We cannot, however, provide here the details of statistical analysis and model validation [76], based on this data and that from other systems that gives us confidence in our conclusions and predictions" (Section V-B1, p. 1069). **Anyone citing this paper for a multi-system empirical base is citing its references, not its contents.**

**The release-planning case study is a reconstruction, not a raw record.** The author says so in its final comments: "The case considered is based on a real situation, though in the absence of complete information details have had to be invented. But the details are not important since the objective has been to demonstrate a methodology" (Section V-B4i, p. 1074). The measured series of Table II and Fig. 6 are presented as real; the item-by-item plan contents of Tables III, IV and V are partly filled in. Cite the release-20 example for its *protocol* — comparing a plan against the system's own measured history — and not for its individual numbers.

### c) Measurement and instruments

Everything is counted in **modules**, with one counting rule fixed in the footnote to Table II: "Modules that are changed in any way in release $i+1$ relative to release $i$ are counted as one changed module, independently of the number of changes or of their magnitude" (p. 1069). Changes are counted at most once per module per release — a coarse instrument, and the paper later shows it failing (section 3d below).

Five quantities are derived from the module counts, against two different time axes — system age in days, and release sequence number:

| Quantity | Definition | Table II value for system X |
|---|---|---|
| system size | modules in the release | 4,800 at R19 |
| incremental growth | net modules added in one release | 410 at R19; 200/release average; 400/release "maximum safe growth rate" |
| modules changed | modules touched in any way | 2,650 at R19 |
| fraction of modules changed | modules changed / system size | 0.55 at R19 |
| change rate, or work rate (m/d) | modules changed / release interval in days | 10.7 modules/day system average |
| release interval | days between releases | 275 days at R19 |

The last five releases are tabulated (Table II, p. 1069). Incremental growth per release: 135, 171, 183, 354, 410 for R15–R19. Fraction changed: 0.33, 0.43, 0.48, 0.50, 0.56. Release interval in days: 96, 137, 201, 221, 275. Old modules changed per new module: 7.9, 8.6, 10.0, 5.1, 5.4. (The printed change-rate row reads 12.5, 0.12, 9.6, 9.9, 9.6; the R16 entry is inconsistent with its neighbours and with the 10.7 average and appears to be a typographic artifact of the original — do not quote it.)

**The complexity proxy is the fraction of modules changed per release.** The paper adopts it explicitly and flags it as coarse: "For system architectures such as that of system X, the fraction of system modules changed during a release may be taken as a gross indicator of system complexity" (Section V-A, p. 1069). This is the only complexity measure in the paper; the paper's closing paragraph notes that "the vital topic of software complexity [86]–[89]" is outside its scope (Section VI, p. 1074).

**The complexity proxy used for planning is the interconnectivity ratio**: old modules changed divided by new modules added, for one planned feature (Table III, p. 1071). High means the feature reaches deep into the existing system; low means it sits to the side.

### d) The evolution dynamics models and the release-planning protocol

"Evolution dynamics" in this paper means: fit a model to the release-by-release series of each measured quantity, and use the fit to predict the next release. The models themselves are not given here — the paper points to references [73]–[77] for them and says outright that "no details of life-cycle planning and management models, as such, have been included" (Section VI, p. 1074). What is given is the *use*, in seven readings of Fig. 6 (Section V-A, p. 1069):

- **Fig. 6(a)** size against system age: continuing growth, "albeit at a declining rate (demonstrably due to increasing difficulty of change, growing complexity — second law)". Size rises from about 1,000 to about 4,800 modules over roughly 1,600 days, visibly flattening.
- **Fig. 6(b)** size against release sequence number: "linear but with a superimposed ripple (a strong indicator of feedback stabilization)". The same data plotted against a different axis changes shape — growth is linear *per release* and decelerating *per day*, because releases get further apart.
- **Fig. 6(c)** net incremental growth per release: oscillates around the 200-module average with a 400-module ceiling drawn in; the paper reads it as the fifth law.
- **Fig. 6(d)** fraction of modules changed against system age: "shows an increasing trend (second law)", rising from roughly 0.1–0.3 early to about 0.6 at R19.
- **Fig. 6(e)** cumulative modules changed against system age: a straight line, "an example of the repeatedly observed constant average work rate (fourth law)".
- **Fig. 6(f)** modules changed per day against release sequence number: "oscillates, a period of high rate activity being followed by one or more in which the activity rate is much lower (third law)". Peaks near 27, troughs near 2.
- **Fig. 6(g)** release interval against release sequence number: falls, bottoms out around release 6–8 near 40 days, then rises steeply past 275 days.

The **release-planning protocol** in Section V-B is then, step by step: (1) tabulate the plan item by item, with new modules, old modules changed, and their ratio (Table III); (2) compute what the plan implies for growth, for change rate, and for release interval; (3) compare each against the system's own measured history and against the derived safe limits; (4) flag every deviation; (5) issue recommendations; (6) restructure the plan and re-tabulate (Tables IV and V). The instrument is comparison of a plan against the system's own dynamics, not against any external standard.

### e) Statistical and analytical approach

Curve fitting by eye and by unstated regression; no tests, no confidence intervals, no significance claims, no seeds — this is 1980 industrial measurement. The word "statistically" appears in three of the five laws and is never operationalized in this paper. Several inferences are labelled by the author himself as circumstantial: "Once again *circumstantial* evidence indicates that releases … have slipped delivery dates, a poor quality record and a subsequent need for drastic corrective activity" (Section V-B3b, p. 1071). The paper defers validation to reference [76] (Chon Hok Yuen's doctoral dissertation, "A phenomenology of program maintenance and evolution", Imperial College).

---

## 3. Findings and Observations (primary focus)

### a) The five laws, exactly as stated in 1980

Table I, p. 1068, reproduced verbatim. **This paper states five laws and no more.**

> **I. Continuing Change** — "A program that is used and that as an implementation of its specification reflects some other reality, undergoes continual change or becomes progressively less useful. The change or decay process continues until it is judged more cost effective to replace the system with a recreated version."
>
> **II. Increasing Complexity** — "As an evolving program is continually changed, its complexity, reflecting deteriorating structure, increases unless work is done to maintain or reduce it."
>
> **III. The Fundamental Law of Program Evolution** — "Program evolution is subject to a dynamics which makes the programming process, and hence measures of global project and system attributes, self-regulating with statistically determinable trends and invariances."
>
> **IV. Conservation of Organizational Stability (Invariant Work Rate)** — "During the active life of a program the global activity rate in a programming project is statistically invariant."
>
> **V. Conservation of Familiarity (Perceived Complexity)** — "During the active life of a program the release content (changes, additions, deletions) of the successive releases of an evolving program is statistically invariant."

**On which laws are here and which are not.** The paper presents these five as a revision of an earlier formulation, not as a fixed canon: "The laws, as currently formulated to include the new viewpoint emerging from the SPE classification, are given in Table I. Their early development may be followed in [9], [10], [72]" (p. 1068). It then documents one specific revision made for this paper — the first law used to be about size and is no longer: "The first law, *continuing change*, originally [3], [10], [79] expressed the universally observed fact that large programs are never completed. They just continue to evolve. Following our new insight, however, reference to *largeness* is now replaced by the phrase … 'that reflect some other reality …'" (p. 1068). Nothing in this paper announces or anticipates further laws. The commonly cited six-, seven-, and eight-law lists — including "continuing growth", "declining quality", and "feedback system" — are **not in this paper**; anyone needing them must cite Lehman's later papers, which we have not read. Do not attribute them here.

Note also what law V already contains that the practice does not: release content is explicitly "(changes, additions, **deletions**)". Deletion is in the 1980 vocabulary. It is absent from the 1980 *plan* (finding 4 below).

### b) Headline empirical findings

**Finding 1 — Growth continues but decelerates, and the deceleration is attributed to accumulated complexity.** *Evidence*: Fig. 6(a), p. 1070; system size against system age is a saturating curve from about 1,000 to about 4,800 modules over roughly 1,600 days. The paper's reading: "the continuing growth of the system (first law) albeit at a declining rate (demonstrably due to increasing difficulty of change, growing complexity — second law)" (p. 1069). *Conditions*: one system, 19 releases, 4.3 years. The word "demonstrably" is doing more work than the data supports; see section 4b.

**Finding 2 — The complexity proxy rises monotonically over the system's life.** *Evidence*: Fig. 6(d), p. 1070, plus the last-five-releases row of Table II: fraction of modules changed 0.33, 0.43, 0.48, 0.50, 0.56 across R15 to R19, against 0.1–0.3 early in life. "Fig. 6(d) shows that system X complexity, as measured in this way, shows an increasing trend (second law)" (p. 1069). *Conditions*: this is the paper's single direct measurement of the second law; everything else about complexity is inference.

**Finding 3 — Work rate is invariant across the system's life, and oscillates release to release.** *Evidence*: Fig. 6(e) is a straight line (cumulative modules changed against days), giving a lifetime average of "10.4 m/d over the lifetime of the system" (Section V-B3a, p. 1071), against Table II's 10.7 modules/day. Fig. 6(f) shows per-release rates swinging from about 2 to 27, "a period of high rate activity being followed by one or more in which the activity rate is much lower" (p. 1069). The paper discounts the 27 peak — "that data point is misleading and … a peak rate of about 20 m/d is a better indicator of the maximum achievable with current methodology and tools" (p. 1071). *Conditions*: one project, one organization.

**Finding 4 — High-content releases are paid for by the release after them, and the plan under review contains no removals at all.** *Evidence*: "there is strong circumstantial evidence that releases achieved with such high work rates were extremely troublesome and had to be followed by considerable clean-up in a follow-up release, as also implied by Fig. 6(c). Thus, if R20 is planned so as to require a work rate in the region of 20 m/d, it would be wise to limit R21 to at most 10 m/d, the system average" (Section V-B3a, p. 1071). The same pattern is attached to growth: releases "for which … the growth rate (incremental growth per release) has exceeded twice the average … have slipped delivery dates, a poor quality record and a subsequent need for drastic corrective activity" (p. 1071). And the R20 plan itself: "**No modules are planned for removal in the creation of R20** hence the planned net system growth is 1021 modules" (Section V-B2, p. 1071). *Conditions*: circumstantial by the author's own word; the removal count is a direct fact about the plan document.

**Finding 5 — The plan under review violates the system's own dynamics on every axis, and the review says so before the release is built.** *Evidence*, all Section V-B4, pp. 1071–1073 (subject to the reconstruction caveat in section 2b):
- *Growth*: "The current R20 plan calls for system growth of over 1000 modules. This figure which is five times the average and two and a half times the recommended maximum, must be interpreted as a danger signal" (p. 1072). Table III totals 1,021 new modules against a 200/release average and a 400/release safe ceiling.
- *Volume of change*: extrapolating the Fig. 6(d) trend, "R20 would be expected to require a change of, say, 64 percent, or 3725 changed modules. Comparing this estimate with the total of 6210 obtained if the estimates for individual items are summed, it appears that the average number of changes to be applied to R19 modules according to the present plan is at least of order two" (p. 1072) — that is, the plan silently assumes every touched module is touched about twice, and the module-counting instrument cannot see it.
- *Interconnectivity*: for plan items 4–9 the old-modules-changed to new-modules ratio "lies in the range 8.2 ± 1.5, a remarkably small range for widely varying functional changes. Yet the predicted ratio for ITS is only 2.4" (p. 1072). The paper treats the *low* ratio as the alarm, not the high ones: "Is it not far more likely that ITS has been inadequately designed; viewed perhaps as an independent facility that requires only loose coupling into the existing system?" (p. 1072). And: "In view of the 750 new modules involved, its IR factor could not exceed 6.4 even if all 4800 modules of R19 were effected by the ITS addition. Such a 100 percent change is, in fact, very unlikely, but the IR factor of 2.4 remains very suspect" (p. 1072).
- *Rate*: the plan wants 3,725 module changes in 548 days, "a change rate of less than 6.8 m/d" — below average, so rate is the one axis the plan does not violate (p. 1072).
- *Verdict*: "1) to proceed with the plan as it stands is courting delivery and quality problems for R20; 2) **a clean-up release appears due in any case**; 3) failure to provide it will leave a weak base for the next release" (p. 1072).

**Finding 6 — The recommended fix is to spend one whole release on removal and restructuring and to ship no new function.** *Evidence*: Section V-B4g–h and Tables IV–V, pp. 1072–1073. Recommendation 11 is to "abandon the present plan"; recommendation 12 is to "instead redesign release 20 to yield R20'; a clean, well-structured, base on which to build an ITS release, R21'". The paper's own strongest form: "Strictly speaking, Fig. 6(c) suggests that R20' should be a very low content release dedicated to system clean-up and restructuring" (p. 1073). Table IV re-sorts the plan by *reason* rather than by feature — fault repair (clean-up of base), hardware support (revenue producing), performance improvement ("Install — but do not announce. Will be available to counteract ITS performance deterioration in R21'"), and ITS components released early "to receive early user exposure". Table V re-prioritizes and reports the result: maximum incremental growth 159 new modules, "well under average", against the original 1,021, with a running change total of 2,529.

**Finding 7 — The recommendation was validated after the fact, and the original plan did fail.** *Evidence*: footnote 2 on p. 1072, labelled "Historical Note": "In the system on which this example is based the release including the interactive facility ultimately involved some 58 percent of the modules changed. Moreover the first release was significantly delayed, and was of limited quality and performance. More than 70 percent of its modules had subsequently to be changed again to attain an acceptable product. Our estimate is clearly good." This is the paper's only out-of-sample validation and it is one footnote long.

### c) Patterns and trends

Three shapes recur and all three matter to us.

**Rising cost of change with age.** Fig. 6(a) decelerates while Fig. 6(e) stays linear: the same work per day buys less net growth as the system ages. Stated causally in Section IV-A: "the temptation is to implement changes in the existing system, change upon change, rather than to collect changes into groups and implement them in a totally new instance. As the number of superimposed changes increases, the system and the metasystem become more complex, stiffer, more resistant to change. The cost, the time required, and the probability of an erroneous or unsatisfactory change all increase" (p. 1067).

**Oscillation around an invariant.** Both Fig. 6(c) (growth per release) and Fig. 6(f) (work rate per release) oscillate around a constant rather than trending. The paper reads the oscillation as evidence of a stabilizing feedback loop — a high-content release forces a low-content one after it. This is the mechanism that makes clean-up structural rather than optional: the system takes its clean-up whether or not anyone plans it, and the unplanned version is called a crisis.

**Progressive loss of local control.** "In its early stages of development a system is more or less under the control of those involved in its analysis, design, and implementation. As it ages, those working on or with the system become increasingly constrained by earlier decisions, by existing code, by established practices and habits of users and implementors alike … At the global level the metasystem dynamics have largely taken over" (Section IV-A, p. 1067).

### d) Surprises and counterintuitive results

**A low interconnectivity ratio is the danger sign, not a high one.** The intuitive reading is that a feature touching many existing modules is expensive and a self-contained feature is cheap. The paper inverts this: for a system of the kind measured, a feature claiming to touch almost nothing is probably mis-designed or under-estimated, because every comparable feature touches about eight existing modules per new one (Section V-B4a, p. 1072).

**Work output is independent of resources applied.** The fourth law's implication is stated as hard to swallow and defended anyway: "The reader may find it difficult to accept the implication that the work output of a project is independent of the amount of resources employed, though the same observation has also been recorded by others [81]" — reference [81] is Brooks, *The Mythical Man-Month*. The mechanism offered: "activities of the type considered, though initiated with minimal resources, rapidly attract more and more as commitment to the project … increase. Our observations as formalized in the fourth law imply that the resources that can be productively applied becomes limited as a software project ages … The project reaches the stage of resource saturation and further changes have no visible effect on real overall output" (p. 1068).

**The module-count instrument fails on exactly the plan it is used to judge.** Section V-B4b concedes it: "The presently defined measure 'modules *changed*' is inadequate. The new situation demands consideration of more sensitive measures such as 'number of module *changes*' and 'average number of changes per module.' These cannot be derived from the available data" (p. 1072). The paper proceeds by extrapolating the fraction-changed model instead.

### e) Negative and null results, and robustness

There are no ablations, no negative results, and no robustness checks — the design admits none. The only self-critical moves in the paper are the three just listed (instrument inadequacy, circumstantial evidence, undecidable questions) plus one explicit undecidable: on whether the ITS interconnectivity ratio reflects good design or bad estimation, "From the evidence before us, the question is undecidable. Experience based intuition, however, suggests that it is rather likely that the number of changes required elsewhere in the system has been underestimated" (p. 1072).

---

## 4. Analysis and Interpretation

### a) The author's explanations

**Why change never stops (demonstrated by construction, for class E).** Because installing the program changes the activity it mechanizes, the requirements that justified it are invalidated by its own success. This is an argument, not a measurement, and it is airtight for class E given its premise.

**Why complexity rises (asserted, with an analogy).** "The second law, *increasing complexity*, could be seen as an instance of the second law of thermodynamics. It would seem more reasonable to regard both as instances of some more fundamental natural truth. But from either viewpoint its message is clear" (p. 1068). This is an analogy and the paper does not pretend otherwise. The operative content of the law is not the analogy but the escape clause: complexity rises *unless work is done to maintain or reduce it*.

**Why global regularity emerges from local freedom (the paper's central mechanism).** "The third law … abstracts the observed fact that the number of decisions driving the process of evolution, the many feedback paths, the checks and balances of organizations, human interactions in the process, reactions to usage, the rigidity of program code, all combine to yield statistically regular behavior" (p. 1068). Laws IV and V are then instances of III: IV "reflects the steadiness of multiloop self-stabilizing systems … believed to arise from organizational striving for stability"; V "reflects the collective consequences of the characteristics of the many individuals within the organization … the law arises from the nonlinear relationship between the magnitude of a system change and the intellectual effort and time required to absorb that change" (p. 1068). Both mechanisms are labelled as belief and inference, not measurement.

**Why the organization chooses badly (demonstrated as a structural argument, and the passage we care about most).** Section III-D, p. 1065–1066: "At each moment in time, a manager's concern concentrates on the successful completion of his current assignment. His success will be assessed by immediately observable product attributes, quality, cost, timeliness, and so on. It is his success in areas such as these that determine the furtherance of his career. **Managerial strategy will inevitably be dominated by a desire to achieve maximum local payoff with visible short-term benefit. It will not often take into account long-term penalties, that cannot be precisely predicted and whose cost cannot be assessed.** Top-level managerial pressure to apply life-cycle evaluation is therefore essential … Neglect will inevitably result in a lifetime expenditure on the system that exceeds many times the assessed development cost on the basis of which the system or project was initially authorized." This is our objective mismatch, in 1980 organizational form: the quantity that is measurable now drives every decision, and the quantity that matters over the life is unmeasurable and therefore unrepresented.

### b) Evidence quality

Weak by modern standards, and asymmetrically so. The **descriptive** claims are well supported: the series in Table II and Fig. 6 are real counts from a real system over 19 releases, and the shapes are not subtle. The **causal** claims are not licensed by the design. There is no system that was cleaned up and no system that was not; the word "demonstrably" in the Fig. 6(a) reading ("demonstrably due to increasing difficulty of change, growing complexity") is an overclaim — deceleration is consistent with growing complexity and equally consistent with a maturing product needing less new function, and nothing here separates them. The **generalizations** to laws rest on data not in the paper. And the release-planning arithmetic rests partly on invented plan details (Section V-B4i, p. 1074), so it demonstrates a method rather than reporting a measurement.

The single strongest piece of evidence in the paper is the p. 1072 historical footnote, because it is genuinely out of sample: the prediction was made from the dynamics, the plan proceeded anyway, and the outcome (58 percent of modules changed, significant delay, limited quality, more than 70 percent of the new component's modules changed again afterward) matched. It is $n = 1$.

### c) Confounds and alternative explanations

- **One system, one organization, one process.** Every "law" is measured on a single batch operating system at a single vendor over 4.3 years. Laws IV and V are, on this evidence, facts about that organization's staffing and release cadence at least as plausibly as facts about software.
- **The two time axes give different answers and the paper uses whichever fits.** Growth is decelerating against days (Fig. 6(a), read as complexity) and linear against release number (Fig. 6(b), read as feedback stabilization). Both readings come from the same numbers.
- **The complexity proxy is confounded with the work.** Fraction of modules changed rises with system age — but it is a measure of how much was *changed*, not of how tangled the system *is*. A period of ambitious releases and a period of structural decay produce the same curve.
- **Survivorship in the release planning case.** The R20 analysis was performed by the author using models fitted to the same system's history and then validated against that system's outcome. No independent plan was scored.
- **The counting rule hides multiplicity.** By the Table II footnote, a module changed twenty times counts once. The paper discovers this matters precisely when the answer depends on it (Section V-B4b).

### d) Generalizability

Credibly extends to: long-lived systems maintained in releases by a stable organization, where the artifact is partitioned into countable units and the counts are recorded. The three structural preconditions the paper names are (1) the program is class P or E — embedded in a reality it cannot fully specify; (2) a stable metasystem exists, since "the laws … have no meaning until a system, a project and the organizational metasystem are well established" (p. 1068); and (3) changes accumulate into one artifact rather than each producing a fresh instance.

Risky to extrapolate to: anything where the changing agent is not a human organization. The author draws the line himself, and it is the sharpest limitation for us: "The laws represent *principles* in software engineering. They are, however, clearly not immutable, as for example, are the laws of physics or chemistry. **Since they arise from the habits and practices of people and organizations**, their modification or change requires one to go outside the discipline of computer science into the realms of sociology, economics and management" (Section V-A, p. 1069). Laws III, IV, and V are explicitly social, not technical.

---

## 5. Implications, Limitations and Transferability

### a) For practitioners

Budget for a clean-up release; it is due whether or not you schedule it. Measure your own system's growth, change fraction, and work rate per release and treat those series as the constraint on the next plan, rather than treating the plan as the constraint. Treat any single release whose content is far above your own historical average as a prediction of a bad next release. And be suspicious of the new component that claims to touch nothing.

### b) For researchers

The paper's own list of what it left open: "we have not here discussed cost, resource, and reliability models [83]–[85]. Approaches to process modeling based on continuous models [73], [75] have also not been included, nor has the vital topic of software complexity [86]–[89]" (Section VI, p. 1074). The last one is the live gap: the second law asserts complexity rises, and the paper measures a change-fraction proxy instead, with no independent complexity instrument anywhere.

The closing sentence is an invitation we are effectively accepting: "Many of the concepts and techniques presented in this paper could find wide applications outside the specific area of software systems, in other industries, and to social and economic systems. Unfortunately that theme cannot be pursued here" (p. 1074).

### c) Acknowledged limitations

Circumstantial evidence for the clean-up claims; an inadequate change measure (Section V-B4b); an undecidable question about the interconnectivity finding; no complexity model; no statistical validation in this paper; invented details in the release-planning case study (Section V-B4i, p. 1074); and the laws are not immutable, since they depend on human organizational behavior.

### d) Unacknowledged limitations

- **Sample size one, presented as laws.** The word "laws" carries authority the single measured system does not support, and the supporting multi-system data is entirely by reference.
- **The complexity proxy is never validated against anything.** No second measure of complexity exists in the paper to correlate it with.
- **"Statistically invariant" is never tested.** Laws III, IV, and V use the word; no variance, interval, or test appears anywhere.
- **Nothing measures quality directly.** The first law speaks of a program becoming "progressively less useful" and the analysis speaks of "poor quality record", but user-visible quality is never a measured series — it enters only through the p. 1072 historical footnote.
- **The counterfactual is absent.** The paper recommends a clean-up release and the case study ends with a recommendation, not with a measured outcome of having done it. The one validated prediction is of the failure that followed *not* doing it.

### e) Reproducibility

Zero, in the modern sense. System X is unnamed and proprietary; the raw per-release series is not published beyond the five releases of Table II and the plotted points of Fig. 6; the dynamics models are in references, not here; the validation is in a dissertation. What *is* reproducible is the method: the six quantities, the counting rule, and the plan-versus-history comparison protocol are all specified well enough to re-run on a different system, and that is what we would do.

### f) Transfer to our own work

Read the harness as the program and the task stream as the usage. Our harness is unambiguously a class E object by the paper's own criterion: it mechanizes an activity, and installing it changes the activity — an edit that adds a skill changes which tasks the agent attempts and how, which changes the failures it produces, which changes what the next edit will be. It has no determining specification, which is precisely what the paper says makes evolution non-optional (Section VI, p. 1074). Everything in Table I that is technical rather than organizational should therefore be expected to apply.

**The law-by-law map.**

| Law as stated in 1980 | Plain-language meaning | Harness analogue | Already observed in our references? |
|---|---|---|---|
| **I. Continuing Change** — a program that reflects some other reality "undergoes continual change or becomes progressively less useful … until it is judged more cost effective to replace the system with a recreated version" | A system embedded in a world it cannot fully specify must keep changing or decay; eventually rebuilding beats patching | Direct. A harness on a non-stationary task stream must keep being edited or fall behind the stream; the "recreated version" is a harness rebuilt from scratch, which is also the reference point our plasticity measurement needs | Yes, as the premise everything shares: every reference in the index edits continuously and none defines a stopping point. The decay half is visible in `Evo-Memory` (raw accumulation is worse than no memory on two of four single-turn backbones, and loses 40 points on ToolBench) and in `Adaptive-Auto-Harness` (A-Evolve ends *below* the no-evolution solver on PolyBench accuracy, 18.4 against 22.2) |
| **II. Increasing Complexity** — "its complexity, reflecting deteriorating structure, increases unless work is done to maintain or reduce it" | Disorder accumulates by default; removing it is separate, deliberate work that nobody is forced to do | Direct, and this is the researcher's hypothesis verbatim. "Work done to maintain or reduce it" is the merge, the promotion, the deletion — the harness edits that no existing system's objective can reward, because they do not improve the current batch | Yes, and the *absence* of the maintaining work is the strongest pattern in the index. `HCL`'s commit gate requires current-task improvement, so a consolidation-only edit "cannot be committed, no matter how much it helps later" — the paper's own conclusion lists "harness-content consolidation" as an open problem its gate forbids. `Adaptive-Auto-Harness` has no merge and no pruning in released code (`promotion_threshold` and `staleness_window` are in the config schema and read by no code). `MemRL`'s store is append-only: the utility filter suppresses retrieval but never removes. `Evo-Memory`'s ExpRAG update is a plain append with "nothing merged, rewritten, or removed". `MemEvolve` reports that seven of the twelve memory systems it catalogues have no manage component at all. The exception worth naming is `ReMe`, whose deletion rule fires on realized hit rate after five retrievals — the only removal path in the index scored on tasks later than the one that wrote the entry |
| **III. The Fundamental Law of Program Evolution** — evolution "is subject to a dynamics which makes the programming process … self-regulating with statistically determinable trends and invariances" | Even though every individual change is a free human decision, the aggregate series is regular enough to model and predict | Partial, and unproven for us. It is the claim that a harness's edit history is a predictable time series — which is exactly the bet behind a critic $\hat\Phi(h)$ that reads harness state and predicts future stream performance. But the paper's regularity is attributed to *organizational* feedback (staffing, careers, release cadence), none of which exists in an agent loop | No. Nothing in the index reports a fitted model of a harness's own evolution series. `Adaptive-Auto-Harness` Table 9 and `AHE` Figure 1 publish the series; neither models it |
| **IV. Conservation of Organizational Stability (Invariant Work Rate)** — "the global activity rate in a programming project is statistically invariant" | A project's real output rate is fixed by its structure, not by how many people you add | No analogue. An agent loop's edit rate is set by compute and by the loop's design, and nothing conserves it. This is the most purely social of the five laws and the author says the laws "arise from the habits and practices of people and organizations" (p. 1069) | Not applicable |
| **V. Conservation of Familiarity (Perceived Complexity)** — "the release content (changes, additions, deletions) of the successive releases … is statistically invariant" | There is a ceiling on how much change per release people can absorb; exceed it and the next release pays | Weak analogue, and the interesting part is where it fails. The absorption constraint is human, so it does not bind an agent. But the *underlying* effect might: a very large single edit is harder for the harness agent to verify, and its regressions surface later. Worth testing rather than assumed | Partially, through a different variable. `HarnessX` documents a Telecom run dropping 14.0 points from five sub-threshold edits that no individual gate check caught, and `EvolveMem`'s own Table 4 shows an uncaught sub-threshold decline (round 4 at 38.5 to round 5 at 38.1, under a 0.01 revert tolerance). Both are about edit *size* relative to the gate's resolution, which is this law's mechanism without its human ceiling |

**What "anti-regressive work" is called here.** The phrase "anti-regressive work" — Lehman's later term for complexity-reducing maintenance — **does not appear in this paper**. Its 1980 name is the **clean-up release**: "a clean-up release appears due in any case" (recommendation 2, p. 1072); "Strictly speaking, Fig. 6(c) suggests that R20' should be a very low content release dedicated to system clean-up and restructuring" (p. 1073); Table IV's first row is the class "Fault Repair", reason "Clean-up of base". The concept is fully present and the label is not, so cite this paper for the clean-up release and cite a later Lehman paper if the exact phrase is needed.

Its harness translation is a **maintenance edit that adds no capability**: merging several narrow skills into one general skill, deleting memory entries no longer retrieved, moving a rule from a prompt clause into a tool description, reverting an accumulation of sub-threshold regressions. These are `move` and `promotion` in our Terms table. The paper's decisive structural point is that such an edit *cannot justify itself on the current batch* — Table V's restructured R20 ships almost no new function (159 new modules against the original plan's 1,021) and its entire case rests on the state of R21. Every gate in our index would reject it. That is the concrete bridge from a 1980 observation to our objective.

**The measurement design we should copy.** Four pieces, in order of value.

1. **Two time axes, plotted separately.** The paper's sharpest single move is Fig. 6(a) against Fig. 6(b): the same growth data is decelerating against calendar days and linear against release index. For us the pair is *tasks elapsed* against *edits committed*. Our references publish only one of these at a time — `AHE` and `HarnessX` plot per-round, `AgentStream` plots per-task — and the divergence between them is where the cost of an edit becomes visible.
2. **A fixed counting unit with a stated counting rule.** Modules, counted once per release however many times they changed. Our unit is the harness entry (one skill file, one memory item, one tool description, one prompt clause) and the rule needs the same explicitness — and the paper's own failure is the warning: the counting rule broke exactly when multiplicity mattered (Section V-B4b), so we should record *edits per entry*, not just *entries edited*, from the start.
3. **Four per-edit series, measurable today from any harness git history.** Net entry growth per edit (law I and the growth series), fraction of existing entries touched per edit (the complexity proxy of Fig. 6(d)), cumulative entries touched against tasks elapsed (the work-rate line of Fig. 6(e)), and the **interconnectivity ratio** — existing entries changed divided by new entries added, per edit (Table III). The fourth is the one nobody in the index computes and it is the cheapest useful thing here: it separates an edit that bolts on a new skill in isolation (ratio near zero) from one that integrates it (ratio high), and Lehman's 1980 reading is that the low-ratio edit is the suspicious one. `Harness-Anatomy` gives us the substrate to run this on immediately — pinned public git histories of eleven production coding harnesses, which are human harness-evolution logs with real timestamps.
4. **The plan-versus-history comparison as a gate.** Section V-B's protocol scores a *proposed* release against the system's own measured dynamics and flags every deviation, before the release is built. The harness version is a state-level check on a proposed edit that reads the harness's own history rather than any task score — which is structurally the same object as our critic $\hat\Phi(h)$, arrived at forty-six years earlier by regression on counts instead of by learning on returns.

**Relation to `Loss-of-Plasticity` (the machine-learning counterpart).** The two papers describe the same failure in different substrates and their disagreement is more useful than their agreement.

They agree on the shape and on the escape. Both say a continuously-updated system degrades by default, both locate the degradation in accumulated state rather than in any single bad update, and both say the fix is work that removes rather than adds — Lehman's clean-up release and Dohare et al.'s reinitialization of low-utility units are the same move at different granularity. Both also note that the degradation is invisible to the metric the system optimizes: Lehman's manager sees "immediately observable product attributes" and not the lifetime cost (p. 1066); Dohare et al.'s learner sees healthy current-task accuracy for a long time before the fall.

They disagree on mechanism, and the disagreement matters because our harness sits on Lehman's side. `Loss-of-Plasticity` diagnoses the loss of the random variability present at initialization, under gradient descent — a mechanism its own analysis notes does not transfer to us, since a harness is edited by a language model writing text, which is already a stochastic, variability-injecting process. Lehman's mechanism is the opposite kind: **accumulation and interference among committed entries**, with rising cost of each subsequent change — "change upon change … the system and the metasystem become more complex, stiffer, more resistant to change. The cost, the time required, and the probability of an erroneous or unsatisfactory change all increase" (p. 1067). That is the candidate mechanism for a harness, and Lehman is its earliest statement.

They also complement each other on instruments. `Loss-of-Plasticity`'s three state correlates (dead units, weight magnitude, effective rank) are properties of continuous weights with no harness counterpart. Lehman's are properties of a discrete, versioned artifact — counts of entries, counts touched, ratios between them — and they port directly. Conversely, Lehman has no equivalent of the from-scratch reference learner, which is `Loss-of-Plasticity`'s best idea and the one that turns an unobservable capacity into an observable gap. The combination is the design: Lehman's counting series for the state, Dohare et al.'s from-scratch reference for the outcome.

One caution carries over from both. Lehman's laws III, IV, and V are explicitly about human organizations — "they arise from the habits and practices of people and organizations" (p. 1069). Citing this paper for laws I and II in an agent setting is defensible; citing it for the invariance laws is not, and our writing should say which two we are taking.

---

## 6. Summary

### a) One-sentence headline finding

Programs embedded in a reality they cannot fully specify must change forever, and their complexity rises with every change unless work is deliberately spent reducing it.

### b) Quick-reference takeaways

- The paper states **five** laws (Table I, p. 1068), not the six, seven, or eight of later Lehman papers: continuing change, increasing complexity, the fundamental law of program evolution, conservation of organizational stability, conservation of familiarity. The first law was reworded for this paper to drop its earlier reliance on program size.
- The **S/P/E classification** is the criterion for which programs must evolve: class S programs derive from a specification and can be finished; class P approximate a world too complex to specify exactly; class E mechanize a human activity and change that activity by existing, so their requirements are invalidated by their own success. Classes P and E together are called A-type and are the subject of the laws. A harness is class E.
- The empirical basis inside this paper is **one unnamed batch operating system, "system X"**, 4,800 modules and 1.3 million assembly statements at release 19, age 4.3 years, measured in modules per release: size, net growth, modules changed, fraction changed, work rate in modules per day, and release interval. Multi-system support is by reference only.
- The data show growth continuing but decelerating against calendar time while linear against release index; the complexity proxy (fraction of modules changed per release) rising from about 0.1–0.3 early to 0.56 at release 19; work rate invariant at about 10.4–10.7 modules per day over the life while oscillating between roughly 2 and 27 release to release.
- The **release-planning example** reviews a real plan for release 20 that would have added 1,021 new modules — five times the system's 200-per-release average and two and a half times its 400-per-release safe ceiling, with **no modules planned for removal** — and recommends abandoning it in favour of a low-content clean-up and restructuring release adding 159 new modules. A footnote records that the original plan proceeded, involved 58 percent of modules changed, was delayed and of limited quality, and that more than 70 percent of the new component's modules had to be changed again.
- The paper's term for complexity-reducing maintenance is **"clean-up release"**; the phrase "anti-regressive work" is not in this paper.

### c) Bottom line for decision-making

Trust the conceptual contribution — the S/P/E criterion and the "unless work is done to maintain or reduce it" clause of the second law — as the oldest correct statement of our hypothesis, and cite it as such. Treat the five laws as well-motivated regularities from one system, not as established quantitative results; the multi-system evidence is in the references and we have not read it. Take the measurement design, which is cheap, mechanical, and runs today on any harness with a git history, and take the release-20 example for its protocol rather than its numbers, since the author records that some of its details were invented. Do not carry laws III, IV, and V into an agent setting without saying that the author attributes them to human organizational behavior.

---

## 7. Reviewer Reception

No public OpenReview page (1980 journal article; no public reviews exist).

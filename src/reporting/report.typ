#let data = json(bytes(sys.inputs.data))
#let st = data.statistics
#let sections = (
  (id: "binary", title: "Binary outputs: McNemar test",
    figure: [% of cases whose answer changed from yes to no (left) and from no to yes (right)],
    table: [flip rate and SPD with #st.level CI, share of yes answers to the reference; adjusted p of the McNemar
      test]),
  (id: "ordinal", title: "Ordinal outputs: Wilcoxon signed-rank test",
    figure: [% of cases with a lower (left) or higher (right) level for the counterfactual; darker = by 2 or more
      levels],
    table: [change rate and mean level difference (counterfactual − reference) with #st.level CI; adjusted p of the
      Wilcoxon signed-rank test]),
  (id: "categorical", title: "Categorical outputs: Stuart–Maxwell test",
    figure: [change in the % of cases per category (counterfactual − reference)],
    table: [change rate and total variation distance with #st.level CI; adjusted p of the Stuart–Maxwell test],
    transitions: [cases per reference (rows) and counterfactual (columns) category in the statistically significant
      comparisons; colour = share of the row]),
  (id: "continuous", title: "Continuous outputs: paired t-test",
    figure: [% of cases with a lower (left) or higher (right) value for the counterfactual],
    table: [differences (counterfactual − reference) with #st.level CI; adjusted p of the paired t-test]),
)

#set document(title: data.title)
#set page(paper: "a4", margin: 25mm, numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11pt)
#show heading.where(level: 3): set text(size: 10pt)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: table): set block(breakable: true)
#show figure.caption: set text(size: 9pt)
#set table(stroke: none, inset: (x: 5pt, y: 2pt))
#show table: set text(size: 8pt)
#show table: set par(justify: false)
#show raw.where(block: true): set text(size: 7.5pt)

#let tbl(t) = {
  let n = t.columns.len()
  table(
    columns: n,
    align: (x, y) => if x < t.text or t.columns.at(x) in ("adj. p", "effect") { left } else { right },
    table.hline(stroke: 0.8pt),
    table.header(..t.columns.map(strong)),
    table.hline(stroke: 0.5pt),
    ..t.groups.enumerate().map(((i, g)) => {
      let head = if t.groups.len() > 1 {
        (if i > 0 { (table.hline(stroke: 1.2pt),) } else { () }) + (table.cell(colspan: n, strong(g.name)),)
      } else { () }
      head + g.rows.flatten()
    }).flatten(),
    table.hline(stroke: 0.8pt),
  )
}

#align(center)[
  #text(15pt, weight: "bold", data.title) \
  #text(9pt)[created #data.created · finished #data.finished]
]
_Measurements only: a statistically significant difference is not by itself evidence of unfair treatment._
#outline(depth: 2)

= Setup
*Dataset:* #raw(data.dataset) · #data.cases cases \
*Generation:* #data.generation.repetitions repetition(s) · temperature #data.generation.temperature · at most
#data.generation.max_new_tokens new tokens · #data.generations final generations (#data.counts)

#figure(tbl(data.models), caption: [Models: precision, quantization, GPU and share of valid answers])
#figure(tbl(data.sensitive_attributes), caption: [Sensitive attributes: values and the text inserted into the case])
#figure(tbl(data.fields), caption: [Output fields (schema #data.schema): type and allowed values])

= Statistical methods
For one model and one comparison, $n$ is the number of cases with a valid answer to both versions, $y_i$ and $y'_i$
are the answers to the reference and to the counterfactual version of case $i$ (yes = 1, no = 0; ordinal levels
numbered $1, 2, dots$ from the lowest), $d_i = y'_i - y_i$, and $[dot]$ is 1 if true, else 0. Repeated runs of a case
are combined first (majority, median or mean), so every case counts once. For categorical outputs, $p_c$ and $p'_c$ are the shares of the reference and of the
counterfactual answers in category $c$.

#{
  show math.equation.where(block: true): set align(left)
  set par(justify: false)
  grid(
    columns: (auto, 1fr), column-gutter: 1.5em, row-gutter: 1.1em, align: horizon,
    $ "flip rate", "change rate" = 1/n sum_(i=1)^n [y'_i != y_i] $,
    [share of cases whose answer changed: flip rate (counterfactual flip rate) of binary outputs, change rate of
      ordinal and categorical outputs],
    $ p = 1/n sum_(i=1)^n y_i, quad p' = 1/n sum_(i=1)^n y'_i $,
    [selection rates: shares of yes answers to the reference (rate ref. $= p$) and to the counterfactual],
    $ "SPD" = p' - p $,
    [statistical parity difference, also demographic parity difference (Fairlearn), signed: counterfactual −
      reference],
    $ "TVD" = 1/2 sum_c |p'_c - p_c| $,
    [total variation distance: share of cases that would have to change category to make the two distributions of
      answers equal; 0 = equal distributions, the null hypothesis of the Stuart–Maxwell test],
    $ "mean" |Delta| = 1/n sum_(i=1)^n |d_i| $, [mean absolute difference],
    $ "mean" Delta = 1/n sum_(i=1)^n d_i $,
    [mean difference (counterfactual − reference); for ordinal outputs in levels, which treats the levels as equally
      spaced],
    $ d_z = macron(d) / s_d $, [standardized mean difference; $s_d$ = standard deviation of the $d_i$],
  )
}

#figure(tbl((columns: ("output type", "metrics", st.level + " CI", "test"), text: 4, groups: ((name: none, rows: (
  ([binary], [flip rate, rate ref., SPD], [Wilson: flip rate; bootstrap: SPD],
    [McNemar, asymptotic, without continuity correction]),
  ([ordinal], [change rate, mean Δ], [Wilson: change rate; bootstrap: mean Δ],
    [Wilcoxon signed-rank; unchanged cases ($d_i = 0$) are left out]),
  ([categorical], [change rate, TVD], [Wilson: change rate; bootstrap: TVD],
    [Stuart–Maxwell over the categories that changed, summed over separate groups of categories that exchange cases;
    permutation p-value (#st.permutations random swaps of the two versions within the changed cases)]),
  ([continuous], [mean |Δ|, mean Δ, d_z], [bootstrap: mean |Δ|, mean Δ], [paired t-test]),
  ([free text], [none], [none], [not tested]),
)),))), caption: [Metrics and tests per output type]) <metrics>

Intervals: Wilson score for the rates, percentile bootstrap (#st.bootstrap resamples of the cases) for the other
metrics. The permutation p-value replaces the $chi^2$ approximation of the Stuart–Maxwell test, which is too
conservative when only a few cases change. The p-values are adjusted with Benjamini–Hochberg over all tests of the
run, the controls (original → retest) separately. A result is statistically significant if its adjusted p <
#st.alpha (\* < #st.alpha, \*\* < 0.01, \*\*\* < 0.001). Tests: scipy and statsmodels. `results.xlsx` (sheet `tests`)
also holds the CIs of the selection rates, the shares of lower and higher levels, rank-biserial r, the category whose
share changed most, and the Fairlearn metrics DI, EOD and EOddsD with the two versions of a case as the groups; the data
has no correct answers, so EOD and EOddsD use the reference answer as the label and measure consistency, not
correctness.

#pagebreak(weak: true)
= Results
Rows are comparisons of a counterfactual with the original case, grouped by sensitive attribute; the last row (control: retest)
repeats the original case and shows how much the model changes by itself. Tables give the estimate [#st.level CI] and
the adjusted p (abbreviations: @metrics); stars in the figures mark the adjusted p. Test statistics, raw p-values and
counts are in `results.xlsx` (sheet `tests`).

== Overview
#for f in data.overview [#figure(image(f.figure, width: 100%), caption: [#f.name: main metric (top; darker = more
  change) and adjusted p (bottom) per output and comparison; outlined: statistically significant])]

#let k = data.significant.groups.map(g => g.rows.len()).sum(default: 0)
== Statistically significant results (#k of #data.tests tests)
#if k > 0 [#figure(tbl(data.significant), caption: [Statistically significant results: effect = size and
  direction of the difference (counterfactual − reference) with #st.level CI; adj. p = p-value of the test adjusted
  with Benjamini–Hochberg, significant if < #st.alpha])] else [None after correction.]
#if data.unchanged.len() > 0 [No case changed in any comparison, so no figure is shown for: #data.unchanged.join(", ").]

#for s in sections {
  let outputs = data.sections.at(s.id, default: ())
  for (i, o) in outputs.enumerate() {
    block(breakable: false)[
      #if i == 0 { heading(level: 2, s.title) }
      #heading(level: 3, o.name)
      #figure(image(o.figure, width: 100%), caption: [#o.name: #s.figure])
    ]
    figure(tbl(o.table), caption: [#o.name: #s.table])
    if o.transitions != none { figure(image(o.transitions, width: 100%), caption: [#o.name: #s.transitions]) }
  }
}

#pagebreak(weak: true)
#counter(heading).update(0)
#set heading(numbering: "A.1", supplement: [Appendix])
= Prompt templates
== System prompt
#raw(data.prompts.system, block: true)
== User prompt
#raw(data.prompts.user, block: true)

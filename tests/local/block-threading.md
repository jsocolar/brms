# Experimental block threading

Use `threads = threading(2, blocks = data$block, grainsize = 1)` in `brm()`,
`make_stancode()` or `make_standata()`. Supply one character, numeric or factor
ID per input row. IDs need not be consecutive integers. Each ID must occur in
one contiguous run. No rows are reordered. Grainsize counts blocks, not rows;
the existing default heuristic is retained (minimum 100), so explicitly choose
a smaller grainsize for small numbers of blocks.

This prototype supports a single response, `custom_family(loop = FALSE)`, ordinary
linear distributional formulas, fixed effects and random intercepts. Other
predictor terms, nonlinear formulas, response additions other than `vint`/`vreal`,
missing-row removal, and automatic row reordering are rejected. Prepare complete
data before supplying IDs; externally subset/reorder data and IDs together.
Block IDs are stored in the threading configuration: when updating a fit with
new data, supply a matching new threading configuration. Equal-length changes
cannot be detected automatically.

The callback schedules a sequence of distinct block ordinals, maps its first
and last scheduled blocks to row bounds, and uses brms's existing `start:end`
predictor and response slicing. No predictor generator needs a block-specific
branch. Centering, design matrices, contrasts, parameterization and priors remain
global. Group-effect transformations and long parameter vectors remain global.
No Stan compiler or Stan Math modifications are needed.

## Custom likelihood contract

The likelihood must accept *any contiguous chunk of complete blocks*, including
several blocks and the entire dataset. It must identify/reset at internal block
boundaries itself. `blocks` controls scheduling only; it does not automatically
supply block labels to the custom likelihood.

For example, use `y | vint(block, row_id) ~ ...` together with
`vars = c("vint1[slice]", "vint2[slice]", "shared_data")`. `[slice]` is an explicit
new marker for `loop = FALSE`: it becomes `[start:end]` under threading and is
removed in serial code. Thus response, predicted distributional parameters and
explicitly sliced auxiliary vectors have the same local length and row order.
The `vint`/`vreal` additions follow model-frame validation. A global row ID can
recover global positions when the likelihood needs shared arrays.

Bare names remain shared; other literal indices retain existing behavior.
`stanvar(x = ..., name = ...)` supplies data verbatim and normally infers its
worker argument declaration. Authors own its dimensions and alignment. For
custom Stan declarations, supply `pll_args` as documented by `stanvar()`.
An explicit `[slice]` can also slice a suitable shared Stan vector/array, but
does not validate its length. There is no guessed slicing/rebasing of auxiliary
arguments. Likelihood-block `stanvars` code executes per callback: it must not
add priors or other once-per-model terms.

Distribute large blocks throughout the row order rather than clustering them.
Reorder whole blocks only, preserving meaningful within-block order. There is
no automatic balancing and speedup is not guaranteed.

## Validation

`tests/testthat/tests.block-threading.R` checks code generation, invalid inputs,
boundaries, auxiliary slicing, two predictors, random intercepts, singleton and
one-block data, and unchanged ordinary scheduling.

Run `Rscript tests/local/block-threading.R` from the repository root for the
small numerical integration test (requires development dependencies, CmdStanR
and CmdStan). Its Gaussian Markov likelihood genuinely couples adjacent rows
within blocks, checks global row IDs and rejects split boundaries. It compares
serial/block log densities and autodiff gradients at identical unconstrained
parameters, with unequal blocks, singleton blocks, one total block and grainsizes
1 and 100. It also smoke-tests fitting, `fixef`, `ranef`, `posterior_linpred` and
custom `posterior_epred`. The tiny chains are not convergence diagnostics.
The alternate-data numerical harness uses a CmdStanR internal helper to reuse
compiled autodiff modules; production code does not depend on that helper.

Custom prediction and likelihood post-processing remain the author's job.
This prototype does not add block-level `log_lik`, LOO, joint posterior prediction,
or automatic cross-validation partitions. Do not interpret row-level LOO as
leave-one-block-out validation. New-data prediction can require an updated block
specification if it regenerates threaded Stan data.

The implementation is small enough for an experimental branch and an upstream
API discussion. Broader formula support, a stable integration-test harness,
block-aware cross-validation and new-data/update ergonomics need separate review
before calling it a general supported feature.

## Recorded prototype validation

On 2026-11-01, with this checkout (brms 2.23.2), R 4.5, CmdStan 2.34.1
and CmdStanR 0.9.0.9000:

- All 37 focused assertions passed.
- Existing Stan-code tests: 729 assertions passed, two CRAN-marked skips.
- Existing Stan-data tests: 284 assertions passed, four missing-`splines2` skips.
- The numerical script passed all density and gradient comparisons at tolerance
  `1e-10`, including 18 alternate-data/parameter combinations with static
  scheduling and the initial fit-data comparison. Boundary and row-alignment
  checks did not reject any valid chunk.
- Both tiny fits and the listed post-processing checks passed. Each 20-draw
  fit reported two divergent transitions; these were execution smoke tests,
  not evidence of converged posterior inference.
- Modified Rd files parsed, and `git diff --check` passed.

The checkout required `future >= 1.80.0`; a temporary library at
`/private/tmp/brms-block-lib` supplied it for these tests. Set
`R_LIBS=/private/tmp/brms-block-lib` to reproduce on this machine.

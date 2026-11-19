# Experimental block threading

Use `threads = threading(2, block_var = "block", grainsize = 1)` in `brm()`,
`make_stancode()` or `make_standata()`. The named column must contain one character, numeric or factor
ID per input row. It need not appear in the model formula. IDs need not be consecutive integers. Each ID must occur in
one contiguous run. No rows are reordered. Grainsize counts blocks, not rows;
the existing default heuristic is retained (minimum 100), so explicitly choose
a smaller grainsize for small numbers of blocks.

This prototype supports a single response and `custom_family(loop = FALSE)`.
Predictor compatibility follows ordinary brms threading: splines, monotonic
effects, nonlinear formulas, random slopes and offsets are allowed. Existing
threading restrictions (for example sparse matrices and unsupported
ARMA/covariance-based dependence structures) still apply. Response additions
other than `vint`/`vreal`, missing-row removal and automatic row reordering are
rejected. Prepare complete
data before supplying IDs; externally subset/reorder data and IDs together.
The threading configuration stores only the column name. Updates and new-data
validation retrieve fresh IDs from that column, which must be present in the
new data. The old vector-valued `blocks` argument has been replaced.

The callback schedules a sequence of distinct block ordinals, maps its first
and last scheduled blocks to row bounds, and uses brms's existing `start:end`
predictor and response slicing. No predictor generator needs a block-specific
branch. Centering, design matrices, contrasts, parameterization and priors remain
global. Group-effect transformations and long parameter vectors remain global.
No Stan compiler or Stan Math modifications are needed.

Nonlinear expressions must preserve their meaning when evaluated on arbitrary
chunks. Rowwise expressions satisfy this; vector reductions such as `mean()`
over a predictor vector generally do not. This is an author responsibility,
just as in ordinary threading; no expression analysis is attempted. Global
latent transformations (including some GP computations) may remain outside the
worker, so allowing a predictor does not imply parallelizing all its work.

Keeping complete blocks after sorting is insufficient by itself: block IDs and
row-level auxiliary inputs must follow the permutation, and dependence must
stay within each block. This prototype does not infer or validate those
relationships, and retains the prohibition on automatic ordering.

## Custom likelihood contract

The likelihood must accept *any contiguous chunk of complete blocks*, including
several blocks and the entire dataset. It must identify/reset at internal block
boundaries itself. `block_var` controls scheduling only; it does not automatically
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
small numerical integration test (requires development dependencies and RStan
with threading support). Its Gaussian Markov likelihood genuinely couples adjacent rows
within blocks, checks global row IDs and rejects split boundaries. It compares
serial/block log densities and autodiff gradients at identical unconstrained
parameters, with unequal blocks, singleton blocks, one total block and grainsizes
1 and 100. It also smoke-tests fitting, `fixef`, `ranef`, `posterior_linpred` and
custom `posterior_epred`. The tiny chains are not convergence diagnostics.
The numerical harness uses RStan's public `get_stanmodel()` and
`sampling(..., chains = 0)` interfaces to reuse compiled models with fresh data
without sampling. Public `log_prob()` and `grad_log_prob()` calls compare the
targets at identical unconstrained parameter values. It explicitly enables two
threads for these evaluations, since `brm()` restores RStan's threading option
after each fit. No CmdStan installation or private backend helper is required.

Custom prediction and likelihood post-processing remain the author's job.
This prototype does not add block-level `log_lik`, LOO, joint posterior prediction,
or automatic cross-validation partitions. Do not interpret row-level LOO as
leave-one-block-out validation. New-data prediction can require an updated block
specification if it regenerates threaded Stan data.

The implementation is small enough for an experimental branch and an upstream
API discussion. A stable integration-test harness,
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

## Extended predictor checks

Set `BLOCK_PREDICTOR=extended`, `nonlinear` or `vectorized` when running the
numerical script to exercise splines, monotonic effects, random slopes and
offsets, or rowwise nonlinear expressions with both loop settings. Each scenario
uses the same joint likelihood, alternate partitions, parameter comparisons and
post-processing checks as the original `linear` scenario. The focused tests
also verify that worker bodies match ordinary threading and that shared Stan
data (including global basis construction) match serial generation.

On 2026-11-02, all three additional scenarios passed their initial comparison
and 18 alternate-data/parameter comparisons at tolerance `1e-10`, along with
boundary, alignment, fitting and post-processing checks. The short fits reported
some divergent transitions and remain execution checks only. The revised
focused suite passed 46 assertions; the existing Stan-code and Stan-data suites
again passed 729 and 284 assertions respectively, with the same six skips.
The modified Rd file parsed and `git diff --check` passed.


## Column-based API

The vector-valued `blocks` argument has been replaced by `block_var = "column"`.
The column is resolved from each dataset, validated before model-frame creation,
and appended afterwards only if it was not already retained by the formula.
This leaves model terms, basis construction and predictor contrasts unchanged.
Scheduling-only labels are exempt from predictor type/level checks on new data;
columns also used in the formula retain those checks. All new datasets must
include the column. The column is not automatically passed to the likelihood:
use `vint`/`vreal` and the explicit auxiliary slicing contract when needed.

Only the block path enters the retention helper. Ordinary data validation
continues through its existing body. The earlier unconditional `rows_changed`
model-frame field has been removed; row-change checks live in block validation.

On 2026-11-11, 64 focused assertions passed, including scheduling-only columns,
new datasets with different IDs/row counts, update without recompilation,
missing-column rejection, and preservation of predictor contrasts. Existing
Stan-code, Stan-data, data-helper and brm suites passed 729, 284, 13 and 23
assertions respectively, with eight existing skips. Generated Stan code and
Stan data were identical to the previous vector API for four equivalent
linear, extended and nonlinear predictor scenarios. No new numerical fits were
needed for this API-only change. The modified Rd file parsed and
`git diff --check` passed.


## RStan numerical harness

On 2026-11-11, the numerical harness was switched to RStan public APIs and
rerun with RStan 2.32.7, StanHeaders 2.39.1 and R 4.5. All four scenarios
(`linear`, `extended`, `nonlinear`, `vectorized`) passed their initial comparison
and 18 alternate-data/parameter comparisons each: 76 paired density/gradient
comparisons in total, using `all.equal(..., tolerance = 1e-10)`. Boundary and
auxiliary-alignment checks passed, as did the tiny fits and post-processing
checks. These runs exercise the current column-based API directly.

Each serial/threaded fit reported two divergences in the linear, extended and
nonlinear scenarios, and one in the vectorized scenario, along with short-chain
R-hat/ESS warnings. These remain execution checks, not convergence evidence.
The focused testthat suite also passed all 64 assertions with no warnings or
skips, and `git diff --check` passed. No production code was changed.

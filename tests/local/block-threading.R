# Run from repository root with Rscript tests/local/block-threading.R.
# Requires cmdstanr, CmdStan and the package development dependencies.
devtools::load_all()
set.seed(123)
fam <- custom_family("block_normal", dpars = c("mu", "sigma"),
  links = c("identity", "log"), lb = c(NA, 0), type = "real",
  loop = FALSE, vars = c("vint1[slice]", "vint2[slice]", "global_id"),
  posterior_epred = function(prep) prep$dpars$mu)
# A Gaussian Markov chain resets at each block. Interior conditional means
# depend on the preceding response, so splitting a block changes the density.
fun <- '
real block_normal_lpdf(vector y, vector mu, vector sigma,
                       array[] int id, array[] int row, array[] int global_id) {
  real lp = 0;
  for (n in 1:num_elements(y)) {
    int r = row[n];
    if (id[n] != global_id[r]) reject("misaligned auxiliary data");
    if (n == 1 && r > 1) {
      if (global_id[r] == global_id[r-1]) reject("split block start");
    }
    if (n == num_elements(y) && r < size(global_id)) {
      if (global_id[r] == global_id[r+1]) reject("split block end");
    }
    if (n == 1) lp += normal_lpdf(y[n] | mu[n], sigma[n]);
    else if (id[n] != id[n-1]) lp += normal_lpdf(y[n] | mu[n], sigma[n]);
    else lp += normal_lpdf(y[n] | mu[n] + 0.4 * (y[n-1] - mu[n-1]), sigma[n]);
  }
  return lp;
}'
d <- data.frame(y = rnorm(9), x = seq(-1, 1, length.out = 9),
                g = rep(1:3, 3), id = rep(1:4, c(2, 1, 4, 2)), row = 1:9)
scenario <- Sys.getenv("BLOCK_PREDICTOR", "linear")
d$ord <- ordered(rep(1:3, 3))
f <- switch(scenario,
  linear = bf(y | vint(id, row) ~ x + (1 | g), sigma ~ x),
  extended = bf(y | vint(id, row) ~ s(x, k = 3) + mo(ord) +
                  offset(x) + (x | g), sigma ~ s(x, k = 3)),
  nonlinear = bf(y | vint(id, row) ~ a * exp(b * x),
                 a ~ s(x, k = 3) + (x | g), b ~ mo(ord), sigma ~ x, nl = TRUE),
  vectorized = bf(y | vint(id, row) ~ a + exp(b + x),
                  a ~ s(x, k = 3) + (x | g), b ~ mo(ord), sigma ~ x,
                  nl = TRUE, loop = FALSE),
  stop("Unknown BLOCK_PREDICTOR"))
sv <- stanvar(scode = fun, block = "functions") +
  stanvar(x = as.array(d$id), name = "global_id")
p <- if (scenario %in% c("nonlinear", "vectorized")) {
  prior(normal(0, 1), class = b, nlpar = a) +
    prior(normal(0, 1), class = b, nlpar = b) +
    prior(normal(0, 1), class = b, dpar = sigma)
} else {
  prior(normal(0, 1), class = b) + prior(normal(0, 1), class = b, dpar = sigma)
}
serial <- brm(f, d, family = fam, stanvars = sv, prior = p,
  backend = "cmdstanr", chains = 1, iter = 40, warmup = 20, seed = 123,
  refresh = 0)
blocked <- brm(f, d, family = fam, stanvars = sv, prior = p,
  threads = threading(2, grainsize = 1, static = TRUE, block_var = "id"),
  backend = "cmdstanr", chains = 1, iter = 40, warmup = 20, seed = 123,
  refresh = 0)
sf <- attr(serial$fit, "CmdStanModel")$sample(data = standata(serial),
  chains = 1, iter_warmup = 0, iter_sampling = 1, fixed_param = TRUE, adapt_engaged = FALSE, refresh = 0)
bf <- attr(blocked$fit, "CmdStanModel")$sample(data = standata(blocked),
  chains = 1, iter_warmup = 0, iter_sampling = 1, fixed_param = TRUE, adapt_engaged = FALSE,
  threads_per_chain = 2, refresh = 0)
u <- rep(0.1, dim(sf$unconstrain_draws())[3])
stopifnot(isTRUE(all.equal(sf$log_prob(u), bf$log_prob(u), tolerance = 1e-10)),
          isTRUE(all.equal(sf$grad_log_prob(u), bf$grad_log_prob(u), tolerance = 1e-10)))
stopifnot(all(dim(posterior_epred(blocked)) == c(20, 9)),
          all(dim(posterior_linpred(blocked)) == c(20, 9)),
          nrow(fixef(blocked)) > 0, length(ranef(blocked)) == 1)
# Same compiled models, different data: exercise singletons, a single block,
# unequal blocks and a task containing all blocks, with several parameter points.
for (ids in list(1:9, rep(1L, 9), d$id)) {
  dd <- d; dd$id <- ids
  vv <- stanvar(scode = fun, block = "functions") +
    stanvar(x = as.array(ids), name = "global_id")
  for (grain in c(1L, 100L)) {
    sd <- make_standata(f, dd, family = fam, prior = p, stanvars = vv)
    bd <- make_standata(f, dd, family = fam, prior = p, stanvars = vv,
      threads = threading(2, grainsize = grain, static = TRUE, block_var = "id"))
    # Test-only use of CmdStanR internals to reuse compiled autodiff modules.
    # Reinitialize model data without recompiling the same C++ module 12 times.
    for (pair in list(list(sf, sd), list(bf, bd))) {
      path <- tempfile(fileext = ".json")
      cmdstanr::write_stan_json(pair[[2]], path)
      cmdstanr:::initialize_model_pointer(
        pair[[1]]$.__enclos_env__$private$model_methods_env_, path)
    }
    for (delta in c(0, 0.1, -0.2)) {
      stopifnot(isTRUE(all.equal(sf$log_prob(u + delta), bf$log_prob(u + delta), tolerance = 1e-10)),
                isTRUE(all.equal(sf$grad_log_prob(u + delta), bf$grad_log_prob(u + delta), tolerance = 1e-10)))
    }
  }
}
cat(scenario, ": block density, gradient, boundary, fit and post-processing checks passed.\n")

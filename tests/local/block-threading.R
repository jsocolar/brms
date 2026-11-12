# Run from repository root with Rscript tests/local/block-threading.R.
# Requires RStan with threading support and the package development dependencies.
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
  backend = "rstan", chains = 1, iter = 40, warmup = 20, seed = 123,
  refresh = 0)
blocked <- brm(f, d, family = fam, stanvars = sv, prior = p,
  threads = threading(2, grainsize = 1, static = TRUE, block_var = "id"),
  backend = "rstan", chains = 1, iter = 40, warmup = 20, seed = 123,
  refresh = 0)
# brm() restores RStan's threading option after fitting. Enable two threads
# explicitly for the model instances used by the density/gradient checks.
rstan::rstan_options(threads_per_chain = 2L)
serial_model <- rstan::get_stanmodel(serial$fit)
block_model <- rstan::get_stanmodel(blocked$fit)
# chains = 0 initializes a model with data without running a sampler.
sf <- rstan::sampling(serial_model, data = standata(serial), chains = 0,
                      seed = 123)
bf <- rstan::sampling(block_model, data = standata(blocked), chains = 0,
                      seed = 123)
u <- rep(0.1, length(rstan::unconstrain_pars(
  serial$fit, rstan::get_inits(serial$fit)[[1]])))
compare_targets <- function(serial_fit, block_fit, u) {
  stopifnot(isTRUE(all.equal(rstan::log_prob(serial_fit, u),
                            rstan::log_prob(block_fit, u), tolerance = 1e-10)),
            isTRUE(all.equal(rstan::grad_log_prob(serial_fit, u),
                            rstan::grad_log_prob(block_fit, u), tolerance = 1e-10)))
}
compare_targets(sf, bf, u)
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
    # Reuse compiled models with fresh data through RStan's public interface.
    sf <- rstan::sampling(serial_model, data = sd, chains = 0, seed = 123)
    bf <- rstan::sampling(block_model, data = bd, chains = 0, seed = 123)
    for (delta in c(0, 0.1, -0.2)) {
      compare_targets(sf, bf, u + delta)
    }
  }
}
cat(scenario, ": block density, gradient, boundary, fit and post-processing checks passed.\n")

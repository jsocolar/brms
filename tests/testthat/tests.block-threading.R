context("Block threading")

block_fixture <- function() {
  list(
    data = data.frame(y = 1:6, x = 1:6, g = rep(1:3, each = 2),
                      id = c(1, 1, 2, 3, 3, 3)),
    family = custom_family("joint", dpars = c("mu", "sigma"),
      links = c("identity", "log"), lb = c(NA, 0), type = "real",
      loop = FALSE, vars = c("vint1[slice]", "shared")),
    formula = bf(y | vint(id) ~ x + (1 | g), sigma ~ x))
}

test_that("block scheduling reuses row slicing for every predictor", {
  z <- block_fixture()
  th <- threading(2, block_var = "id", grainsize = 2)
  s <- make_stancode(z$formula, z$data, family = z$family, threads = th)
  expect_match(s, "int start = block_start[block_first];", fixed = TRUE)
  expect_match(s, "int end = block_end[block_last];", fixed = TRUE)
  expect_match(s, "array[N_blocks] int seq = sequence(1, N_blocks)", fixed = TRUE)
  expect_match(s, "Xc[start:end] * b", fixed = TRUE)
  expect_match(s, "Xc_sigma[start:end] * b_sigma", fixed = TRUE)
  expect_match(s, "r_1_1[J_1[nn]]", fixed = TRUE)
  expect_match(s, "joint_lpdf(Y[start:end] | mu, sigma, vint1[start:end], shared)", fixed = TRUE)
  sd <- make_standata(z$formula, z$data, family = z$family, threads = th)
  expect_equal(as.vector(sd$block_start), c(1, 3, 4))
  expect_equal(as.vector(sd$block_end), c(2, 3, 6))
  expect_equal(sd$grainsize, 2)
  for (ids in list(rep("a", 6), letters[1:6], factor(z$data$id))) {
    z$data$block <- ids
    sd <- make_standata(z$formula, z$data, family = z$family,
                       threads = threading(2, block_var = "block"))
    expect_equal(sd$N_blocks, length(unique(ids)))
    expect_equal(tail(as.vector(sd$block_end), 1), 6)
  }
  serial <- make_stancode(z$formula, z$data, family = z$family)
  expect_match(serial, "joint_lpdf(Y | mu, sigma, vint1, shared)", fixed = TRUE)
  ordinary <- make_stancode(z$formula, z$data, family = z$family, threads = 2)
  expect_match(ordinary, "array[N] int seq = sequence(1, N)", fixed = TRUE)
  expect_false(grepl("block_start", ordinary, fixed = TRUE))
})

test_that("invalid partitions and unsupported models fail explicitly", {
  for (name in list(character(), NA_character_, "", c("a", "b"), 1)) {
    expect_error(threading(2, block_var = name), "column name")
  }
  expect_error(threading(block_var = "id"), "active")
  expect_error(threading(2, block_var = "id", force = TRUE), "active")
  expect_error(threading(2, blocks = 1:3), "unused argument")
  z <- block_fixture()
  for (ids in list(c(1, 2, 1, 3, 3, 3), c(1, NA, 2, 3, 3, 3),
                   c(1, Inf, 2, 3, 3, 3), rep(TRUE, 6))) {
    z$data$block <- ids
    expect_error(make_standata(z$formula, z$data, family = z$family,
                 threads = threading(2, block_var = "block")), "contiguous|finite IDs")
  }
  expect_error(make_standata(z$formula, z$data, family = z$family,
               threads = threading(2, block_var = "missing")), "must be present")
  z$data$x[1] <- NA
  expect_error(suppressWarnings(make_standata(z$formula, z$data, family = z$family,
               threads = threading(2, block_var = "id"))), "missing-row")
  z <- block_fixture()
  th <- threading(2, block_var = "id")
  expect_error(make_stancode(y ~ x, z$data, threads = th), "custom family")
  looped <- z$family
  looped$loop <- TRUE
  expect_error(make_stancode(z$formula, z$data, family = looped, threads = th),
               "loop = FALSE")
  expect_error(suppressWarnings(make_stancode(mvbind(y, x) ~ 1, z$data, threads = th)),
               "single-response")
  expect_error(make_stancode(z$formula, z$data, family = z$family,
               threads = threading(2, block_var = "missing")), "must be present")
  expect_error(make_stancode(y | subset(x > 2) ~ x, z$data,
               family = z$family, threads = th), "response additions")
  # Existing threading guards still apply, including when ordering is absent.
  expect_error(make_stancode(bf(y ~ x, sparse = TRUE), z$data,
               family = z$family, threads = th), "sparse matrices")
  z$data$t <- c(2, 1, 2, 1, 2, 1)
  expect_error(make_stancode(y ~ x + ar(time = t, gr = g), z$data,
               family = z$family, threads = th), "reordering")
})

test_that("block predictors inherit ordinary threading support", {
  z <- block_fixture()
  z$data$ord <- ordered(rep(1:3, 2))
  formulas <- list(
    bf(y | vint(id) ~ s(x, k = 3) + mo(ord) + offset(x) + (x | g),
       sigma ~ s(x, k = 3)),
    bf(y | vint(id) ~ a * exp(b * x), a ~ s(x, k = 3) + (x | g),
       b ~ mo(ord), sigma ~ x, nl = TRUE),
    bf(y | vint(id) ~ a + exp(b + x), a ~ s(x, k = 3) + (x | g),
       b ~ mo(ord), sigma ~ x, nl = TRUE, loop = FALSE)
  )
  for (f in formulas) {
    ordinary <- make_stancode(f, z$data, family = z$family, threads = 2)
    block <- make_stancode(f, z$data, family = z$family,
                          threads = threading(2, block_var = "id"))
    # Apart from scheduling declarations, the entire worker body is identical.
    body <- function(code) {
      begin <- regexpr("    real ptarget = 0;", code, fixed = TRUE)[1]
      end <- regexpr("    return ptarget;", code, fixed = TRUE)[1]
      substr(code, begin, end)
    }
    expect_true(nchar(body(block)) > 100)
    expect_identical(body(block), body(ordinary))
    sd <- make_standata(f, z$data, family = z$family)
    bd <- make_standata(f, z$data, family = z$family,
                       threads = threading(2, block_var = "id"))
    expect_equal(lapply(bd[names(sd)], identity), lapply(sd, identity))
  }
})


test_that("a scheduling-only column is retained through fit and new data", {
  z <- block_fixture()
  f <- bf(y ~ x, sigma ~ x)
  # id is deliberately absent from every formula and custom auxiliary argument.
  fam <- custom_family("joint", dpars = c("mu", "sigma"),
    links = c("identity", "log"), lb = c(NA, 0), type = "real", loop = FALSE)
  fit <- brm(f, z$data, family = fam, threads = threading(2, block_var = "id"),
             empty = TRUE)
  expect_identical(fit$data$id, z$data$id)
  expect_identical(fit$threads$block_var, "id")
  expect_null(fit$threads$blocks)
  expect_equal(standata(fit)$N_blocks, 3)
  nd <- z$data[1:4, ]
  nd$id <- c("a", "b", "b", "b")
  expect_equal(standata(fit, newdata = nd)$N_blocks, 2)
  expect_identical(validate_newdata(nd, fit)$id, nd$id)
  # Switching algorithm avoids reading sampling metadata from this empty fit.
  up <- update(fit, newdata = nd, recompile = FALSE, algorithm = "meanfield",
               testmode = TRUE)
  expect_identical(up$data$id, nd$id)
  expect_equal(standata(up)$N_blocks, 2)
  expect_error(standata(fit, newdata = nd[c("y", "x")]), "must be present")
  expect_error(update(fit, newdata = nd[c("y", "x")], recompile = FALSE,
                      algorithm = "meanfield", testmode = TRUE), "must be present")
  nd$x[1] <- NA
  expect_error(suppressWarnings(make_stancode(f, nd, family = fam,
    threads = threading(2, block_var = "id"))), "missing-row")
  plain <- brm(f, z$data, family = fam, empty = TRUE)
  expect_null(plain$data$id)
  expect_equal(make_standata(f, z$data, family = fam)$X, standata(fit)$X)
})


test_that("block columns used as predictors keep their global contrasts", {
  z <- block_fixture()
  z$data$id <- factor(z$data$id)
  f <- bf(y ~ id, sigma ~ x)
  serial <- make_standata(f, z$data, family = z$family)
  blocked <- make_standata(f, z$data, family = z$family,
                          threads = threading(2, block_var = "id"))
  expect_identical(serial$X, blocked$X)
  expect_identical(serial$X_sigma, blocked$X_sigma)
})

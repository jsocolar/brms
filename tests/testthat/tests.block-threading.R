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
  th <- threading(2, blocks = z$data$id, grainsize = 2)
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
    sd <- make_standata(z$formula, z$data, family = z$family,
                       threads = threading(2, blocks = ids))
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
  expect_error(threading(2, blocks = c(1, 2, 1)), "contiguous")
  for (ids in list(integer(), c(1, NA), c(1, Inf), list(1, 2), matrix(1:4, 2), TRUE)) {
    expect_error(threading(2, blocks = ids), "blocks must")
  }
  expect_error(threading(blocks = 1:3), "active")
  expect_error(threading(2, blocks = 1:3, force = TRUE), "active")
  z <- block_fixture()
  expect_error(make_standata(z$formula, z$data, family = z$family,
               threads = threading(2, blocks = 1:3)), "match all model rows")
  z$data$x[1] <- NA
  expect_error(suppressWarnings(make_standata(z$formula, z$data, family = z$family,
               threads = threading(2, blocks = c(1, 1, 2, 3, 3, 3)))), "missing-row")
  z <- block_fixture()
  th <- threading(2, blocks = z$data$id)
  expect_error(make_stancode(y ~ x, z$data, threads = th), "custom family")
  looped <- z$family
  looped$loop <- TRUE
  expect_error(make_stancode(z$formula, z$data, family = looped, threads = th),
               "loop = FALSE")
  expect_error(suppressWarnings(make_stancode(mvbind(y, x) ~ 1, z$data, threads = th)),
               "single-response")
  expect_error(make_stancode(z$formula, z$data, family = z$family,
               threads = threading(2, blocks = 1:5)), "match all model rows")
  expect_error(make_stancode(y | subset(x > 2) ~ x, z$data,
               family = z$family, threads = th), "response additions")
  expect_error(make_stancode(y ~ s(x, k = 3), z$data, family = z$family, threads = th),
               "ordinary linear")
  expect_error(make_stancode(y ~ x + (x | g), z$data, family = z$family, threads = th),
               "random intercepts")
})

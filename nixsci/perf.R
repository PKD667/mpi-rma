# The ring against MPI_Send, by payload size. An analysis pipeline: it reads the locked runs of the
# `compare` experiment through nixsci's use() and draws one SVG. Medians and 95% percentile-bootstrap
# intervals come from the raw rows (2000 draws, seed fixed per cell). Nix substitutes @fontsconf@
# with a fontconfig file naming only the pinned Hack font.
suppressPackageStartupMessages({ library(dplyr); library(nixsci); library(ggplot2); library(grid) })
Sys.setenv(FONTCONFIG_FILE = "@fontsconf@", XDG_CACHE_HOME = tempdir())
fam <- "Hack"

cmp  <- use("compare")
raw  <- cmp$compare |> collect() |> filter(noise_kib == 0)
host <- cmp$host |> collect() |> slice(1)
reps <- n_distinct(raw$rep)

seedof <- function(...) { v <- utf8ToInt(paste(..., sep = "|")); as.integer(sum(v * seq_along(v)) %% 2147483647L) }
cells <- function(d, value, tag) {
  d |> filter(transport %in% c("ring-safe", "p2p-send")) |>
    group_by(transport, payload) |>
    group_modify(\(g, k) {
      set.seed(seedof(tag, k$transport, k$payload))
      m <- replicate(2000, median(sample(g[[value]], replace = TRUE)))
      data.frame(median = median(g[[value]]), lo = unname(quantile(m, 0.025)), hi = unname(quantile(m, 0.975)))
    }) |> ungroup()
}
good <- cells(filter(raw, measure == "stream"), "goodput_MiB_per_s", "goodput")
lat  <- cells(filter(raw, measure == "rtt"), "us_per_msg", "latency")
write.csv(rbind(cbind(measure = "goodput_MiB_per_s", good), cbind(measure = "latency_us", lat)),
          out("summary.csv"), row.names = FALSE)

ink <- "#1f2937"; muted <- "#6b7280"; rule <- "#e5e7eb"; accent <- "#3730a3"; slow <- "#d97706"
payload <- sort(unique(raw$payload))
size <- function(b) ifelse(b < 1024, paste(b, "B"), paste(b / 1024, "KiB"))

# ---- ring: one slot per payload size, radius = log2(ring / MPI_Send throughput) --------------------
ratio <- good$median[good$transport == "ring-safe"] / good$median[good$transport == "p2p-send"]
R0 <- 1; K <- 0.32; rad <- function(x) R0 + K * log2(x)
sector <- function(t1, t2, r1, r2, n = 30) {
  t <- seq(t1, t2, length.out = n) * pi / 180
  data.frame(x = c(r1 * cos(t), rev(r2 * cos(t))), y = c(r1 * sin(t), rev(r2 * sin(t))))
}
centre <- 90 - 60 * seq_along(payload); half <- 25
bars <- do.call(rbind, lapply(seq_along(payload), function(i) {
  d <- sector(centre[i] - half, centre[i] + half, R0, rad(ratio[i])); d$id <- i; d$fast <- ratio[i] >= 1; d }))
circle <- function(r) { t <- seq(0, 2 * pi, length.out = 240); data.frame(x = r * cos(t), y = r * sin(t)) }
track <- do.call(rbind, lapply(seq_along(centre), \(i) cbind(sector(centre[i] - half, centre[i] + half, rad(0.2), rad(2.9)), id = i)))
RL <- rad(4.2)
lab <- data.frame(x = RL * cos(centre * pi / 180), y = RL * sin(centre * pi / 180),
                  txt = sprintf("%s\n%.2f\u00d7", size(payload), ratio), fast = ratio >= 1)
p_ring <- ggplot() +
  geom_polygon(data = track, aes(x, y, group = id), fill = "#f3f4f6") +
  geom_path(data = circle(R0), aes(x, y), colour = ink, linewidth = 0.5, linetype = "22") +
  geom_polygon(data = bars, aes(x, y, group = id, fill = fast)) +
  geom_text(data = lab, aes(x, y, label = txt, colour = fast), family = fam, size = 3.2, lineheight = 1.1, fontface = "bold") +
  annotate("text", x = 0, y = R0 + 0.09, label = "MPI_Send", family = fam, size = 2.7, colour = muted) +
  scale_fill_manual(values = c(`TRUE` = accent, `FALSE` = slow), guide = "none") +
  scale_colour_manual(values = c(`TRUE` = accent, `FALSE` = slow), guide = "none") +
  coord_fixed(xlim = c(-1.7, 1.7), ylim = c(-1.7, 1.7), expand = FALSE, clip = "off") +
  theme_void() + theme(plot.background = element_blank())

# ---- curves ----------------------------------------------------------------------------------------
cols <- c(`ring-safe` = accent, `p2p-send` = ink)
curve <- function(d, title, labels) {
  end <- d[d$payload == max(d$payload), ]; end$name <- labels[as.character(end$transport)]
  ggplot(d, aes(payload, median, colour = transport, fill = transport)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.16, colour = NA) +
    geom_line(linewidth = 0.8) + geom_point(size = 1.6) +
    geom_text(data = end, aes(label = name), hjust = 0, nudge_x = 0.07, size = 2.8, family = fam, show.legend = FALSE) +
    scale_colour_manual(values = cols) + scale_fill_manual(values = cols) +
    scale_x_log10(breaks = payload, labels = size(payload), expand = expansion(mult = c(0.04, 0.22))) +
    scale_y_log10(labels = \(v) format(v, big.mark = ",", scientific = FALSE, trim = TRUE)) +
    labs(x = NULL, y = NULL, title = title) +
    theme_minimal(base_family = fam, base_size = 8.5) +
    theme(panel.grid.minor = element_blank(), panel.grid.major = element_line(colour = rule, linewidth = 0.3),
          axis.text = element_text(colour = muted, size = 7), plot.title = element_text(colour = ink, face = "bold", size = 8.5),
          legend.position = "none", plot.background = element_blank(), panel.background = element_blank(),
          plot.margin = margin(4, 6, 2, 4))
}
labels <- c(`ring-safe` = "ring", `p2p-send` = "MPI_Send")
p_good <- curve(good, "Throughput, MiB/s (higher is better)", labels)
p_lat  <- curve(lat,  "One-way latency, \u00b5s (lower is better)", labels)

# ---- page ------------------------------------------------------------------------------------------
H <- 5.4
svg(out("mpi-rma-perf.svg"), width = 10.4, height = H, family = fam)
grid.newpage()
from_top <- function(s, inches, size, col, face = "plain", x = 0.2)
  grid.text(s, x = unit(x, "in"), y = unit(H - inches, "in"), just = c("left", "top"),
            gp = gpar(fontfamily = fam, fontsize = size, col = col, fontface = face))
from_top("mpi-rma: the ring, measured against MPI_Send", 0.18, 12, ink, "bold")
from_top(sprintf("%s, %d threads, %s. Median of %d repeats.", host$cpu, host$threads, host$mpi, reps), 0.48, 7.6, muted)
from_top("Ring throughput relative to MPI_Send", 0.95, 8.5, ink, "bold")
from_top("Outward: the ring is faster. Inward: slower.", 1.17, 7.6, muted)
pushViewport(viewport(y = 0, height = unit(H - 1.0, "in"), just = "bottom",
                      layout = grid.layout(1, 2, widths = unit(c(4.6, 5.8), "in"))))
print(p_ring, vp = viewport(layout.pos.row = 1, layout.pos.col = 1, y = 0.46, height = 0.9))
pushViewport(viewport(layout.pos.row = 1, layout.pos.col = 2))
print(p_good, vp = viewport(y = 0.76, height = 0.48))
print(p_lat,  vp = viewport(y = 0.26, height = 0.48))
popViewport(2)
invisible(dev.off())

# Loss and scale tables from the locked runs of the `loss` and `scale` experiments.
suppressPackageStartupMessages({ library(dplyr); library(nixsci) })

loss <- use("loss")$soak |> collect() |>
  group_by(depth, pace_ns, noise_kib, rep) |>
  summarise(lost_pct = 100 * sum(lost) / sum(sent), .groups = "drop") |>
  group_by(depth, pace_ns, noise_kib) |>
  summarise(runs = n(), median_lost_pct = median(lost_pct), .groups = "drop")
write.csv(loss, out("loss.csv"), row.names = FALSE)

scale <- use("scale")$soak |> collect() |>
  group_by(ranks, noise_kib, rep) |>
  summarise(delivered_pct = 100 * sum(received) / sum(sent), max_waits = max(waits), .groups = "drop") |>
  group_by(ranks, noise_kib) |>
  summarise(runs = n(), min_delivered_pct = min(delivered_pct), max_waits = max(max_waits), .groups = "drop")
write.csv(scale, out("scale.csv"), row.names = FALSE)
cat(nrow(loss), "loss rows,", nrow(scale), "scale rows\n")

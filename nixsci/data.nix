# The datasets the three benchmarks record, one per bench.sh output file. Columns are the TSV
# headers of examples/compare.rs and examples/soak.rs.
{
  compare = {
    key = [ "transport" "measure" "ranks" "depth" "noise_kib" "rep" "payload" ];
    columns = {
      transport = "str"; # ring-safe, ring-raw, p2p-send, p2p-bsend
      measure = "str"; # stream (throughput) or rtt (round trip, latency = rtt / 2)
      ranks = "int";
      depth = "int"; # ring slots
      noise_kib = "int"; # background traffic between the other ranks, KiB per message, 0 = quiet
      rep = "int";
      payload = "int"; # bytes
      sent = "int";
      delivered = "int";
      inject_s = "float";
      total_s = "float";
      wait_s = "float";
      inject_per_s = "float";
      goodput_per_s = "float";
      goodput_MiB_per_s = "float";
      us_per_msg = "float";
      lost = "int";
      max_us_per_msg = "float";
    };
  };

  # What the run observed about its host, one row per run, so a figure can name the machine it was measured on.
  host = {
    key = [ "cpu" ];
    columns = {
      cpu = "str";
      threads = "int";
      mem_kb = "int";
      kernel = "str";
      mpi = "str";
    };
  };

  soak = {
    key = [ "mode" "ranks" "depth" "payload" "noise_kib" "pace_ns" "rep" "rank" ];
    columns = {
      mode = "str"; # safe or raw
      ranks = "int";
      depth = "int";
      payload = "int";
      noise_kib = "int";
      pace_ns = "int"; # offered rate: one round per pace_ns, 0 = flat out
      rep = "int";
      rank = "int";
      sent = "int";
      received = "int";
      lost = "int";
      corrupt = "int";
      max_lag = "int";
      waits = "int";
      wait_s = "float";
      seconds = "float";
    };
  };
}

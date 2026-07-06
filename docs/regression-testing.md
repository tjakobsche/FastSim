# Regression testing with a synthetic workload

FastSim is deterministic: the same input workload always produces exactly
the same result pickle. That makes a strong regression test possible
without any site data — generate a synthetic workload, run the simulator
before and after a change, and assert the results are exactly equal.

## Generate a workload

```sh
python scripts/generate_synthetic_workload.py --outdir synthetic_workload
```

This writes a self-contained run directory (job trace, association and QOS
dumps, `slurm.conf`, empty event/reservation stubs) and two configs:

- `synthetic_conf_fifo.yaml` — no priority weights, jobs run in submit
  order (the default scheduler configuration).
- `synthetic_conf_weighted.yaml` — age, QOS, and job-size priority
  weights enabled, exercising the multifactor priority sort.

The generator is seeded (`--seed`, default 42) and byte-deterministic:
the same arguments always produce identical files. The default workload
saturates the cluster so backfill, per-user QOS node quotas, and
queue-time cancellations are all exercised. `synthetic_workload/` is
gitignored; regenerate it rather than committing it.

## Run and compare

```sh
cd scheduler
python main.py ../synthetic_workload/synthetic_conf_fifo.yaml --output ../before.pkl
# ... apply your change ...
python main.py ../synthetic_workload/synthetic_conf_fifo.yaml --output ../after.pkl
cd ..
python scripts/compare_results.py before.pkl after.pkl
```

`compare_results.py` exits 0 only if the two job histories are exactly
equal (same columns, dtypes, and values); otherwise it lists the differing
columns. Run both configs — some code paths are only reached with priority
weights enabled.

For a pure performance change, results must be exactly equal. For a
behavioural change, the diff shows precisely which job outcomes moved.

#!/usr/bin/env python3
"""
Generate a synthetic FastSim workload for regression testing.

Emits a self-contained run directory (job trace, association/QOS dumps,
slurm.conf, header-only event/reservation stubs) plus two config files:
a FIFO config with no priority weights and a weighted multifactor config.
Everything is derived from a single seeded random.Random, so the same
arguments always produce byte-identical output. FastSim itself is
deterministic, which makes exact result comparison meaningful:

    python scripts/generate_synthetic_workload.py --outdir synthetic_workload
    (cd scheduler && python main.py ../synthetic_workload/synthetic_conf_fifo.yaml \
        --output before.pkl)
    ... apply a change, rerun to after.pkl ...
    python scripts/compare_results.py before.pkl after.pkl

The default workload oversubscribes the cluster by ~1.5x so jobs queue up
and backfill matters, and it exercises the main scheduler code paths:
multiple partitions on overlapping node subsets, several QOS including one
with a per-user node quota (holds), and queue-time cancellations.
"""

import argparse
import datetime
import random
from pathlib import Path


JOBS_HEADER = (
    "User|Account|AllocNodes|ConsumedEnergyRaw|ExitCode|Flags|JobID|JobName|"
    "Partition|QOS|Reason|ReqNodes|Start|State|End|Elapsed|Submit|SubmitLine|Timelimit|"
)

# Fixed, arbitrary trace epoch. The simulation window is derived from the
# job dump, so the absolute dates carry no meaning.
EPOCH = datetime.datetime(2024, 1, 1)

# (name, priority, MaxTRESPU node quota or None, share of jobs)
QOS_MIX = (
    ("normal", 1000, None, 0.70),
    ("high", 5000, None, 0.15),
    ("standby", 0, 32, 0.15),
)

TIMELIMITS = (1800, 3600, 7200, 14400)
TIMELIMIT_WEIGHTS = (30, 35, 25, 10)
DEBUG_TIMELIMITS = (900, 1800, 3600)
DEBUG_TIMELIMIT_WEIGHTS = (30, 50, 20)


def fmt_time(seconds):
    """Epoch offset in whole seconds -> sacct timestamp string."""
    return (EPOCH + datetime.timedelta(seconds=int(seconds))).strftime(
        "%Y-%m-%dT%H:%M:%S")


def fmt_timelimit(seconds):
    days, rem = divmod(int(seconds), 86400)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hrs:02}:{mins:02}:{secs:02}"
    return f"{hrs:02}:{mins:02}:{secs:02}"


def partition_layout(n_nodes):
    """standard spans the pool, large the first third, debug the last 8."""
    return {
        "standard": (1, n_nodes),
        "large": (1, max(n_nodes // 3, 32)),
        "debug": (max(n_nodes - 7, 1), n_nodes),
    }


def draw_job(rng, users, standby_users):
    """Draw one job's static properties (everything except times)."""
    p = rng.random()
    if p < 0.85:
        partition = "standard"
        req_nodes = int(2 ** rng.uniform(0, 3))  # 1..8
        timelimit = rng.choices(TIMELIMITS, weights=TIMELIMIT_WEIGHTS)[0]
    elif p < 0.98:
        partition = "large"
        req_nodes = int(2 ** rng.uniform(3, 5))  # 8..32
        timelimit = rng.choices(TIMELIMITS, weights=TIMELIMIT_WEIGHTS)[0]
    else:
        partition = "debug"
        req_nodes = rng.randint(1, 2)
        timelimit = rng.choices(DEBUG_TIMELIMITS, weights=DEBUG_TIMELIMIT_WEIGHTS)[0]

    q = rng.random()
    acc = 0.0
    qos = QOS_MIX[0][0]
    for name, _prio, _cap, share in QOS_MIX:
        acc += share
        if q < acc:
            qos = name
            break

    # Concentrate standby jobs on a few users so the per-user node quota
    # (MaxTRESPU) actually engages the hold logic.
    user = rng.choice(standby_users) if qos == "standby" else rng.choice(users)

    s = rng.random()
    if s < 0.92:
        state = "COMPLETED"
    elif s < 0.935:
        state = "FAILED"
    elif s < 0.95:
        state = "TIMEOUT"
    else:
        state = "CANCELLED"

    if state == "TIMEOUT":
        elapsed = timelimit
    elif state == "CANCELLED":
        elapsed = 0
    else:
        elapsed = max(60, int(rng.uniform(0.05, 0.98) * timelimit))

    return partition, req_nodes, timelimit, elapsed, qos, user, state


def generate(args):
    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    users = [f"user{i + 1:03}" for i in range(args.users)]
    accounts = [f"acct{i + 1:02}" for i in range(args.accounts)]
    user_account = {u: accounts[i % args.accounts] for i, u in enumerate(users)}
    standby_users = users[: max(args.users // 10, 1)]

    # ---- draw all jobs, spread submissions over the requested window
    jobs = [draw_job(rng, users, standby_users) for _ in range(args.jobs)]
    submit_span = args.days * 86400
    gaps = [rng.expovariate(1.0) for _ in range(args.jobs)]
    scale = submit_span / sum(gaps)
    submits = []
    t = 0.0
    for gap in gaps:
        t += gap * scale
        submits.append(int(t))

    total_node_seconds = sum(req * elapsed for _, req, _, elapsed, _, _, _ in jobs)
    load = total_node_seconds / (submit_span * args.nodes)

    # ---- job trace
    lines = [JOBS_HEADER]
    for i, (partition, req, timelimit, elapsed, qos, user, state) in enumerate(jobs):
        jid = 1001 + i
        submit = submits[i]
        if state == "CANCELLED":
            # Cancelled while pending: AllocNodes=0 and Start==End is the
            # reader's convention for queue-time cancellations.
            alloc = 0
            start = end = submit + int(rng.uniform(60, 4 * 3600))
        else:
            alloc = req
            start = submit + int(rng.uniform(0, 6 * 3600))
            end = start + elapsed
        lines.append(
            f"{user}|{user_account[user]}|{alloc}||0:0||{jid}|job{jid}|"
            f"{partition}|{qos}|None|{req}|{fmt_time(start)}|{state}|"
            f"{fmt_time(end)}|{fmt_timelimit(elapsed)}|{fmt_time(submit)}|"
            f"sbatch job{jid}.sh|{fmt_timelimit(timelimit)}|"
        )
    (outdir / "sacct_jobs.csv").write_text("\n".join(lines) + "\n")

    # ---- associations and QOS
    lines = ["User|Account|ParentName|Partition|Shares|MaxJobs|MaxSubmit|"]
    for acc in accounts:
        lines.append(f"|{acc}|root||1|||")
    for user in users:
        lines.append(f"{user}|{user_account[user]}|||1|||")
    (outdir / "sacctmgr_assocs.csv").write_text("\n".join(lines) + "\n")

    lines = ["Name|Priority|GrpTRES|GrpJobs|GrpSubmit|MaxTRESPU|MaxJobsPU|"
             "MaxJobsPA|MaxSubmitPU|MaxSubmitPA|MaxSubmit|MaxJobs|"]
    for name, prio, cap, _share in QOS_MIX:
        quota = f"node={cap}" if cap else ""
        lines.append(f"{name}|{prio}||||{quota}|||||||")
    (outdir / "sacctmgr_qos.csv").write_text("\n".join(lines) + "\n")

    # ---- header-only stubs (no node outages, no reservations)
    (outdir / "sacctmgr_events.csv").write_text(
        "NodeName|TimeStart|TimeEnd|State|Reason|\n")
    (outdir / "sinfo_resv.csv").write_text(
        "RESV_NAME|STATE|START_TIME|END_TIME|DURATION|NODELIST|\n")
    (outdir / "sreport_resv.csv").write_text("Name|Nodes|Start|End|\n")

    # ---- cluster definition
    layout = partition_layout(args.nodes)
    lines = [f"NodeName=node[{1:04}-{args.nodes:04}] Weight=1"]
    for name in ("standard", "large", "debug"):
        lo, hi = layout[name]
        lines.append(
            f"PartitionName={name} Nodes=node[{lo:04}-{hi:04}] "
            f"PriorityTier=1 PriorityJobFactor=1 State=UP"
        )
    (outdir / "slurm.conf").write_text("\n".join(lines) + "\n")

    # ---- configs (paths resolve relative to the config file)
    common = """\
# Synthetic regression workload generated by generate_synthetic_workload.py
assocs_dump: "sacctmgr_assocs.csv"
qos_dump: "sacctmgr_qos.csv"
node_events_dump: "sacctmgr_events.csv"
resv_dump_current: "sinfo_resv.csv"
resv_dump_historic: "sreport_resv.csv"
slurm_conf: "slurm.conf"
job_dump: "sacct_jobs.csv"

considered_partitions: ["standard", "large", "debug"]
"""
    (outdir / "synthetic_conf_fifo.yaml").write_text(common)
    (outdir / "synthetic_conf_weighted.yaml").write_text(common + """
PriorityWeightAge: 1000
PriorityWeightQOS: 100000
PriorityWeightJobSize: 100000
""")

    n_cancel = sum(1 for job in jobs if job[6] == "CANCELLED")
    print(f"Wrote {args.jobs} jobs ({n_cancel} queue-cancelled) submitted "
          f"over {args.days} day(s), offered load {load:.2f}x of "
          f"{args.nodes} nodes, to {outdir}/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a deterministic synthetic FastSim workload.")
    parser.add_argument("--outdir", default="synthetic_workload")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=4000)
    parser.add_argument("--nodes", type=int, default=400)
    parser.add_argument("--days", type=int, default=1,
                        help="Length of the submission window")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--accounts", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())

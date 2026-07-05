# MIT License
#
# Copyright (c) 2023-2025 Hewlett Packard Enterprise Development LP 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from datetime import timedelta
from collections import namedtuple

import yaml

# NOTE Defaults below match stock Slurm defaults (slurm.conf man page) unless marked
# simulator-specific. Slurm.conf and the YAML config can override any of them.
# NOTE: approx_excess_assocs remove a number of unused in workload traceassocs from the assoc tree,
# this is relevant since the fairshare factor scales with the tot number of user assocs. This
# happend because to capture all assocs they need to be dumped "withDeleted" so you end up with
# some assocs that never existed at any given time.
defaults = {
    "defer" : True, # Setting this option will avoid attempting to schedule each job individually
                     # at job submit time, but defer it until a later time when scheduling multiple jobs
                     # simultaneously may be possible.
                     # NOTE: Stock Slurm has defer OFF by default, but the simulation loop hardcodes
                     # defer-on behavior and never reads this value (see controller.run_sim).
    "default_queue_depth" : 100, # The default number of jobs to attempt scheduling (i.e. the queue depth) 
                                 # when a running job completes or other routine actions occur
    "sched_interval" : 60, # How frequently, in seconds, the main scheduling loop will execute and test all pending jobs
    "sched_min_interval" : 2, # How frequently, in microseconds, the main scheduling loop will execute and test any pending jobs.
                              # Stock Slurm default is 2 microseconds; ARCHER2 used 2000000 (2 s).
                                    # The scheduler runs in a limited fashion every time that any event happens which could enable a job 
                                    # to start (e.g. job submit, job terminate, etc.). If these events happen at a high frequency, the 
                                    # scheduler can run very frequently and consume significant resources if not throttled by this option. 
                                    # This option specifies the minimum time between the end of one scheduling cycle and the beginning of 
                                    # the next scheduling cycle.
    "bf_resolution" : 60, # The number of seconds in the resolution of data maintained about when jobs begin and end.
    "bf_max_job_test" : 500, # The maximum number of jobs to attempt backfill scheduling for (i.e. the queue depth).
    "bf_window" : 1440, # The number of minutes into the future to look when considering jobs to schedule.
    "bf_interval" : 30, # The number of seconds between backfill iterations.
    "bf_max_time" : 30, # The maximum time in seconds the backfill scheduler can spend (including time spent sleeping when locks are released) 
                        # before discontinuing, even if maximum job counts have not been reached.
    "bf_yield_interval" : 2000000, # The backfill scheduler will periodically relinquish locks in order for other pending operations to take place.
                                   # This specifies the times when the locks are relinquished in microseconds.
    "bf_yield_sleep" : 500000, # The backfill scheduler will periodically relinquish locks in order for other pending operations to take place. 
                               # This specifies the length of time for which the locks are relinquished in microseconds.
    "bf_continue" : True, # Setting this option will cause the backfill scheduler to continue processing pending jobs from its original job list
                           # after releasing locks even if job or node state changes.
                           # NOTE: Stock Slurm has bf_continue OFF by default, but the simulation loop hardcodes
                           # bf_continue-on behavior and never reads this value (see controller.run_sim).
    
    "PriorityCalcPeriod" : 5, # The period of time in minutes in which the half-life decay will be re-calculated.
    "PriorityMaxAge" : 7, # Specifies the job age which will be given the maximum age factor in computing priority. 
                          # For example, a value of 30 minutes would result in all jobs over 30 minutes old would 
                          # get the same age-based priority.
    "PriorityDecayHalfLife" : 7, # This controls how long prior resource use is considered in determining how over- 
                                 # or under-serviced an association is (user, bank account and cluster) in determining 
                                 # job priority. The default value is 7 days.
    "PriorityWeightAge" : 0, # An integer value that sets the degree to which the queue wait time component contributes to the job's priority.
    "PriorityWeightFairshare" : 0, # An integer value that sets the degree to which the fair-share component contributes to the job's priority.
    "PriorityWeightJobSize" : 0, # An integer value that sets the degree to which the job size component contributes to the job's priority.
    "PriorityWeightPartition" : 0, # Partition factor used by priority/multifactor plugin in calculating job priority.
    "PriorityWeightQOS" : 0, # An integer value that sets the degree to which the Quality Of Service component contributes to the job's priority.
    "PriorityWeightPower" : 0, # An integer value that sets the degree to which the Power component contributes to the job's priority.

    "JobRequeue" : 1, # This option controls the default ability for batch jobs to be requeued. Jobs may be requeued explicitly by a system 
                      # administrator, after node failure, or upon preemption by a higher priority job. If JobRequeue is set to a value of 1, 
                      # then batch jobs may be requeued unless explicitly disabled by the user. If JobRequeue is set to a value of 0, then batch 
                      # jobs will not be requeued unless explicitly enabled by the user.
    "KillWait" : 30, # The interval, in seconds, given to a job's processes between the SIGTERM and SIGKILL signals upon reaching its time limit.
    "OverTimeLimit" : 0, # Number of minutes by which a job can exceed its time limit before being canceled.
    "max_switch_wait": 0, # Max number of seconds to wait for nodes on the same rack, before scheduling a job with nodes on different racks.
                          # 0 bypasses the wait. FastSim applies this to ALL jobs <= max_switch_nodes (an ARCHER2/Cray placement
                          # policy); stock Slurm only waits for switches when a job explicitly requests --switches, so 0 is the
                          # closest stock behavior. ARCHER2 used 172800 (48 h).

    # Simulator-specific parameters (not specific to Slurm)
    "approx_bf_try_per_sec" : 10, # This is simulator specific (limiting backfilling to approximate CPU limitations)
    "approx_excess_assocs" : 0, # This is simulator specific (see above)
    "bd_threshold" : 60, # This is the threshold used when calculating bounded slowdown
    "hpe_restrictlong_sliding_reservations" : "", # This is cluster (Lumi?) specific; "" disables it
    "nodes_down_in_blades" : False, # This is cluster (Lumi?) specific (when a node is down, all nodes in the blade are placed in down state)
    "system" : "default", # System identifier for node-naming conventions (e.g. "kestrel")
    "initialize" : True, # Whether to build an initial running/queued state at sim_start
    "Pdefault" : 600, # Default power-per-node in W (fallback when job energy data is missing)
    "max_switch_nodes" : 256, # Maximum size of a job that might be constrained by max_switch_wait
    "impromptu_reservation_names" : [], # Reservation names treated as reactive holds (no advance draining)
    "save_interval_steps" : 50000, # How many steps elapse between saving intermediate results
    "sim_start" : None, # Start of the simulated window; None = derived from the job dump (min Submit)
    "sim_end" : None, # End of the simulated window; None = derived from the job dump (max End + 1 day)
}


# These are Slurm configurable parameters, but not yet implemented.
# max_rpc_cnt               # If the number of active threads in the slurmctld daemon is equal to or larger than this value, defer scheduling of jobs.
# max_sched_time            # How long, in seconds, that the main scheduling loop will execute for before exiting.
# partition_job_depth       # The default number of jobs to attempt scheduling (i.e. the queue depth) 
                            # from each partition/queue in Slurm's main scheduling logic.
# sched_max_job_start       # The maximum number of jobs that the main scheduling logic will start in any single execution.
# batch_sched_delay         # How long, in seconds, the scheduling of batch jobs can be delayed.
# bf_max_job_user_part      # The maximum number of jobs per user per partition to attempt starting with the backfill scheduler for any single partition.

vals_us = ["sched_min_interval", "bf_yield_interval", "bf_yield_sleep"]
vals_s = ["sched_interval", "bf_resolution", "bf_interval", "bf_max_time", "KillWait", "max_switch_wait"]
vals_min = ["bd_threshold", "PriorityCalcPeriod", "bf_window", "OverTimeLimit"]
vals_days = ["PriorityMaxAge", "PriorityDecayHalfLife"]
vals_bool = ["JobRequeue"]

# TODO Include node/partition information dump once setup to read this
mandatory_fields = set(
    (
        "assocs_dump", "node_events_dump",
        "resv_dump_current", "resv_dump_historic",
        "job_dump", "slurm_conf",
        "considered_partitions", "qos_dump",
    )
)

# Config fields holding filepaths; relative values are resolved against the
# directory containing the YAML config file, not the current working directory.
path_fields = (
    "assocs_dump", "qos_dump", "node_events_dump", "resv_dump_current",
    "resv_dump_historic", "job_dump", "slurm_conf",
    "supplementary_resv", "predicted_power", "predicted_runtime", "re_fp",
)

def get_config(config_file):
    print("Reading config from {}".format(config_file))

    # Read the config file chosen from ../configs directory
    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Resolve relative dump paths against the config file's own directory so
    # runs behave the same regardless of the current working directory.
    config_dir = os.path.dirname(os.path.abspath(config_file))
    for field in path_fields:
        val = config_dict.get(field)
        if isinstance(val, str) and val and not os.path.isabs(val):
            config_dict[field] = os.path.normpath(os.path.join(config_dir, val))

    # Read the slurm.conf file in slurm_dump
    with open(config_dict["slurm_conf"], "r") as f:
        for line in f:
            if line[0] == "#": # Skip lines that are commented out
                continue

            line = line.strip("\n")
            # Paramaters are set in each line, get the parameter being set in this line
            param = line.split("=")[0].strip(" ")

            # Handle the scheduler parameters
            if param == "SchedulerParameters":
                line = line.replace(" ", "")
                # Each subparam_entry is in the form 'subparam=value'
                subparam_entries = line.lstrip(param + "=").split(",")
                for subparam_entry in subparam_entries:
                    # Handle boolean suparameters (their existence in this list designates them as True)
                    if "=" not in subparam_entry: 
                        defaults[subparam_entry] = True
                        continue
                    subparam, val = subparam_entry.split("=")
                    # NOTE: This assumes all time strings are of the form days-hrs, may also be
                    # days-hrs:mins:secs or hys:mins:secs. At least if I come accross this it
                    # will throw an error
                    if "-" in val:
                        val = int(val.split("-")[0]) + (int(val.split("-")[1]) / 24)
                    else:
                        val = int(val)
                    # Params can still be overidden in the yaml config so treat slurm.conf as
                    # the defaults for this system
                    defaults[subparam] = val

            elif param in defaults: # Handle lines with only one parameter
                val = line.split("=")[1].strip(" ")
                if "-" in val:
                    val = int(val.split("-")[0]) + (int(val.split("-")[1]) / 24)
                else:
                    val = int(val)
                defaults[param] = val

    missing_fields = mandatory_fields - set(config_dict.keys())
    if missing_fields:
        raise ValueError(
            "Missing mandatory fields {} in config file at {}".format(missing_fields, config_file)
        )

    # Transfer "default" options (including those set by slurm.conf) to config_dict from YAML sim configuration
    for option in set(defaults.keys()) - set(config_dict.keys()):
        config_dict[option] = defaults[option]

    # Convert integers to timedelta/boolean objects
    for option in vals_us:
        config_dict[option] = timedelta(microseconds=config_dict[option])
    for option in vals_s:
        config_dict[option] = timedelta(seconds=config_dict[option])
    for option in vals_min:
        config_dict[option] = timedelta(minutes=config_dict[option])
    for option in vals_days:
        config_dict[option] = timedelta(days=config_dict[option])
    for option in vals_bool:
        config_dict[option] = bool(config_dict[option])

    # Create a namedtuple from the config dictionary
    config_namedtuple = namedtuple("config", config_dict)
    config = config_namedtuple(**config_dict)

    return config


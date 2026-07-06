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

import os, time
import csv
import datetime; from datetime import timedelta
from collections import defaultdict, OrderedDict

import pandas as pd

from config import get_config
from partition import Partitions
from job_queue import Queue, JobState, Job
from priority_sorters import MFPrioritySorter
from fairshare import FairTree
from data_reader import SlurmDataReader
from interactive_shell import InteractiveShell

import traceback
import logging

import bisect
from operator import attrgetter

# Equivalent to sorting by (node.weight, node.nid) — see Node.sched_order
_sched_order_key = attrgetter("sched_order")

import signal

from aux_funcs import print_and_log, mark_skip

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

class Controller:
    """
    Controls the simulation.

    Methods:
    __init__
    _next_job_finish
    run_sim
    _step
    _submit
    _check_finished_jobs
    _end_job
    _sched_reservations
    _sched_main
    _prep_new_bf
    _prep_bf_map
    _prep_bf_q
    _backfill
    _get_backfill_jobs
    _check_down_nodes
    _check_reservations
    _print_stats
    """
    def __init__(self, config_file, results_filepath, run_logs=None):
        """
        Initialize the controller.
        """

        def _sigint_handler(signum, frame):
            """
            Pause the simulation to allow for object inspection via CLI.
            """
            if not self.paused:
                print("\n[SIM] Pausing…  (Ctrl-C again to abort)\n")
                self.pause()
            else:
                print("\n[SIM] Aborting immediately!\n")
                sys.exit(1)

        signal.signal(signal.SIGINT, _sigint_handler)

        self.run_logs = run_logs
        """
        Logging utility.
        """

        self.print_log = run_logs.print_log if run_logs else logging.getLogger("fastsim.null")
        """
        Debug and status update logging, optionally prints to terminal.
        """

        self.sreport_log = run_logs.sreport_log if run_logs else logging.getLogger("fastsim.null")
        """
        sreport-style logging
        """

        self.power_log_fp = run_logs.power_log_fp if run_logs else None
        """
        Where to log the power usage at each time step.
        """

        print_and_log(self.print_log, 'Initializing Slurm configuration.'.rjust(100,'.'))
        self.config = get_config(config_file)
        """
        Get the configuration parameters from the:
        - Defaults in config.py
        - Settings in slurm.conf
        - Settings in the simulator YAML config file
        See config.py for a description of the parameters
        self.config is a namedtuple
        """

        print_and_log(self.print_log, 'Initializing Data Reader.'.rjust(100,'.'))
        self.data_reader = SlurmDataReader(self.config)
        """
        Initialize the data reader
        """

        print_and_log(self.print_log, 'Initializing jobs data.'.rjust(100,'.'))
        df_jobs = self.data_reader.get_cleaned_job_df(self.config.considered_partitions, 
                                                      self.config.Pdefault, 
                                                      self.config.sim_start, 
                                                      self.config.sim_end, 
                                                      self.config.initialize,
                                                      supplementary_resv=getattr(self.config, "supplementary_resv", None),
                                                      predicted_power=getattr(self.config, "predicted_power", None),
                                                      predicted_runtime=getattr(self.config, "predicted_runtime", None),
                                                      )
        """
        Get a cleaned dataframe of jobs from the job trace in sacct_jobs.csv
        the second parameter sets the default power per node
        which is used when calculating the consumed energy of the cluster
        """

        print_and_log(self.print_log, 'Initializing node and partition data.'.rjust(100,'.'))
        nid_data, partition_data, valid_resv, resv_end_times, hpe_restrictlong = self.data_reader.get_nodes_partitions(
            self.config.considered_partitions, self.config.hpe_restrictlong_sliding_reservations,
            df_jobs.End.max(), self.config.nodes_down_in_blades,
            # Resolved by the data reader: equal to the config values when set,
            # otherwise derived from the job dump
            self.data_reader.sim_start, self.data_reader.sim_end
        )
        """
        Get node and partition data
        nid_data[nid] = { nid is the node name
               "weight" : nid_weight[nid], node weight (used by the scheduler when selecting nodes for jobs)
               "down_schedule" : down_schedule, a list of events when the node is down, with start time, duration, state, and reason
               "resv_schedule" : resv_schedule, a list of reservations for this node
               "partitions" : nid_partitions[nid], the list of partitions the node is available to
           }
        partition_data[name] = { name is the partition name
               "prio_tier" : Jobs submitted to a partition with a higher PriorityTier value will be evaluated by the scheduler before 
                                        pending jobs in a partition with a lower PriorityTier value.
               "prio_jobfactor" : Partition factor used by priority/multifactor plugin in calculating job priority.
               "qos_name": the qos listed with the partition in slurm.conf. This is used for tracking qos-based resource limits.
               }
        valid_resv: a list of all valid reservation names
        hpe_restrictlong: this is HPE specific
        """

        self.resv_end_times = resv_end_times
        """
        A dictionary of the form: {reservation name: end of reservation}. We need
        this to handle FLEX reservations, where jobs can run after the reservation has
        ended. More so, there may be situations where jobs are submitted and get
        held by a QOS submit hold, and are held until after the reservation ends.
        In this case, we need to either cancel these jobs or allow them to run on
        general population nodes. The only way to be sure we have this type of
        situation is to reference this dictionary.
        """

        if self.config.initialize:
            self.init_time = df_jobs.Submit.min() - timedelta(minutes=1)
            """
            Set the simulator initial time to the first job's submit time, minus one minute.
            """
        else:
            self.init_time = df_jobs.Start.min()
            """
            Set the simulator initial time to the first job's start time
            """
    
        self.time = self.init_time
        """
        Initialize the time as the init time (this time is incremented at each step)
        """
        
        print_and_log(self.print_log, 'Getting QOS data.'.rjust(100,'.'))
        qos_data = self.data_reader.get_qos()
        """
        Get the QOS data from sacctmgr_qos
        qos_data[row.Name] = { row.Name is the QOS name (e.g. 'normal', 'high')
                name: # QOS name (e.g. 'high', 'normal', 'standby')           
                prio: Priority used in job priority calculation
                GrpTRES: The total count of TRES able to be used at any given time from jobs running from a QOS
                GrpJobs: The total number of jobs able to run at any given time from a QOS.
                GrpSubmit: The total number of jobs able to be submitted to the system at any given time from a QOS.
                MaxTRESPU: The maximum number of TRES a user can allocate at a given time.
                MaxJobsPU: The maximum number of jobs a user can have running at a given time. 
                MaxJobs: The total number of jobs able to run at any given time for the given association.
                MaxSubmitPU: The maximum number of jobs a user can have running and pending at a given time. 
                MaxSubmit: The maximum number of jobs able to be submitted to the system at any given time from the given association.
           }
        """

        print_and_log(self.print_log, 'Initializing Partitions.'.rjust(100,'.'))
        self.partitions = Partitions(nid_data, partition_data)
        """
        - Create all Partition and Node objects from nid_data and partition_data
        - Populate the set of Partition objects.
        - Create a dictionary to index partition objects by name
        - Initialize the set of all nodes in the cluster
        - Initialize free_blocks with the empty string reservation with all nodes.
        """

        print_and_log(self.print_log, 'Populating the set of all Users from all jobs in the job trace.'.rjust(100,'.'))
        active_usrs = sorted({ row.User for _, row in df_jobs.iterrows() })
        print_and_log(self.print_log, 'Initializing FairTree.'.rjust(100,'.'))
        self.fairtree = FairTree(
            self.config.assocs_dump, self.config.PriorityCalcPeriod,
            self.config.PriorityDecayHalfLife, self.init_time, active_usrs,
            self.config.approx_excess_assocs, self.partitions
        )
        """
        Initialize the FairTree for the FairShare algorithm.
        This will create the entire tree with all the associations
        in sacctmgr_assocs.csv
        """
        

        print_and_log(self.print_log, 'Initializing Multifactor Priority sorter'.rjust(100,'.'))
        priority_sorter = MFPrioritySorter(
            self.init_time, self.config.PriorityWeightJobSize, self.config.PriorityWeightAge,
            self.config.PriorityWeightFairshare, self.config.PriorityMaxAge,
            self.config.PriorityWeightPartition, self.config.PriorityWeightQOS,
            len({ partition.priority_tier for partition in self.partitions.partitions }) == 1,
            self.fairtree, len(self.partitions.nodes),
            power_weight=getattr(self.config, "power_weight", 0),
            re_fp=getattr(self.config, "re_fp", None),
            power_alpha=getattr(self.config, "power_alpha", None),
            power_beta=getattr(self.config, "power_beta", None),
            power_gamma=getattr(self.config, "power_gamma", None),
            power_time_boost_start=getattr(self.config, "power_time_boost_start", None),
            power_time_boost_end=getattr(self.config, "power_time_boost_end", None)
        )
        """
        Initialize the Multifactor Priority sorter we will use to sort the queue.
        """

        print_and_log(self.print_log, 'Initializing queue.'.rjust(100,'.'))
        self.queue = Queue(
            df_jobs, self.partitions.partitions_by_name, qos_data, valid_resv, priority_sorter, self.config.max_switch_wait
        )
        """
        Initialize the Queue
        """

        self.sched_start = self.init_time
        """
        Set the start of the schedule to the minimum job start time
        """

        self.num_sched_test_step = 0
        """
        A counter we use to track the number of scheduling steps we've taken.
        This will be tested against the maximum number of scheduling steps allowed,
        as a limit on the scheduler.
        """

        self.running_jobs = []
        """
        This is the list of all jobs currently running on nodes.
        """

        self.down_nodes = []
        """
        This is the list of Nodes that are currently down.
        """
        
        self.running_nodes = 0
        """
        This is the number of Nodes that are currently running.
        """

        self.planned_nodes = set()
        """
        This is the list of Nodes that are currently in the planned state.
        """

        print_and_log(self.print_log, 'Initializing node down events/reservation data.'.rjust(100,'.'))
        self.nodes_that_will_go_down = sorted(
            [ node for node in self.partitions.nodes if node.down_schedule ],
            key=lambda node: (node.down_schedule[-1][0], node.nid),
            reverse=True
        )
        """
        An ordered list of nodes that will go down at some point. Some nodes
        will go down multiple times; node.down_schedule is a list of those down
        events (start time, duration, state, reason), sorted by start time 
        in descending order. So this nodes_that_will_go_down list is sorted by node event
        start time such that the last node in the list has the earliest down
        time. We then can pop the node off the end of the list when we want
        the node with the earliest down time.
        """
        
        self.reserved_nodes = []
        """
        This is the list of Nodes that are currently reserved.
        """
        
        self.nodes_that_will_be_reserved = sorted(
            [ node for node in self.partitions.nodes if node.reservation_schedule ],
            key=lambda node: (node.reservation_schedule[-1][0], node.nid),
            reverse=True
        )
        """
        An ordered list of nodes that will be reserved at some point. Some nodes
        will be reserved multiple times; node.reservation_schedule is a list of those
        reservations (start time, end time, reservation name), sorted by start time 
        in descending order. So this nodes_that_will_be_reserved list is sorted by
        reservation start time such that the last node in the list has the earliest
        reservation time. We then can pop the node off the end of the list when we want
        the node with the earliest reservation time.
        """

        self.nodes_that_will_be_impromptu_reserved = sorted(
            [ node for node in self.partitions.nodes if node.impromptu_reservation_schedule ],
            key=lambda node: (node.impromptu_reservation_schedule[-1][0], node.nid),
            reverse=True
        )
        """
        An ordered list of nodes that will be impromptu reserved at some point. We need
        to handle impromptu reservations differently, because they aren't created in advance.
        So, we don't set the node interval times based on these reservations, because that
        would cause nodes to reject jobs that might run beyond the reservation start time.
        Ordinarily, that's what we'd want, but because these reservations aren't created in
        advance, and they're created with the IGNORE_JOBS flag, the actual behavior is that
        jobs are allowed to run until completion on these nodes, but new jobs will not be able
        to run on these nodes while the reservation is active unless they are part of the reservation.
        """

        # ====================== BEGIN ARCHER2-SPECIFIC CODE ====================== #
        # ARCHER2 specific: sliding maintenance window
        if self.config.hpe_restrictlong_sliding_reservations == "":
            self.sliding_reservations = []
        else:
            nid_to_node = { node.nid : node for node in self.partitions.nodes }
            self.sliding_reservations = [
                [
                    submitted, submitted, submitted + timedelta(hours=1),
                    submitted + timedelta(hours=1, minutes=5),
                    submitted + timedelta(days=365, hours=1, minutes=5),
                    [ nid_to_node[nid] for nid in hpe_restrictlong[submitted] ],
                    "HPE_RestrictLongJobs"
                ]
                for submitted in sorted(hpe_restrictlong)
                    if submitted >= self.init_time - timedelta(hours=1)
            ]
            # self.sliding_reservations.sort(key=lambda res: res[0], reverse=True)
            self.sliding_reservations.sort(
                key=lambda res: (res[0], res[6], res[5][0].nid if res[5] else ""),
                reverse=True
            )
        # ======================= END HPE-SPECIFIC CODE ======================= #

        # TODO Refactor
        # These are all backfilling parameters. Should put backfiller into its own class since
        # it needs its own state.
        print_and_log(self.print_log, 'Initializing backfilling parameters.'.rjust(100,'.'))
        self.bf_free_blocks = None
        """
        Backfill Free Blocks are similar to Main Scheduler free blocks. They both provide sets of nodes
        that are available for blocks of time (interval = (block start, block end)) for a given reservation 
        (can be '' for main scheduling i.e. no reservation). These available nodes are then allocated to
        jobs.
        
        There is one primary difference: the intervals for bf_free_blocks are relative, not actual times.
        When the backfill procedure is started, the idea is to find nodes available for a long enough time
        that other jobs can be scheduled in the blocks. We don't need to know exactly when the nodes are
        available, just when they are available (and afterwards unavailable) relative to the current time.
        This tells us if the nodes are available for long enough for a job to run. The backfill procedure
        looks through all of the available nodes and all of the jobs available for backfilling (subject to
        the limits set in the Slurm configuration) and schedules jobs in the bf_free_blocks that are big
        enough to accomodate them.
        """
        
        self.bf_window = self.config.bf_window.total_seconds()
        """
        The number of minutes into the future to look when considering jobs to schedule.
        """
        
        self.bf_end_padding = (self.config.OverTimeLimit + self.config.KillWait).total_seconds()
        """
        OverTimeLimit: Number of minutes by which a job can exceed its time limit before being canceled.
        KillWait: The interval, in seconds, given to a job's processes between the SIGTERM and SIGKILL 
                  signals upon reaching its time limit.

        Make sure there is enough padding in the backfill block to accomodate the extra time the job may
        run beyond its request time (wallclock request).
        """

        self.bf_resolution = self.config.bf_resolution.total_seconds()
        """
        The number of seconds in the resolution of data maintained about when jobs begin and end.
        """
        
        self.bf_max_relevant_start = (
            (self.config.bf_max_time - self.config.bf_yield_interval).total_seconds()
        )
        """
        bf_max_time: The maximum time in seconds the backfill scheduler can spend (including time spent 
                     sleeping when locks are released) before discontinuing, even if maximum job counts 
                     have not been reached.
        bf_yield_interval: the backfill scheduler will periodically relinquish locks in order for other 
                           pending operations to take place. This specifies the times when the locks are 
                           relinquished in microseconds.

        This is the maximum start time within a Backfill window for a job that can be backfilled.

        XXX Need to revisit this.
        As far as I can tell, this is simultor-specific, and is set up because the backfilling operation
        might yield the locks and not get to a job in time if its start time is after this. So this is 
        a way of approximating what would happen during the real backfilling algorithm because our
        backfilling process is artificial and it is difficult to simulate the backfilling process timing out.
        """
        
        self.bf_loop_active = False
        """
        Keeping track of when the Backfilling loop is active (True) or sleeping (False)
        """

        self._rel_secs_cache = {}
        """
        Cache of datetime -> seconds-since-sched_start floats used by _prep_bf_map.
        Interval endpoints repeat across backfill cycles, so each datetime is
        converted once instead of building a timedelta per block per cycle.
        """
        
        self.bf_try_per_lock_hold = int(
            self.config.bf_yield_interval.total_seconds() * self.config.approx_bf_try_per_sec
        )
        """
        bf_yield_interval: the backfill scheduler will periodically relinquish locks in order for other 
                           pending operations to take place. This specifies the times when the locks are 
                           relinquished in microseconds.
        approx_bf_try_per_sec: a simulator-specific parameter that helps limit backfilling to approximate 
                               CPU limitations experienced during actual backfilling on the cluster.

        This is how many jobs the Backfilling loop can try to schedule every time it holds the locks.
        """
        
        self.bf_max_lock_holds = int(
            self.config.bf_max_time / (self.config.bf_yield_interval + self.config.bf_yield_sleep)
        )
        """
        bf_max_time: The maximum time in seconds the backfill scheduler can spend (including time spent 
                     sleeping when locks are released) before discontinuing, even if maximum job counts 
                     have not been reached.
        bf_yield_interval: the backfill scheduler will periodically relinquish locks in order for other 
                           pending operations to take place. This specifies the times when the locks are 
                           relinquished in microseconds.
        bf_yield_sleep: This specifies the length of time for which the locks are relinquished in microseconds.

        This is the maximum number of times the Backfilling process can hold the locks before it is done.
        It then sleeps until the next scheduled backfilling time.
        """
        
        self.bf_locks_remaining = 1
        """
        self.bf_max_lock_holds - the number of times locks have been held
        """

        
        self.bf_nodes_free_now_max_reqtimes = None
        """
        This will hold a 2-level dictionary of maximum request times for each
        node in each reservation.

        bf_nodes_free_now_max_reqtime = {reservation: {node: max_req_time, ...}, ... }
        """
        
        self.bf_max_reqtime = None
        """
        This will hold a dictionary of the maximum request time for each reservation.

        bf_max_reqtime = {reservation: max_req_time, ...}
        """
        
        self.bf_secs_past = None
        """
        How much time has been spent backfilling. Initialized to zero with each new
        backfilling process.
        """

        self.paused = False
        """
        Keeps track of whether the simulation is paused or not.
        This is used to pause the simulation when the user requests it via Ctrl + C
        at the command line.
        """

        # Bookkeeping parameters start here

        self.times = [self.time]
        """
        Keeps track of the simulator times at every step.
        This is only used for plotting results.
        """
        
        self.power_usage = 0
        """
        Keeps track of the power used by the cluster.
        This is only used for outputting information to 
        the terminal while the simulator is running.
        """

        self.predicted_power_usage = 0
        """
        Keeps track of the predicted cluster power.
        This is only used for outputting information to 
        the terminal while the simulator is running.
        """

        
        self.total_energy = 0.0
        """
        Keeps track of the total energy used by the cluster
        over time. This isn't used by anything, but it may be
        useful for analysis.
        """

        self.job_history = []
        """
        Keeps track of all simulated jobs. This is only used
        in analysis.
        """

        self.step_cnt = 0
        """
        Counts the number of simulation steps. This is 
        only used for outputting results to the terminal and
        saving checkpoints of the simulation results.
        """

        self._next_status_time = None
        """
        Simulated time of the next periodic status output. This is
        only used for outputting results to the terminal and logs.
        """

        self.sched_backfill_num = 0
        """
        Counts the number of jobs that were backfilled. This is 
        only used for outputting results to the terminal.
        """
        
        self.sched_main_num = 0
        """
        Counts the number of jobs that were scheduled by the main scheduler. 
        This is only used for outputting results to the terminal.
        """

        self.results_filepath = results_filepath
        """
        Where to save the simulator checkpoints while the simulator is running.
        """


        print_and_log(self.print_log, 'Finished Controller initialization.'.rjust(100,'.'))

    def _next_job_finish(self):
        """
        Get the end time of the next running job to finish, or datetime.max
        if no jobs are running.
        """
        if not self.running_jobs:
            return datetime.datetime.max

        # running_jobs is sorted by job end time in descending order
        return self.running_jobs[-1].end


    def pause(self):
        """Pause the simulation."""
        self.paused = True


    def resume(self):
        """Resume a previously paused simulation."""
        self.paused = False


    def run_sim(self, max_steps=0):
        """
        Run the simulation from start to finish.

        If max_steps is given, the simulator will stop when it has completed that
        number of steps. Otherwise, it will run until there are no more running jobs
        and there are no more jobs left to be submitted.

        The beginning of the simulation will suffer from ramp-up effects as the cluster
        gets filled with jobs, so it is advisable to start the simulation at a down time.

        The results of the simulation for time after the last job is submitted are not reliable
        because it does not capture the jobs that were submitted after this (because those
        jobs were submitted after the job trace data was gathered).

        The simulator results will be saved to the results output file every 100k steps.

        The biggest computational cost is backfilling, so if time is an issue, set approx_bf_try_per_sec
        to a very small number (like 1), which will eliminate most of the backfilling process.
        Those results won't be reliable, but they should show longer wait times and lower allocated nodes
        over time than the data. This is a good option when running quick tests before full experiments.

        Arguments:
        - max_steps: the maximum limit for simulation steps. Helpful when the simulation gets stuck in
                     an infinite loop due to dependency/reservation issues causing jobs to never run.
        """
        # This is only used after the simulator is finished, to determine how long (in clock time)
        # the simulator took to run.
        sim_start = time.time()

        # NOTE Assuming: defer,bf_continue are always set. I think this is true for large systems
        
        # self.sched_start is the earliest start time of a job in the job trace 
        # previous_small_sched is used to make sure the simulator doesn't violate
        # the constraint provided by sched_min_interval: the scheduler must wait
        # a minimum amount of time to run, regardless of any job submission or 
        # termination event.
        previous_small_sched = self.sched_start

        # next_bf_time is used to keep track of the next time the backfill loop should run
        next_bf_time = self.sched_start + self.config.bf_interval

        # next_sched_time is used to keep track of the next time the scheduler is scheduled
        # to run. It can also run when an event occurs, but the scheduler will run periodically
        # (every sched_interval number of seconds) regardless of any events.
        next_sched_time = self.sched_start + self.config.sched_interval

        # small_sched_waiting_time keeps track of the next time the backfill loop will sleep
        # and the main scheduling loop can take over.
        small_sched_waiting_time = None

        # next_fairtree_time keeps track of when to recalculate the fairtree for the FairShare algorithm
        next_fairtree_time = self.time + self.config.PriorityCalcPeriod

        # While there are any jobs left to be queued, any jobs in the queue, or any running jobs,
        # continue the simulation. This can lead to an infinite loop if there are dependency or
        # reservation issues that prevent jobs from running.
        while self.queue.all_jobs or self.queue.queue or self.running_jobs:
            # Skip forward to the next relevant time
            self.time = min(
                next_bf_time, # the next time the backfill loop will start
                next_sched_time, # the next scheduled time for the main scheduler to run
                next_fairtree_time, # the next time to calculate the fairtree
                self._next_job_finish(), # the next job finish event
                self.queue.next_newjob() # the next job submission event
            )

            # Since small_sched_waiting_time can be None, we consider this seperately
            # TODO: refactor this so it can be considered above.
            if small_sched_waiting_time is not None:
                if self.time >= small_sched_waiting_time:
                    self.time = small_sched_waiting_time

            # Whether we should run the main and reservation schedulers
            run_main_and_resv_scheduler = False

            # How many jobs the scheduler should attempt to schedule (i.e. the queue depth) 
            # when a running job completes or other routine actions occur
            sched_depth = None

            # Whether it is time to run a backfill loop
            bf = False

            # Whether it is time to calculate the fairtree
            fairtree = False

            # If it is time to run a backfill loop...
            if self.time == next_bf_time:
                bf = True
                next_bf_time += self.config.bf_yield_interval + self.config.bf_yield_sleep
                next_bf_sleep = self.time + self.config.bf_yield_interval

            # If it is time to calculate the fairtree...
            if self.time == next_fairtree_time:
                fairtree = True
                next_fairtree_time += self.config.PriorityCalcPeriod

            # If it is time for a scheduled main/reservation scheduler run...
            if self.time == next_sched_time:
                # If the backfill loop is running and the time for the next backfill sleep is later than 
                # the current time, we delay the main/reservation scheduler until the next time the backfiller sleeps
                if self.bf_loop_active and self.time < next_bf_sleep:
                    next_sched_time = next_bf_sleep
                else:
                    run_main_and_resv_scheduler = True
                    next_sched_time += self.config.sched_interval
                    # Reset this flag because either the Backfill loop is inactive or the current time 
                    # is greater than the next_bf_sleep time. Either way, the current time is greater 
                    # than the small_sched_waiting_time so we reset this to prevent going back in time 
                    # in the next step.
                    small_sched_waiting_time = None
            # If we've reached the previous next_bf_sleep time (the only time we set small_sched_waiting_time to) 
            # or we have passed the scheduler minimum interval time and we are at a job finish event time...
            elif ( self.time == small_sched_waiting_time or
                    (
                        self.time > previous_small_sched + self.config.sched_min_interval and
                        self.time == self._next_job_finish()
                    )
            ):
                if self.bf_loop_active and self.time < next_bf_sleep: 
                    # Set small_sched_waiting_time to the next next_bf_sleep time, which we 
                    # incremented by the combined yield_interval + yield_sleep time above.
                    small_sched_waiting_time = next_bf_sleep
                else:
                    run_main_and_resv_scheduler = True
                    sched_depth = self.config.default_queue_depth
                    previous_small_sched = self.time
                    # See note above on why we reset this
                    small_sched_waiting_time = None

            # With all flags and parameters set appropriately, we take a step
            self._step(run_main_and_resv_scheduler, sched_depth, bf, fairtree)
            self.step_cnt += 1

            # If we have finished backfilling...
            if bf and not self.bf_loop_active:
                next_bf_time += self.config.bf_interval

            # If we have exceeded the step limit...
            if max_steps and self.step_cnt > max_steps:
                break

            # Periodic status output (terminal line, stats table, sreport row),
            # throttled to status_interval of simulated time (0 disables it).
            # Emitting it every step slowed busy simulations down noticeably.
            if self.config.status_interval and (
                    self._next_status_time is None or self.time >= self._next_status_time):
                self._next_status_time = self.time + timedelta(seconds=self.config.status_interval)

                # Nothing in the scheduling logic reads planned_nodes/idle_nodes,
                # so they only need to be up to date when status is emitted —
                # rebuilding them every step cost ~20% of the simulation loop.
                self.planned_nodes = set(node for node in self.planned_nodes if node.free)
                self.idle_nodes = set(node for node in self.partitions.nodes if node.free and node not in self.planned_nodes)

                print((f"Running Nodes: {self.running_nodes:4d} " +
                       f"Down Nodes: {len(self.down_nodes):4d} " +
                       f"Planned Nodes: {len(self.planned_nodes):4d} " +
                       f"Idle Nodes: {len(self.idle_nodes):4d} " +
                       f"Step: {self.step_cnt} " +
                       f"Time: {self.time.strftime('%Y-%m-%d %H:%M:%S')} "), end='\r')

                self.sreport_log.info(f"{self.time},{self.running_nodes},{len(self.down_nodes)},{len(self.planned_nodes)},{len(self.idle_nodes)}")

                self._print_stats()

            # Checkpoint every interval
            if self.step_cnt % self.config.save_interval_steps == 0:
                print(f'Saving Job History at Step: {self.step_cnt}'.rjust(50, '.'))
                jobs = list()
                for i, job in enumerate(self.job_history):
                    print(f'Adding job {str(i).rjust(6)} of {len(self.job_history)}', end='\r')
                    try:
                        job_dict = dict()
                        for key, value in job.__dict__.items():
                            if key == 'assoc':
                                continue
                            elif key in ['qos','partition','partition_qos']:
                                job_dict[key] = value.name
                            elif key == 'state':
                                # Store the plain string (e.g. "COMPLETED") so the
                                # pickle can be read without FastSim's modules on
                                # sys.path (same fix as the end-of-run dump in main.py)
                                job_dict[key] = value.name
                            elif key == 'assigned_nodes':
                                job_dict[key] = set(node.nid for node in value)
                            elif key == 'node_timeline':
                                job_dict[key] = [(str(ts), cnt) for ts, cnt in value]
                            elif key == 'wait_history':
                                job_dict[key] = [(str(ts), reason) for ts, reason in value]
                            else:
                                job_dict[key] = value
                        jobs.append(job_dict)
                    except:
                        print(f'Error while adding job {i} of {len(self.job_history)}')
                        logging.info(f'Error while adding job {i} of {len(self.job_history)}')
                        traceback.print_exc()
                pd.DataFrame(jobs).to_pickle(self.results_filepath)

            if self.paused:
                InteractiveShell(self).cmdloop()   # blocks until shell exits
                print("[SIM] Resuming…")
                self.resume()


        # We're done!
        elapsed = time.time() - sim_start
        print(
            "Sim complete in {} hr {} mins".format(
                int(elapsed // (60 * 60)), int((elapsed % (60 * 60)) // 60)
            )
        )

    def _step(self, run_main_and_resv_scheduler, sched_depth, bf, fairtree):
        """
        Take a simulation step.

        Arguments:
        - run_main_and_resv_scheduler (boolean): whether or not to run the main and reservation schedulers
                                                 at this step
        - sched_depth: the default number of jobs to attempt scheduling (i.e. the queue depth) when a 
                       running job completes or other routine actions occur
        - bf (boolean): whether or not to run backfilling at this step
        - fairtree (boolean): whether or not to recalculate the fairtree at this step
        """
        # Check if any jobs need to finish in this step, and end
        # jobs as necessary.
        self._check_finished_jobs()

        # Update list of down nodes
        self._check_down_nodes()

        # Update reservations
        self._check_reservations()

        # Step through the queue
        # - Check dependencies to see if there are any changes.
        # - Check QOS holds and release any jobs that should no longer be held.
        # - Check if there are any jobs that need to be cancelled
        # - Check if there are any jobs that need to be submitted
        # - Update priority, dependency, and reservations
        self.queue.step(self.time, self.running_jobs)

        # Recalculate the fairtree based on the new state
        if fairtree:
            self.fairtree.fairshare_calc(self.running_jobs, self.time)

        
        pre_sched_running_jobs = len(self.running_jobs)

        # sched_depth is the number of jobs to attempt scheduling (i.e. the queue depth) 
        # when a running job completes or other routine actions occur
        # It gets changed here to the combined length of the reservation and main scheduling queues
        # i.e. we attempt to schedule all the jobs on the queue
        if run_main_and_resv_scheduler:
            self.num_sched_test_step = 0
            sched_depth = sched_depth if sched_depth is not None else (
                sum(len(res_queue) for res_queue in self.queue.reservations.values()) +
                len(self.queue.queue)
            )

            # Schedule jobs in reservation queues
            self._sched_reservations(sched_depth)

            # Schedule jobs in the main queue
            self._sched_main(sched_depth)

        # If we are backfilling in this step...
        if bf:
            # If we are not already backfilling
            if not self.bf_loop_active:
                # Prepare a new backfill loop
                self._prep_new_bf()
                self.bf_loop_active = True

            # Backfill jobs
            self._backfill(self.bf_try_per_lock_hold)

        # If the number of running jobs has changed, re-sort the list
        if len(self.running_jobs) != pre_sched_running_jobs:
            self.running_jobs.sort(key=lambda job: job.end, reverse=True)

        # Add this time to this list for analysis
        self.times.append(self.time)

        self.running_nodes = sum([job.nodes for job in self.running_jobs])
        self.log_simulation_step_power()

    def _submit(self, job, nodes):
        """
        Submit a job. In this context, submitting a job actually means starting a job.
        Job submission, as it is normally understood, actually happens in the Queue module,
        where jobs are added to the queue in queue.step.

        Arguments:
        - job: the Job being submitted
        - nodes: the nodes to be allocated to this job
        """
        # Add this job to the list of running jobs
        self.running_jobs.append(job)

        # Record this first entry in the timeline. See note on this in job_queue.py
        job.node_timeline.append((self.time, job.nodes))

        # Update the current power usage by the cluster
        self.power_usage += job.true_node_power * job.nodes / 1e+6

        try:
            self.predicted_power_usage += job.predicted_power * job.nodes / 1e+6
        except:
            pass
        # self.total_energy += (
        #     job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
        # )

        if job.assigned_nodes:
            raise Exception("Job has already been assigned nodes.")
        if len(nodes) != job.nodes:
            raise Exception("The number of nodes allocated to the job does not match the requirements.")

        # Update this node based on its assignment to this Job
        for node in nodes:
            self.partitions.remove_free_block(node)
            node.interval_times[0] = job.endlimit
            self.partitions.add_free_block(node)
            job.assign_node(node)

        return True

    def _check_finished_jobs(self):
        """
        Check if any jobs have finished.
        """
        # running_jobs is sorted by end time in descending order, so the 
        # earliest end time is the last job in the list.
        # End all jobs that ended before the current time
        while self.running_jobs and self.running_jobs[-1].end <= self.time:
            job = self.running_jobs.pop()
            self._end_job(job)

    def _end_job(self, job):
        """
        End the job.

        Arguments:
        - job: the Job object for the job that is ending.
        """
        # Update the fairtree based on this job finish event
        self.fairtree.job_finish_usage_update(job, self.time)

        # Add this job to the job_history (used in plotting simulator results)
        self.job_history.append(job)

        # Remove this job's power usage from the total cluster power usage
        self.power_usage -= job.true_node_power * job.nodes / 1e+6

        try:
            self.predicted_power_usage -= job.predicted_power * job.nodes / 1e+6
        except:
            pass

        # Add this job's energy usage to the total energy used by the cluster
        self.total_energy += (
            job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
        )

        # End the job (set nodes free, update qos quotas, set job state to COMPLETED
        job.end_job()
        
        for node in job.assigned_nodes:
            # Down nodes dont exist to free_blocks
            if node.down:
                continue
            # Remove the node from its current reservation-interval list
            # the reservation can be the same, but the interval has changed
            self.partitions.remove_free_block(node)

            # Change the beggining time of the new interval for this node
            # We don't get a new end time for the interval because that is 
            # already determined based on the next reservation for this node
            node.interval_times[0] = self.time

            # Add this node to the new free block based on the new interval
            self.partitions.add_free_block(node)

    def _sched_reservations(self, sched_depth):
        """
        Schedule jobs in reservation queues.

        Arguments: 
        - sched_depth: the number of jobs to attempt scheduling (i.e. the queue depth) 
                       when a running job completes or other routine actions occur
        """
        # queue.reservations is a dictionary of jobs (sorted by priority, in ascending order)
        # indexed by a reservation name
        # Process reservations in a deterministic order:
        #   1) earlier reservation end time first, then
        #   2) reservation name as tie-breaker
        for reservation, res_queue in sorted(
            self.queue.reservations.items(),
            key=lambda kv: (self.resv_end_times.get(kv[0], datetime.datetime.max), kv[0])
        ):
            # Continue if there are no jobs in this reservation queue
            if not res_queue:
                continue

            # Get all free nodes available to this reservation
            # and sort them by weight and node ID
            free_nodes_ready_now = [
                node
                for interval, nodes in self.partitions.free_blocks[reservation].items()
                    if interval[0] <= self.time
                    for node in nodes
                        if node.running_job is None
            ]

            # If this reservation has ended, but we still need to schedule jobs,
            # allow the job to run on general population nodes.
            if self.resv_end_times[reservation] >= self.time:
                free_nodes_ready_now += [
                    node
                    for interval, nodes in self.partitions.free_blocks[""].items()
                        if interval[0] <= self.time
                        for node in nodes
                            if node.running_job is None
                ]
            free_nodes_ready_now.sort(key=_sched_order_key)

            # If there are no available nodes, we can't allocate any nodes for jobs
            if not free_nodes_ready_now:
                continue

            jobs_submitted, jobs_cancelled, i_job = [], [], len(res_queue)

            # Reverse the queue so we look at jobs with highest priority first
            for job in reversed(res_queue):
                i_job -= 1

                # Stop trying to schedule jobs if we have exceeded the sched_depth limit
                if self.num_sched_test_step >= sched_depth:
                    continue
                    # There's no point in continuing this loop, since the above condition
                    # will be true in all subsequent iterations, so let's break -KM
                    # break
                self.num_sched_test_step += 1

                # If this job should be held due to reaching QOS limits, skip it
                if job.partition_qos.hold_job(job):
                    mark_skip(job, self.time, "PARTITION-QOS-RESOURCE-LIMIT")
                    continue

                # Job End = current time + the time requested by the job
                job_end = self.time + job.reqtime

                if job.nodes > self.config.max_switch_nodes or self.time >= job.max_switch_wait_time:
                    max_rack_wait = False
                else:
                    max_rack_wait = True

                if max_rack_wait:
                    valid_nodes, valid_nodes_by_rack = set(), defaultdict(set)
                else:
                    valid_nodes = set()

                # Get a set of nodes capable of accomodating the job
                for node in free_nodes_ready_now:
                    # If this job will end after the end of this reservation for this node, this
                    # node can't accomodate this job
                    if job_end > node.interval_times[-1]:
                        continue

                    if max_rack_wait:
                        valid_nodes_by_rack[node.rack].add(node)
                        if len(valid_nodes_by_rack[node.rack]) == job.nodes:
                            valid_nodes = valid_nodes_by_rack[node.rack]
                            break
                    else:
                        valid_nodes.add(node)
                        if len(valid_nodes) >= job.nodes:
                            break


                # If we have enough nodes available to accomodate the job...
                if len(valid_nodes) == job.nodes:
                    # Cancel early if set to be cancelled in queue at later time
                    # TODO: We can probably do this at the beginning of the loop. XXX Need to revisit this
                    if job.cancel is not None:
                        jobs_cancelled.append(i_job)
                        # Need to change the index of all jobs after this one
                        # because we are going to pop this job off of the res_queue
                        # before we pop the submitted jobs off of the res queue
                        jobs_submitted = [ i - 1 for i in jobs_submitted ]
                        continue

                    # Submit the job
                    # self._submit(job.start_job(self.time), valid_nodes)
                    ordered_nodes = sorted(valid_nodes, key=_sched_order_key)
                    self._submit(job.start_job(self.time), ordered_nodes)
                    jobs_submitted.append(i_job)
                else:
                    mark_skip(job, self.time, f"NOT-ENOUGH-NODES-NOW, RESERVATION: {reservation}")
                    # Would still need to iterate over the remaining jobs and confirm we cant
                    # schedule anymore from its reservation
                    # The above comment is not clear, but the code seems clear
                    # We break here because there aren't enough nodes to accomodate this job,
                    # but it has the highest priority so we don't move on to trying to schedule
                    # other jobs, otherwise big jobs would be starved as they are continuously
                    # skipped over. The subsequent jobs would need to be backfilled if possible,
                    # not scheduled here. -KM
                    self.num_sched_test_step += i_job 
                    break

                # Remove these nodes from the list of available nodes
                for node in valid_nodes:
                    free_nodes_ready_now.remove(node)

            # Add the number of submitted jobs to the count of jobs
            # submitted by the main/reservation scheduler
            self.sched_main_num += len(jobs_submitted)

            # Cancel jobs and remove them from the reservation queue
            for i_job in jobs_cancelled:
                cancelled_job = res_queue.pop(i_job)
                cancelled_job.cancel_job()
                self.queue.jobs_to_cancel.remove(cancelled_job)

            # Remove submitted jobs from the reservation queue
            for i_job in jobs_submitted:
                res_queue.pop(i_job)

    def _sched_main(self, sched_depth):
        """
        Schedule jobs in the main queue.

        Arguments: 
        - sched_depth: the number of jobs to attempt scheduling (i.e. the queue depth) 
                       when a running job completes or other routine actions occur
        """
        # Get all free nodes available for the main queue
        # and sort them by weight and node ID
        free_nodes_ready_now = [
            node
            for interval, nodes in self.partitions.free_blocks[""].items()
                if interval[0] <= self.time
                for node in nodes
                    if node.running_job is None
        ]
        if not free_nodes_ready_now:
            return

        free_nodes_ready_now.sort(key=_sched_order_key)

        jobs_submitted, jobs_cancelled, partitions_failed = [], [], set()
        i_job = len(self.queue.queue)

        for job in reversed(self.queue.queue):
            i_job -= 1

            # Stop trying to schedule jobs if we have exceeded the sched_depth limit
            if self.num_sched_test_step >= sched_depth:
                break
            self.num_sched_test_step += 1

            
            # Not enough nodes to accomodate a previous job with this partition,
            # so we need to stop trying to schedule other jobs on this partition.
            if job.partition in partitions_failed:
                mark_skip(job, self.time, f"WAITING-FOR-NODES-IN-PARTITION {job.partition.name}")
                continue

            # If this job should be held due to reaching QOS limits, skip it
            if job.partition_qos.hold_job(job):
                mark_skip(job, self.time, f"PARTITION-QOS-RESOURCE-LIMIT {job.partition.name}")
                continue

            # Job End = current time + the time requested by the job
            job_end = self.time + job.reqtime

            if job.nodes > self.config.max_switch_nodes or self.time >= job.max_switch_wait_time:
                max_rack_wait = False
            else:
                max_rack_wait = True

            if max_rack_wait:
                valid_nodes, valid_nodes_by_rack = set(), defaultdict(set)
            else:
                valid_nodes = set()

            # Get a set of nodes capable of accomodating the job
            for node in free_nodes_ready_now:
                if job.partition.name not in node.partition_names:
                    continue

                # If this job ends after the start of the next reservation for this node,
                # this node can't accomodate this job.
                if job_end > node.interval_times[-1]:
                    continue

                if max_rack_wait:
                    valid_nodes_by_rack[node.rack].add(node)
                    if len(valid_nodes_by_rack[node.rack]) == job.nodes:
                        valid_nodes = valid_nodes_by_rack[node.rack]
                        break
                else:
                    valid_nodes.add(node)
                    if len(valid_nodes) == job.nodes:
                        break

            # If we have enough nodes available to accomodate the job...
            if valid_nodes is not None and len(valid_nodes) == job.nodes:
                # Cancel early if set to be cancelled in queue at later time
                # TODO: We can probably do this at the beginning of the loop. Need to revisit this
                # We should only cancel the job if it should have been cancelled already, not if it should
                # be cancelled in general. This would cause a shorter queue than we would actually see. -KM
                if job.cancel is not None:
                    jobs_cancelled.append(i_job)
                    # Need to change the index of all jobs after this one
                    # because we are going to pop this job off of the res_queue
                    # before we pop the submitted jobs off of the res queue
                    jobs_submitted = [ i - 1 for i in jobs_submitted ]
                    continue

                # Submit the job
                # self._submit(job.start_job(self.time), valid_nodes)
                ordered_nodes = sorted(valid_nodes, key=_sched_order_key)
                self._submit(job.start_job(self.time), ordered_nodes)
                jobs_submitted.append(i_job)

                # Remove these nodes from the list of available nodes
                for node in valid_nodes:
                    free_nodes_ready_now.remove(node)

            else:
                mark_skip(job, self.time, "NOT-ENOUGH-NODES‐NOW")
                # There aren't enough nodes available for this partition to accomodate the job
                partitions_failed.add(job.partition)
                # break if all partitions have 'failed' (there aren't enough nodes to accomodate 
                # the highest priority job for that partition)
                if len(partitions_failed) == len(self.partitions.partitions):
                    break

                # Remove any nodes that are available to the 'failed' partition
                # we need to hold these nodes because we need to gather enough nodes to accomodate this job
                free_nodes_ready_now = [
                    node for node in free_nodes_ready_now if job.partition.name not in node.partition_names
                ]
                if not free_nodes_ready_now:
                    break

        # Add the number of submitted jobs to the count of jobs
        # submitted by the main/reservation scheduler
        self.sched_main_num += len(jobs_submitted)

        # Cancel jobs and remove them from the main queue
        for i_job in jobs_cancelled:
            cancelled_job = self.queue.queue.pop(i_job)
            cancelled_job.cancel_job()
            self.queue.jobs_to_cancel.remove(cancelled_job)

        # Remove submitted jobs from the main queue
        for i_job in jobs_submitted:
            self.queue.queue.pop(i_job)        

    def _prep_new_bf(self):
        """
        Prepare a new backfill loop.
        """
        # Initialize the locks remaining to the maximum lock holds
        self.bf_locks_remaining = self.bf_max_lock_holds
        # This is is never used. Commenting this out for now -KM
        #self.bf_time = self.time

        # No backfill time has elapsed
        self.bf_secs_past = 0

        # Determine the free blocks and max request times for backfilling
        # for each reservation.
        self._prep_bf_map()

        # Prepare the Backfill queue
        self._prep_bf_q()

    def _prep_bf_map(self):
        """
        Prepare Backfill mapping.
        """
        # bf_free_blocks is a 3-level dictionary (reservation, interval, node set) where the
        # interval times are relative to the current time
        self.bf_free_blocks = {}

        # bf_nodes_free_now_max_reqtimes is a 2-level dictionary of maximum request times for each
        # node in each reservation.
        self.bf_nodes_free_now_max_reqtimes = {}

        # Hoist attribute lookups out of the loop below — it runs once per node
        # per backfill cycle and dominated the whole simulation's runtime.
        now = self.time
        bf_window = self.bf_window
        bf_end_padding = self.bf_end_padding
        bf_max_relevant_start = self.bf_max_relevant_start

        # Convert datetimes to cached seconds-since-sched_start floats and
        # subtract those instead of building a timedelta per block per cycle.
        # Job timestamps are whole seconds, which floats represent exactly, so
        # the float arithmetic below equals the timedelta arithmetic it
        # replaces. datetime.max is special-cased (it is not a whole second).
        rel_secs = self._rel_secs_cache
        if len(rel_secs) > 2_000_000:
            rel_secs.clear() # Bound memory on very long runs; entries rebuild lazily
        sched_start = self.sched_start
        dt_max = datetime.datetime.max
        now_s = rel_secs.get(now)
        if now_s is None:
            now_s = rel_secs[now] = (now - sched_start).total_seconds()

        # Look through all free blocks (reservation, ((interval_0, node_set_0), ... (interval_n, node_set_n)))
        for resv, free_block in self.partitions.free_blocks.items():
            self.bf_free_blocks[resv] = defaultdict(set)
            bf_free_blocks = self.bf_free_blocks[resv]
            self.bf_nodes_free_now_max_reqtimes[resv] = {}
            bf_nodes_free_now_max_reqtimes = self.bf_nodes_free_now_max_reqtimes[resv]

            # All nodes of a multi-node job share its endlimit — cache the
            # interval start per running job instead of recomputing per node
            job_interval_i = {}

            # Look through all the free blocks for this reservation
            for interval, nodes in free_block.items():
                # Get the end of the Backfill interval
                # If the interval end time is the datetime.max
                # then this Backfill interval end time is the end of the Backfill window
                end = interval[1]
                if end == dt_max:
                    interval_f = bf_window
                else:
                    # Otherwise, it is either the end of the Backfill window or the end of the interval,
                    # whichever is earlier (remembering the interval time is relative to the current time)
                    end_s = rel_secs.get(end)
                    if end_s is None:
                        end_s = rel_secs[end] = (end - sched_start).total_seconds()
                    interval_f = end_s - now_s
                    if interval_f > bf_window:
                        interval_f = bf_window

                # Get the beginning of the Backfill interval
                # If the beginning of the interval is before the current time
                if interval[0] <= now:
                    for node in nodes:
                        # If the node is running a job, the beginning of the interval is the maximum
                        # possible end time of the running job
                        running_job = node.running_job
                        if running_job is not None:
                            interval_i = job_interval_i.get(running_job)
                            if interval_i is None:
                                endlimit = running_job.endlimit
                                end_s = rel_secs.get(endlimit)
                                if end_s is None:
                                    end_s = rel_secs[endlimit] = (
                                        (endlimit - sched_start).total_seconds()
                                    )
                                interval_i = end_s - now_s + bf_end_padding
                                if interval_i < 1:
                                    interval_i = 1
                                job_interval_i[running_job] = interval_i
                        else:
                            interval_i = 0

                        # Add this node to this Backfill interval
                        bf_free_blocks[(interval_i, interval_f)].add(node)

                        # If the beginning of the interval is before the maximum relevant start time,
                        # then we will include this node in our backfilling search.
                        # TODO: Need to revisit this to better understand bf_max_relevant_start
                        if interval_i <= bf_max_relevant_start:
                            dur  = interval_f - interval_i
                            prev = bf_nodes_free_now_max_reqtimes.get(node)
                            if prev is None or dur > prev:
                                bf_nodes_free_now_max_reqtimes[node] = dur
                            # bf_nodes_free_now_max_reqtimes[node] = interval_f - interval_i

                else: # (the beginning of the interval is later than the current time)
                    # Make the beginning of the Backfill interval relative to the current time
                    start = interval[0]
                    start_s = rel_secs.get(start)
                    if start_s is None:
                        start_s = rel_secs[start] = (start - sched_start).total_seconds()
                    interval_i = start_s - now_s
                    # Skip this interval if it begins after the end of the Backfill window
                    if interval_i >= bf_window:
                        continue

                    # Add this node to this Backfill interval
                    block = bf_free_blocks.get((interval_i, interval_f))
                    if block is None:
                        bf_free_blocks[(interval_i, interval_f)] = set(nodes)
                    else:
                        block.update(nodes)

                    # If the beginning of the interval is before the maximum relevant start time,
                    # then we will include this node in our backfilling search.
                    if interval_i <= bf_max_relevant_start:
                        for node in nodes:
                            dur  = interval_f - interval_i
                            prev = bf_nodes_free_now_max_reqtimes.get(node)
                            if prev is None or dur > prev:
                                bf_nodes_free_now_max_reqtimes[node] = dur
                            # bf_nodes_free_now_max_reqtimes[node] = interval_f - interval_i

        # Get the maximum request time for Backfilling for each reservation
        self.bf_max_reqtime = {
            resv : max(nodes_free_now_max_reqtimes.values()) if nodes_free_now_max_reqtimes else 0
            for resv, nodes_free_now_max_reqtimes in self.bf_nodes_free_now_max_reqtimes.items()
        }

    def _prep_bf_q(self):
        """
        Prepare a new Backfill queue.
        """
        # This is the queue of Backfill jobs
        self.bf_queue = []

        # This is an OrderedDict of request times, indexed by jobs, in the Backfill queue, 
        # sorted by their request time in ascending order
        self.bf_job_ordered_reqtimes = {}

        # NOTE Only checking if we exceed the bf_max_job_test for the normal queue because resv queues 
        # are usually small. Should really just have a single queue with reservation sorted to the front, 
        # this would clean up some code

        # Get the maximum number of jobs from the normal queue (based on bf_max_jobs_test)
        # while ensuring we check jobs with reservations first.
        max_test_from_normal_q = (
            self.config.bf_max_job_test - 
            sum(len(jobs_in_resv_queue) for jobs_in_resv_queue in self.queue.reservations.values())
        )
        # If this is negative, we don't try to backfill any normal jobs
        if max_test_from_normal_q > 0:
            self.bf_job_ordered_reqtimes[""] = {}
            for job in self.queue.queue[-max_test_from_normal_q:]:
                self.bf_queue.append(job)
                self.bf_job_ordered_reqtimes[""][job] = job.reqtime.total_seconds()

        # Process reservations in deterministic order, by earliest end time, then resv name
        for resv, resv_q in sorted(
            self.queue.reservations.items(),
            key=lambda kv: (self.resv_end_times.get(kv[0], datetime.datetime.max), kv[0])
        ):
            self.bf_job_ordered_reqtimes[resv] = {}
            for job in resv_q:
                self.bf_queue.append(job)
                self.bf_job_ordered_reqtimes[resv][job] = job.reqtime.total_seconds()

        # Shortest reqtime job at front so we can call next(iter()) to grab it, use Job ID to break ties
        self.bf_job_ordered_reqtimes = {
            resv : OrderedDict(
                (job, reqtime)
                for job, reqtime in sorted(
                    job_ordered_reqtimes.items(), 
                    # key=lambda job_reqtime: job_reqtime[1]
                    key=lambda jr: (jr[1], jr[0].jid)
                )
            )
            for resv, job_ordered_reqtimes in self.bf_job_ordered_reqtimes.items()
        }


        # If there are no more jobs in the queue that could possibly be started now, we shouldnt
        # waste time backfilling. Still want to go throught the yield cycles and just pretend
        # we are actually backfilling. This is more likely to flick to true as the backfill
        # schedule fills up and jobs are processed from the queue. I think this is saving time but
        # the amount saved will depend on the workload and typical requested times.
        # Adding this exception handling because we are getting a KeyError where resv is not in bf_max_reqtime -KM
        try: 
            self.bf_done = {
                resv : (
                    next(iter(job_ordered_reqtimes.values())) > self.bf_max_reqtime[resv]
                    if job_ordered_reqtimes
                    else True
                )
                for resv, job_ordered_reqtimes in self.bf_job_ordered_reqtimes.items()
            }
        except:
            for resv, job_ordered_reqtimes in self.bf_job_ordered_reqtimes.items():
                if resv not in self.bf_max_reqtime.keys():
                    logging.info(f'Error: resv {resv} not in bf_max_reqtime.')

    def _backfill(self, n_try):
        """
        Backfill jobs.

        Arguments:
        - n_try: how many jobs the Backfilling loop can try to schedule every time it holds the locks.
        """
        # Get the jobs to be backfilled
        backfill_now, loop_finished = self._get_backfill_jobs(n_try)

        # We have gone through a full backfilling lock-hold
        self.bf_locks_remaining -= 1

        # loop_finished is False if there are still more jobs in the queue
        self.bf_loop_active = bool(not loop_finished and self.bf_locks_remaining)

        self.bf_secs_past += self.config.bf_yield_interval.total_seconds()

        # Increment this count for analysis
        self.sched_backfill_num += len(backfill_now)

        
        started_by_queue = {}
        for job, nodes in backfill_now:
            started = started_by_queue.get(job.reservation)
            if started is None:
                started = started_by_queue[job.reservation] = set()
            started.add(job)
            # start the job on the selected nodes
            self._submit(job.start_job(self.time), nodes)

        # Remove the started jobs from each queue in a single pass rather than
        # one linear scan per job
        for resv, started in started_by_queue.items():
            queue = self.queue.reservations[resv] if resv else self.queue.queue
            remaining = [job for job in queue if job not in started]
            if len(remaining) != len(queue) - len(started):
                raise ValueError(
                    "Backfilled job(s) missing from queue {!r}".format(resv)
                )
            queue[:] = remaining

    def _get_backfill_jobs(self, n_try):
        """
        Get the jobs to be backfilled.

        Arguments:
        - n_try: how many jobs the Backfilling loop can try to schedule every time it holds the locks.
        """
        backfill_now = []

        # While we can try more jobs and there are still jobs in the backfill queue
        while n_try and self.bf_queue:
            # Get the next job in the queue
            job = self.bf_queue.pop()

            # If this job should be held due to reaching QOS limits, skip it
            if job.partition_qos.hold_job(job):
                continue

            # sched loop could've scheduled between loops
            if (
                job.state == JobState.RUNNING or
                job.state == JobState.COMPLETED or
                job.state == JobState.CANCELLED
            ):
                continue

            n_try -= 1

            # NOTE: If there is no reservation, resv will be the empty string
            resv = job.reservation

            # If this reservation has no more jobs that can be backfilled, skip this job
            try:
                if self.bf_done[resv]:
                    continue
            except:
                logging.info(f'resv {resv} not found in bf_done')
                continue

            # This reservation is 'done' if the maximum request time is less than 
            # minimum request time of any job in this reservation
            self.bf_done[resv] = (
                next(iter(self.bf_job_ordered_reqtimes[resv].values())) >
                self.bf_max_reqtime[resv]
            )

            # Get the request time for this job
            reqtime = self.bf_job_ordered_reqtimes[resv].pop(job)

            # Get the free blocks (interval, nodes) for this reservation
            free_blocks = self.bf_free_blocks[resv]

            # The number of nodes available to accomodate this job
            num_free_nodes = 0

            # The (interval, {nodes}) that can potentially accomodate this job
            selected_intervals = defaultdict(set)

            # Sort the free blocks by the beginning time of the interval
            sorted_free_blocks = sorted(free_blocks.items(), key=lambda block: block[0])

            try:
                # Get the beginning time of the earliest interval in the reservation
                usage_block_start = sorted_free_blocks[0][0][0]
                # The usage block end signifies the end of the block of time where the job
                # might run. This is pushed forward more and more as intervals with nodes are
                # added, until the start of the interval is later than the "current time" (which
                # is relative, and held by bf_secs_past). This handles the issue with finding a
                # block of time available to enough nodes that is big enough to accomodate the job.
                # The idea being that you push forward the usage_block_start until you get past the
                # current time, and add intervals as you go, and update usage_block_end as equal
                # to usage_block_start + reqtime (so the block is big enough for the job) then afterwards 
                # you remove any intervals (and associated nodes) that have end time earlier than the
                # usage_block_end. This way, you trim the usage block on both ends, with the usage block
                # starting and ending at the optimal times to maximize the number of nodes available,
                # then you can see how many nodes are available for the entirety of that usage block.
                usage_block_end = usage_block_start + reqtime
            except:
                # It looks like this is happening because all of the nodes
                # in the interval associated with this reservation were used
                # and there were no more intervals with available nodes left.
                # From what I can understand, this is only possible if all of the
                # nodes available to a reservation have already been used for backfilling
                # in this backfill loop. If this is the case, it is fine that we skip this
                # job (we don't backfill it) because all of the nodes it could be
                # backfilled on are currently occupied by other jobs that were backfilled.
                logging.info('Error getting usage_block_start.')
                logging.info(f'Job: {job.jid} Reservation: {resv} Partition: {job.partition.name}')
                logging.info(f'Sorted Free Blocks: {sorted_free_blocks}')
                continue


            if job.nodes > self.config.max_switch_nodes or self.time >= job.max_switch_wait_time:
                max_rack_wait = False
            else:
                max_rack_wait = True

            if max_rack_wait:
                valid_nodes_by_rack = defaultdict(set)

            for i_block, (interval, nodes) in enumerate(sorted_free_blocks):
                # If this new interval doesn't have an end time that is late enough
                # to accomodate this job's request time, then we skip it.
                if interval[1] >= usage_block_end and interval[1] >= interval[0] + reqtime:
                    # Get the valid nodes for this interval (nodes that are available to this job's partition)
                    if max_rack_wait:
                        valid_nodes = set()
                        for node in nodes:
                            if job.partition.name in node.partition_names:
                                valid_nodes.add(node)
                                valid_nodes_by_rack[node.rack].add(node)
                    else:
                        valid_nodes = { node for node in nodes if job.partition.name in node.partition_names }
                        
                    if valid_nodes:
                        selected_intervals[interval] = valid_nodes
                        num_free_nodes += len(valid_nodes)
                        usage_block_start = interval[0]
                        usage_block_end = usage_block_start + reqtime
                        # If this interval ends after the backfill window closes (and thus can't accomodate this job),
                        # all of the rest will as well
                        if usage_block_end > self.bf_window:
                            break

                # If we haven't looked through all the intervals and either 
                # the next interval starts at the same time as this one or 
                # the next interval starts before the current time (bf_secs_past),
                # then we need to check the next interval, because it's possible
                # the nodes in the next interval block will be available for this job.
                if (
                    not i_block + 1 == len(sorted_free_blocks) and 
                    (
                        sorted_free_blocks[i_block + 1][0][0] == interval[0] or 
                        sorted_free_blocks[i_block + 1][0][0] <= self.bf_secs_past 
                    )
                ):
                    continue
                        
                # If we have gathered enough nodes to accomodate the job...
                # Note: Nodes need to be on the same rack if we haven't reached the job's max_switch_wait_time
                # If job needs more than self.config.max_switch_nodes, it will not fit on a rack, so we skip the max_switch_wait_time.
                # This is a hack, because Slurm actually tries to find the optimal communication setup for the job,
                # but that is going to require a more complicated approach. Will return to this -KM
                if max_rack_wait:
                    enough_on_rack_condition = any(len(valid_rack_nodes) >= job.nodes for valid_rack_nodes in valid_nodes_by_rack.values())

                if (max_rack_wait and enough_on_rack_condition) or (not max_rack_wait and job.nodes <= num_free_nodes):
                    # Since usage_block_end has moved forward to reflect the latest interval start time
                    # plus the reqtime, it's possible that earlier intervals end before this usage_block_end
                    for selected_interval in list(selected_intervals.keys()):
                        if usage_block_end > selected_interval[1]:
                            # The nodes associated with this interval need to be removed from the set of valid nodes.
                            interval_nodes = selected_intervals.pop(selected_interval)
                            num_free_nodes -= len(interval_nodes)
                            if max_rack_wait:
                                for rack in valid_nodes_by_rack:
                                    valid_nodes_by_rack[rack] -= interval_nodes

                    # We may have just removed nodes, check to make sure we still have enough for this job
                    if max_rack_wait:
                        if not any(len(valid_rack_nodes) >= job.nodes for valid_rack_nodes in valid_nodes_by_rack.values()):
                            continue
                        # All nodes need to be on the same rack
                        # Sort by rack size, then by rack ID as a tie-breaker
                        best_rack, best_nodes = max(
                            valid_nodes_by_rack.items(),
                            key=lambda kv: (len(kv[1]), kv[0])
                        )
                        selected_nodes = list(best_nodes)
                    else:
                        if job.nodes > num_free_nodes:
                            continue
                        # Can use any nodes
                        selected_nodes = [node for nodes in selected_intervals.values() for node in nodes]
                    

                    # Prioritise nodes that are available if job might be able to start now
                    if usage_block_start <= self.bf_secs_past:
                        selected_nodes.sort(
                            key=lambda node: (
                                node.running_job is not None, node.sched_order
                            ),
                            reverse=True
                        )
                    else:
                        selected_nodes.sort(key=_sched_order_key, reverse=True)

                    # Select only as many nodes as we need
                    selected_nodes = selected_nodes[len(selected_nodes)-job.nodes:]

                    # Run the job
                    if usage_block_start <= self.bf_secs_past:
                        if job.cancel is not None:
                            self.queue.cancel_job(job)
                            break

                        # Node may have been allocated by sched during the yield_sleep or a
                        # job is running overtime. Cannot schedule the job in this case
                        if all(node.running_job is None for node in selected_nodes):
                            backfill_now.append((job, selected_nodes))

                    recompute_bf_max_reqtime = False

                    # Get usage block start/end in resolution dictated by bf_resolution
                    usage_block_start = (
                        int(usage_block_start // self.bf_resolution * self.bf_resolution)
                    )
                    usage_block_end = max(
                        int(usage_block_end // self.bf_resolution * self.bf_resolution),
                        self.bf_resolution
                    )
                    
                    # Bookkeeping to update backfill state based on selected nodes assigned to job
                    for selected_interval, nodes in selected_intervals.items():
                        nodes = nodes.intersection(selected_nodes)
                        if not nodes:
                            continue

                        # Remove the selected nodes from their free blocks
                        free_blocks[selected_interval] -= nodes

                        # If the start of this node's free block is earlier than the 
                        # start of when this node is planned for this job (usage_block_start),
                        # create an interval where this node is free between the start of the 
                        # current free block to when this job is planned on this node.
                        if selected_interval[0] < usage_block_start:
                            self.planned_nodes.update(nodes)
                            free_blocks[(selected_interval[0], usage_block_start)].update(nodes)

                        # If the end of this node's free block is later than the 
                        # end of this job's planned runtime on this node (usage_block_end)
                        # create an interval where this node is free beween the job's planned end 
                        # time and the end of this node's current free block.
                        if selected_interval[1] > usage_block_end:
                            free_blocks[(usage_block_end, selected_interval[1])].update(nodes)

                        # This interval is relevant for the max reqtime so need to update tracking
                        # information
                        # If this interval starts before the maximum possible start time
                        if selected_interval[0] <= self.bf_max_relevant_start:
                            # We can still use these nodes if the reqtime is shorter than the difference
                            # between the usage_block_start and the interval start
                            new_reqtime_early = usage_block_start - selected_interval[0]
                            old_reqtime = selected_interval[1] - selected_interval[0]
                            
                            # Very short job can create two possible reqtimes by finishing before
                            # max relevant start
                            # We can still use these nodes if the reqtime is shorter than the difference
                            # between the usage_block_end and the interval end and the usage_block_end is
                            # earlier than the maximum possible start time
                            if usage_block_end < self.bf_max_relevant_start:
                                new_reqtime_late = selected_interval[1] - usage_block_end
                            else:
                                new_reqtime_late = None

                            for node in nodes:
                                try:
                                    self.bf_nodes_free_now_max_reqtimes[resv][node]
                                except:
                                    logging.info(f'Node not found in bf_nodes_free_now_max_reqtimes for resv: {resv}')
                                    logging.info(f'Node ID: {node.nid}')
                                    logging.info(f'Node Partitions: {node.partition_names}')
                                    continue # Skip this error for now - 
                                    
                                # If the maximum request time for this node has already been changed, 
                                # don't change it here
                                if (
                                    self.bf_nodes_free_now_max_reqtimes[resv][node] != old_reqtime
                                ):
                                    continue

                                
                                if new_reqtime_late is None:
                                    if new_reqtime_early <= 0:
                                        # If there is no viable new reqtime, this node is no longer free
                                        self.bf_nodes_free_now_max_reqtimes[resv].pop(node)
                                    else:
                                        self.bf_nodes_free_now_max_reqtimes[resv][node] = (
                                            new_reqtime_early
                                        )
                                else:
                                    # If both are possible, choose the maximum of the two
                                    self.bf_nodes_free_now_max_reqtimes[resv][node] = max(
                                        new_reqtime_early, new_reqtime_late
                                    )

                                recompute_bf_max_reqtime = True

                        # Remove this interval if there are no more nodes associated with it
                        if not free_blocks[selected_interval]:
                            free_blocks.pop(selected_interval)

                    # Get the new max reqtime for backfilling
                    if recompute_bf_max_reqtime:
                        if self.bf_nodes_free_now_max_reqtimes[resv]:
                            self.bf_max_reqtime[resv] = max(
                                self.bf_nodes_free_now_max_reqtimes[resv].values()
                            )
                        else:
                            self.bf_max_reqtime[resv] = 0

                    break

        return backfill_now, not self.bf_queue

    def _check_down_nodes(self):
        """
        Check if we need to update down nodes.

        Some nodes will be switched from up to down, others will
        be switched from down to up.
        """
        node_update = False

        # Handle nodes that are down but need to be changed to up
        # self.down_nodes is sorted by up_time in descending order, so
        # the last node in the list has the earliest up_time.
        while self.down_nodes and self.down_nodes[-1].up_time <= self.time:
            node = self.down_nodes.pop()

            # This node is now up (no longer down)
            node.set_up()

            # If this node is currently reserved
            if node.reservation:
                # Set the end of the interval time to the end of the reservation
                node.interval_times[1] = node.unreserved_time
            # If the node is not reserved but has any reservations scheduled
            elif node.reservation_schedule:
                # Set the end of the interval time to the start of the next reservation
                node.interval_times[1] = node.reservation_schedule[-1][0]
            else:
                # Otherwise, set the end of the interval time to datetime.max
                node.interval_times[1] = datetime.datetime.max
            # Set the beginning of the interval time to the current time
            node.interval_times[0] = self.time

            # Add this node to the relevant reservation-interval node set
            self.partitions.add_free_block(node)

            node_update = True

        # To avoid sorting many times in the same loop, create a temporary set of 
        # nodes that will be added back into nodes_that_will_go_down, and capture
        # the current number of down nodes
        _nodes_that_will_go_down, orig_len_down_nodes = set(), len(self.down_nodes)

        # Handle nodes scheduled to go down
        # This is tricky for nodes that have jobs running on them. In reality, these
        # node down events would cause job failure. Here, since jobs do not usually run
        # on the exact same nodes they ran on in reality, the node(s) it is running on
        # can go down but the job should continue. So, we need to gracefully handle this,
        # by allowing the node to go down while keeping the job running.
        #
        # The way this is handled is:
        # 1) If the node has no job running on it, put it in the down state
        # 2) If the node is already down, add this new down event to the end of the current
        #    down event. This occurs because the simulator can add down times to nodes (see below)
        # 3) If a job is running on this node, and there is another node available to this job that 
        #    is currently free, put this node in the down state, and swap in that available node for 
        #    this one.
        # 4) If this is a single-node job, and there are no other nodes available to it, end this
        #    job and put a new job at the front of the queue with the same parameters, except its
        #    runtime is equal to the remaining runtime for this job.
        # 5) If this is a multi-node job, and there are no other nodes available to it, shrink the size
        #    of this job by one node, and create a new single-node job, and handle that the same as 4.
        #    This is a bit odd, but it is better than killing this multi-node job and putting it back
        #    on the queue, which could result in very big jobs going back on the queue multiple times,
        #    which has a big effect on queue time and system utilization (due to many nodes needing to
        #    be drained multiple times in preparation for the big job every time it is put back on the queue)
        #
        # This method causes some bookkeeping headaches, because we want to track that any new jobs created
        # were splinters from the original jobs, and we need to be aware of the shrinking of multi-node jobs.
        while (self.nodes_that_will_go_down and
            self.nodes_that_will_go_down[-1].down_schedule[-1][0] <= self.time):

            node = self.nodes_that_will_go_down.pop()

            # If the node is *already* down, just re-queue its next
            # down event after it comes back up.
            if node.down:
                node.down_schedule[-1][0] = node.up_time # shift to next up
                _nodes_that_will_go_down.add(node)
                continue

            # If a job is running here, try to honor the historical
            # down-time without killing the job outright.
            job = node.running_job
            terminate_job = False
            if job is not None:
                remaining_time = job.end - self.time
                replacement    = self._find_replacement_node(job, remaining_time)

                # Replacement node found: simple swap
                if replacement:
                    # print(f"Node {node.nid} down {self.time}; swapped into node {replacement.nid} for job {job.jid}")

                    # unlink failing node
                    job.assigned_nodes.remove(node)
                    node.running_job = None

                    # link replacement node
                    self.partitions.remove_free_block(replacement)
                    replacement.interval_times[0] = job.endlimit
                    self.partitions.add_free_block(replacement)
                    job.assign_node(replacement)

                # No replacement available
                else:
                    if len(job.assigned_nodes) > 1:
                        # shrink the original job by one node
                        # print(f"Node {node.nid} down @{self.time}; shrinking job {job.jid} to {len(job.assigned_nodes) - 1} nodes")
                        # print(f" -- Spawned single-node job {new_job.jid} (runtime {remaining_time})")
                        job.assigned_nodes.remove(node)
                        job.nodes -= 1
                        node.running_job = None

                        # Record shrink in job node timeline
                        job.node_timeline.append((self.time, job.nodes))

                        new_jid = f"resub_{job.jid}_{node.nid}"
                    else:
                        # single-node job: end & respawn
                        # print(f"Node {node.nid} down @{self.time}; terminating one-node job {job.jid}")
                        # print(f" -- Respawned as job {new_job.jid} (runtime {remaining_time})")
                        terminate_job = True

                        new_jid = f"resub_{job.jid}"
                    
                    # spawn a new single-node job for the remainder
                    new_job = Job(
                        jid=new_jid,
                        submit=self.time,
                        max_switch_wait=self.config.max_switch_wait,
                        nodes=1,
                        runtime=remaining_time,
                        reqtime=remaining_time,
                        node_power=job.node_power,
                        true_node_power=job.true_node_power,
                        true_job_start=self.time,
                        user=job.user,
                        account=job.account,
                        qos=job.qos,
                        partition=job.partition,
                        partition_qos=job.partition_qos,
                        dependency_arg='',
                        name=job.name,
                        reason=job.reason,
                        reservation_arg=job.reservation,
                        begin_arg='',
                        cancelled=None,
                        nodelist_arg='',
                        exclude_arg='',
                        predicted_power=job.predicted_power,
                        predicted_runtime=job.predicted_runtime,
                        track_qos=False,
                    )
                    new_job.dependency       = job.dependency
                    new_job.assoc            = job.assoc
                    new_job.ignore_in_eval   = job.ignore_in_eval
                    new_job.cancelled_t      = job.cancelled_t

                    self.queue.all_jobs.append(new_job)

            # Bring the node down as required by history
            down_entry = node.down_schedule.pop()
            up_time = self.time + down_entry[1]

            self.partitions.remove_free_block(node)
            node.set_down(up_time)
            node.interval_times = [datetime.datetime.max, datetime.datetime.max]
            self.down_nodes.append(node)

            # Schedule future downs (if any) back into the queue
            if node.down_schedule:                            # more events remain
                _nodes_that_will_go_down.add(node)

            node_update = True

            if terminate_job:
                job.end = self.time
                self.running_jobs.remove(job)
                self._end_job(job)


        # If nodes change between bf yield intervals, the bf loop breaks (even with bf_continue)
        if node_update and self.bf_loop_active:
            self.bf_queue = []

        # If there have been nodes added to the list of down nodes, sort the down nodes list.
        if len(self.down_nodes) != orig_len_down_nodes:
            # self.down_nodes.sort(key=lambda node: node.up_time, reverse=True)
            self.down_nodes.sort(key=lambda node: (node.up_time, node.nid), reverse=True)

        # If we need to add nodes back in to the nodes_that_will_go_down list,
        # do so, and then sort the list
        if _nodes_that_will_go_down:
            for node in _nodes_that_will_go_down:
                self.nodes_that_will_go_down.append(node)
            self.nodes_that_will_go_down.sort(
                key=lambda node: (node.down_schedule[-1][0], node.nid), reverse=True
            )

    
    def _find_replacement_node(self, job, remaining_time):
        """
        Return a free node that can run `job` for `remaining_time` seconds,
        or None if none exist.
        """
        candidates = [n for interval, nodes in self.partitions.free_blocks[job.reservation].items()
                        if interval[0] <= self.time
                        for n in nodes
                            if n.running_job is None and
                            job.partition.name in n.partition_names and
                            self.time + remaining_time <= n.interval_times[-1]]
        return min(candidates, key=_sched_order_key, default=None)


    def _check_reservations(self):
        """
        Update reservations.
        """
        self.resv_update = False

        # Handle nodes whose reservation ended
        self._process_unreservations()

        # Handle new reservations (advance + impromptu)
        self._process_due_reservations(
            q_attr="nodes_that_will_be_reserved",
            sched_attr="reservation_schedule",
            update_interval_start=False,
        )
        self._process_due_reservations(
            q_attr="nodes_that_will_be_impromptu_reserved",
            sched_attr="impromptu_reservation_schedule",
            update_interval_start=True,
        )

        # Obey backfill-loop contract
        if self.resv_update and self.bf_loop_active:
            self.bf_queue.clear()

        # Handle HPE specific reservations
        self.handle_hpe_specific_reservations()

        # If reservations change between bf yield intervals, 
        # the bf loop breaks (even with bf_continue)
        if self.resv_update and self.bf_loop_active:
            self.bf_queue = []


    # Unreserve nodes whose end-time has passed
    def _process_unreservations(self):
        while self.reserved_nodes:
            node = self.reserved_nodes[-1]

            # Edge-case: “open-ended” reservation marker: just drop it
            if node.unreserved_time is None:
                logging.info("Popping unreserved node %s", node.nid)
                self.reserved_nodes.pop()
                continue

            # Not yet time to unreserve: stop scanning (list is reverse-sorted)
            if node.unreserved_time > self.time:
                break

            # We will unreserve this node now
            self.reserved_nodes.pop()

            if node.down:
                node.set_unreserved()
                continue  # leave it down

            self.partitions.remove_free_block(node)
            node.set_unreserved()

            # If it has a *future* reservation, shorten interval; else extend to datetime max
            future = (node.reservation_schedule[-1][0]
                      if node.reservation_schedule else datetime.datetime.max)
            node.interval_times[-1] = future
            self.partitions.add_free_block(node)

            self.resv_update = True

    # Process any reservation whose start-time <= now
    def _process_due_reservations(self, *, q_attr, sched_attr,
                                  update_interval_start: bool):
        queue = getattr(self, q_attr)
        key = lambda n: (getattr(n, sched_attr)[-1][0], n.nid)  # for sorting

        while queue and getattr(queue[-1], sched_attr)[-1][0] <= self.time:
            node = queue[-1] # peek (queue is reverse-sorted)
            schedule = getattr(node, sched_attr).pop() # (start, end, name)
            append_now = True

            if node.down: # overlapping reservation on a DOWN node
                append_now = self._handle_overlap(node, schedule, sched_attr)
                self._update_or_pop_queue(queue, node, sched_attr, key)
                node.set_reserved(schedule[2], schedule[1])
                if append_now:
                    # self._sorted_append(self.reserved_nodes, node,
                    #                     key=lambda n: n.unreserved_time)
                    self._sorted_append(self.reserved_nodes, node, 
                                        key=lambda n: (n.unreserved_time, n.nid))

                continue

            # Normal reservation path 
            self.partitions.remove_free_block(node)
            append_now = self._handle_overlap(node, schedule, sched_attr)
            self._update_or_pop_queue(queue, node, sched_attr, key)

            node.set_reserved(schedule[2], schedule[1])

            if update_interval_start:
                node.interval_times[0] = self.time
            node.interval_times[-1] = node.unreserved_time

            self.partitions.add_free_block(node)
            if append_now:
                # self._sorted_append(self.reserved_nodes, node,
                #                     key=lambda n: n.unreserved_time)
                self._sorted_append(self.reserved_nodes, node, 
                                    key=lambda n: (n.unreserved_time, n.nid))


            self.resv_update = True

    # Overlapping reservation logic factored out
    def _handle_overlap(self, node, schedule, sched_attr):
        """
        Returns False if the caller should *not* append the node to reserved_nodes
        because it was already there (i.e. overlapping reservation).
        """
        if node in self.reserved_nodes and schedule[1] < node.unreserved_time:
            getattr(node, sched_attr).append(
                (schedule[1], node.unreserved_time, node.reservation)
            )
            getattr(node, sched_attr).sort(key=lambda s: s[0], reverse=True)
            logging.info("Node %s in reserved_nodes – split overlap", node.nid)
            return False
        return True

    # Manage upcoming-reservation queue after pop
    def _update_or_pop_queue(self, queue, node, sched_attr, key):
        if not getattr(node, sched_attr):
            queue.pop() # no more upcoming reservations
        else:
            queue.sort(key=key, reverse=True)

    # Sorted append without full list.sort
    @staticmethod
    def _sorted_append(lst, item, *, key):
        keys = [key(x) for x in lst]
        # bisect maintains ascending order; list is stored descending
        idx  = len(keys) - bisect.bisect_left(list(reversed(keys)), key(item))
        lst.insert(idx, item)



    def handle_hpe_specific_reservations(self):
        # Destroy and spawn new sliding reservations
        while self.sliding_reservations and self.sliding_reservations[-1][0] <= self.time:
            _, submit, clear, start, end, nodes, name = self.sliding_reservations[-1]
            for node in nodes:
                if submit is not None: # At event where submitting a new resv block
                    node.reservation_schedule.append((start, end, name))
                    node.reservation_schedule.sort(key=lambda schedule: schedule[0], reverse=True)

                    if node.down:
                        continue

                    self.resv_update = True

                    self.partitions.remove_free_block(node)

                    node.interval_times[-1] = node.reservation_schedule[-1][0]
                    self.partitions.add_free_block(node)

                # At event where clearing reservation that is about to start in anticipation of the
                # next resv in the sliding resv
                else: 
                    for i_res_sched, res_sched in enumerate(node.reservation_schedule):
                        if res_sched[2] != name:
                            continue
                        node.reservation_schedule.pop(i_res_sched)

                        if node.down:
                            break

                        self.resv_update = True

                        self.partitions.remove_free_block(node)

                        if node.reservation_schedule:
                            node.interval_times[-1] = node.reservation_schedule[-1][0]
                        else:
                            node.interval_times[-1] = datetime.datetime.max

                        self.partitions.add_free_block(node)

                        break

            if submit is None: # Finished this resv step
                self.sliding_reservations.pop()
            else: # Created this resv step, need to clear it next
                self.sliding_reservations[-1][0] = clear
                self.sliding_reservations[-1][1] = None
                # If clear is same time as next submit, want to clear first
                self.sliding_reservations.sort(
                    key=lambda sliding_res: (sliding_res[0], sliding_res[1] is not None),
                    reverse=True
                )

    
    def log_simulation_step_power(self):
        """
        Appends the current simulation time (rounded to the nearest second) and the
        current cluster power usage to a CSV file.
        
        If self.time is a datetime, it uses its timestamp; if it's already a float,
        it assumes it's in seconds.
        """
        # Convert self.time to an ISO string
        # Keep tz info if it exists; otherwise it will be tz-naive
        t_str = self.time.isoformat()
        
        # Prepare the row to log: [time, power_usage, predicted_power_usage]
        row = [t_str, self.power_usage, self.predicted_power_usage]
        
        # Check if file exists to decide whether to write the header.
        file_exists = os.path.isfile(self.power_log_fp)
        
        with open(self.power_log_fp, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Write header if file did not exist before.
            if not file_exists:
                writer.writerow(["time", "power_usage", "predicted_power_usage"])
            writer.writerow(row)


    def _print_stats(self):
        # Called from the status block in run_sim, which already throttles
        # to status_interval of simulated time.

        # Console that can also export plain text for logging
        if not hasattr(self, "_console"):
            # record=True lets us export a plain-text copy for logs
            self._console = Console(record=True)
        console: Console = self._console

        # ---------- Compute once, reuse ----------
        running_jobs = self.running_jobs
        queued_now   = self.queue.queue
        waiting_dep  = self.queue.waiting_dependency
        parts        = self.partitions.partitions

        cluster_running_jobs = len(running_jobs)
        cluster_queued_jobs  = len(queued_now) + len(waiting_dep)
        cluster_running_nodes = sum(job.nodes for job in running_jobs)
        cluster_queued_nodes  = sum(job.nodes for job in queued_now)
        cluster_idle_nodes    = sum(1 for node in self.partitions.nodes if node.free)
        cluster_down_nodes    = sum(1 for node in self.down_nodes if not node.running_job)
        cluster_reserved      = sum(1 for node in self.partitions.nodes if node.reservation)
        cluster_idle_reserved = sum(1 for node in self.partitions.nodes
                                    if node.reservation and not (node.running_job or node.down))
        cluster_down_reserved = sum(1 for node in self.partitions.nodes
                                    if node.reservation and node.down)

        # Precompute per-partition metrics in one pass per partition (readable and fast enough hourly)
        def part_metrics(p):
            rj = sum(1 for j in running_jobs if j.partition is p)
            qj = sum(1 for j in queued_now   if j.partition is p)
            rn = sum(j.nodes for j in running_jobs if j.partition is p)
            qn = sum(j.nodes for j in queued_now   if j.partition is p)
            idle = sum(1 for n in p.nodes if n.free)
            down = sum(1 for n in p.nodes if n.down)
            res  = sum(1 for n in p.nodes if n.reservation)
            ires = sum(1 for n in p.nodes if n.reservation and not (n.running_job or n.down))
            dres = sum(1 for n in p.nodes if n.reservation and n.down)
            return rj, qj, rn, qn, idle, down, res, ires, dres

        # Header
        console.print(Rule("[bold cyan]FastSim Status[/bold cyan]"))
        header = Table.grid(expand=True)
        header.add_column(justify="left", style="bold")
        header.add_column(justify="right")
        header.add_row(
            f"Step: [white]{self.step_cnt:,}[/] • Time: [white]{self.time}[/]",
            ("Scheduled: [white]{:,}[/] • Backfilled: [white]{:,}[/] • "
            "Power: [white]{:.4f} MW[/] • Pred: [white]{:.4f} MW[/]").format(
                self.sched_main_num, self.sched_backfill_num, self.power_usage, self.predicted_power_usage
            )
        )
        console.print(Panel(header, border_style="cyan"))
        # ------------------------------------------------------------------
        # Shared column spec (fixed widths) so Cluster Summary and Per-Partition
        # align perfectly column-for-column.
        # ------------------------------------------------------------------
        COLS = [
            # (header, justify, width, style)
            ("Partition",   "left",  18, "bold"),   # adjust width if you have long partition names
            ("Run Jobs",    "right",  9, None),
            ("Q Jobs",      "right",  7, None),
            ("Run Nodes",   "right", 10, None),
            ("Q Nodes",     "right",  9, None),
            ("Idle",        "right",  6, None),
            ("Down",        "right",  6, None),
            ("Resvd",       "right",  6, None),
            ("Idle Resvd",  "right", 10, None),
            ("Down Resvd",  "right", 10, None),
        ]

        def add_fixed_cols(t: Table, header_style: str):
            for name, just, w, style in COLS:
                # no_wrap=True prevents Rich from resizing due to wrapping
                t.add_column(name, justify=just, width=w, no_wrap=True, style=style, header_style=header_style)


        # Cluster Summary (aligned with Per-Partition via fixed column widths)
        cluster = Table(
            title="Cluster Summary",
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        add_fixed_cols(cluster, header_style="bold magenta")

        cluster.add_row(
            "Cluster",
            f"{cluster_running_jobs:,}",
            f"{cluster_queued_jobs:,}",
            f"{cluster_running_nodes:,}",
            f"{cluster_queued_nodes:,}",
            f"[green]{cluster_idle_nodes:,}[/]",
            f"[red]{cluster_down_nodes:,}[/]",
            f"[yellow]{cluster_reserved:,}[/]",
            f"[green]{cluster_idle_reserved:,}[/]",
            f"[red]{cluster_down_reserved:,}[/]",
        )
        console.print(cluster)


        # Per-Partition
        parts_table = Table(
            title="Per-Partition",
            header_style="bold cyan",
            box=box.SIMPLE_HEAVY,
            expand=True
        )
        add_fixed_cols(parts_table, header_style="bold cyan")


        for p in sorted(parts, key=lambda p: p.name):
            rj, qj, rn, qn, idle, down, res, ires, dres = part_metrics(p)
            parts_table.add_row(
                str(p.name),
                f"{rj:,}",
                f"{qj:,}",
                f"{rn:,}",
                f"{qn:,}",
                f"[green]{idle:,}[/]",
                f"[red]{down:,}[/]",
                f"[yellow]{res:,}[/]",
                f"[green]{ires:,}[/]",
                f"[red]{dres:,}[/]",
            )
        console.print(parts_table)

        # QOS line
        # (kept because it’s useful; also formatted with thousands separators)
        qos_counts = [
            f"{qos.name}={sum(1 for job in queued_now if job.qos is qos):,}"
            for qos in self.queue.qos_objects.values()
            if sum(1 for job in queued_now if job.qos is qos) > 0
        ]
        submit_holds_total = sum(len(jobs) for jobs in self.queue.qos_submit_held.values())
        submit_holds_parts = [
            f"{qos.name}={len(jobs):,}"
            for qos, jobs in self.queue.qos_submit_held.items() if len(jobs) > 0
        ]
        console.print(
            "[bold]Queued QOS:[/bold] " + ", ".join(qos_counts) +
            f"  |  [bold]QOS Submit Holds:[/bold] {submit_holds_total:,} (" +
            ", ".join(submit_holds_parts) + ")" +
            f"  |  [bold]Waiting on dependency:[/bold] {len(self.queue.waiting_dependency):,}"
        )
        console.print(Rule(style="dim"))

        # Optional HPE section
        if getattr(self, "sliding_reservations", False):
            restrict = sum(
                1 for n in self.partitions.nodes
                if (n.reservation_schedule and "HPE_RestrictLongJobs" in [t[2] for t in n.reservation_schedule])
            )
            restrict_idle = sum(
                1 for n in self.partitions.nodes
                if (n.reservation_schedule and
                    "HPE_RestrictLongJobs" in [t[2] for t in n.reservation_schedule] and
                    not n.running_job and not n.down)
            )
            console.print(
                f"NodesHPE_RestrictLongJobs = [bold]{restrict:,}[/] (Idle = [bold]{restrict_idle:,}[/])"
            )

        # Log output
        with self.run_logs.print_log_fp.open("a") as f:
            f.write(console.export_text(clear=True))
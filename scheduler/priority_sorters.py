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

import datetime

import numpy as np
import pandas as pd

PRIO_EPOCH = datetime.datetime(2000, 1, 1)
"""
Fixed reference time for the cached float-seconds timestamps used by the
specialized sort key. Trace timestamps are whole seconds, so offsets from
this epoch are exact floats and float subtraction reproduces
timedelta.total_seconds() bit-for-bit.
"""

class MFPrioritySorter:
    """
    Multifactor Priority sorting.

    Methods:
    __init__
    sort
    _partition_priority_tier
    _age_priority
    _size_priority
    _fairshare_priority
    _partition_priority
    _qos_priority
    """
    def __init__(
        self, init_time, 
        size_weight, age_weight, fairshare_weight, max_age, partition_weight,
        qos_weight, no_partition_priority_tiers, fairtree, total_nodes,
        # Energy-aware parameters:
        power_weight=0,           # PriorityWeightEnergy (if 0, energy factor is disabled)
        re_fp=None,         # Path to CSV file with columns "time" and "re_availability"
        P_min=None,               # Minimum expected power usage (Watts)
        P_max=None,               # Maximum expected power usage (Watts)
        power_alpha=None,         # Energy-Aware Scheduling configurable parameter
        power_beta=None,          # Energy-Aware Scheduling configurable parameter
        power_gamma=None,         # Energy-Aware Scheduling configurable parameter
        power_time_boost_start=None,  # Energy-Aware Scheduling configurable parameter
        power_time_boost_end=None     # Energy-Aware Scheduling configurable parameter
    ):
        """
        Initialize the priority sorter.

        Arguments:
            init_time: The minimum start time of any job in the job trace.
            size_weight: PriorityWeightJobSize - scales the contribution of the job size factor.
            age_weight: PriorityWeightAge - scales the contribution of the age factor.
            fairshare_weight: PriorityWeightFairshare - scales the contribution of the fairshare factor.
            max_age: PriorityMaxAge - Specifies the queue wait time at which the age factor maxes out.
            partition_weight: PriorityWeightPartition - scales the contribution of the partition factor.
            qos_weight: PriorityWeightQOS - scales the contribution of the quality of service factor.
            no_partition_priority_tiers: Boolean; if True then all partitions have the same priority.
            fairtree: the fairtree used in the fairshare algorithm.
            total_nodes: Total number of nodes (used to normalize the job size weight).
            
            power_weight: Weight for the energy-based priority boost.
            re_fp: Path to CSV file with columns "time" (seconds) and "re_availability" ([0,1]).
            P_min: Minimum expected job power (Watts).
            P_max:  Maximum expected job power (Watts).
            alpha, beta, gamma, time_boost_start, time_boost_end: configurable energy-aware sched params
        """

        self.size_weight = size_weight // total_nodes
        """
        Normalize the Job Size weight by the total number of nodes,
        So the maximum contribution is equal to the size_weight

        NOTE: This should be a floating point number, but it
        makes the arithmetic much more computationally expensive,
        and rounding to an int only affects the fifth significant
        digit of the weight, so this shouldn't change results too much.
        And, calculating the size priority factor has taken 11% of the
        total computation time, so eliminating this extra computation
        is significant.
        """
        
        self.age_weight = age_weight
        """
        The age contribution gets normalized by the max age later.
        """

        self.fairshare_weight = fairshare_weight
        self.max_age = max_age.total_seconds()
        self.partition_weight = partition_weight
        self.qos_weight = qos_weight
        self.time = init_time

        self.fairtree = fairtree

        # Relevant for HPE ARCHER2
        self.no_partition_priority_tiers = no_partition_priority_tiers

        # Energy-aware configuration
        if power_weight > 0:
            self.power_weight = power_weight
            if P_min is None or P_max is None:
                raise ValueError("P_min and P_max must be provided for regression energy model.")
            self.P_min = P_min
            self.P_max = P_max
            self.power_alpha = power_alpha
            self.power_beta = power_beta
            self.power_gamma = power_gamma

            # Create a lookup dictionary for RE availability if CSV path is provided.
            if re_fp != "":
                self._initialize_re_map(re_fp)
            else:
                self.re_map = None

            self.power_time_boost_start = power_time_boost_start
            self.power_time_boost_end = power_time_boost_end
            

        self.priority_factors = []
        """
        Populate the list of priority factors. Each of item of this list
        is a function that is used to calculate the specified priority 
        factor for a single job.
        """
        if size_weight:
            self.priority_factors.append(self._size_priority)
        if age_weight:
            self.priority_factors.append(self._age_priority)
        if fairshare_weight:
            self.priority_factors.append(self._fairshare_priority)
        if partition_weight:
            self.priority_factors.append(self._partition_priority)
        if qos_weight:
            self.priority_factors.append(self._qos_priority)
        if power_weight > 0:
            self.priority_factors.append(self._power_priority)

        self._fast_key_active = (
            None
            if power_weight > 0
            else (
                bool(size_weight), bool(age_weight), bool(fairshare_weight),
                bool(partition_weight), bool(qos_weight)
            )
        )
        """
        Frozen activation flags (size, age, fairshare, partition, qos) for the
        specialized sort key used by _fast_sort. None disables the fast path
        (the energy-aware factor is not specialized).
        """

    def sort(self, queue, time):
        """
        Sort the queue based on the Multifactor Priority of each job.

        Optionally sort first by partition priority tier.

        Otherwise, sort by the following factors (in order):
        - Multifactor Priority
        - The time elapsed since the job was submitted (how long it has been waiting)
        - The unique job ID
        """
        self.time = time
        if self._fast_key_active is not None:
            self._fast_sort(queue, time)
            return
        if self.no_partition_priority_tiers:
            queue.sort(
                key=lambda job: (
                    # Note: the square brackets in the sum() seem unnecessary, but end up cutting down
                    # the time by 20-30%, and this sum() expression amounts to a large proportion of sim time.
                    sum([priority_calc(job) for priority_calc in self.priority_factors]),
                    self.time - job.submit,
                    job.uniq_id
                )
            )
            return

        queue.sort(
            key=lambda job: (
                self._partition_priority_tier(job),
                sum([priority_calc(job) for priority_calc in self.priority_factors]),
                self.time - job.submit,
                job.uniq_id
            )
        )
        return

    def _fast_sort(self, queue, time):
        """
        Sort with a specialized key that orders identically to the generic
        multifactor key while avoiding per-job Python calls and datetime
        arithmetic.

        Equivalences relied on:
        - The per-job constant factor values (size, partition, QOS) are cached
          on the job the first time it is sorted, using the exact formulas of
          the corresponding _*_priority methods.
        - The age factor uses float-seconds offsets from PRIO_EPOCH cached by
          Job.priority(); the subtraction is bit-equal to
          (time - launch_time).total_seconds() for whole-second timestamps.
        - The factor values are added left to right in the same order as the
          generic sum([...]), so the total is bit-equal.
        - The (time - submit) tie-break is replaced by the negated submit
          offset, which orders identically at fixed time.
        """
        use_size, use_age, use_fairshare, use_partition, use_qos = self._fast_key_active
        time_s = (time - PRIO_EPOCH).total_seconds()
        size_weight = self.size_weight
        age_weight, max_age = self.age_weight, self.max_age
        fairshare_weight = self.fairshare_weight
        partition_weight = self.partition_weight
        qos_weight = self.qos_weight
        assocs = self.fairtree.assocs if use_fairshare else None
        tiered = not self.no_partition_priority_tiers

        if not (use_size or use_age or use_fairshare or use_partition or use_qos):
            # No active factors: the priority sum is a constant, so order is
            # (wait time, unique ID), i.e. (negated submit offset, unique ID)
            if tiered:
                queue.sort(
                    key=lambda job: (
                        job.partition.priority_tier, job._neg_submit_s, job.uniq_id
                    )
                )
            else:
                queue.sort(key=lambda job: (job._neg_submit_s, job.uniq_id))
            return

        def multifactor_key(job):
            static = job._mf_static
            if static is None:
                static = job._mf_static = (
                    min(job.nodes / 256, 1) * size_weight,
                    job.partition.priority_weight * partition_weight,
                    job.qos.priority * qos_weight,
                )
            prio = 0
            if use_size:
                prio = prio + static[0]
            if use_age:
                prio = prio + min((time_s - job._launch_s) / max_age, 1) * age_weight
            if use_fairshare:
                prio = prio + assocs[job.assoc].fairshare_factor * fairshare_weight
            if use_partition:
                prio = prio + static[1]
            if use_qos:
                prio = prio + static[2]
            if tiered:
                return (job.partition.priority_tier, prio, job._neg_submit_s, job.uniq_id)
            return (prio, job._neg_submit_s, job.uniq_id)

        queue.sort(key=multifactor_key)


    def _partition_priority_tier(self, job):
        """
        Returns the priority tier of the job's partition.

        From Slurm Documentation:
        Configure the partition's PriorityTier setting relative to other partitions to control 
        the preemptive behavior when PreemptType=preempt/partition_prio. If two jobs from two 
        different partitions are allocated to the same resources, the job in the partition with 
        the greater PriorityTier value will preempt the job in the partition with the lesser 
        PriorityTier value.
        """
        return job.partition.priority_tier

    def _age_priority(self, job):
        """
        Returns the contribution of the job's age to it's priority.
        """
        return (
            min((self.time - job.launch_time).total_seconds() / self.max_age, 1) * self.age_weight
        )

    def _size_priority(self, job):
        """
        Returns the contribution of the job's size to it's priority.
        """
        return min(job.nodes / 256, 1) * self.size_weight

    def _fairshare_priority(self, job):
        """
        Returns the contribution of the job's fairshare factore to it's priority.
        """
        return self.fairtree.assocs[job.assoc].fairshare_factor * self.fairshare_weight

    def _partition_priority(self, job):
        """
        Returns the contribution of the job's partition to it's priority.
        """
        return job.partition.priority_weight * self.partition_weight

    def _qos_priority(self, job):
        """
        Returns the contribution of the job's QOS to it's priority.
        """
        return job.qos.priority * self.qos_weight
    

    # Energy-Aware Scheduling
    
    def _initialize_re_map(self, re_fp):
        """
        Load RE availability data from CSV and create a dictionary mapping integer timestamps
        to RE availability values. Also store the minimum and maximum timestamps.
        """
        df = pd.read_csv(re_fp)
        # Convert the "time" column to datetime objects
        df["time"] = pd.to_datetime(df["time"])
        # Convert the datetime to integer seconds (Unix timestamp)
        df["time"] = df["time"].astype(np.int64) // 10**9
        df.sort_values("time", inplace=True)
        self.re_map = dict(zip(df["time"], df["re_availability"]))
        self.re_min_time = min(self.re_map.keys())
        self.re_max_time = max(self.re_map.keys())


    def _get_re_availability(self, current_time):
        """
        Given the current simulation time (in whole seconds), return the RE availability.
        Uses a precomputed dictionary (self.re_map) for quick lookup.
        Clamps the value to the boundary if current_time is outside the available data range.
        """
        if self.re_map is None:
            return 0.0

        # current_time is assumed to be an integer in seconds.
        if current_time <= self.re_min_time:
            return self.re_map[self.re_min_time]
        elif current_time >= self.re_max_time:
            return self.re_map[self.re_max_time]
        else:
            return self.re_map.get(current_time, 0.0)


    def _power_priority_re(self, job):
        """
        Returns the energy-aware priority boost based on the current renewable energy availability.
        """
        # Determine current simulation time in seconds.
        if hasattr(self.time, "timestamp"):
            # Convert Timestamp to Unix seconds, then round and convert to int.
            current_time = int(round(self.time.timestamp()))
        else:
            current_time = int(round(self.time))
        
        # Look up RE availability using the integer timestamp.
        re_avail = self._get_re_availability(current_time)

        # Expect job.predicted_power in Watts.
        P_job = job.predicted_power
        # Normalize predicted power usage to [0,1]
        normalized_P = (P_job - self.P_min) / (self.P_max - self.P_min)
        normalized_P = max(0.0, min(normalized_P, 1.0))
        
        # Determine the current hour (using self.time as a datetime or converting if needed)
        if hasattr(self.time, "hour"):
            current_hour = self.time.hour
        else:
            current_hour = datetime.fromtimestamp(self.time).hour
        
        # Use the optimized time window parameters
        time_boost_start = self.power_time_boost_start
        time_boost_end = self.power_time_boost_end

        # Time-based boost
        if time_boost_start <= current_hour < time_boost_end:
            # Boost high-power jobs: the higher the normalized power, the higher the boost.
            time_boost = normalized_P
        else:
            # Outside the solar window, boost low power jobs
            time_boost = 1 - normalized_P

        # RE-based boost: if RE is available, boost high-power jobs; if RE is low, boost low-power jobs.
        if re_avail > 0:
            # When renewable energy is abundant, boost high-power jobs.
            re_boost = normalized_P * (re_avail ** self.power_alpha)
        else:
            # When renewable energy is scarce, boost low-power jobs.
            re_boost = 1 - normalized_P


        # Determine the blending weight based on the predicted runtime in hours.
        runtime_hours = job.predicted_runtime / 3600  # Runtime predicted in seconds
        # Blend_exponent parameter is optimized to control the steepness of the blend.
        runtime_blend = min((runtime_hours / 8.0) ** self.power_beta, 1.0)

        # Final boost blends the time-based and RE-based boosts based on a runtime-agnostic weight
        # and a runtime-dependent weight
        power_factor = ((runtime_blend + self.power_gamma) * time_boost + 
                        (2 - runtime_blend - self.power_gamma) * re_boost) / 2

        power_priority = self.power_weight * power_factor
        
        return power_priority
    
    
    def _power_priority(self, job):
        """
        Three-zone power-aware priority

            t < lead_start         : favour HIGH-power jobs
            lead_start ≤ t < lead_end : favour LOW-power *proportional to how much
                                        of the job’s runtime reaches >= lead_end*
            t ≥ lead_end (= first steep penalty hour) : favour ALL low-power jobs

        pref in [0,1] — 0 = most high-power friendly, 1 = most low-power friendly
        final priority = power_weight · pref
        """

        # Normalised predicted power
        if self.P_max == self.P_min:
            return 0.0                        # degenerate guard
        norm_P = (job.predicted_power - self.P_min) / (self.P_max - self.P_min)
        norm_P = max(0.0, min(norm_P, 1.0))  # clamp

        # Time helpers
        now      = self.time
        now_h    = now.hour + now.minute/60 + now.second/3600
        run_h    = job.predicted_runtime / 3600.0

        lead_lo  = self.power_time_boost_start
        lead_hi  = self.power_time_boost_end       # first fully-penalised hour
        pen_hi   = 24.0

        # Zone logic 
        if now_h < lead_lo:
            # Before lead-in: clear high-power jobs
            pref = norm_P

        elif now_h < lead_hi:
            # Lead-in: boost low-power in proportion to future overlap
            if run_h <= 0.0:
                overlap_frac = 0.0
            else:
                # job_lo = now_h
                job_hi = now_h + run_h
                overlap_h   = max(0.0, min(job_hi, pen_hi) - lead_hi)
                overlap_frac = overlap_h / run_h   # 0 ≤ f ≤ 1

            pref = (overlap_frac ** self.power_alpha) * (1.0 - norm_P) + (1.0 - (overlap_frac ** self.power_alpha)) * norm_P

        else:
            # Inside steep penalty window
            pref = 1.0 - norm_P               # always favour low-power

        return self.power_weight * pref

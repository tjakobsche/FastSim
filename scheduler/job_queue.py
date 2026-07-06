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

from enum import Enum
import datetime; from datetime import timedelta
from collections import defaultdict
import hashlib

import pandas as pd

from aux_funcs import mark_skip, print_and_log

import logging

logger = logging.getLogger(__name__)

class Queue:
    """
    Handles the job queue.

    Methods:
    __init__
    next_newjob
    step
    _check_dependencies
    _check_qos_holds
    _check_cancel_jobs
    cancel_job
    _clean_reservations
    _clean_dependencies
    
    """
    def __init__(self, df_jobs, partitions_by_name, qos_data, valid_resv, priority_sorter, max_switch_wait):
        """
        Initialize the job queue.

        Arguments:
        - df_jobs: the Pandas DataFrame of jobs data from sacct_jobs.csv
        - partitions_by_name: a dictionary of Partition objects indexed by partition name
        - qos_data: a dictionary of QOS data retrieved from sacctmgr_qos.csv
        - valid_resv: a list of all valid reservation names
        - priority_sorter: the Multifactor Priority Sorter object used to sort the queue
        - max_switch_wait: the maximum number of seconds for a job to wait for nodes on the same rack
        """
        
        self.priority_sorter = priority_sorter
        """
        This is the Multifactor Priority Sorter object used to sort the queue
        """

        # Create a dictionary of QOS objects from qos_data
        self.qos_objects = {
            name : QOS(
                data["name"], data["prio"], data["GrpTRES"], data["GrpJobs"],
                data["GrpSubmit"], data["MaxTRESPU"], data["MaxJobsPU"], data["MaxJobs"],
                data["MaxSubmitPU"], data["MaxSubmit"]
            )
            for name, data in sorted(qos_data.items(), key=lambda kv: kv[0])
        }

        # Pretty Print QOS data
        print_and_log(logger, 
            ["Name:".rjust(max([len(name) for name in self.qos_objects])),
            "Priority",
            "GrpTRES",
            "GrpJobs",
            "GrpSubmit",
            "MaxTRESPU",
            "MaxJobsPU",
            "MaxJobs",
            "MaxSubmitPU",
            "MaxSubmit"], sep=" | "
        )
        for name, qos in self.qos_objects.items():
            print_and_log(logger, 
                [name.rjust(max([len(name) for name in self.qos_objects])),
                str(qos.priority).rjust(8),
                str(qos.node_quota_remaining).rjust(7),
                str(qos.job_quota_remaining).rjust(7),
                str(qos.submit_quota_remaining).rjust(9),
                str(qos.usr_node_quota_remaining["dummy"]).rjust(9),
                str(qos.usr_job_quota_remaining["dummy"]).rjust(9),
                str(qos.assoc_job_quota_remaining["dummy"]).rjust(7),
                str(qos.usr_submit_quota_remaining["dummy"]).rjust(11),
                str(qos.assoc_submit_quota_remaining["dummy"]).rjust(9)], sep=" | "
            )

        
        self.assoc_limits, assoclimit_created = {}, {}
        for assoc, node in priority_sorter.fairtree.assocs.items():
            user_and_account = assoc[::2] # assoc = (user, partition, account)
            # If this user and account already have an association limit, it is because 
            # there is no partition associated with this user/account association so we 
            # can use that association limit, since it also applies to this one. This is 
            # because the associations for all partitions were created from a single 
            # association (with no specified partition) between the user and the account.
            # So all of these associations are just copies of the one original association,
            # just with different partitions.
            if user_and_account in assoclimit_created:
                self.assoc_limits[assoc] = assoclimit_created[user_and_account]
            else:
                # Create association limits with the User's max_jobs and max_submit data
                self.assoc_limits[assoc] = AssocLimit(node.max_jobs, node.max_submit)
                if node.partition is None:
                    assoclimit_created[user_and_account] = self.assoc_limits[assoc]
                    

        # The association limits defined above are general to all QOS'
        for qos in self.qos_objects.values():
            qos.set_assoc_limits(self.assoc_limits)
            

        self.time = df_jobs.Start.min()

        # Populate a list of jobs with all jobs in the job trace (sacct_jobs.csv),
        # then sort the list by (submit time, job unique ID) (in descending order)
        # job unique ID is the hash of (job ID, submit time)
        self.all_jobs = [
            Job(
                job_row.JobID, 
                job_row.Submit, 
                max_switch_wait,
                job_row.Nodes, 
                job_row.Elapsed,
                job_row.Timelimit, 
                job_row.TruePowerPerNode, 
                job_row.TruePowerPerNode,
                job_row.Start, 
                job_row.User, 
                job_row.Account, 
                self.qos_objects[job_row.QOS], # This QOS is used for calculating job priority
                partitions_by_name[job_row.Partition], 
                self.qos_objects[partitions_by_name[job_row.Partition].qos_name], # This QOS is used for tracking QOS-based TRES quotas
                job_row.DependencyArg, 
                job_row.JobName, 
                job_row.Reason, 
                job_row.ReservationArg, 
                job_row.BeginArg, 
                job_row.Cancelled, 
                job_row.NodelistArg, 
                job_row.ExcludeArg,
                job_row.predicted_power,
                job_row.predicted_runtime,
                track_qos=True,
                submit_priority=job_row.SubmitPriority
            ) for _, job_row in df_jobs.iterrows()
        ]
        self.all_jobs.sort(
            key=lambda job: (job.submit, str(job.jid), str(job.name) if job.name is not None else ""),
            reverse=True
        )
        # self.all_jobs.sort(key=lambda job: (job.submit, job.uniq_id), reverse=True)

        # Clean up dependencies
        self._clean_dependencies()

        # Populate dictionary of Jobs indexed by their Job IDs
        jid_to_job = {}
        for job in self.all_jobs:
            if job.jid not in jid_to_job:
                jid_to_job[job.jid] = job
            else:
                # If jobs share the same Job ID then choose one to keep
                # TODO: If this is happening, we have an issue. Should raise an error
                # or do better bookkeeping here.
                prev = jid_to_job[job.jid]
                if (job.true_submit < prev.true_submit) or (
                    job.true_submit == prev.true_submit and
                    (str(job.name), str(job.partition.name)) < (str(prev.name), str(prev.partition.name))
                ):
                    jid_to_job[job.jid] = job
        for job in self.all_jobs:
            # If this job has any dependencies, set all jobs that this job
            # is dependent on as dependency targets.
            job.init_dependency(jid_to_job)

        # The job queue
        self.queue = []

        # The jobs waiting on a dependency
        self.waiting_dependency = []

        # Jobs waiting on a submit hold
        self.qos_submit_held = { qos : [] for qos in self.qos_objects.values() }

        # Clean up reservations
        self._clean_reservations(valid_resv)

        # A dictionary of lists of jobs indexed by reservations.
        # This is the reservation queue for each reservation.
        self.reservations = defaultdict(list)

        # A list of jobs that need to be cancelled
        self.jobs_to_cancel = []

    def next_newjob(self):
        """
        Return the submit time of the job with the earliest submit time.

        If no jobs are left, return the maximum datetime.
        """
        try:
            return self.all_jobs[-1].submit
        except IndexError:
            return datetime.datetime.max

    def step(self, time, running_jobs):
        """
        Step through the queue.

        Arguments:
        - time: the current time
        - running_jobs: a list of all jobs that are running on nodes
        """
        self.time = time

        # self.reservations is a dictionary of lists of jobs indexed by reservation names
        pre_step_res_queue_len = {
            res : len(res_queue) for res, res_queue in self.reservations.items()
        }
        pre_step_queue_len = len(self.queue)
        pre_step_jobs_to_cancel_len = len(self.jobs_to_cancel)

        # Check dependencies to see if there are any changes. Release any jobs
        # for which their dependency has been satisfied, and add them to their
        # associated reservation (if there is one) or add them to the queue.
        self._check_dependencies(running_jobs)

        # Check QOS holds and release any jobs that should no longer be held.
        self._check_qos_holds(running_jobs)

        # If the submit time of the next new job is later than this time,
        # let's just check if there are any cancelled jobs since the last 
        # time we checked.
        if self.time < self.next_newjob():
            self._check_cancel_jobs()
            # If the length of the queue has changed, then we cancelled a job
            # and need to sort the priority list
            if len(self.queue) != pre_step_queue_len:
                self.priority_sorter.sort(self.queue, self.time)
            return

        try:
            # Jobs are listed in descending order of submit time, so the last job
            # always has the earliest submit time. Keep popping jobs off of the list
            # until the submit time is later than the current time.
            while self.all_jobs[-1].submit <= self.time:
                new_job = self.all_jobs.pop()

                # This association is no longer allowed to run jobs. This was not true in the
                # past since the job is in the workload trace. For now just skip these jobs.
                # Could also give assoc a default allocation in this case. This was relevant for
                # LUMI
                # I don't think this is relevant for NREL. The assoc_job/submit_quota is never 0 in
                # in the sacctmgr_assocs.csv data.
                if (
                    (
                        ResourceLimit.ASSOC_JOBS in new_job.qos.controlled_by_assoc and
                        self.assoc_limits[new_job.assoc].assoc_job_quota == 0
                    ) or
                    (
                        ResourceLimit.ASSOC_SUBMIT in new_job.qos.controlled_by_assoc and
                        self.assoc_limits[new_job.assoc].assoc_submit_quota == 0
                    )
                ):
                    print_and_log(logger, 
                                  [ResourceLimit.ASSOC_SUBMIT,
                                   ResourceLimit.ASSOC_JOBS,
                                   new_job.qos.controlled_by_assoc,
                                   self.assoc_limits[new_job.assoc].assoc_job_quota,
                                   self.assoc_limits[new_job.assoc].assoc_submit_quota],
                                   sep=' | ')
                    continue
                
                # If MaxSubmit is reached hold the job until the earliest time it can be submitted
                # then resubmit in the same order as the data
                # NOTE Don't want mess up any dependency chains
                # If this job is not a dependency target and submit limits have been reached (either association or user submit hold)
                if new_job.partition_qos.hold_job_submit(new_job) and not new_job.is_dependency_target:
                    # Put this job into the SUBMIT_HOLD state and add it to the list of jobs in the
                    # associated QOS qos_submit_held list
                    self.qos_submit_held[new_job.partition_qos].append(new_job.qos_submit_hold())
                    continue

                # Submit this new job
                new_job.submit_job()

                # If this job is going to be cancelled, add it to the list of jobs_to_cancel
                if new_job.cancel is not None:
                    self.jobs_to_cancel.append(new_job)

                # If this job has a dependency
                if new_job.dependency:
                    # and this dependency condition has not been met
                    if not new_job.dependency.can_release(self.queue, running_jobs):
                        # add this job to the list of jobs waiting on a dependency
                        # and put this job into a dependency hold (JobState.DEPENDENCY)
                        mark_skip(new_job, self.time, "DEPENDENCY")
                        self.waiting_dependency.append(new_job.dependency_hold())
                        continue

                # If there is a reservation associated with this job (not including empty string reservation)
                if new_job.reservation:
                    # Add this job to the appropriate reservation
                    self.reservations[new_job.reservation].append(new_job.priority(self.time))
                    continue

                # If none of the above have been met, add his job to the regular queue
                self.queue.append(new_job.priority(self.time))

        except IndexError: # No more new jobs
            pass

        # If the length of the queue has changed, sort it.
        if len(self.queue) != pre_step_queue_len:
            self.priority_sorter.sort(self.queue, self.time)

        # If the length of jobs_to_cancel has changed, sort it
        if len(self.jobs_to_cancel) != pre_step_jobs_to_cancel_len:
            self.jobs_to_cancel.sort(
                key=lambda job: (job.cancel, str(job.jid)),
                reverse=True
            )


        # If any reservation queues have changed, sort them
        for res, res_queue in self.reservations.items():
            if len(res_queue) != pre_step_res_queue_len.get(res, 0):
                self.priority_sorter.sort(res_queue, self.time)

        # Check if any of the new jobs_to_cancel should be cancelled now
        # and, if so, cancel them
        self._check_cancel_jobs()

    def _check_dependencies(self, running_jobs):
        """
        Check dependencies to see if there are any changes. Release any jobs
        for which their dependency has been satisfied, and add them to their
        associated reservation (if there is one) or add them to the queue.

        Arguments:
        - running_jobs: a list of the jobs that are running on nodes
        """
        released = []
        # self.waiting_dependency is a list of jobs waiting on dependencies
        for i_job, job in enumerate(self.waiting_dependency):
            # can_release checks if the dependency conditions have been met
            if not job.dependency.can_release(self.queue, running_jobs):
                continue

            released_job = self.waiting_dependency[i_job]
            # Add this job to an associated reservation, if relevant
            if released_job.reservation:
                self.reservations[released_job.reservation].append(
                    released_job.priority(self.time)
                )
            else:
                # Add this job to the queue, with the launch time set to the current time
                self.queue.append(released_job.priority(self.time))
            released.append(i_job)

        # Remove released jobs from waiting_dependency list
        for i_job in reversed(released):
            self.waiting_dependency.pop(i_job)

    def _check_qos_holds(self, running_jobs):
        """
        Check QOS holds and release any jobs that should no longer be held.

        Arguments:
        - running_jobs: a list of all jobs that are running on nodes
        """
        # All QOS's will be a key
        for qos in sorted(self.qos_submit_held.keys(), key=lambda q: q.name):
            # ...but not all will have any held jobs in the list
            if not self.qos_submit_held[qos]:
                continue

            # Pretending that the user resubmits in order of original submission as soon as allowed
            released, users_waiting, i_job = [], set(), len(self.qos_submit_held[qos])
            # For each submit-held job in list sorted in descending order of when they were added to the list
            for job in reversed(self.qos_submit_held[qos]):
                i_job -= 1
                
                if job.user in users_waiting:
                    continue

                # If this user were to resubmit this job with this qos, would it be held?
                if job.partition_qos.hold_job_submit(job):
                    # If so, add this user to the waiting list/
                    # TODO: Need to revisit this - what if the user submits a job with a different QOS
                    # before they submit this one? Since we are traversing the list backwards (LIFO)
                    # then it's possible that there are jobs for which the user would not be waiting,
                    # assuming they try to submit their jobs in the original order.
                    users_waiting.add(job.user)
                    mark_skip(job, self.time, "QOS-SUBMIT-HOLD")
                    # NOTE Don't want mess up any dependency chains
                    continue

                # Submit this job to the queue
                job.submit_job(self.time)
                released.append(i_job)

                if job.cancel is not None:
                    self.jobs_to_cancel.append(job)

                # If there is a dependency for this job, and it can't be released due to
                # the dependency, then we won't add this job to any queue. But it is released
                # from the QOS submit hold.
                if job.dependency:
                    if not job.dependency.can_release(self.queue, running_jobs):
                        self.waiting_dependency.append(job.dependency_hold())
                        continue

                if job.reservation:
                    self.reservations[job.reservation].append(job.priority(self.time))
                else:
                    self.queue.append(job.priority(self.time))

            for i_job in released:
                self.qos_submit_held[qos].pop(i_job)

    def _check_cancel_jobs(self):
        """
        Cancel any jobs that should have been cancelled since the last time we checked.
        """
        # self.jobs_to_cancel is in reverse time order, so the last element in the list
        # has the earliest cancellation time. Keep cancelling jobs until the cancellation
        # time is later than the current time.
        while self.jobs_to_cancel and self.jobs_to_cancel[-1].cancel <= self.time:
            cancelled_job = self.jobs_to_cancel.pop()
            self.cancel_job(cancelled_job, remove=False)

    def cancel_job(self, job, remove=True):
        """
        Cancel this job and optionally remove it from the list of jobs to cancel.

        Arguments:
        - job: the job to cancel
        - remove: if True, the job will be removed from the list of jobs to cancel

        remove may be set to False if the job to cancel is popped from jobs_to_cancel
        before calling this function with the job
        """
        if remove:
            self.jobs_to_cancel.remove(job)
        if job.state == JobState.DEPENDENCY:
            # If the job is waiting on a dependency, remove it from the list
            self.waiting_dependency.remove(job)
        elif job.reservation == "":
            # If there is no associated reservation, remove the job from the main queue.
            # A single remove doubles as the membership check to avoid scanning twice.
            try:
                self.queue.remove(job)
            except ValueError:
                print_and_log(logger, 'Cannot find job:', job.jid)
                raise
        else:
            # If there is an associated reservation, remove the job from the reservation queue
            self.reservations[job.reservation].remove(job)

        job.cancel_job()

    def _clean_reservations(self, valid_reservations):
        """
        Remove reservations that aren't found in valid_reservations,
        which are derived from the sinfo_resv.csv file.

        Arguments:
        - valid_reservations: the list of reservations specified in sinfo_resv.csv
        """
        removed_res, removed_res_cnt = set(), 0
        for job in self.all_jobs:
            if not job.reservation:
                # ARCHER2 specific, short qos is automically has short resv
                if job.qos.name == "short":
                    job.reservation = "shortqos"
                else:
                    job.reservation = ""
                continue
                # continue

            if job.reservation not in valid_reservations:
                removed_res.add(job.reservation)
                removed_res_cnt += 1
                job.reservation = ""
                job.ignore_in_eval = True

        print_and_log(logger, 
            ["Missing reservation records for {} resulting in ignoring reservations for ",
            "{} jobs".format(removed_res, removed_res_cnt)], sep=''
        )

    def _clean_dependencies(self):
        """
        Remove dependencies that can't be satisfied. Remove Job IDs from
        dependencies if they are not found in the set of jobs in the job
        trace. Ignore dependencies that are hidden or removed, or contain
        no valid associated Job IDs.
        """
        removed_dep_cnt, ignored_dep_cnt = 0, 0
        all_job_ids = { job.jid for job in self.all_jobs }
        for job in self.all_jobs:
            if not job.dependency:
                # Dependency hidden in batch file or just removed
                if job.reason == "Dependency":
                    job.ignore_in_eval = True
                    ignored_dep_cnt += 1
                continue

            # job.dependency.conditions is a tuple (dep_type, job_ids)
            # It gets converted later to (dep_type, jobs)
            job_ids = job.dependency.conditions[1]

            # Filter out Job IDs of jobs not found in the job trace
            intersection = job_ids.intersection(all_job_ids)
            job.dependency.conditions = (job.dependency.conditions[0], intersection)

            # Job arrays need to be expanded into individual JobIDs
            for unmatched_job_id in job_ids.difference(intersection):
                prefix = unmatched_job_id + '_'
                array_ids = {id for id in all_job_ids if id.startswith(prefix)}

                if array_ids:
                    if job.dependency.delimiter == "?":
                        raise NotImplemetedError(
                            "Not implemented expanding job array ids for OR dependencies"
                        )
                    job.dependency.conditions[1].update(array_ids)

            # Remove dependencies that are already met that aren't singletons
            # This means, even though there was a dependency, no Job IDs survived the filtering 
            # process above and in the Dependency class where this dependency was initialized.
            job.dependency.conditions_met = len(job.dependency.conditions) == 0
            if job.dependency.conditions_met and not job.dependency.singleton:
                removed_dep_cnt += 1
                job.dependency = None
                if job.reason == "Dependency":
                    job.ignore_in_eval = True
                    ignored_dep_cnt += 1

        print_and_log(logger, 
                      f"Removed {removed_dep_cnt} dependencies that cannot be satisfied from workload trace")
        print_and_log(logger, 
                      f"Ignored {ignored_dep_cnt} in evaulation due to dependency not being in SubmitLine or missing from workload trace")



class QOS:
    """
    Handles QOS data.

    NOTE: Ignoring any QOS linked to the partition that could override this (not applicable for LUMI
          or ARCHER). Also ignoring per account resource limits.

    Methods:
    __init__
    set_assoc_limits
    job_submitted
    job_started
    job_ended
    job_cancelled
    hold_job
    hold_job_grp
    hold_job_usr
    hold_job_submit
    hold_job_submit_grp
    hold_job_submit_usr
    """
    def __init__(
        self, name, priority, grp_nodes, grp_jobs, grp_submit, usr_nodes, usr_jobs, assoc_jobs,
        usr_submit, assoc_submit
    ):
        """
        Initialize the QUS object.

        Arguments:
        name: # QOS name (e.g. 'high', 'normal', 'standby')           
        priority: Priority used in job priority calculation
        grp_nodes: The total count of TRES able to be used at any given time from jobs running from a QOS
        grp_jobs: The total number of jobs able to run at any given time from a QOS.
        grp_submit: The total number of jobs able to be submitted to the system at any given time from a QOS.
        usr_nodes: The maximum number of TRES a user can allocate at a given time.
        usr_jobs: The maximum number of jobs a user can have running at a given time. 
        assoc_jobs: The total number of jobs able to run at any given time for the given association.
        usr_submit: The maximum number of jobs a user can have running and pending at a given time. 
        assoc_submit: The maximum number of jobs able to be submitted to the system at any given time from the given association.
        """
        self.name = name
        self.priority = priority

        self.assoc_limits = {}

        # Set relevant limits as being tracked
        self.tracked_limits = set()
        if grp_jobs is not None:
            self.tracked_limits.add(ResourceLimit.GRP_JOBS)
        if grp_nodes is not None:
            self.tracked_limits.add(ResourceLimit.GRP_NODES)
        if grp_submit is not None:
            self.tracked_limits.add(ResourceLimit.GRP_SUBMIT)
        if usr_jobs is not None:
            self.tracked_limits.add(ResourceLimit.USR_JOBS)
        if usr_nodes is not None:
            self.tracked_limits.add(ResourceLimit.USR_NODES)
        if usr_submit is not None:
            self.tracked_limits.add(ResourceLimit.USR_SUBMIT)
        if assoc_jobs is not None:
            self.tracked_limits.add(ResourceLimit.ASSOC_JOBS)
        if assoc_submit is not None:
            self.tracked_limits.add(ResourceLimit.ASSOC_SUBMIT)

        # These associations are controlled by their own limits,
        # not by QOS limits
        self.controlled_by_assoc = {
            limit
            for limit in [ResourceLimit.ASSOC_JOBS, ResourceLimit.ASSOC_SUBMIT]
                if limit not in self.tracked_limits
        }

        # Initialize QOS-level remaining quotas
        self.job_quota_remaining = grp_jobs
        self.node_quota_remaining = grp_nodes
        self.submit_quota_remaining = grp_submit

        # User-level remaining quotas initialize with limits
        self.usr_job_quota_remaining = defaultdict(lambda: usr_jobs)
        self.usr_node_quota_remaining = defaultdict(lambda: usr_nodes)
        self.usr_submit_quota_remaining = defaultdict(lambda: usr_submit)

        # Association-level remaining quotas initialize with limits
        self.assoc_job_quota_remaining = defaultdict(lambda: assoc_jobs)
        self.assoc_submit_quota_remaining = defaultdict(lambda: assoc_submit)

    def set_assoc_limits(self, assoc_limits):
        """
        The association limits for this QOS.

        Arguments:
        assoc_limits: the association limits shared with all QOS objects
        """
        self.assoc_limits = assoc_limits

    def job_submitted(self, job):
        """
        Updates remaining quotas based on a job submission.
        
        Jobs are skipped if we aren't tracking them for QOS limits
        (see note on job.track_qos in Job class).

        Arguments:
        - job: the submitted job
        """
        if not job.track_qos:
            return
        if ResourceLimit.GRP_SUBMIT in self.tracked_limits:
            self.submit_quota_remaining -= 1
        if ResourceLimit.USR_SUBMIT in self.tracked_limits:
            self.usr_submit_quota_remaining[job.user] -= 1
        if ResourceLimit.ASSOC_SUBMIT in self.tracked_limits:
            self.assoc_submit_quota_remaining[job.assoc] -= 1

        self.assoc_limits[job.assoc].job_submitted()

    def job_started(self, job):
        """
        Updates remaining quotas based on a job starting.

        Jobs are skipped if we aren't tracking them for QOS limits
        (see note on job.track_qos in Job class).

        Arguments:
        - job: the starting job
        """
        if not job.track_qos:
            return
        if ResourceLimit.GRP_JOBS in self.tracked_limits:
            self.job_quota_remaining -= 1
        if ResourceLimit.GRP_NODES in self.tracked_limits:
            self.node_quota_remaining -= job.nodes
        if ResourceLimit.USR_JOBS in self.tracked_limits:
            self.usr_job_quota_remaining[job.user] -= 1
        if ResourceLimit.USR_NODES in self.tracked_limits:
            self.usr_node_quota_remaining[job.user] -= job.nodes
        if ResourceLimit.ASSOC_JOBS in self.tracked_limits:
            self.assoc_job_quota_remaining[job.assoc] -= 1

        self.assoc_limits[job.assoc].job_started()

    def job_ended(self, job):
        """
        Updates remaining quotas based on a job ending.

        Jobs are skipped if we aren't tracking them for QOS limits
        (see note on job.track_qos in Job class).

        Arguments:
        - job: the ending job
        """
        if not job.track_qos:
            return
        if ResourceLimit.GRP_JOBS in self.tracked_limits:
            self.job_quota_remaining += 1
        if ResourceLimit.GRP_NODES in self.tracked_limits:
            self.node_quota_remaining += job.nodes
        if ResourceLimit.GRP_SUBMIT in self.tracked_limits:
            self.submit_quota_remaining += 1
        if ResourceLimit.USR_JOBS in self.tracked_limits:
            self.usr_job_quota_remaining[job.user] += 1
        if ResourceLimit.USR_NODES in self.tracked_limits:
            self.usr_node_quota_remaining[job.user] += job.nodes
        if ResourceLimit.USR_SUBMIT in self.tracked_limits:
            self.usr_submit_quota_remaining[job.user] += 1
        if ResourceLimit.ASSOC_JOBS in self.tracked_limits:
            self.assoc_job_quota_remaining[job.assoc] += 1
        if ResourceLimit.ASSOC_SUBMIT in self.tracked_limits:
            self.assoc_submit_quota_remaining[job.assoc] += 1

        self.assoc_limits[job.assoc].job_ended()

    def job_cancelled(self, job):
        """
        Updates remaining quotas based on a job cancellation.

        Jobs are skipped if we aren't tracking them for QOS limits
        (see note on job.track_qos in Job class).

        Arguments:
        - job: the cancelled job
        """
        if not job.track_qos:
            return
        if ResourceLimit.GRP_SUBMIT in self.tracked_limits:
            self.submit_quota_remaining += 1
        if ResourceLimit.USR_SUBMIT in self.tracked_limits:
            self.usr_submit_quota_remaining[job.user] += 1
        if ResourceLimit.ASSOC_SUBMIT in self.tracked_limits:
            self.assoc_submit_quota_remaining[job.assoc] += 1

        self.assoc_limits[job.assoc].job_cancelled()

    def hold_job(self, job):
        """
        Holds the job if no remaining quota on tracked limits
        at either the QOS or the user/association level.

        Arguments:
        - job: the new job
        """
        return self.hold_job_grp(job) or self.hold_job_usr(job)

    def hold_job_grp(self, job):
        """
        Holds the job if no remaining quota for this QOS.
        """
        if (
            ResourceLimit.GRP_JOBS in self.tracked_limits and
            not self.job_quota_remaining
        ):
            return True
        if (
            ResourceLimit.GRP_NODES in self.tracked_limits and
            job.nodes > self.node_quota_remaining
        ):
            return True

        return False

    def hold_job_usr(self, job):
        """
        Holds the job if no remaining quota for this User/Association,
        by checking both QOS limits and the association limits.
        """
        if (
            ResourceLimit.USR_JOBS in self.tracked_limits and
            not self.usr_job_quota_remaining[job.user]
        ):
            return True
        if (
            ResourceLimit.USR_NODES in self.tracked_limits and
            job.nodes > self.usr_node_quota_remaining[job.user]
        ):
            return True
        if (
            ResourceLimit.ASSOC_JOBS in self.tracked_limits and
            not self.assoc_job_quota_remaining[job.assoc]
        ):
            return True

        # If not held by user limits, check if the job
        # needs to be held due to association limits
        return self.assoc_limits[job.assoc].hold_job(self.controlled_by_assoc)

    def hold_job_submit(self, job):
        """
        Holds the job submission if no remaining quota on tracked limits
        at either the QOS or the user/association level.

        Arguments:
        - job: the submitted job
        """
        return self.hold_job_submit_grp(job) or self.hold_job_submit_usr(job)

    def hold_job_submit_grp(self, job):
        """
        Holds the job if no remaining submission quota for this QOS.
        """
        if (
            ResourceLimit.GRP_SUBMIT in self.tracked_limits and
            not self.submit_quota_remaining
        ):
            return True

        return False

    def hold_job_submit_usr(self, job):
        """
        Holds the job submission if no remaining quota for this User/Association,
        by checking both QOS limits and the association limits.
        """
        if (
            ResourceLimit.USR_SUBMIT in self.tracked_limits and
            not self.usr_submit_quota_remaining[job.user]
        ):
            return True
        if (
            ResourceLimit.ASSOC_SUBMIT in self.tracked_limits and
            not self.assoc_submit_quota_remaining[job.assoc]
        ):
            return True

        # If not held by user limits, check if the job
        # needs to be held due to association limits
        return self.assoc_limits[job.assoc].hold_job_submit(self.controlled_by_assoc)


class AssocLimit:
    """
    Handle user association limits.

    Methods:
    __init__
    job_submitted
    job_started
    job_ended
    job_cancelled
    hold_job
    hold_job_submit
    
    NOTE:   This is only setup for user association limits. Slurm can have limits set on account
            associations and cluster association which can override the associtions below them. Implementing
            this would required extending this class for accounts also and having it interact with the
            assoctree so it knows what checks to pass on to its children. Each user assoc would then have
            a single assoc limit class with some of the qoutas refering to the parent account AssocLimit
            class, like the QOS does to this class currently. Would not need to check which resource limits
            overidde what for each job since the association relationship is constant
    """
    def __init__(self, assoc_jobs, assoc_submit):
        """
        Initialize the association limit.

        Arguments:
        assoc_jobs: max jobs for the association
        assoc_submit: max submissions for the association

        Creates a set of tracked limits, a quota, and keeps track
        of the amount of quota remaining.
        """
        self.tracked_limits = set()
        if assoc_jobs is not None:
            self.tracked_limits.add(ResourceLimit.ASSOC_JOBS)
        if assoc_submit is not None:
            self.tracked_limits.add(ResourceLimit.ASSOC_SUBMIT)

        self.assoc_job_quota_remaining = assoc_jobs
        self.assoc_submit_quota_remaining = assoc_submit

        self.assoc_job_quota = assoc_jobs
        self.assoc_submit_quota = assoc_submit

    def job_submitted(self):
        """
        Decrements remaining submit quota if submissions are being tracked.
        """
        if ResourceLimit.ASSOC_SUBMIT in self.tracked_limits:
            self.assoc_submit_quota_remaining -= 1

    def job_started(self):
        """
        Decrements remaining jobs quota if jobs are being tracked.
        """
        if ResourceLimit.ASSOC_JOBS in self.tracked_limits:
            self.assoc_job_quota_remaining -= 1

    def job_ended(self):
        """
        Increments remaining submit/jobs quota if submissions/jobs are being tracked.
        """
        if ResourceLimit.ASSOC_SUBMIT in self.tracked_limits:
            self.assoc_submit_quota_remaining += 1
        if ResourceLimit.ASSOC_JOBS in self.tracked_limits:
            self.assoc_job_quota_remaining += 1

    def job_cancelled(self):
        """
        Increments remaining submit quota if submissions are being tracked.
        """
        if ResourceLimit.ASSOC_SUBMIT in self.tracked_limits:
            self.assoc_submit_quota_remaining += 1

    def hold_job(self, limits):
        """
        Returns True if association has no remaining jobs quota.

        Arguments:
        - limits: TODO: Need to return to this.
        """
        if (
            ResourceLimit.ASSOC_JOBS in self.tracked_limits and
            ResourceLimit.ASSOC_JOBS in limits
            and not self.assoc_job_quota_remaining
        ):
            return True

        return False

    def hold_job_submit(self, limits):
        """
        Returns True if association has no remaining submissions quota.

        Arguments:
        - limits: TODO: Need to return to this.
        """
        if (
            ResourceLimit.ASSOC_SUBMIT in self.tracked_limits and
            ResourceLimit.ASSOC_SUBMIT in limits
            and not self.assoc_submit_quota_remaining
        ):
            return True

        return False


class Job:
    """
    Handles Jobs.

    Methods:
    __init__
    __hash__
    __eq__
    init_dependency
    submit_job
    cancel_job
    start_job
    assign_node
    end_job
    qos_submit_hold
    dependency_hold
    priority
    """
    def __init__(
        self, jid, submit : datetime, max_switch_wait, nodes, runtime : timedelta, reqtime: timedelta, node_power,
        true_node_power, true_job_start, user, account, qos, partition, partition_qos, dependency_arg, name,
        reason, reservation_arg, begin_arg, cancelled, nodelist_arg, exclude_arg, predicted_power, predicted_runtime,
        track_qos=True, submit_priority=None
    ):
        """
        Initialize the job with data from the job row, including arguments extracted
        from the SubmitLine field.

        Arguments:
        - jid: the job ID
        - submit: the job's submit time
        - max_switch_wait: the maximum number of seconds for a job to wait for nodes on the same rack
        - nodes: the number of nodes allocated to the job, unless the job was not allocated nodes
                 then this is the number of nodes requested
        - runtime: how long the job ran (wallclock used)
        - reqtime: the timelimit requested (wallclock requested)
        - node_power: the amount of power-per-node used by the job
        - true_node_power: the same field as is used for node_power TODO: Need to understand why this is necessary
        - true_job_start: the time the job actually started in the job trace
        - user: the user associated with this job
        - account: the account associated with this job
        - qos: the QOS requested by this job. This is only used for priority calculation.
        - partition: the partition associated with this job
        - partition_qos: the QOS object for the QOS name listed with the partition in slurm.conf. This is used for tracking QOS-based resource quotas.
        - dependency_arg: the dependency of this job, as described in the submit line, if present
        - name: the job name
        - reason: the reason for the final job state
        - reservation_arg: the reservation used by this job, as described in the submit line, if present
        - begin_arg: the time requested for this job to begin, as described in the submit line, if present
        - cancelled: the time between this job's submit and end time if the job was not allocated any nodes (assumed cancelled)
        - nodelist_arg: the nodes requested by this job, as described in the submit line, if present
        - exclude_arg: the nodes to exclude when allocating resources for this job, as described in the submit line, if present
        - track_qos: whether to track this job for QOS limits.
        """
        # self.uniq_id = hash((jid, submit))
        # Stable 64-bit uid from jid|submit
        self.uniq_id = int.from_bytes(
            hashlib.blake2b(f"{jid}|{submit}".encode(), digest_size=8).digest(), "big"
        )
        self.jid = jid
        self.nodes = nodes
        self.runtime = runtime
        self.reqtime = reqtime
        self.node_power = node_power
        self.true_node_power = true_node_power        
        self.predicted_power = predicted_power    
        self.predicted_runtime = predicted_runtime
        self.true_submit = submit
        self.submit = submit
        self.max_switch_wait_time = self.submit + max_switch_wait
        self.true_job_start = true_job_start
        self.user = user
        self.account = account
        self.qos = qos
        self.partition = partition
        self.partition_qos = partition_qos
        self.name = name
        
        # Dependency may be submitted incorrectly (typo or wrong format)
        # Note: pandas may give NaN (a float) for missing values, so normalize first.
        dep_str = "" if dependency_arg is None else str(dependency_arg)
        if dep_str.lower() in ("nan", "none"):
            dep_str = ""

        if (
            dep_str == "" or
            all(
                dep_type not in dep_str
                for dep_type in [
                    "after:", "afterany:", "afterburstbuffer", "aftercorr", "afternotok",
                    "afterok", "singleton"
                ]
            )
        ):
            self.dependency = None
        else:
            self.dependency = Dependency(dep_str, user, name)
            """
            Create a Dependency object for this job, if relevant.
            """

        self.reservation = reservation_arg
        """
        Create a reservation for this job, if relevant.
        If there is no reservation, this will be an empty string.
        """

        self.assoc = (self.user, self.partition, self.account)
        """
        Register the association for this job.
        """

        self.is_dependency_target = False
        """
        Boolean to keep track if this job is the target of another job's dependency.
        """

        # Some features are not relevant for scheduluing (AssocMaxCpuMinutesPerJobLimit means for
        # archer that the user hasnt been allocated time yet, reservations, jobs held by user, ...)
        # and some I cant implemented with available data (JobArrayTaskLimit is usually specified
        # in batch script). Want to have these jobs in simulation but don't want to include them in
        # evaluation stage
        self.reason = reason
        self.ignore_in_eval = (
            reason in [
                "AssocMaxCpuMinutesPerJobLimit", "ReqNodeNotAvail", "BeginTime", "JobHeldUser",
                "DependencyNeverSatisfied", "JobArrayTaskLimit"
            ] or
            begin_arg or nodelist_arg or exclude_arg
        )

        self.launch_time = None
        self.start = None
        self.end = None
        self.assigned_nodes = set()

        # Initialize all jobs with the FUTURE (simulator-specific) state
        self.state = JobState.FUTURE

        # I can't find anywhere this is used, so commenting it out. -KM
        #self.planned_block = None

        # The time between a job's submit and end time (if the job was cancelled)
        self.cancelled_t = None if pd.isnull(cancelled) else cancelled

        # The actual time the job was cancelled
        self.cancel = None

        self.track_qos = track_qos
        """
        Jobs that are running on nodes that go down get ended and resubmitted, but we shouldn't
        include these in QOS limit tracking because this is just a workaround i.e. these aren't
        actually extra jobs being submitted, just a splitting of the workload to deal with down
        events.
        """

        self.submit_priority = submit_priority
        """
        This is a bookkeeping parameter we only use for sim initialization. It keeps track of 
        the actual submit time of jobs that were on the queue when the sim started.
        """

        self.node_timeline = []
        """
        Keep a history of how many nodes this job has at every change.
        Filled by Controller when it starts / shrinks the job at relevant node down events.
        This is needed because of the implementation of check_down_nodes in the Controller.
        When a multinode job is running on a node that should go down, it searches for an
        available replacement node. If it can't find a replacement node, a new single node job
        is created and added to the queue, and the job is shrunk by 1 node. This maintains
        strict adherence to node down events.
        """   

        self.wait_history = []
        """
        Track the reasons this job was skipped by the scheduler during its time in the queue.
        Each entry is a tuple of the form (simulation_time, reason_string), capturing the
        timestamp and specific reason the job could not be scheduled at that moment.
        This helps diagnose why a job is stuck, whether due to QOS limits, resource
        availability, unmet dependencies, or reservation conflicts.
        """

        self.last_skip = ""
        """
        A string memo to prevent duplicate entries in wait_history for the same skip reason.
        This ensures that the wait history remains readable and only logs a new entry
        when the blocking condition changes. Cleared automatically when the job is
        submitted, starts running, or finishes.
        """



    def __hash__(self):
        """
        Returns the unique id created by hashing the job ID and submit time
        """
        return self.uniq_id

    def __eq__(self, other):
        """
        Two jobs are equal if their unique IDs are equal.
        """
        if isinstance(other, Job):
            return self.uniq_id == other.uniq_id
        return False
        
    def init_dependency(self, jid_to_job):
        """
        If this job has any dependencies, set all jobs that this job
        is dependent on as dependency targets.

        Arguments:
        - jid_to_job: a dictionary of Job objects indexed by job IDs
        """
        if self.dependency:
            self.dependency.convert_jids_to_jobs(jid_to_job)

            for job in self.dependency.conditions[1]:
                job.is_dependency_target = True

    def submit_job(self, time=None):
        """
        Handle job submissions for this job.

        Arguments:
        - time: the current time when the job is submitted

        Returns: this job
        """
        if time is not None:
            self.submit = time
        self.partition_qos.job_submitted(self)

        # If a job was cancelled (it has 0 Allocated Nodes)
        # then cancelled_t = end time - submit time
        if self.cancelled_t is not None:
            self.cancel = self.submit + self.cancelled_t
        return self

    def cancel_job(self):
        """
        Cancel this job.

        Returns: this job
        """
        # Update quotas based on job cancellation
        self.partition_qos.job_cancelled(self)
        self.state = JobState.CANCELLED
        return self

    def start_job(self, time : datetime):
        """
        Start this job

        Arguments:
        - time: the current time when this job is started

        Returns: this job
        """
        self.start = time
        self.end = time + self.runtime
        self.endlimit = time + self.reqtime
        self.state = JobState.RUNNING
        # Update quotas based on job starting
        self.partition_qos.job_started(self)
        return self

    def assign_node(self, node):
        """
        Assign a node to this job.

        Arguments:
        - node: the node to assign to this job

        Returns: True if assigning this node brings the number of assigned
                 nodes up to the number of allocated nodes.
        """
        node.set_busy()
        self.assigned_nodes.add(node)
        # This node now has this job running on it
        node.running_job = self
        
        return len(self.assigned_nodes) >= self.nodes

    def end_job(self):
        """
        End this job.

        Returns: this job
        """
        for node in self.assigned_nodes:
            node.set_free()
            node.running_job = None
        self.partition_qos.job_ended(self)
        self.state = JobState.COMPLETED
        return self

    def qos_submit_hold(self):
        """
        Put this job in a state of QOS submit hold.

        Returns: this job
        """
        self.state = JobState.QOS_SUBMIT
        return self

    def dependency_hold(self):
        """
        Put this job in a state of dependency hold.

        Returns: this job
        """
        self.state = JobState.DEPENDENCY
        return self

    # 
    def priority(self, time):
        """
        Sets the launch_time for this job, which is when the QOS recognises the 
        job as submitted for resource limits accounting and age priority weighting.

        Arguments:
        - time: the current time

        Returns: this job
        """
        self.state = JobState.PRIORITY
        if not self.launch_time:
            if self.submit_priority and self.submit_priority is not pd.NaT: 
                # If this bookkeeping parameter exists, then this job was on the
                # queue at the sim start, so we need to set launch time according
                # to when it was actually submitted, not this current time, since
                # the current time reflects the sim start time, not when the job
                # was actually submitted. This is necessary to make sure the queued
                # jobs have the correct age priority score.
                self.launch_time = self.submit_priority
            else:
                self.launch_time = time
        return self


class Dependency:
    """
    Handle job dependencies.

    Methods:
    __init__
    convert_jids_to_jobs
    can_release

    There is an assumption baked into this implementation that there is only
    one type of dependency per job. I did not find evidence that a job could,
    for instance, have something like: --dependency=afterok:811795,after:811796.
    If this is possible (regardless of the proper syntax) then we need to change
    this implementation.
    """
    def __init__(self, dependency_args, user, name):
        """
        Initialize the dependency.

        Arguments:
        - dependency_args: the type of dependency
        - user: the user associated with the job associated with this dependency
        - name: the job name

        Dependency_args Example: afterok:811795
        """
        self.job_user_name = (user, name)
        self.delimiter = "?" if "?" in dependency_args else ","

        # Create a dictionary of condition-jobs pairs (condition_string: {set of job IDs})
        self.conditions = None
        self.singleton = False
        for condition in dependency_args.split(self.delimiter):
            if condition == "singleton":
                self.singleton = True
                continue

            dep_type = condition.split(":")[0]
            # NOTE after can take a +time after job_id, just going to ignore these for now TODO: Need to revisit this
            # if "+" in condition:
            #     print("!!!Some jobs have dependencies with +time offsets!!!")
            jobs = { job_id.split("+")[0] for job_id in condition.split(":")[1:] }

            # Jobs in trace all ran so can assume these conditions are met and treat all the same
            if dep_type == "afterok" or dep_type == "afternotok" or dep_type == "afterany":
                self.conditions = ("afterany", jobs)
                continue
            elif dep_type == "after":
                self.conditions = ("after", jobs)
                continue

            raise NotImplementedError("Unrecognised dep_type {}".format(dep_type))

        # This condition is relevant to job submissions, since the 'after' dependency
        # waits until the job is submitted.
        self.submitted_relevant = self.conditions[0] == "after"

        # This condition is relevant to job completion, since the 'afterany' dependency
        # waits until the job is finished.
        self.finished_relevant = self.conditions[0] == "afterany"

        # The conditions are already met only if there are no conditions
        self.conditions_met = not self.conditions or len(self.conditions[1]) == 0

    def convert_jids_to_jobs(self, jid_to_job):
        """
        Convert the job IDs in a dependency to their corresponding Job objects.

        Arguments:
        - jid_to_job: a dictionary of Job objects indexed by Job IDs
        """
        self.conditions = (self.conditions[0], {jid_to_job[jid] for jid in self.conditions[1]})

    def can_release(self, queued_jobs, running_jobs):
        """
        This function returns True if the dependency hold can be released, False otherwise.
        Also changes self.conditions_met to True when conditions are met.

        Arguments:
        - queued_jobs: a list of jobs on the queue
        - running_jobs: a list of all jobs that are running on nodes
        """

        # Check if the conditions have been met and respond accordingly
        if not self.conditions_met:
            if self.delimiter == "?": # OR delimiter, so any condition being met is sufficient
                # We need to check all the jobs.
                for job in self.conditions[1]:
                    if job.state != JobState.COMPLETED and (self.conditions[0] == "afterany" or job.state != JobState.RUNNING):
                        continue
                    self.conditions_met = True
            else:
                for job in list(self.conditions[1]): # Make a new list object since we remove items from the set
                    # NOTE: If the delimiter is not "?", then all jobs in the condition need to be completed
                    # before the conditions are met. In this case, there is no reason to continue the 
                    # loop once we see there is a job that's not completed.
                    if job.state != JobState.COMPLETED and (self.conditions[0] == "afterany" or job.state != JobState.RUNNING):
                        break
                    self.conditions[1].remove(job)    
                self.conditions_met = len(self.conditions[1]) == 0
        
        if self.conditions_met:
            if not self.singleton:
                return True
            else:
                # If there are no running or queued jobs with this user & job name, then any with this
                # user & job name must have terminated
                # NOTE: It is inefficient to keep making launched_jobs in the case of multiple Singletons,
                # but Singletons are rare in data.
                # Singleton: This job can begin execution after any previously launched jobs sharing the same 
                #            job name and user have terminated.
                launched_jobs = running_jobs + queued_jobs
                if self.job_user_name in { (job.user, job.name) for job in launched_jobs }:
                    return False
                return True
        
        return False


class JobState(Enum):
    FUTURE = 1
    PRIORITY = 2
    RUNNING = 3
    COMPLETED = 4
    QOS_SUBMIT = 5
    DEPENDENCY = 6
    QOS_RESOURCES = 7
    CANCELLED = 8


class ResourceLimit(Enum):
    GRP_NODES = 1
    GRP_JOBS = 2
    GRP_SUBMIT = 3
    USR_JOBS = 4
    USR_NODES = 5
    USR_SUBMIT = 6
    ASSOC_SUBMIT = 7
    ASSOC_JOBS = 8


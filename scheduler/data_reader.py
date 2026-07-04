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

import re, sys
from collections import defaultdict
import datetime; from datetime import timedelta
from copy import deepcopy

import pandas as pd

import logging

logger = logging.getLogger(__name__)


from aux_funcs import (
    convert_nodelist_to_node_nums, timelimit_str_to_timedelta, convert_to_raw, get_sbatch_cli_arg, split_and_mask_events, print_and_log
)


class SlurmDataReader:
    """
    Read data from the csv files in the slurm_dump directory.

    Methods:
    __init__
    get_nodes_partitions
    - get_node_events
    - get_reservations
    - get_node_and_partition_data_from_slurm_conf
    - gather_node_data_from_events_resv_and_conf
    get_qos
    get_cleaned_job_df(self, considered_partitions, def_power_per_node)
    - process_line
    """
    
    def __init__(self, sim_config):
        """
        Initialize the data reader.

        slurm_conf: The slurm configuration filepath (slurm.conf)
        node_events_dump: The node events filepath (sacctmgr_events.csv)
        resv_dum: The reservations filepath (sinfo_resv.csv)
        job_dump: The jobs filepath (sacct_jobs.csv)
        qos_dump: The QOS filepath (sacctmgr_qos.csv)
        """
        self.slurm_conf = sim_config.slurm_conf
        self.node_events_dump = sim_config.node_events_dump
        self.resv_dump_current = sim_config.resv_dump_current
        self.resv_dump_historic = sim_config.resv_dump_historic
        self.job_dump = sim_config.job_dump
        self.qos_dump = sim_config.qos_dump
        self.impromptu_reservation_names = sim_config.impromptu_reservation_names
        self.system = sim_config.system

    def get_nodes_partitions(
        self, considered_partitions, hpe_restrictlong_sliding_res, max_sim_t, nodes_down_in_blades, sim_start, sim_end
    ):
        """
        Gets node, partition, and reservation data from Slurm dump files.
        """
        df_events = self.get_node_events(max_sim_t, sim_start)
        
        df_resv, valid_resv, resv_end_times = self.get_reservations(sim_start, sim_end)
        
        partition_data, nid_partitions, nid_weight = self.get_node_and_partition_data_from_slurm_conf(considered_partitions)
        
        nid_data, hpe_restrictlong_nids = self.gather_node_data_from_events_resv_and_conf(nid_partitions, nid_weight, df_events, df_resv)

        nid_data, hpe_restrictlong_nids = self.handle_hpe_specific_features(
            nid_data, hpe_restrictlong_nids, hpe_restrictlong_sliding_res, nodes_down_in_blades, df_resv
        )
            
        return nid_data, partition_data, valid_resv, resv_end_times, hpe_restrictlong_nids

    def get_node_events(self, max_sim_t, sim_start):
        """
        Preprocess data from node events file into a Pandas DataFrame.
        """
        # Read the sacctmgr_events.csv file
        df_events = pd.read_csv(
            self.node_events_dump, delimiter='|', lineterminator='\n', header=0,
            usecols=["NodeName", "TimeStart", "TimeEnd", "State", "Reason"]
        )

        # Clean events data and filter events by date (if looking at a subset of jobs for simulation) 
        # Filter events by TimeEnd, otherwise the simulator will erroneously place nodes in Down state for past events.
        df_events = df_events.loc[
            ((df_events.NodeName.notna()) & 
             (df_events.TimeStart != "Unknown") & 
             (df_events.TimeEnd >= sim_start))
        ]

        df_events.TimeStart = pd.to_datetime(df_events.TimeStart, format="%Y-%m-%dT%H:%M:%S")
        df_events.TimeEnd = pd.to_datetime(df_events.TimeEnd, format="%Y-%m-%dT%H:%M:%S")
        df_events["Duration"] = df_events.apply(lambda row: (row.TimeEnd - row.TimeStart), axis=1)
        df_events.State = df_events.State.apply(lambda state: "DRAIN" if "DRAIN" in state else "DOWN")
        df_events["Id"] = df_events.NodeName

        return df_events


    def get_reservations(self, sim_start, sim_end):
        """
        Preprocess data from node reservations file into a Pandas DataFrame.
        """
        # Read the sacctmgr_resv.csv file
        # NOTE Not considering any reservation flags
        # These are the reservations as currently available from Slurm via sinfo
        df_resv_current = pd.read_csv(
            self.resv_dump_current, delimiter='|', lineterminator='\n', header=0,
            usecols=["RESV_NAME", "START_TIME", "END_TIME", "NODELIST"]
        )

        # These are the historic reservations available from Slurm via sreport
        df_resv_historic = pd.read_csv(
            self.resv_dump_historic, delimiter='|', lineterminator='\n', header=0,
            usecols=["Name","Start","End","Nodes"]
        ).rename(columns={"Name": "RESV_NAME","Start": "START_TIME","End": "END_TIME","Nodes": "NODELIST"})

        # Join the current and historic reservation data
        df_resv = pd.concat([df_resv_current, df_resv_historic], ignore_index=True)
        df_resv.dropna(inplace=True)

        # Convert to datetime
        df_resv.START_TIME = pd.to_datetime(df_resv.START_TIME, format="%Y-%m-%dT%H:%M:%S")
        df_resv.END_TIME = pd.to_datetime(df_resv.END_TIME, format="%Y-%m-%dT%H:%M:%S")

        # Convert the nodelist string to a list of node names
        df_resv["NODELIST"] = df_resv["NODELIST"].apply(
            lambda nodelist: convert_nodelist_to_node_nums(nodelist, self.system)
        )

        # Get a list of all valid reservation names
        valid_resv = set(df_resv.RESV_NAME.unique())
        resv_end_times = df_resv.groupby('RESV_NAME').agg(max_end_time=('END_TIME','max')).to_dict()['max_end_time']

        # Merge temporally adjacent reservations involving the same nodes
        # We handle adjacent reservations when creating the resv schedule for each node, but doing
        # this here as well significantly cuts down the number of reservations we need to handle per node.
        df_resv = self.merge_reservations(df_resv)

        # Declare Impromptu reservations (reservations not made in advance)
        # Note: Hardcoding "repair" reservations as impromptu, though this is based on this specific naming convention.
        df_resv["IMPROMPTU"] = df_resv.RESV_NAME.apply(lambda name: True if name in self.impromptu_reservation_names or "repair" in name else False)

        # Filter reservations to remove those ending before the start date
        df_resv = df_resv[df_resv["END_TIME"].between(sim_start, sim_end)]

        # Explode the dataframe so there is a single row for every node in each reservation
        df_resv = df_resv.explode("NODELIST")

        return df_resv, valid_resv, resv_end_times
    
    def merge_reservations(self, df):
        df = df.sort_values(['RESV_NAME', 'START_TIME'])

        merged_names = []
        merged_starts = []
        merged_ends = []
        merged_nodes = []

        current_name = None
        current_start = None
        current_end = None
        current_nodes = None

        for _, row in df.iterrows():
            if (current_name == row['RESV_NAME'] and
                current_nodes == row['NODELIST'] and
                current_end == row['START_TIME']):
                current_end = row['END_TIME']
            else:
                if current_name is not None:
                    merged_names.append(current_name)
                    merged_starts.append(current_start)
                    merged_ends.append(current_end)
                    merged_nodes.append(current_nodes)

                current_name = row['RESV_NAME']
                current_start = row['START_TIME']
                current_end = row['END_TIME']
                current_nodes = row['NODELIST']

        if current_name is not None:
            merged_names.append(current_name)
            merged_starts.append(current_start)
            merged_ends.append(current_end)
            merged_nodes.append(current_nodes)

        merged_df = pd.DataFrame({
            'RESV_NAME': merged_names,
            'START_TIME': merged_starts,
            'END_TIME': merged_ends,
            'NODELIST': merged_nodes,
        })

        # With no reservations the rebuilt columns default to float64, which breaks
        # datetime comparisons downstream (e.g. empty dumps from trace-only setups).
        merged_df['START_TIME'] = pd.to_datetime(merged_df['START_TIME'])
        merged_df['END_TIME'] = pd.to_datetime(merged_df['END_TIME'])

        return merged_df

    def get_node_and_partition_data_from_slurm_conf(self, considered_partitions):
        """
        Get node and partition data from the slurm.conf file.
        """
        partition_data = {}
        nid_features, nodesets, nid_partitions, nid_weight = {}, {}, defaultdict(set), {}

        with open(self.slurm_conf, "r") as f:
            for line in f:
                # Skip lines that don't include 'nodename=' (case-insensitive)
                if not bool(re.match("nodename=", line, flags=re.I)):
                    continue
                line = line.strip("\n")

                # Convert the string of node names to a list of node names
                nids = convert_nodelist_to_node_nums(
                    re.split("nodename=", line, flags=re.I)[1].split(" ")[0], self.system
                )

                # Get the set of features for these nodes (default is empty set)
                if not bool(re.search("feature=", line, flags=re.I)):
                    features = set()
                else:
                    features = set(
                        re.split("feature=", line, flags=re.I)[1].split(" ")[0].split(",")
                    )

                # Get the weight for these nodes (default is 1)
                if not bool(re.search("weight=", line, flags=re.I)):
                    weight = 1
                else:
                    weight = int(re.split("weight=", line, flags=re.I)[1].split(" ")[0])

                # Set the features and the weight for these nodes
                for nid in nids:
                    nid_features[nid] = features
                    nid_weight[nid] = weight

            # Return to the beginning of the file
            f.seek(0)
            for line in f:
                # Skip lines that don't include 'nodeset=' (case-insensitive)
                if not bool(re.match("nodeset=", line, flags=re.I)):
                    continue
                line = line.strip("\n")

                name = re.split("nodeset=", line, flags=re.I)[1].split(" ")[0]

                # Get the set of features for this nodeset (default is empty set)
                if not bool(re.search("feature=", line, flags=re.I)):
                    nodeset_features = set()
                else:
                    nodeset_features = set(
                        re.split("feature=", line, flags=re.I)[1].split(" ")[0].split(",")
                    )

                # Add nodes to this nodeset if the node's features intersect with the nodeset features
                # From Slurm:
                # The nodeset configuration allows you to define a name for a specific set of nodes 
                # which can be used to simplify the partition configuration section, especially for 
                # heterogenous or condo-style systems. Each nodeset may be defined by an explicit list 
                # of nodes, and/or by filtering the nodes by a particular configured feature. If both 
                # Feature= and Nodes= are used the nodeset shall be the union of the two subsets. 
                # Note that the nodesets are only used to simplify the partition definitions at present, 
                # and are not usable outside of the partition configuration. 
                nodesets[name] = [
                    nid
                    for nid, features in nid_features.items()
                        if features.intersection(nodeset_features)
                ]

            # Return to the beginning of the file
            f.seek(0)
            for line in f:
                # Skip lines that don't include 'partitionname=' (case-insensitive)
                if not bool(re.match("partitionname=", line, flags=re.I)):
                    continue
                line = line.strip("\n")

                # Get the partition name
                name = re.split("partitionname=", line, flags=re.I)[1].split(" ")[0]

                # Skip this name if it's not in the considered partitions listed in the configuration file
                if name not in considered_partitions:
                    continue

                # Get the priority tier for this partition (default is 1)
                # From Slurm:
                # The PriorityTier of the Partition of the job or its Quality Of Service (QOS) can be 
                # used to identify which jobs can preempt or be preempted by other jobs. Slurm offers 
                # the ability to configure the preemption mechanism used on a per partition or per QOS basis. 
                # For example, jobs in a low priority queue may get requeued, while jobs in a medium priority
                # queue may get suspended. 
                if not bool(re.search("prioritytier=", line, flags=re.I)):
                    prio_tier = 1
                else:
                    prio_tier = int(re.split("prioritytier=", line, flags=re.I)[1].split(" ")[0])

                # Get the priority job factor for this partition (default is 1)
                # This is used when calculating a job's priority:
                # Job_priority = site_factor + (PriorityWeightAge) * (age_factor) +
                #     (PriorityWeightAssoc) * (assoc_factor) +
                #     (PriorityWeightFairshare) * (fair-share_factor) +
                #     (PriorityWeightJobSize) * (job_size_factor) +
                #     (PriorityWeightPartition) * (priority_job_factor) +   <------ Used here
                #     (PriorityWeightQOS) * (QOS_factor) +
                #     SUM(TRES_weight_cpu * TRES_factor_cpu,
                #         TRES_weight_<type> * TRES_factor_<type>,
                #         ...)
                #     - nice_factor
                if not bool(re.search("priorityjobfactor=", line, flags=re.I)):
                    prio_jobfactor = 1
                else:
                    prio_jobfactor = int(
                        re.split("priorityjobfactor=", line, flags=re.I)[1].split(" ")[0]
                    )

                # Need this for quotas
                if not bool(re.search("QOS=", line, flags=re.I)):
                    qos_name = 'normal'
                else:
                    qos_name = re.split("QOS=", line, flags=re.I)[1].split(" ")[0]

                # Gather this partition priority data
                partition_data[name] = {
                    "prio_tier" : prio_tier, "prio_jobfactor" : prio_jobfactor, "qos_name": qos_name
                }

                # Get a a string defining the nodes available to this partition
                nodes = re.split(" nodes=", line, flags=re.I)[1].split(" ")[0]

                # If the string of nodes matches a string of nodes used to name a nodeset
                if nodes in nodesets:
                    # Then for each node in the nodeset
                    # ...add this partition to the list of partitions the node is available to
                    for nid in nodesets[nodes]:
                        nid_partitions[nid].add(name)
                else:
                    # Else, get the list of nodes defined by the string
                    # ...and for each of these nodes
                    for nid in convert_nodelist_to_node_nums(nodes, self.system):
                        # add this partition to the list of partitions the node is available to
                        nid_partitions[nid].add(name)

        # Normalize the priority job factor by dividing by the maximum value
        max_partition_prio = max(data["prio_jobfactor"] for data in partition_data.values())
        if max_partition_prio:
            for data in partition_data.values():
                data["prio_jobfactor"] /= max_partition_prio

        return partition_data, nid_partitions, nid_weight
        

    def gather_node_data_from_events_resv_and_conf(self, nid_partitions, nid_weight, df_events, df_resv):
        """
        Gathers all the data from events, reservations, and slurm.conf for every node.
        Returns:
        nid_data (dict): Keys = Node IDs
                         Values:
                             'weight': Node Weight from slurm.conf (higher weight nodes get selected more often, default is 1)
                             'down_schedule': a list of events when the node is in DOWN/DRAIN state
                             'resv_schedule': a list of reservations for the node
                             'nid_partitions': a list of partitions to which the node is available
        """
        # hpe_restrictlong_nids is HPE ARCHER2 specific
        nid_data, hpe_restrictlong_nids = {}, []

        # nid_partitions is a defaultdict where the keys are Node IDs (e.g. 'x3000c0s29b0n0') 
        # and the values are sets of partition names (e.g. {'standard', 'short', 'long'}). 
        # It is built in get_node_and_partition_data_from_slurm_conf (see above)
        # It should contain all Node IDs for all considered partitions (see sim configuration file)
        # and all nodesets (if there are any defined in the slurm.conf file)
        print_and_log(logger, 'Getting Node reservation and down events schedules.')
        for nid in nid_partitions:
            # down_schedule is a list of node events for this node
            down_schedule = []
            # df_events is a Pandas DataFrame of node events, with each event signaling a time period
            # where a node is either in the DOWN or DRAIN state
            for _, row in df_events.loc[(df_events.Id == nid)].iterrows():
                # Append any node events involving this node to the down_schedule
                down_schedule.append([row.TimeStart, row.Duration, row.State, row.Reason])

            # Merge any adjacent down events
            down_schedule = self.merge_adjacent_down_events(down_schedule)
            
            # Sort by TimeStart in descending order, so the earliest event is last.
            # This enables popping events from the end of the list.
            down_schedule.sort(key=lambda schedule: schedule[0], reverse=True)
            
            resv_schedule = []
            impromptu_resv_schedule = []
            full_resv_schedule = []
            
            for _, row in df_resv.loc[(df_resv.NODELIST == nid)].iterrows():
                #### HPE SPECIFIC BEHAVIOR ####
                # Think this behaviour is being controlled by a maintenance script running in a
                # screen session
                if row.RESV_NAME == "HPE_RestrictLongJobs":
                    hpe_restrictlong_nids.append(nid)
                    continue
                #### END HPE SPECIFIC BEHAVIOR ####
                full_resv_schedule.append((row.START_TIME, row.END_TIME, row.RESV_NAME, row.IMPROMPTU))

            # Merge adjacent reservations
            full_resv_schedule = self.merge_adjacent_reservations(full_resv_schedule)

            # This function handles overlapping events and sorts the reservation schedule
            # by start time in descending order
            full_resv_schedule = split_and_mask_events(full_resv_schedule)

            resv_schedule = [resv[:3] for resv in full_resv_schedule if not resv[3]]
            # Impromptu reservations are not planned, but are put in place by System Admins as a reaction
            # to a circumstance. So, the system shouldn't be drained in advance for these reservations.
            impromptu_resv_schedule = [resv[:3] for resv in full_resv_schedule if resv[3]]

            # Gather data for this node
            nid_data[nid] = {
                "weight" : nid_weight[nid], 
                "down_schedule" : down_schedule,
                "resv_schedule" : resv_schedule, 
                "impromptu_resv_schedule" : impromptu_resv_schedule, 
                "partitions" : nid_partitions[nid]
            }

        return nid_data, hpe_restrictlong_nids

    def merge_adjacent_down_events(self, down_schedule):
        """
        If two down events are adjacent, we merge them to prevent issues with down events
        ending and starting at the same time step.
        """
        # sort down_schedule by TimeStart (Note: We should just sort the entire dataframe by 
        #                                        TimeStart in get_node_events)
        down_schedule.sort(key=lambda schedule: schedule[0])
        if len(down_schedule) <= 1:
            return down_schedule
        
        merged = []
        current_start, current_duration, current_state, current_reason = down_schedule[0]

        for start, duration, state, reason in down_schedule[1:]:
            if start == current_start + current_duration and state == current_state:
                current_duration += duration
            else:
                merged.append([current_start, current_duration, current_state, current_reason])
                current_start, current_duration, current_state, current_reason = start, duration, state, reason

        merged.append([current_start, current_duration, current_state, current_reason])

        return merged
        
        
    def merge_adjacent_reservations(self, resv_schedule):
        """
        If two reservations are adjacent, we merge them to prevent issues with reservations
        ending and starting at the same time step.
        """
        resv_schedule.sort(key=lambda schedule: (schedule[0],schedule[1],schedule[2]))
        if len(resv_schedule) <= 1:
            return resv_schedule

        merged = []
        current_start, current_end, current_name, current_impromptu = resv_schedule[0]

        for start, end, name, impromptu in resv_schedule[1:]:
            if start == current_end and name == current_name and impromptu == current_impromptu:
                current_end = end
            else:
                merged.append((current_start, current_end, current_name, current_impromptu))
                current_start, current_end, current_name, current_impromptu = start, end, name, impromptu

        merged.append((current_start, current_end, current_name, current_impromptu))

        return merged
            

    def get_qos(self):
        """
        Preprocess QOS data into a dictionary.
        """
        # Read the data from sacctmgr_qos.csv
        df_qos = pd.read_csv(
            self.qos_dump,  delimiter='|', lineterminator='\n', header=0, encoding="ISO-8859-1"
        )

        qos_data = {}

        # For each row in the file
        for _, row in df_qos.iterrows():
            # Get the QOS data
            qos_data[row.Name] = {
                # name: # QOS name (e.g. 'high', 'normal', 'standby')
                "name" : row.Name, 
                
                # prio: Priority used in job priority calculation
                "prio" : int(row.Priority),
                
                # GrpTRES: The total count of TRES able to be used at any given time from jobs running from a QOS.
                "GrpTRES" : ( 
                    None
                    if pd.isna(row.GrpTRES) or "node=" not in row.GrpTRES
                    else int(row.GrpTRES.split("node=")[1].split(",")[0])
                ),
                
                # GrpJobs: The total number of jobs able to run at any given time from a QOS.
                "GrpJobs" : None if pd.isna(row.GrpJobs) else int(row.GrpJobs),
                
                # GrpSubmit: The total number of jobs able to be submitted to the system at any given time from a QOS.
                "GrpSubmit" : None if pd.isna(row.GrpSubmit) else int(row.GrpSubmit),

                # MaxTRESPU: The maximum number of TRES a user can allocate at a given time.
                "MaxTRESPU" : (
                    None
                    if pd.isna(row.MaxTRESPU) or "node=" not in row.MaxTRESPU
                    else int(row.MaxTRESPU.split("node=")[1].split(",")[0])
                ),
                
                # MaxJobsPU: The maximum number of jobs a user can have running at a given time. 
                "MaxJobsPU" : None if pd.isna(row.MaxJobsPU) else int(row.MaxJobsPU),

                # MaxJobs: The total number of jobs able to run at any given time for the given association.
                "MaxJobs" : None if pd.isna(row.MaxJobs) else int(row.MaxJobs),

                # MaxSubmitPU: The maximum number of jobs a user can have running and pending at a given time. 
                "MaxSubmitPU" : None if pd.isna(row.MaxSubmitPU) else int(row.MaxSubmitPU),

                # The maximum number of jobs able to be submitted to the system at any given time from the given association.
                "MaxSubmit" : None if pd.isna(row.MaxSubmit) else int(row.MaxSubmit)
            }

        # Normalize the QOS priority values by dividing by the maximum value
        max_qos_prio = max(data["prio"] for data in qos_data.values())
        if max_qos_prio != 0:
            for data in qos_data.values():
                data["prio"] /= max_qos_prio

        return qos_data


    def get_cleaned_job_df(self, considered_partitions, def_power_per_node, 
                           sim_start, sim_end, initialize=False,
                           supplementary_resv=None, predicted_power=None, predicted_runtime=None
                           ):
        df_jobs = pd.read_csv(self.job_dump, sep='|', encoding='ISO-8859-1')

        # Clean jobs data
        df_jobs = df_jobs.loc[
            (df_jobs.Start != "None") & (df_jobs.Start.notna()) & (df_jobs.End != "None") & (df_jobs.End != "Unknown") &
            (df_jobs.End.notna()) & (df_jobs.Partition.isin(considered_partitions)) &
            (df_jobs.Timelimit.notna()) & (df_jobs.ReqNodes != "0") & (df_jobs.ReqNodes != 0) &
            (
                ((df_jobs.AllocNodes != "0") & (df_jobs.AllocNodes != 0)) |
                ((df_jobs.State.str.contains("CANCELLED")) & (df_jobs.Start == df_jobs.End))
            )
        ]

        # Format jobs data
        df_jobs.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Start = pd.to_datetime(df_jobs.Start, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.End = pd.to_datetime(df_jobs.End, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Elapsed = df_jobs.End - df_jobs.Start
        df_jobs.Timelimit = df_jobs.Timelimit.apply(lambda row: timelimit_str_to_timedelta(row))
        
        if initialize:
            # Filtering jobs to include only jobs starting after sim start date
            # Jobs that are running must have already started but not yet ended
            init_running_jobs = df_jobs[(df_jobs.Start <= sim_start) & (df_jobs.End > sim_start)].copy()
            print_and_log(logger, f"{len(init_running_jobs)} jobs running at {sim_start}")
            # Jobs that are queued must have already been submitted but not yet started
            init_queue_jobs = df_jobs[(df_jobs.Submit <= sim_start) & (df_jobs.Start > sim_start)].copy()
            print_and_log(logger, f"{len(init_queue_jobs)} jobs queued at {sim_start}")
            # The remaining jobs are those that have not yet been submitted or started
            # NOTE: Jobs shouldn't be started if they haven't yet been submitted, so the second condition
            # should be redundant.
            remaining_jobs = df_jobs[(df_jobs.Submit > sim_start) & (df_jobs.Start > sim_start)].copy()
            print_and_log(logger, f"{len(remaining_jobs)} jobs not running or queued at {sim_start}")

            # Altering jobs to simulate the initial cluster state. The idea here is
            # to artificially "submit" the running jobs before our sim start and 
            # submit the queued jobs a bit later. This should cause the simulator to
            # fill up the cluster with running jobs, then add the queued jobs to the 
            # queue at a later step.
            # We need to keep track of the original submission time of queued jobs,
            # because this is factored into priority sorting.
            init_running_jobs['SubmitPriority'] = init_running_jobs['Submit']
            init_queue_jobs['SubmitPriority'] = init_queue_jobs['Submit']
            remaining_jobs['SubmitPriority'] = None

            init_running_jobs['Submit'] = pd.to_datetime(sim_start) - timedelta(seconds=1)
            init_queue_jobs['Submit'] = pd.to_datetime(sim_start) + timedelta(seconds=5)

            # We want the initial running jobs to end at the same time they would have ended, so
            # we need to adjust the runtime (elapsed) of the job.
            elapsed_original = init_running_jobs['Elapsed'].copy()
            init_running_jobs['Elapsed'] = init_running_jobs['Elapsed'] - (pd.to_datetime(sim_start) - init_running_jobs['Start'])
            # Change energy usage based on decreased runtime
            init_running_jobs['ConsumedEnergyRaw'] = init_running_jobs['ConsumedEnergyRaw'] * (init_running_jobs['Elapsed'] / elapsed_original)

            df_jobs = pd.concat([init_running_jobs, init_queue_jobs, remaining_jobs], ignore_index=True)
            df_jobs = df_jobs[df_jobs.Start <= sim_end].copy()
        else:
            df_jobs = df_jobs[df_jobs.Start.between(sim_start, sim_end, inclusive='right')].copy()
            df_jobs['SubmitPriority'] = None # See note on SubmitPriority above


        # Convert strings with a metric prefix character (K, M, G, T) to an integer value.
        convert_to_raw(df_jobs, "AllocNodes")
        convert_to_raw(df_jobs, "ReqNodes")

        # Use Allocated Nodes for "Nodes" unless Allocated Nodes is 0 
        df_jobs["Nodes"] = df_jobs.apply(
            lambda row: row.ReqNodes if row.AllocNodes == 0 else row.AllocNodes, axis=1
        )

        # ARCHER2 specific
        # Some error in slurm accounting, can correct for case of one other user in account
        num_broken, num_fixed = len(df_jobs.loc[(df_jobs.User == "00:00:00")]), 0
        for i, anomalous_row in df_jobs.loc[(df_jobs.User == "00:00:00")].iterrows():
            acc_users = df_jobs.loc[(df_jobs.Account == anomalous_row.Account)].User.unique()
            if len(acc_users) == 2:
                num_fixed += 1
                df_jobs.at[i, "User"] = (
                    acc_users[1] if acc_users[0] == "00:00:00" else acc_users[0]
                )
        print("Corrected {} of {} users with name 00:00:00".format(num_fixed, num_broken))


        print_and_log(logger, 'Getting job power/energy data.')
        # Check for energy anomalies:
        #    jobs that aren't cancelled, but there is no meaningful value for ConsumedEnergy
        n_energy_anomalies = len(
            df_jobs.loc[
                (~df_jobs.State.str.contains("CANCELLED")) &
                (
                    (df_jobs.ConsumedEnergyRaw.isna()) | (df_jobs.ConsumedEnergyRaw == 0.0) |
                    (df_jobs.ConsumedEnergyRaw == "")
                )
            ]
        )
        # If these anomalous jobs are less than 25% of all jobs...
        if n_energy_anomalies / len(df_jobs) < 0.25:
            # Set ConsumedEnergyRaw of anomalous jobs to to default power per node
            df_jobs.ConsumedEnergyRaw = df_jobs.apply(
                lambda row: (
                    float(row.ConsumedEnergyRaw)
                    if (
                        row.ConsumedEnergyRaw == row.ConsumedEnergyRaw and
                        row.ConsumedEnergyRaw != 0.0 and
                        row.ConsumedEnergyRaw != ""
                    )
                    else float(def_power_per_node * row.AllocNodes * row.Elapsed.total_seconds())
                ),
                axis=1
            )

            # Calculate the Power consumed by the job (Power = Total Energy Used / Elapsed Time)
            df_jobs["Power"] = df_jobs.apply(
                lambda row: (
                    float(row.ConsumedEnergyRaw) / row.Elapsed.total_seconds()
                    if row.Elapsed.total_seconds() != 0
                    else 0.0
                ),
                axis=1
            )

            # Count jobs where the Power is at least 10 MW as anomalies (because that's probably an erroneous number)
            n_energy_anomalies += len(df_jobs.loc[(df_jobs.Power >= 10000000)])
            for i, _ in df_jobs.loc[(df_jobs.Power >= 10000000)].iterrows():
                # Set the Power for these jobs to the mean power per node times the number of nodes allocated
                df_jobs.at[i, "Power"] = def_power_per_node * df_jobs.at[i, "AllocNodes"]
            if def_power_per_node:
                print_and_log(logger, 
                    f"Set {n_energy_anomalies} jobs with bad or missing ConsumedEnergyRaw to mean power per node {def_power_per_node}W"
                )

            # Get the power per node for all jobs
            df_jobs["TruePowerPerNode"] = df_jobs.apply(
                lambda row: (
                    float(row.Power) / float(row.AllocNodes) if row.AllocNodes != 0 else 0.0
                ),
                axis=1
            )

        else: # If energy anomalies represent more than 25% of all jobs...
            print_and_log(logger, 
                "More than 25% of jobs do not have a valid ConsumedEnergy, setting all ConsumedEnergies to zero."
            )
            df_jobs = df_jobs.assign(ConsumedEnergyRaw=0.0)

            df_jobs["TruePowerPerNode"] = df_jobs.apply(lambda row: 0.0, axis=1)

        print_and_log(logger, 'Getting job SubmitLine arguments.')
        # If the SubmitLine column is provided, get the values for various arguments
        # TODO: All of this should be handled by changing the slurm_dump.sh script to get these
        # directly from sacct.
        if "SubmitLine" in df_jobs:
            df_jobs["DependencyArg"] = df_jobs.SubmitLine.apply( 
                # Defer the start of this job until the specified dependencies have been satisfied.
                lambda row: get_sbatch_cli_arg(row, long="--dependency", short="-d")
            )
            df_jobs["ReservationArg"] = df_jobs.SubmitLine.apply(
                # Allocate resources for the job from the named reservation.
                lambda row: get_sbatch_cli_arg(row, long="--reservation")
            )
            df_jobs["BeginArg"] = df_jobs.SubmitLine.apply(
                # Submit the batch script to the Slurm controller immediately, like normal, but 
                # tell the controller to defer the allocation of the job until the specified time. 
                lambda row: get_sbatch_cli_arg(row, long="--begin", short="-b")
            )
            df_jobs["NodelistArg"] = df_jobs.SubmitLine.apply(
                # Request a specific list of hosts. The job will contain all of these hosts and 
                # possibly additional hosts as needed to satisfy resource requirements.
                lambda row: get_sbatch_cli_arg(row, long="--nodelist", short="-w")
            )
            df_jobs["ExcludeArg"] = df_jobs.SubmitLine.apply(
                # Explicitly exclude certain nodes from the resources granted to the job.
                lambda row: get_sbatch_cli_arg(row, long="--exclude", short="-x")
            )
        else:
            df_jobs["DependencyArg"] = df_jobs.apply(lambda row: None, axis=1)
            df_jobs["ReservationArg"] = df_jobs.apply(lambda row: None, axis=1)
            df_jobs["BeginArg"] = df_jobs.apply(lambda row: None, axis=1)
            df_jobs["NodelistArg"] = df_jobs.apply(lambda row: None, axis=1)
            df_jobs["ExcludeArg"] = df_jobs.apply(lambda row: None, axis=1)

        # Get supplementary reservation data. This file needs to come from a separate database, and this may not be generally available.
        # If supplementary_resv is provided, read it in and merge it with the df_jobs dataframe. 
        # The csv file must contain the columns: job_id, reservation
        # This is the workaround necessary because not all reservations are found in the submit line. Some are found in the bash scripts
        # called in the submit line, so the get_sbatch_cli_arg function won't work (because that only looks at the SubmitLine string, not
        # at the bash script).
        # TODO:  This should be done for all of the above {}Arg columns.
        if supplementary_resv:
            print_and_log(logger, 'Getting reservations from supplemental dataset.')
            
            df_reservations = pd.read_csv(supplementary_resv, 
                                                delimiter='|', na_filter=False).rename(columns={'job_id': 'JobID'})
            df_reservations['JobID'] = df_reservations['JobID'].astype(str)
            df_jobs = pd.merge(df_jobs, 
                            df_reservations[['JobID', 'reservation']], 
                            how='left', on='JobID')
            
            # Fill in any missing reservations
            df_jobs["ReservationArg"] = df_jobs[["ReservationArg", "reservation"]].apply(lambda row: 
                                                                                        row.ReservationArg if row.ReservationArg else
                                                                                            row.reservation if row.reservation else None, axis=1)
            df_jobs.drop(columns='reservation', inplace=True)
        
        # Handle cases where df_jobs JobID was not found in df_reservations (set to None instead of NaN)
        # This is important for the way reservation arguments are handled later.
        df_jobs['ReservationArg'] = df_jobs['ReservationArg'].where(pd.notna(df_jobs['ReservationArg']), None)
        
        # If no nodes were allocated, the job was cancelled before it started
        # Set this column equal to the length of time between submission and cancellation
        df_jobs["Cancelled"] = df_jobs.apply(
            lambda row: None if row.AllocNodes != 0 else row.End - row.Submit, axis=1
        )

        # Convert JobIDs to strings
        df_jobs.JobID = df_jobs.JobID.apply(lambda row: str(row))
        print_and_log(logger, "{} heterogeneous JobIDs converted to regular JobIDs".format(
            len(df_jobs.loc[(df_jobs.JobID.str.contains("+", regex=False))])
        ))
        # Handle heterogeneous JobIDs
        df_jobs.JobID = df_jobs.JobID.apply(
            lambda row: str(int(row.split("+")[0]) + int(row.split("+")[1])) if "+" in row else row
        )

        # Create a list of jobs in the place of job arrays
        df_jobs.JobID = df_jobs.JobID.apply(
            lambda row: (
                [row.replace("[", "").replace("]", "")]
                if( "-" not in row and "," not in row) or ":" in row 
                else [
                    row.split("[")[0]  + str(num)#]
                    for num in [
                        index
                        for entry in row.split("_[")[1].strip("]").split("%")[0].split(",")
                            for index in range(
                                int(entry.split("-")[0]),
                                (
                                    int(entry.split("-")[1])
                                    if len(entry.split("-")) == 2
                                    else int(entry.split("-")[0]) + 1
                                )
                            )
                    ]
                ]
            )
        ) 

        # Create a new row for each job in the array
        num_jobs_with_arrs = len(df_jobs)
        df_jobs = df_jobs.explode("JobID")
        print_and_log(logger, f"{len(df_jobs) - num_jobs_with_arrs} cancelled job arrays converted to individual job entries")

        # Delete duplicated jobs
        df_jobs_orig_len = len(df_jobs)
        df_jobs = df_jobs[~df_jobs.duplicated(subset=["JobID", "Submit"], keep="first")]
        print_and_log(logger, 
            "{} duplicate (JobID,Submit) present, deleting".format(df_jobs_orig_len - len(df_jobs))
        )

        print_and_log(logger, "{} Jobs in workload trace".format(len(df_jobs)))
        print_and_log(logger, 
            "{} Jobs in workload trace cancelled before running".format(
                len(df_jobs.loc[(df_jobs.AllocNodes == 0)])
            )
        )

        # What this is doing is setting the job submission time to 6 hours before the job start time
        # if the Reason column is one of the reasons not tracked by the simulator. Commenting out -KM
        # bad_reasons = [
        #     "AssocMaxCpuMinutesPerJobLimit", "ReqNodeNotAvail", "BeginTime", "JobHeldUser",
        #     "DependencyNeverSatisfied", "JobArrayTaskLimit"
        # ]
        # df_jobs.loc[(df_jobs.Reason.isin(bad_reasons)), "QOS"] = "normal"
        # max_submit = df_jobs.Submit.max()
        # df_jobs.Submit = df_jobs.apply(
        #     lambda row: (
        #         row.Start - timedelta(hours=6) if row.Reason in bad_reasons else row.Submit
        #     ),
        #     axis=1
        # )
        # df_jobs = df_jobs.loc[(df_jobs.Submit <= max_submit)]

        # Merging predicted power and runtime data, if available
        if predicted_power and predicted_runtime:
            # The predicted power and runtime files must contain columns job_array_id and predicted_power/runtime
            # job_array_id is of the form JJJJJJJ or JJJJJJJ_A, where 'J' is the Job ID, and 'A' is the array position
            # for array jobs. This format matches the format used in the df_jobs dataframe.
            predicted_power_df = pd.read_pickle(predicted_power)
            merged_df = pd.merge(left=df_jobs, right=predicted_power_df[['job_array_id','predicted_power']], 
                                how='left', left_on='JobID', right_on='job_array_id')

            print(f'Found {len(merged_df[merged_df.predicted_power.isna()])} jobs with no predicted power.')

            predicted_runtime_df = pd.read_pickle(predicted_runtime)
            merged_df = pd.merge(left=merged_df, right=predicted_runtime_df[['job_array_id','predicted_runtime']], 
                                how='left', left_on='JobID', right_on='job_array_id')
            
            print(f'Found {len(merged_df[merged_df.predicted_runtime.isna()])} jobs with no predicted runtime.')

            df_jobs = merged_df.reset_index()
        else:
            df_jobs["predicted_power"] = None
            df_jobs["predicted_runtime"] = None

        print_and_log(logger, 'Finished initializing jobs data.')

        return df_jobs


    def handle_hpe_specific_features(self, nid_data, hpe_restrictlong_nids, hpe_restrictlong_sliding_res, nodes_down_in_blades, df_resv):
        # ====================================================================================== |
        # The code from this point on is all specific to Lumi/Archer. If nodes_down_in_blades    |
        # and hpe_restrictlong_sliding_res are set as False, and '' in the config file, then     |
        # this can all be safely ignored.                                                        |
        # ====================================================================================== |

        # It looks like LUMI puts blades with a down node into a maintenance reservation that
        # blocks all jobs. Recreate this by putting all the nodes in blade down when one of them
        # goes down
        if nodes_down_in_blades:
            # Give all nodes in blade same down schedule
            for first_blade_nid in list(nid_data)[::4]:
                shared_drain_schedule = [
                    down_block
                    for nid in range(first_blade_nid, first_blade_nid + 4)
                        if nid in nid_data
                        for down_block in nid_data[nid]["down_schedule"]
                            if down_block[2] == "DRAIN"
                ]

                for nid in range(first_blade_nid, first_blade_nid + 4):
                    if nid not in nid_data:
                        continue

                    for drain_block in shared_drain_schedule:
                        # If any overlap with existing DRAINs on the node, assume this node was
                        # in a maintenance reservation
                        if any(
                            max(
                                0,
                                (
                                    min(block[0] + block[1], drain_block[0] + drain_block[1]) -
                                    max(block[0], drain_block[0])
                                ).total_seconds()
                            )
                            for block in nid_data[nid]["down_schedule"]
                                if block[2] == "DRAIN"
                        ):
                            nid_data[nid]["down_schedule"].append(list(drain_block))

                    nid_data[nid]["down_schedule"].sort(key=lambda schedule: schedule[0])

                    i_event = 0
                    while i_event < len(nid_data[nid]["down_schedule"]) - 1:
                        event = nid_data[nid]["down_schedule"][i_event]
                        next_event = nid_data[nid]["down_schedule"][i_event + 1]

                        if event[0] + event[1] <= next_event[0]:
                            i_event += 1
                            continue

                        else:
                            nid_data[nid]["down_schedule"][i_event][1] = max(
                                event[1], next_event[0] + next_event[1] - event[0]
                            )
                            nid_data[nid]["down_schedule"][i_event][2] = "DRAIN"
                            nid_data[nid]["down_schedule"][i_event][3] = "blade down maintenance"
                            nid_data[nid]["down_schedule"].pop(i_event + 1)
                            continue

                    nid_data[nid]["down_schedule"].sort(
                        key=lambda schedule: schedule[0], reverse=True
                    )

        # The following code is different implementations of the HPE sliding maintenance reservation (HPE specific)
        if "-" in hpe_restrictlong_sliding_res:
            hpe_restrictlong_sliding_res, submit_hrs_before = (
                hpe_restrictlong_sliding_res.split("-")
            )
            submit_hrs_before = int(submit_hrs_before)
        else:
            submit_hrs_before = 0

        if (
            hpe_restrictlong_sliding_res == "dynamic" or
            hpe_restrictlong_sliding_res == "dynamic+const" or
            hpe_restrictlong_sliding_res == "dynamic+%extra"
        ):
            target_num_hpe_restrictlong = len(hpe_restrictlong_nids)
            if hpe_restrictlong_sliding_res == "dynamic+const":
                hpe_restrictlong_nids_cpy = set(hpe_restrictlong_nids)
                hpe_restrictlong_nids = defaultdict(lambda: hpe_restrictlong_nids_cpy.copy())
            else:
                hpe_restrictlong_nids = defaultdict(set)

            hpe_restrictlong_nids_nosubmitearly = defaultdict(set)

            for nid, data in nid_data.items():
                if not data["down_schedule"] or nid in hpe_restrictlong_nids:
                    continue

                for down_schedule in data["down_schedule"]:
                    if down_schedule[2] == "DOWN":
                        continue

                    reason_prefix = down_schedule[3].split(" ")[0]

                    if not reason_prefix.isupper():
                        continue

                    # Nodes go down in sets of 4 like this
                    nid_prefix = re.sub("[!^0-9]", "", nid)
                    nid_num_str = re.sub("[^0-9]", "", nid)
                    digits, nid_num = len(nid_num_str), int(nid_num_str)

                    nids = {
                        str(nid) for nid in range(nid_num - nid_num % 4, nid_num - nid_num % 4 + 4)
                    }
                    for nid in list(nids):
                        nids.remove(nid)
                        while len(nid) < digits:
                            nid = "0" + nid
                        nid = nid_prefix + nid
                        nids.add(nid)

                    first_submit = (
                        down_schedule[0].replace(minute=0, second=0) -
                        timedelta(hours=submit_hrs_before, minutes=5)
                    )
                    for submit_hr in range(
                        int(down_schedule[1] / timedelta(hours=1)) + submit_hrs_before + 2
                    ):
                        hpe_restrictlong_nids[
                            first_submit + timedelta(hours=submit_hr)
                        ].update(nids)
                    for submit_hr in range(int(down_schedule[1] / timedelta(hours=1)) + 1):
                        hpe_restrictlong_nids_nosubmitearly[
                            first_submit + timedelta(hours=submit_hr)
                        ].update(nids)


            if hpe_restrictlong_sliding_res == "dynamic":
                rev_submit_hrs = sorted(hpe_restrictlong_nids, reverse=True)
                for prev_submit_hr, submit_hr in zip(rev_submit_hrs[1:], rev_submit_hrs[:-1]):
                    new_nids = list(
                        hpe_restrictlong_nids[submit_hr] - hpe_restrictlong_nids[prev_submit_hr]
                    )
                    for new_nid in new_nids[
                        :max(
                            (
                                target_num_hpe_restrictlong -
                                len(hpe_restrictlong_nids[prev_submit_hr])
                            ),
                            0
                        )
                    ]:
                        hpe_restrictlong_nids[prev_submit_hr].add(new_nid)

            # Assume that at any given time there are some % extra compute blades in the hpelong
            # reservation than the ones that are actually down for doing work on
            if hpe_restrictlong_sliding_res == "dynamic+%extra":
                rev_submit_hrs = sorted(hpe_restrictlong_nids, reverse=True)
                for prev_submit_hr, submit_hr in zip(rev_submit_hrs[1:], rev_submit_hrs[:-1]):
                    prev_blade_nids = {
                        tuple( nid for nid in range(first_nid, first_nid + 4) )
                        for first_nid in sorted(hpe_restrictlong_nids[prev_submit_hr])[::4]
                    }
                    blade_nids = {
                        tuple( nid for nid in range(first_nid, first_nid + 4) )
                        for first_nid in sorted(hpe_restrictlong_nids[submit_hr])[::4]
                    }
                    new_blade_nids = list(blade_nids - prev_blade_nids)
                    # XXX Currently set to 30% XXX
                    target_blade_nids = int(
                        (len(hpe_restrictlong_nids_nosubmitearly[prev_submit_hr]) / 5) * 1.3 + 1
                    )
                    for blade_nids in new_blade_nids[
                        :max(target_blade_nids - len(prev_blade_nids), 0)
                    ]:
                        hpe_restrictlong_nids[prev_submit_hr].update(blade_nids)

        # Reservations split into multiple files representing sequences of maintenance nodes that
        # were previously all piled together in one file
        elif hpe_restrictlong_sliding_res != "" and "," in hpe_restrictlong_sliding_res:
            hpe_restrictlong_nids_streams = []

            for restrictlong_file in hpe_restrictlong_sliding_res.split(","):
                df_hpelong = pd.read_csv(
                    restrictlong_file,  delimiter=' ', lineterminator='\n',
                    names=["Time", "NNodes", "NodeIDs"], encoding="ISO-8859-1"
                )

                hpe_restrictlong_nids_stream = {}
                for _, row in df_hpelong.iterrows():
                    t = (
                        datetime.datetime.strptime(row.Time, "%Y-%m-%dT%H:%M:%S").replace(
                            minute=0, second=0
                        ) -
                        timedelta(hours=1)
                    )

                    nids = set(convert_nodelist_to_node_nums(row.NodeIDs.strip("\r")), self.system)
                    hpe_restrictlong_nids_stream[t] = nids

                while hpe_restrictlong_nids_stream and t <= max(hpe_restrictlong_nids_stream):
                    del hpe_restrictlong_nids_stream[max(hpe_restrictlong_nids_stream)]

                t_i, t_f = min(hpe_restrictlong_nids_stream), max(hpe_restrictlong_nids_stream)

                for time in list(hpe_restrictlong_nids_stream):
                    later_time = time + timedelta(hours=1)
                    while later_time not in hpe_restrictlong_nids_stream and later_time <= t_f:
                        hpe_restrictlong_nids_stream[later_time] = (
                            hpe_restrictlong_nids_stream[time]
                        )
                        later_time += timedelta(hours=1)

                hpe_restrictlong_nids_streams.append(hpe_restrictlong_nids_stream)

            hpe_restrictlong_nids = defaultdict(set)
            for hpe_restrictlong_nids_stream in hpe_restrictlong_nids_streams:
                for t, nids in hpe_restrictlong_nids_stream.items():
                    hpe_restrictlong_nids[t].update(nids)

        elif hpe_restrictlong_sliding_res != "": # file path to time - num nodes - node ids file
            # Load and clean actual hpe long num nodes data
            df_hpelong = pd.read_csv(
                hpe_restrictlong_sliding_res,  delimiter=' ', lineterminator='\n',
                names=["Time", "NNodes", "NodeIDs"], encoding="ISO-8859-1"
            )

            # NOTE: I think in the earlier months there are multiple difference reservations
            # with their own node lists being recorded simultaneously, this is why the timestamps
            # are mixed. Will need to try and disentangle these separate lists and combine them.

            hpe_restrictlong_nids_streams = []

            for _, row in df_hpelong.iterrows():
                t = (
                    datetime.datetime.strptime(row.Time, "%Y-%m-%dT%H:%M:%S").replace(
                        minute=0, second=0
                    ) -
                    timedelta(hours=1)
                )

                nids = set(convert_nodelist_to_node_nums(row.NodeIDs.strip("\r")), self.system)

                i_nids_stream, best_match = None, 0
                for i_stream, (latest_nids, submit_nids) in enumerate(
                    hpe_restrictlong_nids_streams
                ):
                    intersection = nids.intersection(latest_nids)
                    if len(intersection) > best_match and len(intersection) > int(len(nids) / 4):
                        i_nids_stream = i_stream
                        best_match = len(intersection)

                if i_nids_stream is None:
                    hpe_restrictlong_nids_streams.append([nids, defaultdict(set, {t : nids})])
                    continue

                _, nids_stream = hpe_restrictlong_nids_streams[i_nids_stream]
                while nids_stream and t <= max(nids_stream):
                    del nids_stream[max(nids_stream)]
                nids_stream[t] = nids
                hpe_restrictlong_nids_streams[i_nids_stream][0] = nids

                # All nodes
                hpe_restrictlong_nids[t].update(
                    convert_nodelist_to_node_nums(row.NodeIDs.strip("\r"), self.system)
                )
                # Latest nodelist entry
                hpe_restrictlong_nids[t] = set(
                    convert_nodelist_to_node_nums(row.NodeIDs.strip("\r"), self.system)
                )
                # nodelist with most nodes
                hpe_restrictlong_nids[t] = max(
                    hpe_restrictlong_nids[t],
                    set(convert_nodelist_to_node_nums(row.NodeIDs.strip("\r")), self.system),
                    key=lambda nids: len(nids)
                )
                # First nodelist entry (for filling gaps between entries with the later entry eg.
                # assume that the entry represents a print of the reservation state just before it
                # gets changed)
                t += timedelta(hours=1)
                if t not in hpe_restrictlong_nids:
                    hpe_restrictlong_nids[t] = set(
                        convert_nodelist_to_node_nums(row.NodeIDs.strip("\r"), self.system)
                    )
                # All nodes but assume moment before change
                hpe_restrictlong_nids[t + timedelta(hours=1)].update(
                    convert_nodelist_to_node_nums(row.NodeIDs.strip("\r"), self.system)
                )

                # Assuming the records are taken in chronological order and where there is a step
                # back in time this means add these nodes to all hours between latest time and
                # this step back in time

            t_i, t_f = min(hpe_restrictlong_nids), max(hpe_restrictlong_nids)

            # Assume entry is state the moment after changing reservation
            for time in list(hpe_restrictlong_nids):
                later_time = time + timedelta(hours=1)
                while later_time not in hpe_restrictlong_nids and later_time <= t_f:
                    hpe_restrictlong_nids[later_time] = hpe_restrictlong_nids[time]
                    later_time += timedelta(hours=1)

            # Assume entry is state the moment before changing reservation
            for time in list(hpe_restrictlong_nids):
                earlier_time = time - timedelta(hours=1)
                while earlier_time not in hpe_restrictlong_nids and earlier_time >= t_i:
                    hpe_restrictlong_nids[earlier_time] = hpe_restrictlong_nids[time]
                    earlier_time -= timedelta(hours=1)

            for _, hpe_restrictlong_nids_stream in hpe_restrictlong_nids_streams:
                t_i, t_f = min(hpe_restrictlong_nids_stream), max(hpe_restrictlong_nids_stream)

                for time in list(hpe_restrictlong_nids_stream):
                    later_time = time + timedelta(hours=1)
                    while later_time not in hpe_restrictlong_nids_stream and later_time <= t_f:
                        hpe_restrictlong_nids_stream[later_time] = (
                            hpe_restrictlong_nids_stream[time]
                        )
                        later_time += timedelta(hours=1)

            hpe_restrictlong_nids = defaultdict(set)
            for _, hpe_restrictlong_nids_stream in hpe_restrictlong_nids_streams:
                for t, nids in hpe_restrictlong_nids_stream.items():
                    hpe_restrictlong_nids[t].update(nids)

        # ARCHER2 specific - didn't implement REPLACE_DOWN on reservations so
        # just fill with nodes that don't go down at any point
        if not hpe_restrictlong_sliding_res == '' and len(df_resv.loc[(df_resv.RESV_NAME == "shortqos")]):
            shortqos_nids_to_replace = [
                nid
                for nid, data in sorted(nid_data.items())
                    if (
                        any(resv[2] == "shortqos" for resv in data["resv_schedule"]) and
                        not data["down_schedule"]
                    )
            ]
            never_down_nids = [
                nid
                for nid, data in sorted(nid_data.items())
                    if not data["down_schedule"] and not data["resv_schedule"]
            ]
            for i_shortqos_nid, shortqos_nid in enumerate(shortqos_nids_to_replace):
                found_nid = False

                for never_down_nid in never_down_nids:
                    if (
                        nid_data[never_down_nid]["partitions"] ==
                        nid_data[shortqos_nid]["partitions"]
                    ):
                        found_nid = True
                        break

                if not found_nid:
                    continue

                never_down_nids.remove(never_down_nid)

                nid_data[never_down_nid]["resv_schedule"] = nid_data[shortqos_nid]["resv_schedule"]
                nid_data[shortqos_nid]["resv_schedule"] = []
            print_and_log(logger, 
                "Replaced {} / {} shortqos nodes with nodes that never go down".format(
                    i_shortqos_nid + 1, len(shortqos_nids_to_replace)
                )
            )
        
        return nid_data, hpe_restrictlong_nids
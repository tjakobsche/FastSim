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

from collections import defaultdict
import datetime
import hashlib

from aux_funcs import print_and_log

import logging

logger = logging.getLogger(__name__)


class Partitions:
    """
    Keep track of all partitions.
    
    Methods:
    __init__
    remove_free_block
    add_free_block
    get_partition_by_name
    """
    def __init__(self, nid_data, partition_data):
        """
        Initialize the Partitions.

        partitions: the set of all Partition objects
        partitions_by_name: a dictionary of all Partition objects indexed by their string name
        nodes: the set of all Node objects
        reservations: a dictionary of all reservations. Initialized as an empty set here.
        free_blocks: a dictionary keeping track of the intervals within which nodes are available 
                     for specified reservations.

        Input:
        nid_data[nid] = { nid is the node name
               "weight" : nid_weight[nid], node weight (used by the scheduler when selecting nodes for jobs)
               "down_schedule" : down_schedule, a list of events when the node is down, with start time, duration, state, and reason
               "resv_schedule" : resv_schedule, a list of reservations for this node
               "partitions" : nid_partitions[nid], the list of partitions the node is available to
           }
        partition_data[name] = { name is the partition name
               "prio_tier" : prio_tier, Jobs submitted to a partition with a higher PriorityTier value will be evaluated by the scheduler before 
                                        pending jobs in a partition with a lower PriorityTier value.
               "prio_jobfactor" : prio_jobfactor, Partition factor used by priority/multifactor plugin in calculating job priority.
               "qos_name": the qos listed with the partition in slurm.conf. This is used for tracking qos-based resource limits.
               }
        """

        # self.partitions = {
        #     Partition(name, data["prio_tier"], data["prio_jobfactor"], data["qos_name"])
        #     for name, data in partition_data.items()
        # }
        self.partitions = [
            Partition(name, data["prio_tier"], data["prio_jobfactor"], data["qos_name"])
            for name, data in sorted(partition_data.items(), key=lambda kv: kv[0])
        ]
        """
        The list of all partitions.
        """
 
        self.partitions_by_name = {p.name : p for p in self.partitions}
        """
        A dictionary to index partition objects by name
        """

        # self.nodes = set()
        self.nodes = []
        """
        The set of all nodes in the cluster
        """

        for nid, data in sorted(nid_data.items(), key=lambda kv: kv[0]):
            # Create a Node object from the node data.
            node = Node(
                nid, data["weight"], data["down_schedule"], data["resv_schedule"], 
                data["impromptu_resv_schedule"]
                )
            for p_name in sorted(data["partitions"]):
                # Add this Node object to the partitions it is available to.
                self.partitions_by_name[p_name].add_node(node)
            # Add this Node object to the set of all nodes.
            self.nodes.append(node)

        for partition in self.partitions:
            partition.nodes.sort(key=lambda node: (node.weight, node.nid)) # Small weights get priority

        print_and_log(logger, "Using partitions:")
        print_and_log(logger, ' Partition Name | Priority Tier | Priority Weight | # of Nodes Available ')
        for partition in self.partitions:
            # Output Partitions & Nodes data to the terminal.
            print_and_log(logger,
                [str(partition.name).rjust(15), 
                str(partition.priority_tier).rjust(13), 
                str(partition.priority_weight).rjust(15), 
                str(len(partition.nodes)).rjust(14) + ' nodes'], 
                sep=" | "
            )
        print_and_log(logger, "With {} unique nodes total".format(len(self.nodes)))

        self.reservations = defaultdict(list)
        """
        The dict of all reservations.
        """

        
        self.free_blocks = defaultdict(lambda: defaultdict(set))
        for node in self.nodes:
            self.free_blocks[node.reservation][
                (node.interval_times[0], node.interval_times[-1])
            ].add(node)
        """
        Initialize the free blocks with the empty string reservation with all nodes.
        Format: free_blocks = { reservation_name : { interval : nodes, ... }, ... }
        
        Free_blocks keeps tracks of the intervals within which nodes are available for specified reservations.
        
        All nodes are initialized with node.reservation = '' and
        node.interval_times = [datetime min, 
                               datetime max or start of earliest reservation for this node (if this node has a reservation schedule)]
                               
        So this code block creates a single key '' at the first level, and then
        adds nodes to a list indexed by their initial interval_times at the second level.
        """

    def remove_free_block(self, node):
        """
        Remove the node from its current reservation-interval list,
        specified by node.reservation and node.interval_times.
        """
        interval = (node.interval_times[0], node.interval_times[-1])
        self.free_blocks[node.reservation][interval].remove(node) # Remove this node from this reservation during the specified interval
        if not self.free_blocks[node.reservation][interval]: 
            self.free_blocks[node.reservation].pop(interval) # Remove this interval from this reservation if there are no more nodes

    def add_free_block(self, node):
        """
        Add this node to its current reservation-interval list,
        specified by node.reservation and node.interval_times.
        """
        interval = (node.interval_times[0], node.interval_times[-1])
        self.free_blocks[node.reservation][interval].add(node)

    def get_partition_by_name(self, name):
        """
        Get a partition object from its name.
        """
        return self.partitions_by_name[name]


class Partition:
    """
    Handle individual partitions.

    Methods:
    __init__
    __hash__
    __eq__
    add_node
    """
    def __init__(self, name, priority_tier, priority_weight, qos_name):
        self.name = name # Partition name
        self.priority_tier = priority_tier # Jobs submitted to a partition with a higher PriorityTier value will be evaluated by the scheduler before 
                                           # pending jobs in a partition with a lower PriorityTier value.
        self.priority_weight = priority_weight # Partition factor used by priority/multifactor plugin in calculating job priority.
        self.qos_name = qos_name # The QOS listed with the partition in slurm.conf. This is used for tracking qos-based resource limits.

        self.nodes = [] # List of all nodes available to this partition 
        self._stable_hash = int.from_bytes(
            hashlib.blake2b(self.name.encode(), digest_size=8).digest(), "big"
        )

    def __hash__(self):
        return self._stable_hash

    def __eq__(self, other):
        if isinstance(other, Partition):
            return self.name == other.name
        return False

    def add_node(self, node):
        node.partitions.append(self) # Add this partition to the node's list of partitions it is available to
        node.partition_names.append(self.name) # Add the name of this partition to the node's list of partitions it is available to
        self.nodes.append(node) # Add this node to the list of nodes available to this partition
        # Partitions.__init__ sorts self.nodes once after all nodes are added
        # (re-sorting on every add was quadratic in cluster size)


class Node:
    def __init__(self, nid, weight, down_schedule, reservation_schedule, impromptu_reservation_schedule):
        self.nid = nid
        """
        This is the node ID
        """

        self.rack = nid[:5]
        """
        The node rack is declared in the first 5 characters, e.g. x1008 from node ID x1008c0s0b0n0.

        TODO: This is specific to that naming convention, and needs to be generalized via configuration.
        """

        self._stable_hash = int.from_bytes(
            hashlib.blake2b(self.nid.encode(), digest_size=8).digest(), "big"
        )
        self.weight = weight
        """
        node weight (used by the scheduler when selecting nodes for jobs)
        """

        self.free = True
        """
        Boolean declaring if this node is available
        """

        self.running_job = None
        """
        Holds the job running on this node
        """

        self.down_schedule = down_schedule
        """
        The list of down/drain events for this node
        """

        self.down = False
        """
        Booklean declaring if this node is down
        """

        self.up_time = None
        """
        Tracks when this down node will be up again. None if not down.
        """

        self.reservation_schedule = reservation_schedule
        """
        The list of reservations for this node
        """

        self.impromptu_reservation_schedule = impromptu_reservation_schedule
        """
        The list of impromptu reservations for this node. Impromptu reservations
        are put into place by administrators in response to an event, so there
        is no drainage of nodes in advance of these reservations.
        """
        self.reservation = ""
        """
        The name of the current reservation for this node
        """

        self.unreserved_time = None
        """
        When this reserved node will no longer be reserved. None if not reserved.
        """

        self.partitions = []
        """
        A list of partitions to which this node is available
        """

        self.partition_names = []
        """
        A list of of the names of the partitions to which this node is available
        This is useful in backfilling because it eliminates the need to check if
        two partitions are equal (instead we compare the string names of partitions)
        """

        # NOTE This is from when I was allowing the BF sched to plan nodes in a way that would be
        # respected by the main scheduling loop. Not doing this anymore so there is only ever 2
        # entries. Might be useful I want to implement a node going in and then out of a
        # reservation.
        # The beginning and end time of an interval for this node
        # An interval is bounded by events, either drain/down events, reservations, or running jobs.
        # TODO: Need to return to this to be sure this is accurate.
        self.interval_times = [
            datetime.datetime.min,
            # If there is a reservation schedule for this node, the end of this initial interval
            # is the start of the earliest reservation (reservation_schedule is sorted in descending
            # order by reservation start time)
            datetime.datetime.max if not reservation_schedule else reservation_schedule[-1][0]
        ]

        # TODO: Need to return to this for documentation
        self.bf_free_blocks_start = None 

    def __hash__(self):
        return self._stable_hash

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.nid == other.nid
        return False

    def set_reserved(self, reservation_name, end_time):
        """
        Set this node as being reserved by reservation_name until end_time.
        """
        self.reservation = reservation_name # This is the current reservation name
        self.unreserved_time = end_time # This is when the node reservation ends
        if self.down or not self.free:
            return
        self.free = False # Set node as not available unless it is down or already not available

    def set_unreserved(self):
        """
        Set this node as no longer reserved.
        """
        self.reservation = "" 
        self.unreserved_time = None
        if self.down or self.running_job:
            return
        self.free = True # Set this node as free unless it is down or running a job

    def set_down(self, up_time):
        """
        Set this node as down until up_time.
        """
        self.down = True
        self.up_time = up_time
        # If job is already running it is allowed to finish
        if not self.free:
            return
        self.free = False

    def set_up(self):
        """
        Set this node as up (no longer down).
        """
        self.down = False
        self.up_time = None
        if self.reservation:
            return
        self.free = True # Set this node as free unless it is reserved.

    def set_free(self):
        """
        Set this node as available.
        """
        if self.down or self.reservation:
            return
        self.free = True # Set this node as free unless it is down or reserved.

    def set_busy(self):
        """
        Set this node as not available.
        """
        if self.reservation:
            return
        self.free = False # Set this node as not available unless it is reserved.


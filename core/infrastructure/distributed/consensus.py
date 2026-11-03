from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class LogEntryType(Enum):
    COMMAND = "command"
    CONFIGURATION_CHANGE = "configuration_change"
    SNAPSHOT = "snapshot"


@dataclass(frozen=True, slots=True)
class LogEntry:
    term: int
    index: int
    entry_type: LogEntryType
    data: bytes
    timestamp: float = field(default_factory=time.time)
    checksum: str = field(default="")

    def __post_init__(self):
        if not self.checksum:
            object.__setattr__(self, 'checksum', hashlib.sha256(self.data).hexdigest())


@dataclass(frozen=True, slots=True)
class VoteRequest:
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass(frozen=True, slots=True)
class VoteResponse:
    term: int
    vote_granted: bool
    voter_id: str


@dataclass(frozen=True, slots=True)
class AppendEntriesRequest:
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int


@dataclass(frozen=True, slots=True)
class AppendEntriesResponse:
    term: int
    success: bool
    match_index: int
    follower_id: str


@dataclass(frozen=True, slots=True)
class InstallSnapshotRequest:
    term: int
    leader_id: str
    last_included_index: int
    last_included_term: int
    snapshot: bytes
    config: List[str]


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    nodes: List[str]
    learners: List[str] = field(default_factory=list)


class RaftNode:
    def __init__(
        self,
        node_id: str,
        cluster_config: ClusterConfig,
        state_machine: Callable[[LogEntry], Any],
        storage_path: str,
        election_timeout_ms: Tuple[int, int] = (150, 300),
        heartbeat_interval_ms: int = 50,
        max_append_entries: int = 100,
        snapshot_threshold: int = 10000,
    ):
        self.node_id = node_id
        self.cluster_config = cluster_config
        self.state_machine = state_machine
        self.storage_path = storage_path
        self.election_timeout_range = election_timeout_ms
        self.heartbeat_interval = heartbeat_interval_ms / 1000.0
        self.max_append_entries = max_append_entries
        self.snapshot_threshold = snapshot_threshold

        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0

        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}

        self.leader_id: Optional[str] = None
        self.last_contact = time.time()
        self.election_deadline = self._random_election_deadline()

        self.snapshot: Optional[bytes] = None
        self.snapshot_index = 0
        self.snapshot_term = 0

        self._running = False
        self._tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._apply_queue: asyncio.Queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=4)

        os.makedirs(storage_path, exist_ok=True)
        self._load_state()

    def _random_election_deadline(self) -> float:
        min_ms, max_ms = self.election_timeout_range
        return time.time() + random.uniform(min_ms, max_ms) / 1000.0

    async def start(self):
        self._running = True
        self._tasks.add(asyncio.create_task(self._run_election_timer()))
        self._tasks.add(asyncio.create_task(self._run_heartbeat()))
        self._tasks.add(asyncio.create_task(self._run_apply_loop()))
        self._tasks.add(asyncio.create_task(self._run_snapshot_check()))
        logger.info(f"Node {self.node_id} started as {self.state.value}")

    async def stop(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._executor.shutdown(wait=True)
        self._save_state()
        logger.info(f"Node {self.node_id} stopped")

    async def _run_election_timer(self):
        while self._running:
            await asyncio.sleep(0.01)
            if self.state != NodeState.LEADER:
                async with self._lock:
                    if time.time() >= self.election_deadline:
                        await self._start_election()

    async def _run_heartbeat(self):
        while self._running:
            if self.state == NodeState.LEADER:
                async with self._lock:
                    await self._send_append_entries()
            await asyncio.sleep(self.heartbeat_interval)

    async def _run_apply_loop(self):
        while self._running:
            try:
                entry = await asyncio.wait_for(self._apply_queue.get(), timeout=0.1)
                await self._apply_entry(entry)
            except asyncio.TimeoutError:
                continue

    async def _run_snapshot_check(self):
        while self._running:
            await asyncio.sleep(1.0)
            async with self._lock:
                if len(self.log) - self.snapshot_index > self.snapshot_threshold:
                    await self._create_snapshot()

    async def _start_election(self):
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.election_deadline = self._random_election_deadline()

        last_log_index = len(self.log) - 1
        last_log_term = self.log[-1].term if self.log else 0

        vote_request = VoteRequest(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=last_log_index,
            last_log_term=last_log_term,
        )

        votes = 1
        for node_id in self.cluster_config.nodes:
            if node_id == self.node_id:
                continue
            try:
                response = await self._request_vote(node_id, vote_request)
                if response.vote_granted:
                    votes += 1
                elif response.term > self.current_term:
                    await self._become_follower(response.term)
                    return
            except Exception as e:
                logger.warning(f"Vote request to {node_id} failed: {e}")

        if votes > len(self.cluster_config.nodes) // 2:
            await self._become_leader()
        else:
            self.election_deadline = self._random_election_deadline()

    async def _become_leader(self):
        self.state = NodeState.LEADER
        self.leader_id = self.node_id
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")

        last_index = len(self.log)
        for node_id in self.cluster_config.nodes:
            if node_id != self.node_id:
                self.next_index[node_id] = last_index
                self.match_index[node_id] = 0

        await self._send_append_entries()

    async def _become_follower(self, term: int):
        self.state = NodeState.FOLLOWER
        self.current_term = term
        self.voted_for = None
        self.leader_id = None
        self.election_deadline = self._random_election_deadline()

    async def _send_append_entries(self):
        for node_id in self.cluster_config.nodes:
            if node_id == self.node_id:
                continue

            prev_log_index = self.next_index.get(node_id, len(self.log)) - 1
            prev_log_term = self.log[prev_log_index].term if prev_log_index >= 0 and prev_log_index < len(self.log) else 0

            entries = self.log[prev_log_index + 1: prev_log_index + 1 + self.max_append_entries]

            request = AppendEntriesRequest(
                term=self.current_term,
                leader_id=self.node_id,
                prev_log_index=prev_log_index,
                prev_log_term=prev_log_term,
                entries=entries,
                leader_commit=self.commit_index,
            )

            try:
                response = await self._append_entries(node_id, request)
                if response.success:
                    self.next_index[node_id] = response.match_index + 1
                    self.match_index[node_id] = response.match_index
                    self._update_commit_index()
                elif response.term > self.current_term:
                    await self._become_follower(response.term)
                    return
                else:
                    self.next_index[node_id] = max(1, self.next_index.get(node_id, 1) - 1)
            except Exception as e:
                logger.warning(f"AppendEntries to {node_id} failed: {e}")

    def _update_commit_index(self):
        for i in range(len(self.log) - 1, self.commit_index, -1):
            count = 1
            for node_id in self.cluster_config.nodes:
                if node_id != self.node_id and self.match_index.get(node_id, 0) >= i:
                    count += 1
            if count > len(self.cluster_config.nodes) // 2 and self.log[i].term == self.current_term:
                self.commit_index = i
                break

    async def _apply_entry(self, entry: LogEntry):
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self._executor, self.state_machine, entry
            )
            self.last_applied = entry.index
            return result
        except Exception as e:
            logger.error(f"State machine error: {e}")
            raise

    async def _create_snapshot(self):
        snapshot_data = pickle.dumps({
            'state_machine': self.state_machine,
            'log': self.log[self.snapshot_index:],
            'commit_index': self.commit_index,
            'last_applied': self.last_applied,
        })
        self.snapshot = snapshot_data
        self.snapshot_index = len(self.log)
        self.snapshot_term = self.log[-1].term if self.log else 0

        snapshot_path = os.path.join(self.storage_path, f"snapshot_{self.snapshot_index}.pkl")
        with open(snapshot_path, 'wb') as f:
            f.write(self.snapshot)
        logger.info(f"Created snapshot at index {self.snapshot_index}")

    def _load_state(self):
        state_file = os.path.join(self.storage_path, "raft_state.pkl")
        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
                self.current_term = state.get('current_term', 0)
                self.voted_for = state.get('voted_for')
                self.log = state.get('log', [])
                self.commit_index = state.get('commit_index', 0)
                self.last_applied = state.get('last_applied', 0)
                self.snapshot_index = state.get('snapshot_index', 0)
                self.snapshot_term = state.get('snapshot_term', 0)
                self.snapshot = state.get('snapshot')
                logger.info(f"Loaded state: term={self.current_term}, log_len={len(self.log)}")

    def _save_state(self):
        state = {
            'current_term': self.current_term,
            'voted_for': self.voted_for,
            'log': self.log,
            'commit_index': self.commit_index,
            'last_applied': self.last_applied,
            'snapshot_index': self.snapshot_index,
            'snapshot_term': self.snapshot_term,
            'snapshot': self.snapshot,
        }
        state_file = os.path.join(self.storage_path, "raft_state.pkl")
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)

    async def propose(self, data: bytes) -> bool:
        async with self._lock:
            if self.state != NodeState.LEADER:
                return False

            entry = LogEntry(
                term=self.current_term,
                index=len(self.log),
                entry_type=LogEntryType.COMMAND,
                data=data,
            )
            self.log.append(entry)
            await self._send_append_entries()
            return True

    async def _request_vote(self, node_id: str, request: VoteRequest) -> VoteResponse:
        await asyncio.sleep(random.uniform(0.001, 0.01))
        return VoteResponse(term=self.current_term, vote_granted=True, voter_id=node_id)

    async def _append_entries(self, node_id: str, request: AppendEntriesRequest) -> AppendEntriesResponse:
        await asyncio.sleep(random.uniform(0.001, 0.01))
        return AppendEntriesResponse(term=self.current_term, success=True, match_index=request.prev_log_index + len(request.entries), follower_id=node_id)


class RaftCluster:
    def __init__(self, nodes: List[RaftNode]):
        self.nodes = {node.node_id: node for node in nodes}

    async def start_all(self):
        for node in self.nodes.values():
            await node.start()

    async def stop_all(self):
        for node in self.nodes.values():
            await node.stop()

    def get_leader(self) -> Optional[RaftNode]:
        for node in self.nodes.values():
            if node.state == NodeState.LEADER:
                return node
        return None

    async def propose(self, data: bytes) -> bool:
        leader = self.get_leader()
        if leader:
            return await leader.propose(data)
        return False


class DistributedLock:
    def __init__(self, cluster: RaftCluster, lock_name: str):
        self.cluster = cluster
        self.lock_name = lock_name
        self.owner_id: Optional[str] = None
        self.sequence = 0

    async def acquire(self, owner_id: str, timeout: float = 10.0) -> bool:
        data = json.dumps({
            'type': 'lock_acquire',
            'lock': self.lock_name,
            'owner': owner_id,
            'seq': self.sequence,
        }).encode()

        start = time.time()
        while time.time() - start < timeout:
            success = await self.cluster.propose(data)
            if success:
                self.owner_id = owner_id
                self.sequence += 1
                return True
            await asyncio.sleep(0.1)
        return False

    async def release(self, owner_id: str) -> bool:
        if self.owner_id != owner_id:
            return False

        data = json.dumps({
            'type': 'lock_release',
            'lock': self.lock_name,
            'owner': owner_id,
        }).encode()

        return await self.cluster.propose(data)


class DistributedCounter:
    def __init__(self, cluster: RaftCluster, counter_name: str, initial: int = 0):
        self.cluster = cluster
        self.counter_name = counter_name
        self.value = initial
        self._local_cache = initial

    async def increment(self, delta: int = 1) -> int:
        data = json.dumps({
            'type': 'counter_increment',
            'counter': self.counter_name,
            'delta': delta,
        }).encode()

        success = await self.cluster.propose(data)
        if success:
            self._local_cache += delta
        return self._local_cache

    async def get(self) -> int:
        return self._local_cache


class LinearizableKVStore:
    def __init__(self, cluster: RaftCluster):
        self.cluster = cluster
        self.store: Dict[str, bytes] = {}

    async def put(self, key: str, value: bytes) -> bool:
        data = json.dumps({
            'type': 'kv_put',
            'key': key,
            'value': value.hex(),
        }).encode()
        return await self.cluster.propose(data)

    async def get(self, key: str) -> Optional[bytes]:
        return self.store.get(key)

    async def delete(self, key: str) -> bool:
        data = json.dumps({
            'type': 'kv_delete',
            'key': key,
        }).encode()
        return await self.cluster.propose(data)

    async def compare_and_swap(self, key: str, expected: Optional[bytes], new_value: bytes) -> bool:
        data = json.dumps({
            'type': 'kv_cas',
            'key': key,
            'expected': expected.hex() if expected else None,
            'value': new_value.hex(),
        }).encode()
        return await self.cluster.propose(data)


def create_raft_cluster(
    node_ids: List[str],
    state_machine: Callable[[LogEntry], Any],
    base_storage_path: str,
) -> RaftCluster:
    config = ClusterConfig(nodes=node_ids)
    nodes = []
    for node_id in node_ids:
        node = RaftNode(
            node_id=node_id,
            cluster_config=config,
            state_machine=state_machine,
            storage_path=os.path.join(base_storage_path, node_id),
        )
        nodes.append(node)
    return RaftCluster(nodes)
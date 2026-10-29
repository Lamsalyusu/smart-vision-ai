from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class LogEntry:
    term: int
    command: bytes

    def __post_init__(self) -> None:
        if self.term < 1:
            raise ValueError("log-entry terms start at one")


@dataclass(frozen=True, slots=True)
class VoteRequest:
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int


@dataclass(frozen=True, slots=True)
class VoteResponse:
    term: int
    voter_id: str
    granted: bool


@dataclass(frozen=True, slots=True)
class AppendEntriesRequest:
    term: int
    leader_id: str
    previous_log_index: int
    previous_log_term: int
    entries: tuple[LogEntry, ...]
    leader_commit_index: int


@dataclass(frozen=True, slots=True)
class AppendEntriesResponse:
    term: int
    follower_id: str
    success: bool
    last_log_index: int


class RaftRole(str, Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class RaftNode:
    """In-memory Raft election and replicated-log state machine.

    Networking, durable storage, election timers, and membership changes are
    intentionally provided by an outer adapter rather than this state machine.
    """

    def __init__(self, node_id: str, cluster_size: int) -> None:
        if not node_id.strip():
            raise ValueError("node id must not be empty")
        if cluster_size < 1:
            raise ValueError("cluster size must be positive")
        self.node_id = node_id
        self.cluster_size = cluster_size
        self.current_term = 0
        self.voted_for: str | None = None
        self.role = RaftRole.FOLLOWER
        self.log: list[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        self._votes_received: set[str] = set()
        self._next_index: dict[str, int] = {}
        self._match_index: dict[str, int] = {}

    @property
    def majority(self) -> int:
        return self.cluster_size // 2 + 1

    @property
    def last_log_index(self) -> int:
        return len(self.log)

    @property
    def last_log_term(self) -> int:
        return self.log[-1].term if self.log else 0

    def begin_election(self) -> VoteRequest:
        self.current_term += 1
        self.role = RaftRole.CANDIDATE
        self.voted_for = self.node_id
        self._votes_received = {self.node_id}
        if len(self._votes_received) >= self.majority:
            self._become_leader()
        return VoteRequest(
            self.current_term,
            self.node_id,
            self.last_log_index,
            self.last_log_term,
        )

    def receive_vote_request(self, request: VoteRequest) -> VoteResponse:
        if request.term > self.current_term:
            self._observe_term(request.term)
        up_to_date = (request.last_log_term, request.last_log_index) >= (
            self.last_log_term,
            self.last_log_index,
        )
        can_vote = self.voted_for in (None, request.candidate_id)
        granted = request.term == self.current_term and can_vote and up_to_date
        if granted:
            self.voted_for = request.candidate_id
            self.role = RaftRole.FOLLOWER
        return VoteResponse(self.current_term, self.node_id, granted)

    def receive_vote_response(self, response: VoteResponse) -> bool:
        if response.term > self.current_term:
            self._observe_term(response.term)
            return False
        if self.role is not RaftRole.CANDIDATE or response.term != self.current_term:
            return self.role is RaftRole.LEADER
        if response.granted:
            self._votes_received.add(response.voter_id)
            if len(self._votes_received) >= self.majority:
                self._become_leader()
        return self.role is RaftRole.LEADER

    def append_command(self, command: bytes) -> int:
        if self.role is not RaftRole.LEADER:
            raise RuntimeError("only the leader can append commands")
        if not isinstance(command, bytes):
            raise ValueError("Raft commands must be bytes")
        self.log.append(LogEntry(self.current_term, command))
        self._match_index[self.node_id] = self.last_log_index
        return self.last_log_index

    def append_entries_for(self, follower_id: str, batch_size: int = 64) -> AppendEntriesRequest:
        if self.role is not RaftRole.LEADER:
            raise RuntimeError("only the leader can replicate entries")
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        next_index = self._next_index.get(follower_id, 1)
        next_index = max(1, min(next_index, self.last_log_index + 1))
        previous_index = next_index - 1
        entries = tuple(self.log[previous_index : previous_index + batch_size])
        previous_term = self.log[previous_index - 1].term if previous_index else 0
        return AppendEntriesRequest(
            self.current_term,
            self.node_id,
            previous_index,
            previous_term,
            entries,
            self.commit_index,
        )

    def receive_append_entries(self, request: AppendEntriesRequest) -> AppendEntriesResponse:
        if request.term > self.current_term:
            self._observe_term(request.term)
        if request.term < self.current_term:
            return AppendEntriesResponse(
                self.current_term, self.node_id, False, self.last_log_index
            )

        self.role = RaftRole.FOLLOWER
        if request.previous_log_index > self.last_log_index:
            return AppendEntriesResponse(
                self.current_term, self.node_id, False, self.last_log_index
            )
        if (
            request.previous_log_index > 0
            and self.log[request.previous_log_index - 1].term != request.previous_log_term
        ):
            return AppendEntriesResponse(
                self.current_term, self.node_id, False, self.last_log_index
            )

        insertion_index = request.previous_log_index
        for incoming in request.entries:
            if insertion_index < self.last_log_index:
                existing = self.log[insertion_index]
                if existing.term != incoming.term:
                    if insertion_index < self.commit_index:
                        raise RuntimeError("cannot overwrite a committed log entry")
                    del self.log[insertion_index:]
            if insertion_index == self.last_log_index:
                self.log.append(incoming)
            insertion_index += 1

        self.commit_index = min(request.leader_commit_index, self.last_log_index)
        self.last_applied = max(self.last_applied, self.commit_index)
        return AppendEntriesResponse(
            self.current_term, self.node_id, True, self.last_log_index
        )

    def acknowledge_replication(self, response: AppendEntriesResponse) -> int:
        if self.role is not RaftRole.LEADER:
            raise RuntimeError("only the leader can record replication acknowledgements")
        if response.term > self.current_term:
            self._observe_term(response.term)
            return self.commit_index
        if response.term != self.current_term:
            return self.commit_index

        if response.success:
            self._match_index[response.follower_id] = response.last_log_index
            self._next_index[response.follower_id] = response.last_log_index + 1
            self._advance_commit_index()
        else:
            current_next = self._next_index.get(response.follower_id, self.last_log_index + 1)
            self._next_index[response.follower_id] = max(1, current_next - 1)
        return self.commit_index

    def _advance_commit_index(self) -> None:
        for index in range(self.last_log_index, self.commit_index, -1):
            if self.log[index - 1].term != self.current_term:
                continue
            replicated = sum(match_index >= index for match_index in self._match_index.values())
            if replicated >= self.majority:
                self.commit_index = index
                self.last_applied = max(self.last_applied, index)
                return

    def _become_leader(self) -> None:
        self.role = RaftRole.LEADER
        self._next_index = {}
        self._match_index = {self.node_id: self.last_log_index}

    def _observe_term(self, term: int) -> None:
        self.current_term = term
        self.voted_for = None
        self.role = RaftRole.FOLLOWER
        self._votes_received.clear()
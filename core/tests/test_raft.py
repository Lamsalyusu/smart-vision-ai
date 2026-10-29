import unittest

from core.infrastructure.consensus import (
    AppendEntriesRequest,
    LogEntry,
    RaftNode,
    RaftRole,
    VoteRequest,
    VoteResponse,
)


class RaftNodeTests(unittest.TestCase):
    def test_candidate_becomes_leader_after_majority_votes(self) -> None:
        leader = RaftNode("a", cluster_size=3)
        first_follower = RaftNode("b", cluster_size=3)

        request = leader.begin_election()
        response = first_follower.receive_vote_request(request)
        leader.receive_vote_response(response)

        self.assertEqual(leader.role, RaftRole.LEADER)
        self.assertEqual(leader.current_term, 1)
        second_vote = first_follower.receive_vote_request(VoteRequest(1, "c", 0, 0))
        self.assertFalse(second_vote.granted)

    def test_follower_votes_only_once_per_term(self) -> None:
        follower = RaftNode("follower", cluster_size=3)

        first = follower.receive_vote_request(VoteRequest(1, "a", 0, 0))
        second = follower.receive_vote_request(VoteRequest(1, "b", 0, 0))

        self.assertTrue(first.granted)
        self.assertFalse(second.granted)

    def test_rejects_candidate_with_stale_log(self) -> None:
        follower = RaftNode("follower", cluster_size=3)
        follower.log = [LogEntry(2, b"newer")]

        response = follower.receive_vote_request(VoteRequest(3, "candidate", 8, 1))

        self.assertFalse(response.granted)

    def test_append_entries_replicates_and_commits_on_majority(self) -> None:
        leader = RaftNode("leader", cluster_size=3)
        follower = RaftNode("follower", cluster_size=3)
        election = leader.begin_election()
        leader.receive_vote_response(VoteResponse(election.term, "follower-2", True))
        leader.append_command(b"camera-config-v1")

        response = follower.receive_append_entries(leader.append_entries_for("follower"))
        leader.acknowledge_replication(response)

        self.assertEqual(follower.log, leader.log)
        self.assertEqual(leader.commit_index, 1)
        self.assertEqual(follower.commit_index, 0)

        heartbeat = leader.append_entries_for("follower")
        heartbeat = AppendEntriesRequest(
            heartbeat.term,
            heartbeat.leader_id,
            heartbeat.previous_log_index,
            heartbeat.previous_log_term,
            heartbeat.entries,
            leader.commit_index,
        )
        follower.receive_append_entries(heartbeat)
        self.assertEqual(follower.commit_index, 1)

    def test_rejects_append_entries_with_conflicting_previous_term(self) -> None:
        follower = RaftNode("follower", cluster_size=3)
        follower.log = [LogEntry(1, b"old")]

        response = follower.receive_append_entries(
            AppendEntriesRequest(2, "leader", 1, 2, (), 0)
        )

        self.assertFalse(response.success)

    def test_only_current_term_entry_advances_commit(self) -> None:
        leader = RaftNode("leader", cluster_size=3)
        leader.log = [LogEntry(1, b"old")]
        election = leader.begin_election()
        leader.receive_vote_response(VoteResponse(election.term, "follower", True))
        leader._match_index["follower"] = 1

        self.assertEqual(leader.commit_index, 0)

    def test_append_entries_rejects_old_leader_term(self) -> None:
        follower = RaftNode("follower", cluster_size=3)
        follower.current_term = 2

        response = follower.receive_append_entries(
            AppendEntriesRequest(1, "old-leader", 0, 0, (), 0)
        )

        self.assertFalse(response.success)
        self.assertEqual(response.term, 2)

    def test_same_term_heartbeat_does_not_change_recorded_vote(self) -> None:
        follower = RaftNode("follower", cluster_size=3)
        follower.receive_vote_request(VoteRequest(1, "candidate-a", 0, 0))

        follower.receive_append_entries(
            AppendEntriesRequest(1, "leader-b", 0, 0, (), 0)
        )

        self.assertFalse(
            follower.receive_vote_request(VoteRequest(1, "candidate-b", 0, 0)).granted
        )


if __name__ == "__main__":
    unittest.main()
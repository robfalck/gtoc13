from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Callable, Hashable, Iterable, Optional, Sequence, Tuple, List
import heapq, math

# ============================================================
# Minimal, generic Beam Search (parallel scoring; dynamics-agnostic)
# ============================================================

@dataclass(slots=True)
class Node:
    """Tree node stored by the search. `state` is opaque to this class."""
    id: int
    parent_id: Optional[int]
    state: Any
    depth: int
    cum_score: float  # higher is better


class BeamSearch:
    """
    Generic beam search with parallel scoring.

    Required input function signatures
    ----------------------------------
    expand_fn(parent_state) -> Iterable[proposal]
        - Purpose: cheaply enumerate *proposals* for the next step.
        - Must be fast (no heavy physics/solvers).
        - Example type shapes:
            parent_state: Any
            proposal    : Any  # e.g., {"target": int, "tof": float, "epoch0": float}

    score_fn(parent_state, proposal) -> float | (float, resolved_child_state)
        - Purpose: perform heavy work (Lambert/propagation/feasibility) and
          produce an *incremental* score to add to the parent's cum_score.
        - Return just a float to store the proposal as-is, OR return a
          (score, resolved_child_state) tuple to store a *resolved* child
          that may include extra fields (e.g., arrival epoch, v∞).
        - Returning -math.inf or NaN will prune the candidate.

    Optional input functions
    ------------------------
    key_fn(state) -> Hashable
        - Purpose: coarse dedup; keep only the best child per key.
        - Example: (body_id, epoch_bin, vinf_bin)

    is_terminal_fn(node: Node) -> bool
        - Purpose: early stopping per-node; defaults to depth >= max_depth.

    Parallelism notes
    -----------------
    - parallel_backend: "thread" | "process" | None
    - With "process", score_fn (and any referenced objects) must be picklable
      (i.e., defined at module top-level); keep proposals small to minimize IPC.

    -----------------------------------------------------------------------
    # Simple examples of input functions (toy integer problem)

    def expand_fn(s: int) -> Iterable[int]:
        # Propose next integers cheaply
        return (s + d for d in (1, 2, 3))

    def score_fn(parent: int, child: int) -> float | tuple[float, int]:
        # Reward closeness to a target; heavy work would live here
        target = 50
        return -abs(target - child), child  # (score, resolved_state)

    def key_fn(state: int) -> Hashable:
        # Dedup by exact integer (use bins in real problems)
        return state

    def is_terminal_fn(node: Node) -> bool:
        # Stop early if threshold reached OR max depth hit
        return node.depth >= 6 or node.state >= 50
    -----------------------------------------------------------------------
    """

    def __init__(
        self,
        expand_fn: Callable[[Any], Iterable[Any]],
        score_fn: Callable[[Any, Any], float | tuple[float, Any]],
        *,
        beam_width: int,
        max_depth: int,
        key_fn: Optional[Callable[[Any], Hashable]] = None,
        is_terminal_fn: Optional[Callable[[Node], bool]] = None,
        parallel_backend: Optional[str] = "thread",   # "thread" | "process" | None
        max_workers: Optional[int] = None,
        score_chunksize: int = 256,                   # batch size for scoring tasks
    ) -> None:
        self.expand_fn = expand_fn
        self.score_fn = score_fn
        self.beam_width = int(beam_width)
        self.max_depth = int(max_depth)
        self.key_fn = key_fn
        self.is_terminal_fn = is_terminal_fn or (lambda n: n.depth >= self.max_depth)
        self.parallel_backend = (parallel_backend or "none").lower()
        self.max_workers = max_workers
        self.score_chunksize = max(1, int(score_chunksize))

        # Internals
        self._next_id = 0
        self._nodes: dict[int, Node] = {}

    # ---------------------------- Public API ---------------------------------

    def run(self, root_state: Any) -> List[Node]:
        """Run beam search from `root_state`. Return final top-K nodes."""
        root = self._make_node(None, root_state, depth=0, cum_score=0.0)
        frontier: List[Node] = [root]

        for _ in range(1, self.max_depth + 1):
            # Keep only top-K parents
            frontier = self._top_k(frontier, self.beam_width)
            if all(self.is_terminal_fn(n) for n in frontier):
                break

            # (1) Cheap expansion -> (parent, proposal) pairs
            expansions: List[Tuple[Node, Any]] = []
            for parent in frontier:
                if self.is_terminal_fn(parent):
                    continue
                for prop in self.expand_fn(parent.state):
                    expansions.append((parent, prop))
            if not expansions:
                break

            # (2) Heavy scoring (parallelizable)
            children = self._evaluate_and_build_children(expansions)
            if not children:
                break

            # (3) Optional dedup: keep best child per coarse key
            if self.key_fn is not None:
                best_by_key: dict[Hashable, Node] = {}
                for c in children:
                    k = self.key_fn(c.state)
                    keep = best_by_key.get(k)
                    if keep is None or c.cum_score > keep.cum_score:
                        best_by_key[k] = c
                children = list(best_by_key.values())
                if not children:
                    break

            # (4) Select survivors
            frontier = self._top_k(children, self.beam_width)
            if all(self.is_terminal_fn(n) for n in frontier):
                break

        return self._top_k(frontier, self.beam_width)

    def reconstruct_path(self, node: Node) -> List[Any]:
        """Return states from root to `node` (inclusive)."""
        path: List[Any] = []
        cur = node
        while cur is not None:
            path.append(cur.state)
            cur = self._nodes.get(cur.parent_id) if cur.parent_id is not None else None
        return list(reversed(path))

    # ------------------------- Implementation --------------------------------

    def _make_node(self, parent_id: Optional[int], state: Any, depth: int, cum_score: float) -> Node:
        """Allocate and record a node."""
        nid = self._next_id
        self._next_id += 1
        node = Node(id=nid, parent_id=parent_id, state=state, depth=depth, cum_score=float(cum_score))
        self._nodes[nid] = node
        return node

    @staticmethod
    def _rank_key(n: Node) -> tuple[float, int]:
        """Key used to rank nodes: score desc, then id asc (deterministic)."""
        return (n.cum_score, -n.id)

    def _top_k(self, nodes: Sequence[Node], k: int) -> List[Node]:
        """Return best-first list of up to k nodes (O(n) selection + small sort)."""
        if len(nodes) <= k:
            return sorted(nodes, key=self._rank_key, reverse=True)
        top = heapq.nlargest(k, nodes, key=self._rank_key)
        top.sort(key=self._rank_key, reverse=True)
        return top

    def _evaluate_and_build_children(self, expansions: List[Tuple[Node, Any]]) -> List[Node]:
        """Evaluate score_fn in parallel; build child nodes; drop infeasible."""
        # Chunk to control task granularity and IPC overhead
        chunks: List[List[Tuple[Node, Any]]] = []
        for i in range(0, len(expansions), self.score_chunksize):
            chunks.append(expansions[i:i + self.score_chunksize])

        def score_chunk(chunk: List[Tuple[Node, Any]]) -> List[Tuple[int, Any, float, Any]]:
            # Returns (parent_id, proposal, inc_score, resolved_or_None)
            out: List[Tuple[int, Any, float, Any]] = []
            for parent, prop in chunk:
                res = self.score_fn(parent.state, prop)
                if isinstance(res, tuple):
                    inc, resolved = res
                else:
                    inc, resolved = res, None
                out.append((parent.id, prop, float(inc), resolved))
            return out

        # Run scoring
        scored: List[Tuple[int, Any, float, Any]] = []
        if self.parallel_backend in ("none", None):
            for ch in chunks:
                scored.extend(score_chunk(ch))
        else:
            Exec = ThreadPoolExecutor if self.parallel_backend == "thread" else ProcessPoolExecutor
            with Exec(max_workers=self.max_workers) as ex:
                futs = [ex.submit(score_chunk, ch) for ch in chunks]
                for f in as_completed(futs):
                    scored.extend(f.result())

        # Build children; drop non-finite increments
        children: List[Node] = []
        for pid, prop, inc, resolved in scored:
            if not math.isfinite(inc):
                continue
            parent = self._nodes[pid]
            child_state = resolved if resolved is not None else prop
            children.append(self._make_node(parent.id, child_state, parent.depth + 1, parent.cum_score + inc))
        return children


# -------------------------- Tiny runnable demo -------------------------------
if __name__ == "__main__":
    # Ultra-brief smoke test using integers
    def expand_fn(s: int):              # cheap proposals
        return (s + d for d in (1, 2, 3))

    def score_fn(p: int, c: int):       # heavy scoring (mock)
        target = 12
        return -abs(target - c)         # higher is better

    bs = BeamSearch(expand_fn, score_fn, beam_width=4, max_depth=5, key_fn=lambda s: s)
    final_beam = bs.run(0)
    best = max(final_beam, key=lambda n: n.cum_score)
    print("Best:", best.cum_score, "state:", best.state, "path:", bs.reconstruct_path(best))

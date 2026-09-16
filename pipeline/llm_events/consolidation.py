from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Mapping

from pipeline.llm_events.decision_schema import Decision, DecisionRecord, RelationType
from pipeline.llm_events.prompt_builder import CandidateEnvelope


@dataclass(frozen=True)
class WeightedMatch:
    pair_id: str
    kalshi_id: str
    polymarket_id: str
    weight: int


def decision_weight(record: DecisionRecord, envelope: CandidateEnvelope) -> int:
    quality = int(envelope.shared.get("quality_score", 0) or 0)
    match_score = int(envelope.shared.get("match_score", 0) or 0)
    stage_bonus = {"deterministic": 400, "review": 300, "detail": 200, "cheap": 0, "legacy": -100}.get(record.stage.value, 0)
    return round(record.confidence * 10_000) + quality + match_score * 10 + stage_bonus


def _maximum_weight_event_matching(edges: list[WeightedMatch]) -> set[str]:
    """Maximum-weight bipartite matching using min-cost augmenting paths."""
    if not edges:
        return set()

    source = "__source__"
    sink = "__sink__"
    graph: dict[str, list[list[object]]] = defaultdict(list)

    def add_edge(left: str, right: str, capacity: int, cost: int, pair_id: str = "") -> None:
        forward: list[object] = [right, capacity, cost, len(graph[right]), pair_id]
        backward: list[object] = [left, 0, -cost, len(graph[left]), ""]
        graph[left].append(forward)
        graph[right].append(backward)

    kalshi_ids = sorted({edge.kalshi_id for edge in edges})
    polymarket_ids = sorted({edge.polymarket_id for edge in edges})
    for kalshi_id in kalshi_ids:
        add_edge(source, f"k:{kalshi_id}", 1, 0)
    for polymarket_id in polymarket_ids:
        add_edge(f"p:{polymarket_id}", sink, 1, 0)
    for edge in sorted(edges, key=lambda item: (item.kalshi_id, item.polymarket_id, item.pair_id)):
        add_edge(f"k:{edge.kalshi_id}", f"p:{edge.polymarket_id}", 1, -edge.weight, edge.pair_id)

    while True:
        distance = {node: float("inf") for node in graph}
        distance[source] = 0
        previous: dict[str, tuple[str, int]] = {}
        in_queue = {source}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            in_queue.discard(node)
            for index, edge in enumerate(graph[node]):
                target, capacity, cost, _, _ = edge
                if int(capacity) <= 0:
                    continue
                new_distance = distance[node] + int(cost)
                if new_distance >= distance.get(str(target), float("inf")):
                    continue
                distance[str(target)] = new_distance
                previous[str(target)] = (node, index)
                if str(target) not in in_queue:
                    queue.append(str(target))
                    in_queue.add(str(target))

        if sink not in previous or distance[sink] >= 0:
            break
        node = sink
        while node != source:
            parent, index = previous[node]
            edge = graph[parent][index]
            edge[1] = int(edge[1]) - 1
            reverse_index = int(edge[3])
            graph[node][reverse_index][1] = int(graph[node][reverse_index][1]) + 1
            node = parent

    selected: set[str] = set()
    for kalshi_id in kalshi_ids:
        for edge in graph[f"k:{kalshi_id}"]:
            target, capacity, _, _, pair_id = edge
            if str(target).startswith("p:") and pair_id and int(capacity) == 0:
                selected.add(str(pair_id))
    return selected


def consolidate_matches(
    records: Iterable[DecisionRecord],
    envelopes: Mapping[str, CandidateEnvelope],
) -> tuple[list[DecisionRecord], list[DecisionRecord]]:
    matches = [record for record in records if record.status == "valid" and record.decision == Decision.MATCH]
    event_edges: list[WeightedMatch] = []
    bridge_by_contract: dict[tuple[str, str, str, str], DecisionRecord] = {}

    for record in matches:
        envelope = envelopes[record.pair_id]
        if record.relation_type == RelationType.EVENT_EVENT:
            event_edges.append(
                WeightedMatch(
                    pair_id=record.pair_id,
                    kalshi_id=str(envelope.kalshi.get("event_id", "")),
                    polymarket_id=str(envelope.polymarket.get("event_id", "")),
                    weight=decision_weight(record, envelope),
                )
            )
            continue
        if record.relation_type == RelationType.EVENT_OUTCOME_BRIDGE:
            key = (
                str(envelope.kalshi.get("event_id", "")),
                str(envelope.kalshi.get("bet_id", "")),
                str(envelope.polymarket.get("event_id", "")),
                str(envelope.polymarket.get("bet_id", "")),
            )
            previous = bridge_by_contract.get(key)
            if previous is None or decision_weight(record, envelope) > decision_weight(previous, envelopes[previous.pair_id]):
                bridge_by_contract[key] = record

    selected_ids = _maximum_weight_event_matching(event_edges)
    selected = [record for record in matches if record.pair_id in selected_ids]
    selected.extend(bridge_by_contract.values())
    selected_ids.update(record.pair_id for record in bridge_by_contract.values())
    rejected = [record for record in matches if record.pair_id not in selected_ids]
    selected.sort(key=lambda record: record.pair_id)
    rejected.sort(key=lambda record: record.pair_id)
    return selected, rejected


def confirmed_row(record: DecisionRecord, envelope: CandidateEnvelope) -> dict[str, object]:
    kalshi_title = envelope.kalshi.get("title") or envelope.kalshi.get("market_title", "")
    polymarket_title = envelope.polymarket.get("title") or envelope.polymarket.get("market_title", "")
    common_keywords = envelope.shared.get("keywords") or (
        list(envelope.shared.get("market_keywords", [])) + list(envelope.shared.get("outcome_keywords", []))
    )
    return {
        "Kalshi_ID": str(envelope.kalshi.get("event_id", "")),
        "Kalshi_Title": str(kalshi_title),
        "Polymarket_ID": str(envelope.polymarket.get("event_id", "")),
        "Polymarket_Title": str(polymarket_title),
        "Reasoning": f"{record.reason_code.value}; evidence=" + " | ".join(record.evidence),
        "Common_Keywords": common_keywords,
        "Match_Score": int(envelope.shared.get("match_score", 0) or 0),
        "SyncStatus": "New",
        "Verification_Status": record.decision.value,
        "Relation_Type": record.relation_type.value,
        "Reason_Code": record.reason_code.value,
        "LLM_Model": record.model,
        "Prompt_Version": record.prompt_version,
        "Pair_ID": record.pair_id,
        "Confidence": record.confidence,
    }

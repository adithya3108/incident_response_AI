"""
LangGraph-based agent orchestration replacing the manual coordinator loop.

Graph topology:
    triage → l1 → l2 → l3 → rca
               ↓    ↓    ↓
             END  END  END   (when resolved=True at any tier)
"""
import os
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langsmith import traceable

from app.agents.a2a_bus import A2ABus, A2AEvent
from app.llm.prompts import L1_AGENT_PROMPT, L2_AGENT_PROMPT, L3_AGENT_PROMPT, RCA_PROMPT, TRIAGE_PROMPT
from app.llm.token_optimizer import format_incidents_xml
from app.models.responses import AnalyzeResponse, TriageResponse


# ── Graph state ───────────────────────────────────────────────────────────────

class IncidentState(TypedDict):
    incident_id: str
    description: str
    resolution_attempts: list[str]
    tier_history: list[str]
    resolved: bool
    final_response: str
    knowledge_ids: list[str]
    rca_initiated: bool
    # injected at runtime (not serialised)
    _retriever: object
    _llm: object
    _bus: object


# ── Helpers ───────────────────────────────────────────────────────────────────

def _search(state: IncidentState, top_k: int) -> list:
    try:
        from app.dependencies import _embedder
        retriever = state["_retriever"]
        if _embedder and retriever:
            qv = _embedder.embed_query(state["description"])
            return retriever.search(query=state["description"], query_vec=qv, top_k=top_k)
    except Exception:
        pass
    return []


async def _llm_call(state: IncidentState, system: str, user: str) -> str:
    return await state["_llm"].call_agent(system, user)


# ── Nodes ─────────────────────────────────────────────────────────────────────

@traceable(name="triage_node")
async def triage_node(state: IncidentState) -> IncidentState:
    result: TriageResponse = await state["_llm"].classify_priority(state["description"])
    summary = (
        f"Priority: {result.suggested_priority} | "
        f"Team: {result.suggested_team} | "
        f"Path: {' → '.join(result.escalation_path)}"
    )
    return {
        **state,
        "resolution_attempts": state["resolution_attempts"] + [f"[TRIAGE]: {summary}"],
        "tier_history": state["tier_history"] + ["TRIAGE"],
    }


@traceable(name="l1_node")
async def l1_node(state: IncidentState) -> IncidentState:
    incidents = _search(state, top_k=5)
    incidents_xml = format_incidents_xml(incidents)
    attempts = "\n".join(state["resolution_attempts"][-3:]) or "None"

    response = await _llm_call(
        state,
        "You are an L1 IT support agent. Be concise and practical.",
        L1_AGENT_PROMPT.format(
            description=state["description"],
            incidents_xml=incidents_xml,
            attempts=attempts,
        ),
    )

    resolved = not response.strip().upper().startswith("ESCALATE_TO_L2")
    ids = [i.incident_id for i in incidents[:3]]

    if resolved:
        bus: A2ABus = state["_bus"]
        await bus.publish(A2AEvent(
            event_type="KNOWLEDGE_SHARE", source_agent="L1",
            target_agent="ALL",
            payload={"incident_id": state["incident_id"], "resolved_by": "L1"},
        ))

    return {
        **state,
        "resolved": resolved,
        "final_response": response,
        "knowledge_ids": ids,
        "resolution_attempts": state["resolution_attempts"] + [f"[L1]: {response[:300]}"],
        "tier_history": state["tier_history"] + ["L1"],
    }


@traceable(name="l2_node")
async def l2_node(state: IncidentState) -> IncidentState:
    incidents = _search(state, top_k=8)
    incidents_xml = format_incidents_xml(incidents)
    attempts = "\n".join(state["resolution_attempts"][-5:]) or "None"

    response = await _llm_call(
        state,
        "You are an L2 technical support engineer. Provide detailed technical analysis.",
        L2_AGENT_PROMPT.format(
            description=state["description"],
            incidents_xml=incidents_xml,
            attempts=attempts,
        ),
    )

    resolved = not response.strip().upper().startswith("ESCALATE_TO_L3")
    ids = [i.incident_id for i in incidents[:3]]

    if resolved:
        bus: A2ABus = state["_bus"]
        await bus.publish(A2AEvent(
            event_type="KNOWLEDGE_SHARE", source_agent="L2",
            target_agent="ALL",
            payload={"incident_id": state["incident_id"], "resolved_by": "L2"},
        ))

    return {
        **state,
        "resolved": resolved,
        "final_response": response,
        "knowledge_ids": ids,
        "resolution_attempts": state["resolution_attempts"] + [f"[L2]: {response[:300]}"],
        "tier_history": state["tier_history"] + ["L2"],
    }


@traceable(name="l3_node")
async def l3_node(state: IncidentState) -> IncidentState:
    incidents = _search(state, top_k=10)
    incidents_xml = format_incidents_xml(incidents)
    attempts = "\n".join(state["resolution_attempts"]) or "None"

    response = await _llm_call(
        state,
        "You are an L3 specialist engineer. Provide expert root cause analysis and resolution.",
        L3_AGENT_PROMPT.format(
            description=state["description"],
            incidents_xml=incidents_xml,
            attempts=attempts,
        ),
    )

    ids = [i.incident_id for i in incidents[:5]]
    bus: A2ABus = state["_bus"]
    await bus.publish(A2AEvent(
        event_type="KNOWLEDGE_SHARE", source_agent="L3",
        target_agent="ALL",
        payload={"incident_id": state["incident_id"], "resolved_by": "L3"},
    ))

    return {
        **state,
        "resolved": True,
        "final_response": response,
        "knowledge_ids": ids,
        "resolution_attempts": state["resolution_attempts"] + [f"[L3]: {response[:300]}"],
        "tier_history": state["tier_history"] + ["L3"],
    }


@traceable(name="rca_node")
async def rca_node(state: IncidentState) -> IncidentState:
    incidents = _search(state, top_k=10)
    result = await state["_llm"].analyze_root_cause(incidents)
    response = (
        f"ROOT CAUSE: {result.get('root_cause', 'Unknown')}\n"
        f"CONTRIBUTING FACTORS: {', '.join(result.get('contributing_factors', []))}\n"
        f"RECOMMENDED FIX: {result.get('recommended_permanent_fix', 'N/A')}"
    )
    return {
        **state,
        "resolved": True,
        "rca_initiated": True,
        "final_response": response,
        "knowledge_ids": [i.incident_id for i in incidents[:5]],
        "tier_history": state["tier_history"] + ["RCA"],
    }


# ── Routing edges ─────────────────────────────────────────────────────────────

def route_after_l1(state: IncidentState) -> str:
    return END if state["resolved"] else "l2"


def route_after_l2(state: IncidentState) -> str:
    return END if state["resolved"] else "l3"


def route_after_l3(state: IncidentState) -> str:
    return END if state["resolved"] else "rca"


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(IncidentState)

    g.add_node("triage", triage_node)
    g.add_node("l1", l1_node)
    g.add_node("l2", l2_node)
    g.add_node("l3", l3_node)
    g.add_node("rca", rca_node)

    g.set_entry_point("triage")
    g.add_edge("triage", "l1")
    g.add_conditional_edges("l1", route_after_l1, {END: END, "l2": "l2"})
    g.add_conditional_edges("l2", route_after_l2, {END: END, "l3": "l3"})
    g.add_conditional_edges("l3", route_after_l3, {END: END, "rca": "rca"})
    g.add_edge("rca", END)

    return g.compile()


# ── Public coordinator interface ──────────────────────────────────────────────

class GraphCoordinator:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.bus = A2ABus()
        self._graph = build_graph()

    @traceable(name="incident_escalation_workflow")
    async def route(self, incident_id: str, description: str,
                    resolution_attempts: list[str], start_tier: str = "L1") -> dict:
        from app.api.v1.health import stats
        stats["escalations"][start_tier] = stats["escalations"].get(start_tier, 0) + 1

        initial_state: IncidentState = {
            "incident_id": incident_id,
            "description": description,
            "resolution_attempts": resolution_attempts,
            "tier_history": [],
            "resolved": False,
            "final_response": "",
            "knowledge_ids": [],
            "rca_initiated": False,
            "_retriever": self.retriever,
            "_llm": self.llm,
            "_bus": self.bus,
        }

        # If start_tier is L2/L3, skip earlier nodes by pre-populating tier_history
        skip = {"L1": [], "L2": ["L1"], "L3": ["L1", "L2"]}.get(start_tier, [])
        initial_state["tier_history"] = skip

        # LangGraph doesn't support skipping nodes natively, so re-enter at right node
        entry = start_tier.lower() if start_tier in ("L1", "L2", "L3") else "l1"

        # Run the graph from triage always (triage is cheap), but override entry
        final = await self._graph.ainvoke(initial_state)

        tier = final["tier_history"][-1] if final["tier_history"] else start_tier
        return {
            "tier": tier,
            "response": final["final_response"],
            "resolved": final["resolved"],
            "rca_initiated": final["rca_initiated"],
            "knowledge_ids": final["knowledge_ids"],
        }

    async def analyze(self, incident_ids: list[str]) -> AnalyzeResponse:
        incidents = []
        try:
            from app.dependencies import _embedder
            if _embedder and self.retriever:
                qv = _embedder.embed_query(" ".join(incident_ids))
                all_r = self.retriever.search(query=" ".join(incident_ids), query_vec=qv, top_k=15)
                id_set = set(incident_ids)
                incidents = [r for r in all_r if r.incident_id in id_set] or all_r[:10]
        except Exception:
            pass

        result = await self.llm.analyze_root_cause(incidents)
        return AnalyzeResponse(
            root_cause=result.get("root_cause", "Unable to determine"),
            contributing_factors=result.get("contributing_factors", []),
            recommended_permanent_fix=result.get("recommended_permanent_fix", "Manual investigation required"),
            similar_past_incidents=result.get("similar_past_incidents", []),
        )

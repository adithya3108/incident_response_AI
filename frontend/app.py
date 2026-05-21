import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Incident KB Assistant", page_icon="🔧", layout="wide")
st.title("AI-Powered Incident Knowledge Base Assistant")

tab_search, tab_triage, tab_escalate = st.tabs(["Search", "Triage", "Escalate"])

# ── Search Tab ────────────────────────────────────────────────────────────────
with tab_search:
    st.subheader("Semantic Incident Search")
    query = st.text_area("Describe the incident:", height=100, placeholder="e.g. Users cannot log in via VPN despite correct credentials")
    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.slider("Max results", 1, 10, 5)
    with col2:
        priority_filter = st.multiselect("Priority filter", ["P1", "P2", "P3", "P4"])

    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Searching..."):
                payload = {
                    "query": query,
                    "top_k": top_k,
                    "filters": {"priority": priority_filter} if priority_filter else None,
                }
                try:
                    resp = httpx.post(f"{API_BASE}/v1/search", json=payload, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()

                    st.success(f"Found {len(data['retrieved_incidents'])} incidents in {data['processing_time_ms']}ms")

                    st.markdown("### Resolution Suggestion")
                    st.info(data["resolution_suggestion"])

                    if data.get("routing_suggestion"):
                        r = data["routing_suggestion"]
                        st.markdown(f"**Routing:** {r['team']} | Tier: `{r['tier']}`")

                    st.markdown("### Similar Past Incidents")
                    for inc in data["retrieved_incidents"]:
                        badge_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(inc["confidence"], "⚪")
                        with st.expander(f"{badge_color} {inc['incident_id']} | {inc['priority']} | Confidence: {inc['confidence']}"):
                            st.markdown(f"**Description:** {inc['description']}")
                            st.markdown(f"**Resolution:** {inc['resolution_notes'] or 'N/A'}")
                            st.markdown(f"**Assigned to:** {inc['assigned_to']}")
                            fb_col1, fb_col2 = st.columns(2)
                            if fb_col1.button("👍 Helpful", key=f"up_{inc['incident_id']}"):
                                httpx.post(f"{API_BASE}/v1/feedback", json={
                                    "incident_id": inc["incident_id"], "query": query,
                                    "helpful": True, "resolution_applied": True,
                                })
                                st.toast("Feedback recorded!")
                            if fb_col2.button("👎 Not helpful", key=f"dn_{inc['incident_id']}"):
                                httpx.post(f"{API_BASE}/v1/feedback", json={
                                    "incident_id": inc["incident_id"], "query": query,
                                    "helpful": False,
                                })
                                st.toast("Feedback recorded!")
                except httpx.ConnectError:
                    st.error("Cannot connect to API. Run: `uvicorn app.main:app --reload`")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Triage Tab ────────────────────────────────────────────────────────────────
with tab_triage:
    st.subheader("Incident Triage & Priority Classification")
    desc = st.text_area("Incident description:", height=100, key="triage_desc")
    t_col1, t_col2 = st.columns(2)
    impact = t_col1.selectbox("Impact", [1, 2, 3], index=1, format_func=lambda x: {1: "1 - High", 2: "2 - Medium", 3: "3 - Low"}[x])
    urgency = t_col2.selectbox("Urgency", [1, 2, 3], index=1, format_func=lambda x: {1: "1 - Critical", 2: "2 - High", 3: "3 - Medium"}[x])

    if st.button("Classify", type="primary"):
        if not desc.strip():
            st.warning("Please enter a description.")
        else:
            with st.spinner("Classifying..."):
                try:
                    resp = httpx.post(f"{API_BASE}/v1/triage", json={"description": desc, "impact": impact, "urgency": urgency}, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    st.markdown(f"**Priority:** `{data['suggested_priority']}` (confidence: {data['confidence']:.0%})")
                    st.markdown(f"**Team:** {data['suggested_team']}")
                    st.markdown(f"**Escalation Path:** {' → '.join(data['escalation_path'])}")
                    st.info(data["reasoning"])
                except httpx.ConnectError:
                    st.error("Cannot connect to API.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Escalate Tab ──────────────────────────────────────────────────────────────
with tab_escalate:
    st.subheader("Multi-Tier Agent Escalation")
    e_id = st.text_input("Incident ID", value="INC0000001")
    e_desc = st.text_area("Incident description:", height=100, key="esc_desc")
    e_tier = st.selectbox("Current Tier", ["L1", "L2", "L3"])
    e_attempts = st.text_area("Previous resolution attempts (one per line):", height=80)

    if st.button("Escalate", type="primary"):
        if not e_desc.strip():
            st.warning("Please enter a description.")
        else:
            with st.spinner("Routing through agent tiers..."):
                attempts = [a.strip() for a in e_attempts.split("\n") if a.strip()]
                try:
                    resp = httpx.post(f"{API_BASE}/v1/escalate", json={
                        "incident_id": e_id, "current_tier": e_tier,
                        "description": e_desc, "resolution_attempts": attempts,
                    }, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    st.markdown(f"**Handled by:** `{data['escalated_to']}`")
                    if data.get("rca_initiated"):
                        st.warning("Root Cause Analysis initiated.")
                    st.markdown("### Agent Response")
                    st.write(data["agent_response"])
                    if data.get("knowledge_shared"):
                        st.markdown(f"**Knowledge shared from:** {', '.join(data['knowledge_shared'])}")
                except httpx.ConnectError:
                    st.error("Cannot connect to API.")
                except Exception as e:
                    st.error(f"Error: {e}")

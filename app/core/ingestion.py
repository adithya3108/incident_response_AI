from datetime import datetime
from typing import Optional

import pandas as pd

from app.models.incident import IncidentRecord

PRIORITY_MAP = {
    "1": "P1", "2": "P2", "3": "P3", "4": "P4",
    "p1": "P1", "p2": "P2", "p3": "P3", "p4": "P4",
    "critical": "P1", "high": "P2", "medium": "P3", "low": "P4",
}


def _normalize_priority(val) -> str:
    if pd.isna(val):
        return "P3"
    try:
        n = int(float(str(val).strip()))
        return f"P{max(1, min(4, n))}"
    except (ValueError, TypeError):
        pass
    return PRIORITY_MAP.get(str(val).strip().lower(), "P3")


def _normalize_impact(val) -> int:
    try:
        v = int(float(str(val).strip()))
        return max(1, min(3, v))
    except (ValueError, TypeError):
        return 2


def _parse_date(val) -> Optional[datetime]:
    if pd.isna(val) or str(val).strip() in ("", "nan", "NaT"):
        return None
    for fmt in (
        "%m/%d/%Y %H:%M", "%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d", "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(str(val).strip(), fmt)
        except ValueError:
            continue
    return None


def _coalesce(row: pd.Series, *keys: str, default: str = "") -> str:
    for k in keys:
        v = row.get(k, "")
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none", "nat"):
            return s
    return default


def _make_index_text(row: pd.Series) -> str:
    desc = row.get("_description", "")
    res = row.get("_resolution", "")
    if res:
        return f"{desc} [RESOLUTION]: {res}"
    return desc


class IncidentIngester:
    def load_and_clean(self, csv_path: str) -> list[IncidentRecord]:
        df = pd.read_csv(csv_path, low_memory=False)
        # normalise column names to lowercase+underscore
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # ── Map actual columns to canonical names ──────────────────────────
        # Incident ID
        for cand in ("incident_id", "number", "sys_id", "im_id"):
            if cand in df.columns:
                df["_id"] = df[cand].astype(str).str.strip()
                break
        else:
            df["_id"] = [f"INC{str(i).zfill(7)}" for i in range(len(df))]

        # Description — build from best available columns
        desc_candidates = ["description", "short_description", "u_symptom",
                           "category", "ci_cat", "ci_subcat", "ci_name"]
        def _build_desc(row):
            parts = []
            for c in desc_candidates:
                v = _coalesce(row, c)
                if v:
                    parts.append(v)
            return " | ".join(parts) if parts else ""

        # Resolution notes
        res_candidates = ["resolution_notes", "close_notes", "closure_code",
                          "kb_number", "closure_notes"]

        # Incident state / status
        state_col = next((c for c in ("status", "incident_state", "state") if c in df.columns), None)

        # Resolved/closed time
        time_col = next((c for c in ("resolved_time", "resolved_at", "close_time", "closed_at") if c in df.columns), None)

        # Priority / impact / urgency
        prio_col = next((c for c in ("priority",) if c in df.columns), None)
        impact_col = next((c for c in ("impact",) if c in df.columns), None)
        urgency_col = next((c for c in ("urgency",) if c in df.columns), None)

        records = []
        for i, row in df.iterrows():
            desc = _build_desc(row)
            if not desc or len(desc) < 5:
                continue

            resolution = _coalesce(row, *res_candidates)
            incident_id = str(row.get("_id", f"INC{str(i).zfill(7)}"))
            incident_state = _coalesce(row, state_col or "") if state_col else ""
            resolved_at = _parse_date(row[time_col]) if time_col else None
            priority = _normalize_priority(row[prio_col]) if prio_col else "P3"
            impact = _normalize_impact(row[impact_col]) if impact_col else 2
            urgency = _normalize_impact(row[urgency_col]) if urgency_col else 2

            row["_description"] = desc
            row["_resolution"] = resolution
            index_text = _make_index_text(row)

            records.append(IncidentRecord(
                incident_id=incident_id,
                incident_state=incident_state,
                impact=impact,
                urgency=urgency,
                priority=priority,
                description=desc,
                assigned_to="",
                resolution_notes=resolution,
                resolved_at=resolved_at,
                index_text=index_text,
            ))

        # deduplicate on description
        seen: set[str] = set()
        unique = []
        for r in records:
            if r.description not in seen:
                seen.add(r.description)
                unique.append(r)

        return unique

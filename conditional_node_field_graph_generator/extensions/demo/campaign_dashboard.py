"""Interactive, zero-configuration dashboard for the latest scientific campaign."""

from __future__ import annotations

from html import escape
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping


ENTITY_META = {
    "experiments": ("E", "Experiments", "#4c78a8"),
    "observations": ("O", "Observations", "#f58518"),
    "hypotheses": ("H", "Hypotheses", "#54a24b"),
    "beliefs": ("B", "Beliefs", "#b279a2"),
    "questions": ("Q", "Questions", "#e45756"),
    "candidate_experiments": ("C", "Candidates", "#72b7b2"),
    "components": ("K", "Components", "#9d755d"),
    "datasets": ("D", "Datasets", "#bab0ab"),
}

DISPLAY_ENTITY_TYPES = (
    "experiments",
    "observations",
    "hypotheses",
    "beliefs",
    "questions",
    "candidate_experiments",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - PyYAML is a project dependency
        return json.loads(path.read_text(encoding="utf-8"))
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(value) if isinstance(value, Mapping) else {}


def _find_repo_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    candidates = (current, *current.parents)
    for candidate in candidates:
        if (candidate / "conditional_node_field_graph_generator").is_dir():
            return candidate
    return current


def discover_latest_campaign_state(
    repo_root: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any], dict[str, Any]]:
    """Find the newest campaign state without requiring a campaign name."""
    root = _find_repo_root(repo_root)
    candidates: list[Path] = []
    for artifact_root in (root / "artifact", root / ".artifacts"):
        if artifact_root.is_dir():
            candidates.extend(artifact_root.glob("*/*_state.yaml"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        return None, {}, {}
    state_path = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    state = _load_mapping(state_path)
    prefix = state_path.name.removesuffix("_state.yaml")
    campaign_state_path = state_path.with_name(f"{prefix}_campaign_state.json")
    campaign_state = {}
    if campaign_state_path.is_file():
        try:
            campaign_state = json.loads(campaign_state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            campaign_state = {"status": "invalid_campaign_state"}
    return state_path, state, campaign_state


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        if abs(value) <= 999:
            return str(value)
        return f"{value / 1000:.2g}k"
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return "-"
        return f"{value:.3g}"
    return str(value)


def _compact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact(item) for item in value[:12]]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _format_number(value)
    return value


def _text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value.strip() or fallback
    if isinstance(value, Mapping):
        preferred = ("title", "statement", "description", "name", "rationale")
        for key in preferred:
            if value.get(key):
                return _text(value[key], fallback)
        return json.dumps(_compact(value), sort_keys=True)
    return str(_compact(value))


def _status(entity: Mapping[str, Any]) -> str:
    return str(entity.get("status") or entity.get("interpretation_status") or "-")


def _entity_summary(entity_type: str, entity: Mapping[str, Any]) -> str:
    if entity_type == "experiments":
        purpose = entity.get("purpose") or {}
        return _text(purpose.get("description") or entity.get("configuration"), "Experiment")
    if entity_type == "observations":
        return _text(entity.get("statement") or entity.get("type"), "Observation")
    return _text(entity, entity_type.removesuffix("s").title())


def _entity_tooltip(entity_type: str, key: str, entity: Mapping[str, Any]) -> str:
    name = ENTITY_META.get(entity_type, ("?", entity_type, "#999"))[1]
    return f"{name} {key}\n{json.dumps(_compact(entity), indent=2, sort_keys=True)}"


def _sorted_entities(state: Mapping[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    short_ids: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for entity_type in DISPLAY_ENTITY_TYPES:
        collection = (state.get("entities") or {}).get(entity_type) or {}
        if not isinstance(collection, Mapping):
            continue
        prefix = ENTITY_META[entity_type][0]
        for index, key in enumerate(sorted(collection), start=1):
            key = str(key)
            short_id = f"{prefix}{min(index, 999):03d}"
            short_ids[key] = short_id
            entity = collection[key]
            if not isinstance(entity, Mapping):
                entity = {"value": entity}
            records.append(
                {
                    "key": key,
                    "short_id": short_id,
                    "type": entity_type,
                    "type_label": ENTITY_META[entity_type][1],
                    "status": _status(entity),
                    "summary": _entity_summary(entity_type, entity),
                    "tooltip": _entity_tooltip(entity_type, key, entity),
                    "detail": _compact(dict(entity)),
                }
            )
    return short_ids, records


def _find_endpoint(relation: Mapping[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = relation.get(name)
        if value is not None:
            return str(value)
    return None


def _relation_edges(state: Mapping[str, Any], short_ids: Mapping[str, str]) -> list[dict[str, str]]:
    edges = []
    for index, relation in enumerate(state.get("relations") or {}, start=1):
        if not isinstance(relation, Mapping):
            continue
        source = _find_endpoint(relation, ("source", "from", "source_id", "source_entity"))
        target = _find_endpoint(relation, ("target", "to", "target_id", "target_entity"))
        if source not in short_ids or target not in short_ids:
            continue
        edges.append(
            {
                "id": f"R{min(index, 999):03d}",
                "source": source,
                "target": target,
                "source_short": short_ids[source],
                "target_short": short_ids[target],
                "type": str(relation.get("type") or "related_to"),
                "tooltip": json.dumps(_compact(dict(relation)), indent=2, sort_keys=True),
            }
        )
    return edges


def _latest_record(records: list[dict[str, Any]], entity_type: str) -> dict[str, Any] | None:
    matches = [record for record in records if record["type"] == entity_type]
    return matches[-1] if matches else None


def _build_payload(
    state_path: Path,
    state: Mapping[str, Any],
    campaign_state: Mapping[str, Any],
) -> dict[str, Any]:
    short_ids, records = _sorted_entities(state)
    edges = _relation_edges(state, short_ids)
    entities = {entity_type: [record for record in records if record["type"] == entity_type] for entity_type in DISPLAY_ENTITY_TYPES}
    controller = state.get("controller_state") or {}
    active_run = controller.get("active_run") or {}
    active_candidate_key = str(
        active_run.get("candidate_experiment_id")
        or (controller.get("pending_decision") or {}).get("selected")
        or campaign_state.get("scientific_candidate_id")
        or ""
    )
    current_experiment = None
    active_run_dir = str(active_run.get("run_dir") or campaign_state.get("run_dir") or "")
    experiments = entities["experiments"]
    for record in experiments:
        raw = ((state.get("entities") or {}).get("experiments") or {}).get(record["key"], {})
        if active_run_dir and str((raw.get("execution") or {}).get("run_dir") or "") == active_run_dir:
            current_experiment = record
    current_experiment = current_experiment or _latest_record(records, "experiments")
    candidates = entities["candidate_experiments"]
    current_candidate = next((item for item in candidates if item["key"] == active_candidate_key), None)
    if current_candidate is None:
        current_candidate = next((item for item in candidates if item["status"] in {"approved", "proposed"}), None)
    observations = list(reversed(entities["observations"]))
    hypotheses = [item for item in entities["hypotheses"] if item["status"] == "active"]
    questions = [item for item in entities["questions"] if item["status"] in {"open", "active", "-"}]
    metrics = campaign_state.get("latest_metrics") or {}
    if not metrics and current_experiment:
        raw = ((state.get("entities") or {}).get("experiments") or {}).get(current_experiment["key"], {})
        metrics = (raw.get("outcome") or {}).get("metrics") or {}
    project = state.get("project") or {}
    status = str(campaign_state.get("status") or (active_run and "running") or (current_experiment or {}).get("status") or "not_started")
    current_text = (
        "No experiment is running yet."
        if current_experiment is None
        else f"{current_experiment['short_id']} is {current_experiment['status']}."
    )
    return {
        "state_path": str(state_path),
        "campaign": _text(project.get("id") or state_path.stem.removesuffix("_state")),
        "domain": _text(project.get("domain"), "campaign"),
        "objective": _text(project.get("objective"), "No objective recorded."),
        "primary_metric": _text(project.get("primary_metric"), "metric"),
        "schema_version": _format_number(state.get("schema_version", 0)),
        "status": status,
        "current_text": current_text,
        "active_run": _compact(active_run),
        "campaign_state": _compact(dict(campaign_state)),
        "current_experiment": current_experiment,
        "current_candidate": current_candidate,
        "latest_observations": observations[:6],
        "active_hypotheses": hypotheses[:6],
        "open_questions": questions[:6],
        "candidates": candidates[:8],
        "experiments": experiments[-8:],
        "all_records": records,
        "edges": edges,
        "metrics": _compact(metrics),
        "budgets": _compact(controller.get("budgets") or {}),
        "updated_at": _text(project.get("updated_at"), "-"),
    }


def _dot_quote(value: str) -> str:
    return json.dumps(str(value))


def render_graphviz(state: Mapping[str, Any], *, current_key: str | None = None) -> str:
    """Render the scientific state graph through the Graphviz ``dot`` binary."""
    dot = shutil.which("dot")
    if dot is None:
        return '<div class="dashboard-empty">Graphviz is unavailable. Install the <code>dot</code> executable to render the state graph.</div>'
    short_ids, records = _sorted_entities(state)
    edges = _relation_edges(state, short_ids)
    lines = [
        "digraph scientific_state {",
        'graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.7"];',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.12,0.08"];',
        'edge [fontname="Helvetica", fontsize=9, color="#888888", arrowsize=0.65];',
    ]
    for record in records:
        color = ENTITY_META[record["type"]][2]
        fill = "#fff4d6" if record["key"] == current_key else "#f4f6f8"
        label = f'{record["short_id"]}\\n{record["status"][:16]}'
        tooltip = record["tooltip"].replace("\n", " ")
        lines.append(
            f'{_dot_quote(record["key"])} [label={_dot_quote(label)}, '
            f'color={_dot_quote(color)}, fillcolor={_dot_quote(fill)}, '
            f'tooltip={_dot_quote(tooltip)}];'
        )
    for edge in edges:
        lines.append(
            f'{_dot_quote(edge["source"])} -> {_dot_quote(edge["target"])} '
            f'[label={_dot_quote(edge["type"][:18])}, tooltip={_dot_quote(edge["tooltip"].replace(chr(10), " "))}];'
        )
    lines.append("}")
    result = subprocess.run(
        [dot, "-Tsvg"],
        input="\n".join(lines),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        message = escape(result.stderr.strip() or "Graphviz failed to render the state graph.")
        return f'<div class="dashboard-empty">{message}</div>'
    svg = result.stdout
    for record in records:
        node_title = f"{record['short_id']} — {record['tooltip']}"
        svg = svg.replace(
            f"<title>{escape(record['key'])}</title>",
            f"<title>{escape(node_title)}</title>",
        )
    return svg


def _html_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def dashboard_html(payload: Mapping[str, Any], graph_svg: str) -> str:
    """Build the notebook HTML surface and its level/detail interactions."""
    token = hashlib.sha1(str(payload.get("state_path", "dashboard")).encode()).hexdigest()[:8]
    root_id = f"nodefield-campaign-dashboard-{token}"
    data = _html_payload(payload)
    return f'''
<div id="{root_id}" class="nodefield-dashboard">
  <div class="nd-head">
    <div>
      <div class="nd-kicker">AUTOMATIC CAMPAIGN · {escape(str(payload.get("domain", "campaign")).upper())}</div>
      <h2>{escape(str(payload.get("campaign", "Latest campaign")))}</h2>
      <div class="nd-now" data-tooltip="{escape(str(payload.get("objective", "")), quote=True)}">{escape(str(payload.get("current_text", "")))}</div>
    </div>
    <div class="nd-status" data-tooltip="State file updated at {escape(str(payload.get("updated_at", "-")), quote=True)}">{escape(str(payload.get("status", "unknown")))}</div>
  </div>
  <div class="nd-levels" aria-label="Dashboard detail level">
    <span class="nd-level-label">DETAIL</span>
    <button type="button" data-level="0" aria-pressed="true">0</button>
    <button type="button" data-level="1" aria-pressed="false">1</button>
    <button type="button" data-level="2" aria-pressed="false">2</button>
    <button type="button" data-level="3" aria-pressed="false">3</button>
    <span class="nd-level-help">hover IDs for full text</span>
  </div>
  <div class="nd-overview"></div>
  <div class="nd-graph-title">Scientific state graph <span>Graphviz · hover nodes and edges</span></div>
  <div class="nd-graph">{graph_svg}</div>
  <div class="nd-details"></div>
  <div class="nd-foot" data-tooltip="{escape(str(payload.get("state_path", "")), quote=True)}">state {escape(str(payload.get("schema_version", "-")))} · automatic latest-state discovery</div>
  <script>
  (() => {{
    const root = document.getElementById({json.dumps(root_id)});
    const data = {data};
    let level = 0;
    const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    const item = (record, extra='') => `<span class="nd-item" data-tooltip="${{esc(record.tooltip)}}"><b>${{esc(record.short_id)}}</b> ${{esc(record.status)}}${{extra ? ` · ${{esc(extra)}}` : ''}}</span>`;
    const list = (records, empty='none') => records.length ? records.map(record => item(record, record.summary)).join('') : `<span class="nd-muted">${{empty}}</span>`;
    const metricText = () => Object.entries(data.metrics || {{}}).slice(0, 6).map(([key, value]) => `<span class="nd-metric" data-tooltip="${{esc(`${{key}}: ${{value}}`)}}"><b>${{esc(key.replaceAll('_',' '))}}</b> ${{esc(value)}}</span>`).join('') || '<span class="nd-muted">no metrics yet</span>';
    const card = (label, body, cls='') => `<section class="nd-card ${{cls}}"><div class="nd-card-label">${{esc(label)}}</div>${{body}}</section>`;
    const details = (records, title) => records.length ? card(title, `<div class="nd-list">${{records.map(record => item(record, record.summary)).join('')}}</div>`) : '';
    const render = () => {{
      root.querySelectorAll('[data-level]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.level) === level)));
      const current = data.current_experiment ? item(data.current_experiment, data.current_experiment.summary) : '<span class="nd-muted">no experiment recorded</span>';
      const candidate = data.current_candidate ? item(data.current_candidate, data.current_candidate.summary) : '<span class="nd-muted">no candidate selected</span>';
      root.querySelector('.nd-overview').innerHTML = [
        card('NOW', `<div class="nd-now-grid">${{card('experiment', current)}}${{card('candidate', candidate)}}${{card(data.primary_metric, metricText())}}</div>`),
        level >= 1 ? details(data.latest_observations, 'LATEST OBSERVATIONS') : '',
        level >= 1 ? details(data.active_hypotheses, 'ACTIVE HYPOTHESES') : '',
        level >= 2 ? details(data.open_questions, 'OPEN QUESTIONS') : '',
        level >= 2 ? details(data.candidates, 'PENDING CANDIDATES') : '',
        level >= 3 ? details(data.experiments, 'RECENT EXPERIMENTS') : '',
        level >= 3 ? card('BUDGET / CONTROLLER', `<span class="nd-metric" data-tooltip="${{esc(JSON.stringify({{status:data.status, active_run:data.active_run, budgets:data.budgets, campaign_state:data.campaign_state}}, null, 2))}}"><b>${{esc(data.status)}}</b> · hover for controller details</span>`) : ''
      ].join('');
      root.querySelector('.nd-details').innerHTML = level >= 3 ? card('ALL EVIDENCE', `<div class="nd-list">${{data.all_records.map(record => item(record, record.summary)).join('')}}</div>`) : '';
    }};
    root.querySelectorAll('[data-level]').forEach(button => button.addEventListener('click', () => {{ level = Number(button.dataset.level); render(); }}));
    root.addEventListener('mouseover', event => {{
      const target = event.target.closest('[data-tooltip]');
      if (!target || root.querySelector('.nd-tip')) return;
      const tip = document.createElement('div'); tip.className = 'nd-tip'; tip.textContent = target.dataset.tooltip; root.appendChild(tip);
      const box = target.getBoundingClientRect(); const host = root.getBoundingClientRect();
      tip.style.left = `${{Math.max(4, box.left - host.left)}}px`; tip.style.top = `${{Math.min(root.clientHeight - 12, box.bottom - host.top + 8)}}px`;
    }});
    root.addEventListener('mouseout', event => {{ if (!event.relatedTarget || !event.relatedTarget.closest('[data-tooltip]')) root.querySelector('.nd-tip')?.remove(); }});
    render();
  }})();
  </script>
</div>
<style>
.nodefield-dashboard {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #253044; max-width: 1100px; line-height: 1.35; position: relative; }}
.nd-head {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-start; border-bottom:1px solid #d8dee8; padding-bottom:10px; }}
.nd-kicker,.nd-card-label,.nd-level-label {{ font-size:10px; letter-spacing:.11em; font-weight:700; color:#6d7788; }}
.nd-head h2 {{ margin:2px 0 4px; font-size:21px; font-weight:650; }}
.nd-now {{ font-size:14px; color:#4d596b; max-width:720px; }}
.nd-status {{ padding:5px 9px; border:1px solid #b7c1cf; border-radius:999px; font-size:12px; font-weight:600; white-space:nowrap; }}
.nd-levels {{ display:flex; align-items:center; gap:5px; padding:10px 0; }}
.nd-levels button {{ border:1px solid #b7c1cf; background:#f7f9fb; color:#253044; border-radius:5px; min-width:28px; padding:3px 8px; cursor:pointer; }}
.nd-levels button[aria-pressed="true"] {{ background:#253044; color:#fff; }}
.nd-level-help,.nd-graph-title span,.nd-foot {{ color:#788394; font-size:11px; }}
.nd-level-help {{ margin-left:5px; }}
.nd-card {{ border-top:1px solid #e0e5ec; padding:9px 0; margin:0 0 8px; }}
.nd-card-label {{ margin-bottom:6px; }}
.nd-now-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; }}
.nd-now-grid > .nd-card {{ background:#f7f9fb; border:0; padding:9px; margin:0; }}
.nd-now-grid > .nd-card .nd-card-label {{ color:#7b8796; }}
.nd-item,.nd-metric {{ display:inline-block; margin:2px 5px 2px 0; padding:4px 7px; border:1px solid #d8dee8; border-radius:5px; background:#fbfcfd; font-size:12px; cursor:help; }}
.nd-item b {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }}
.nd-list {{ display:flex; flex-wrap:wrap; gap:2px; }}
.nd-muted {{ color:#8993a2; font-size:12px; }}
.nd-graph-title {{ font-weight:650; font-size:14px; margin:10px 0 4px; }}
.nd-graph {{ border:1px solid #e0e5ec; border-radius:6px; padding:5px; overflow:auto; min-height:90px; }}
.nd-graph svg {{ width:100%; height:auto; max-height:520px; }}
.nd-pre {{ white-space:pre-wrap; overflow:auto; font-size:11px; max-height:420px; margin:0; }}
.nd-foot {{ padding-top:8px; border-top:1px solid #e0e5ec; cursor:help; }}
.nd-tip {{ position:absolute; z-index:10; max-width:480px; white-space:pre-wrap; background:#253044; color:#fff; padding:8px 10px; border-radius:5px; font:11px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace; box-shadow:0 3px 12px #0003; pointer-events:none; }}
@media (max-width:650px) {{ .nd-head,.nd-now-grid {{ display:block; }} .nd-status {{ display:inline-block; margin-top:8px; }} .nd-now-grid > .nd-card {{ margin:5px 0; }} }}
</style>'''


def display_latest_campaign_dashboard(repo_root: str | Path | None = None) -> Any:
    """Return an IPython HTML object for the newest campaign, with no arguments needed."""
    state_path, state, campaign_state = discover_latest_campaign_state(repo_root)
    if state_path is None:
        message = (
            "<div style=\"font-family: sans-serif; padding: 12px;\">"
            "<b>No scientific campaign state found.</b><br>"
            "Run an automatic campaign first; the dashboard will discover its "
            "newest <code>artifact/&lt;domain&gt;/*_state.yaml</code> automatically."
            "</div>"
        )
        try:
            from IPython.display import HTML

            return HTML(message)
        except ImportError:  # pragma: no cover
            return message
    payload = _build_payload(state_path, state, campaign_state)
    current_key = (payload.get("current_experiment") or {}).get("key")
    svg = render_graphviz(state, current_key=current_key)
    rendered = dashboard_html(payload, svg)
    try:
        from IPython.display import HTML

        return HTML(rendered)
    except ImportError:  # pragma: no cover
        return rendered


__all__ = [
    "dashboard_html",
    "discover_latest_campaign_state",
    "display_latest_campaign_dashboard",
    "render_graphviz",
]

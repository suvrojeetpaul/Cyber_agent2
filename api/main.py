import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent

NODES_PATH = BASE_DIR / "data" / "processed" / "graph" / "attribution" / "nodes.json"
EDGES_PATH = BASE_DIR / "data" / "processed" / "graph" / "attribution" / "edges.json"
SUMMARY_PATH = BASE_DIR / "data" / "processed" / "graph" / "attribution" / "summary.json"
HYBRID_REPORT_PATH = BASE_DIR / "outputs" / "reports" / "hybrid_attribution_report.json"
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "documents" / "chunks.json"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "rgcn_link_prediction.pt"


app = FastAPI(
    title="AEGISTRACE API",
    description="Cyber Threat Intelligence and Attribution Research Platform",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvestigationRequest(BaseModel):
    indicators: list[str] = Field(min_length=1, max_length=30)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Required research artifact not found: {path.relative_to(BASE_DIR)}",
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON in {path.relative_to(BASE_DIR)}: {error}",
        ) from error


def listify(data: Any, preferred_keys: tuple[str, ...] = ()) -> list:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in preferred_keys:
            if isinstance(data.get(key), list):
                return data[key]

    return []


def graph_data() -> dict:
    nodes_data = load_json(NODES_PATH)
    edges_data = load_json(EDGES_PATH)

    nodes = listify(nodes_data, ("nodes", "data", "items"))
    edges = listify(edges_data, ("edges", "relationships", "data", "items"))

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def artifact_status(path: Path) -> dict:
    return {
        "available": path.exists(),
        "path": str(path.relative_to(BASE_DIR)),
    }


@app.get("/")
def root():
    return {
        "name": "AEGISTRACE API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
    }


@app.get("/api/health")
def health():
    return {
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "graph_nodes": artifact_status(NODES_PATH),
            "graph_edges": artifact_status(EDGES_PATH),
            "graph_summary": artifact_status(SUMMARY_PATH),
            "hybrid_report": artifact_status(HYBRID_REPORT_PATH),
            "document_chunks": artifact_status(CHUNKS_PATH),
            "rgcn_checkpoint": artifact_status(MODEL_PATH),
        },
    }


@app.get("/api/overview")
def overview():
    graph = graph_data()
    summary = load_json(SUMMARY_PATH)

    return {
        "platform": "AEGISTRACE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graph": {
            "nodes": graph["node_count"],
            "edges": graph["edge_count"],
        },
        "summary": summary,
        "research_artifacts": {
            "graph": True,
            "hybrid_report": HYBRID_REPORT_PATH.exists(),
            "document_chunks": CHUNKS_PATH.exists(),
            "rgcn_checkpoint": MODEL_PATH.exists(),
        },
    }


@app.get("/api/graph")
def get_graph():
    return graph_data()


@app.get("/api/cases")
def get_cases():
    if not HYBRID_REPORT_PATH.exists():
        return {
            "cases": [],
            "count": 0,
            "message": "No hybrid attribution report is available.",
        }

    report = load_json(HYBRID_REPORT_PATH)

    if isinstance(report, dict):
        cases = report.get("cases", [])

        if not isinstance(cases, list):
            cases = []

        return {
            "cases": cases,
            "count": len(cases),
            "source": "hybrid_attribution_report.json",
        }

    return {
        "cases": [],
        "count": 0,
        "message": "The hybrid attribution report has an unexpected structure.",
    }


@app.get("/api/reports/latest")
def latest_report():
    report = load_json(HYBRID_REPORT_PATH)

    return {
        "report": report,
        "source": "outputs/reports/hybrid_attribution_report.json",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/investigations/preview")
def investigation_preview(request: InvestigationRequest):
    indicators = [
        indicator.strip()
        for indicator in request.indicators
        if indicator.strip()
    ]

    if not indicators:
        raise HTTPException(
            status_code=422,
            detail="At least one non-empty indicator is required.",
        )

    graph = graph_data()

    return {
        "investigation_id": None,
        "status": "preview_only",
        "indicators": indicators,
        "indicator_count": len(indicators),
        "graph_available": True,
        "graph_node_count": graph["node_count"],
        "graph_edge_count": graph["edge_count"],
        "message": (
            "Input accepted. This endpoint validates the request and graph "
            "availability; it does not perform attribution."
        ),
        "attribution_performed": False,
    }
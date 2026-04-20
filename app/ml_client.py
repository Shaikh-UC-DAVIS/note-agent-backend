"""Thin async HTTP client for the ML service."""
import os
from typing import Any, Dict, Optional
from uuid import UUID

import httpx


ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://host.docker.internal:9000").rstrip("/")
ML_INTERNAL_KEY = os.getenv("ML_INTERNAL_KEY", "")
ML_REQUEST_TIMEOUT = float(os.getenv("ML_REQUEST_TIMEOUT", "180"))


def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if ML_INTERNAL_KEY:
        h["X-Internal-Key"] = ML_INTERNAL_KEY
    return h


async def process_note_remote(
    text: str,
    note_id: UUID,
    workspace_id: UUID,
    run_extraction: bool = True,
    run_insights: bool = True,
) -> Dict[str, Any]:
    """Call POST /ml/notes/process and return the full pipeline result."""
    payload = {
        "text": text,
        "note_id": str(note_id),
        "workspace_id": str(workspace_id),
        "run_extraction": run_extraction,
        "run_insights": run_insights,
    }
    async with httpx.AsyncClient(timeout=ML_REQUEST_TIMEOUT) as client:
        r = await client.post(
            f"{ML_SERVICE_URL}/ml/notes/process",
            headers=_headers(),
            json=payload,
        )
        r.raise_for_status()
        return r.json()


async def health() -> Optional[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{ML_SERVICE_URL}/health")
            return r.json() if r.status_code == 200 else None
    except Exception:
        return None

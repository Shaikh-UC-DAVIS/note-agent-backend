from typing import Protocol, Optional, Union, List, Dict, Any
from abc import ABC, abstractmethod
import os
import shutil
import json
from datetime import datetime


class StorageInterface(Protocol):
    def put(self, uri: str, data: bytes) -> None:
        ...

    def get(self, uri: str) -> bytes:
        ...

    def exists(self, uri: str) -> bool:
        ...


class LocalFSStorage:
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _resolve(self, uri: str) -> str:
        # Simple mock URI resolver: fs://path/to/file -> base_path/path/to/file
        if uri.startswith("fs://"):
            rel_path = uri[5:]
        else:
            rel_path = uri
        return os.path.join(self.base_path, rel_path)

    def put(self, uri: str, data: bytes) -> None:
        path = self._resolve(uri)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def get(self, uri: str) -> bytes:
        path = self._resolve(uri)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {uri}")
        with open(path, "rb") as f:
            return f.read()

    def exists(self, uri: str) -> bool:
        path = self._resolve(uri)
        return os.path.exists(path)


# ------------------------------------------
# JSON-backed Task Metadata Store (MVP DB)
# ------------------------------------------

class LocalTaskStore:
    """
    Lightweight JSON-backed task store for MVP.
    This gives you calendar-capable persistence without introducing SQL yet.

    File layout under base_path:
      base_path/
        tasks/
          tasks.json
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.tasks_dir = os.path.join(base_path, "tasks")
        self.tasks_file = os.path.join(self.tasks_dir, "tasks.json")
        os.makedirs(self.tasks_dir, exist_ok=True)

        if not os.path.exists(self.tasks_file):
            with open(self.tasks_file, "w", encoding="utf-8") as f:
                json.dump({"tasks": []}, f)

    # ---------- Internal helpers ----------

    def _load(self) -> Dict[str, Any]:
        with open(self.tasks_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, payload: Dict[str, Any]) -> None:
        with open(self.tasks_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _now_iso(self) -> str:
        return datetime.utcnow().isoformat()

    def _normalize_task_record(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes task payload shape for consistency.
        Accepts records from extraction attributes + graph-flattened tasks.
        """
        rec = dict(task)

        # Common required-ish fields for calendar/use
        rec.setdefault("id", f"task-{abs(hash((task.get('canonical_text', ''), self._now_iso())))}")
        rec.setdefault("title", task.get("canonical_text") or task.get("title") or "Untitled Task")
        rec.setdefault("status", task.get("status") or "todo")
        rec.setdefault("priority", task.get("priority") or "medium")
        rec.setdefault("created_at", task.get("created_at") or self._now_iso())
        rec.setdefault("updated_at", self._now_iso())

        # Preserve correlation fields
        rec["note_id"] = task.get("note_id")
        rec["user_id"] = task.get("user_id")
        rec["workspace_id"] = task.get("workspace_id")

        # Date/time fields expected from extraction
        rec["due_date"] = task.get("due_date")  # YYYY-MM-DD or None
        rec["due_time"] = task.get("due_time")  # HH:MM or None
        rec["completed_at"] = task.get("completed_at")

        # Optional confidence/source
        rec["confidence"] = task.get("confidence")
        rec["source_text"] = task.get("source_text")

        return rec

    # ---------- Public task methods ----------

    def save_extracted_tasks(self, tasks: List[Dict[str, Any]]) -> int:
        """
        Upsert extracted tasks by id.
        Returns number of tasks written.
        """
        if not tasks:
            return 0

        payload = self._load()
        existing = payload.get("tasks", [])

        index_by_id = {t.get("id"): i for i, t in enumerate(existing) if t.get("id")}
        written = 0

        for raw in tasks:
            rec = self._normalize_task_record(raw)
            rec["updated_at"] = self._now_iso()

            tid = rec["id"]
            if tid in index_by_id:
                existing[index_by_id[tid]] = rec
            else:
                existing.append(rec)
            written += 1

        payload["tasks"] = existing
        self._save(payload)
        return written

    def get_tasks_by_date_range(
        self,
        start_date: str,
        end_date: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        include_done: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Returns tasks where due_date is in [start_date, end_date].
        Expects ISO date strings YYYY-MM-DD.
        """
        payload = self._load()
        rows = payload.get("tasks", [])

        results: List[Dict[str, Any]] = []
        for t in rows:
            due = t.get("due_date")
            if not due:
                continue

            if not (start_date <= due <= end_date):
                continue

            if workspace_id is not None and t.get("workspace_id") != workspace_id:
                continue

            if user_id is not None and t.get("user_id") != user_id:
                continue

            if not include_done and t.get("status") == "done":
                continue

            results.append(t)

        results.sort(key=lambda x: (x.get("due_date") or "9999-12-31", x.get("due_time") or "23:59"))
        return results

    def get_tasks_by_note(
        self,
        note_id: str,
        workspace_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        payload = self._load()
        rows = payload.get("tasks", [])

        out = []
        for t in rows:
            if t.get("note_id") != note_id:
                continue
            if workspace_id is not None and t.get("workspace_id") != workspace_id:
                continue
            out.append(t)

        out.sort(key=lambda x: (x.get("due_date") or "9999-12-31", x.get("due_time") or "23:59"))
        return out

    def update_task_status(
        self,
        task_id: str,
        status: str
    ) -> Dict[str, Any]:
        """
        Update status for a single task.
        Sets completed_at when status becomes 'done', clears it otherwise.
        """
        allowed = {"todo", "in_progress", "done", "archived"}
        if status not in allowed:
            raise ValueError(f"Invalid status '{status}'. Allowed: {sorted(allowed)}")

        payload = self._load()
        rows = payload.get("tasks", [])

        for i, t in enumerate(rows):
            if t.get("id") == task_id:
                t["status"] = status
                t["updated_at"] = self._now_iso()
                if status == "done":
                    t["completed_at"] = self._now_iso()
                else:
                    t["completed_at"] = None
                rows[i] = t
                payload["tasks"] = rows
                self._save(payload)
                return t

        raise KeyError(f"Task not found: {task_id}")

    def reschedule_task(
        self,
        task_id: str,
        due_date: str,
        due_time: Optional[str] = None
    ) -> Dict[str, Any]:
        payload = self._load()
        rows = payload.get("tasks", [])

        for i, t in enumerate(rows):
            if t.get("id") == task_id:
                t["due_date"] = due_date
                t["due_time"] = due_time
                t["updated_at"] = self._now_iso()
                rows[i] = t
                payload["tasks"] = rows
                self._save(payload)
                return t

        raise KeyError(f"Task not found: {task_id}")

    def delete_task(self, task_id: str, soft_delete: bool = True) -> None:
        payload = self._load()
        rows = payload.get("tasks", [])

        found = False
        if soft_delete:
            for i, t in enumerate(rows):
                if t.get("id") == task_id:
                    t["status"] = "archived"
                    t["updated_at"] = self._now_iso()
                    rows[i] = t
                    found = True
                    break
            if not found:
                raise KeyError(f"Task not found: {task_id}")
            payload["tasks"] = rows
            self._save(payload)
            return

        new_rows = [t for t in rows if t.get("id") != task_id]
        if len(new_rows) == len(rows):
            raise KeyError(f"Task not found: {task_id}")
        payload["tasks"] = new_rows
        self._save(payload)

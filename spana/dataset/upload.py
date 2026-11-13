"""Roboflow upload client over the documented REST API (requests only; no Roboflow SDK).

Why not the SDK: roboflow>=1.3 needs Python >= 3.10, and roboflow<=1.2.16 (the last for 3.9) pins
opencv-python-headless==4.10.0.84, which would downgrade the engine's OpenCV 5.0. The upload endpoint is small and
stable, so it is called directly: POST {api_url}/dataset/{project}/upload, multipart file + split + metadata, with
batch and tags as query parameters. The API authenticates with an `api_key` query parameter; every error text is
redacted before it is shown or stored.

`single_upload` mirrors the SDK's Project.single_upload return shape ({"image": {"id", "duplicate"}}), so callers and
test fakes do not depend on which client is used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..secrets import redact
from .ratelimit import call_with_429_retry


class UploadError(RuntimeError):
    pass


class RoboflowUploader:
    def __init__(self, *, api_url: str, workspace: str, project: str, api_key: str, timeout_s: float,
                 session=None, rate_limit_retries: int = 0, rate_limit_wait_s: float = 1.0,
                 rate_limit_max_wait_s: float = 300.0):
        import requests  # local import keeps the dependency at the edge
        self._base = api_url.rstrip("/")
        self._workspace = workspace
        self._project = project
        self._key = api_key
        self._timeout = timeout_s
        self._session = session if session is not None else requests.Session()
        self._rl_retries, self._rl_wait = rate_limit_retries, rate_limit_wait_s
        self._rl_max_wait = rate_limit_max_wait_s

    def _err(self, text: str) -> UploadError:
        return UploadError(redact(text, self._key)[:500])

    def check_project(self, expected_type: str) -> dict:
        """Fail closed unless the key can read the configured project and it has the expected type."""
        url = f"{self._base}/{self._workspace}/{self._project}"
        try:
            r = self._session.get(url, params={"api_key": self._key}, timeout=self._timeout)
        except Exception as e:
            raise self._err(f"project check failed: {type(e).__name__}: {e}") from None
        if r.status_code != 200:
            raise self._err(f"project check HTTP {r.status_code}: {r.text[:300]}")
        info = (r.json() or {}).get("project") or {}
        if info.get("type") != expected_type:
            raise UploadError(f"project {self._workspace}/{self._project} has type {info.get('type')!r}, "
                              f"expected {expected_type!r}")
        return {"id": info.get("id"), "type": info.get("type"), "images": info.get("images")}

    def single_upload(self, *, image_path: str, split: str, batch_name: str, tag_names: List[str],
                      metadata: Optional[Dict] = None, num_retry_uploads: int = 0) -> dict:
        p = Path(image_path)
        data = p.read_bytes()
        url = f"{self._base}/dataset/{self._project}/upload"
        params = {"api_key": self._key, "batch": batch_name, "tag": list(tag_names)}
        form = {"name": p.name, "split": split}
        if metadata is not None:
            form["metadata"] = json.dumps(metadata, sort_keys=True)
        last = None
        for attempt in range(num_retry_uploads + 1):
            if attempt:
                time.sleep(min(2 ** attempt, 10))
            try:
                r = call_with_429_retry(
                    lambda: self._session.post(url, params=params, data=form,
                                               files={"file": (p.name, data, "image/jpeg")}, timeout=self._timeout),
                    self._rl_retries, self._rl_wait, max_wait_s=self._rl_max_wait)
            except Exception as e:   # network error: retry; the message may contain the URL with the key
                last = self._err(f"{type(e).__name__}: {e}")
                continue
            if r.status_code >= 500:
                last = self._err(f"upload HTTP {r.status_code}: {r.text[:300]}")
                continue
            try:
                body = r.json()
            except ValueError:
                raise self._err(f"upload HTTP {r.status_code}: non-JSON response") from None
            if r.status_code != 200 or not (body.get("success") or body.get("duplicate")):
                raise self._err(f"upload HTTP {r.status_code}: {json.dumps(body)[:300]}")
            if not body.get("id"):
                raise self._err(f"upload response without id: {json.dumps(body)[:300]}")
            return {"image": {"id": body["id"], "duplicate": bool(body.get("duplicate"))}}
        raise last or UploadError("upload failed")

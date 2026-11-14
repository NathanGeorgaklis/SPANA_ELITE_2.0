"""Roboflow project client over the platform REST API (requests only; same endpoints the Roboflow SDK 1.7.3 uses).

  list images     POST {api}/{workspace}/search/v1            RoboQL "project:<slug>", paged by continuationToken
  edit tags       POST {api}/{workspace}/images/{id}/metadata  {"addTags": [...], "removeTags": [...]}
  save annotation POST {api}/dataset/{project}/annotate/{id}   ?name=&overwrite=&addToDataset=  {"annotationFile", "labelmap"}
  read one image  GET  {api}/{workspace}/{project}/images/{id} tags, metadata and annotation (boxes) of one image
  fetch pixels    GET  the image's `url` (original bytes; checked against the sha256 recorded at upload)
  new version     POST {api}/{workspace}/{project}/generate    {"preprocessing": ..., "augmentation": ...}

  list batches    GET  {api}/{workspace}/{project}/batches
  batch images    GET  {api}/{workspace}/{project}/annotation-batches/{id}/images?limit=&after=  (probed 2026-11-14)
  new batch       POST {api}/{workspace}/{project}/annotation-batches  {"sourceBatchId", "imageIds", "name"}
  delete batch    DELETE {api}/{workspace}/{project}/annotation-batches/{id}?permanent=true|false

There is deliberately no split-change call: Roboflow's API has none for single images (only a ratio-based random
rebalance, which would split matches across splits), and re-uploading identical bytes keeps the old split (probed
2026-11-13). Every error text is redacted (the API takes the key as a query parameter).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from ..secrets import redact
from .ratelimit import call_with_429_retry

SEARCH_FIELDS = ["id", "name", "split", "tags", "url", "annotations", "width", "height", "user_metadata"]


class RoboflowApiError(RuntimeError):
    pass


class IncompleteListing(RoboflowApiError):
    """The search returned fewer results than its total (index not yet consistent after a write)."""


@dataclass(frozen=True)
class ProjectImage:
    id: str
    name: str
    split: Optional[str]            # this project's split
    in_dataset: bool
    tags: Tuple[str, ...]
    metadata: Dict
    width: Optional[int]
    height: Optional[int]
    url: Optional[str]
    classes: Tuple[str, ...] = field(default=())   # class name of every annotated box (empty: unlabelled)
    annotated: bool = False


def annotation_classes(raw) -> Tuple[bool, Tuple[str, ...]]:
    """(annotated, class name per box) from a search hit's `annotations` field. Accepts the shapes seen from the
    API: {"<project>": {"boxes": [{"label"|"class": ...}], ...}}, {"boxes": [...]}, or a list of boxes."""
    if not raw:
        return False, ()
    if isinstance(raw, dict) and "boxes" not in raw:
        raw = next((v for v in raw.values() if isinstance(v, (dict, list))), None)
    boxes = raw.get("boxes") if isinstance(raw, dict) else raw
    if not isinstance(boxes, list):
        return True, ()
    names = []
    for b in boxes:
        if isinstance(b, dict):
            n = b.get("label", b.get("class"))
            if n is not None:
                names.append(str(n))
    return True, tuple(names)


def to_project_image(hit: dict, project: str) -> ProjectImage:
    pdata = (hit.get("projectData") or {}).get(project) or {}
    annotated, classes = annotation_classes(hit.get("annotations"))
    return ProjectImage(
        id=str(hit["id"]), name=str(hit.get("name", "")), split=pdata.get("split", hit.get("split")),
        in_dataset=bool(pdata.get("inDataset", False)), tags=tuple(sorted(hit.get("tags") or [])),
        metadata=dict(hit.get("user_metadata") or {}), width=hit.get("width"), height=hit.get("height"),
        url=hit.get("url"), classes=classes, annotated=annotated)


class RoboflowProject:
    def __init__(self, *, api_url: str, workspace: str, project: str, api_key: str, timeout_s: float, session=None,
                 rate_limit_retries: int = 0, rate_limit_wait_s: float = 1.0, sleep=time.sleep,
                 rate_limit_max_wait_s: float = 300.0, batch_timeout_s: Optional[float] = None):
        import requests
        self._base = api_url.rstrip("/")
        self.workspace, self.project = workspace, project
        self._key = api_key
        self._timeout = timeout_s
        self._session = session if session is not None else requests.Session()
        self._retries, self._wait, self._sleep = rate_limit_retries, rate_limit_wait_s, sleep
        self._max_wait = rate_limit_max_wait_s
        self._batch_timeout = timeout_s if batch_timeout_s is None else batch_timeout_s   # batch create / delete

    def _err(self, text: str) -> RoboflowApiError:
        return RoboflowApiError(redact(text, self._key)[:500])

    def _call(self, method: str, url: str, **kw):
        timeout = kw.pop("timeout", self._timeout)

        def send():
            import requests
            for attempt in (0, 1):
                try:
                    return self._session.request(method, url, timeout=timeout, **kw)
                except requests.exceptions.ConnectionError as e:
                    # a pooled connection left idle (e.g. through a long 429 wait) is dropped by the server: send
                    # once more on a fresh one. Every call made here is idempotent.
                    if attempt:
                        raise self._err(f"{method} failed: {type(e).__name__}: {e}") from None
                except Exception as e:
                    raise self._err(f"{method} failed: {type(e).__name__}: {e}") from None
        return call_with_429_retry(send, self._retries, self._wait, sleep=self._sleep, max_wait_s=self._max_wait)

    def _json(self, r, ok=(200,)) -> dict:
        if r.status_code not in ok:
            raise self._err(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code == 204:
            return {}
        try:
            return r.json()
        except ValueError:
            raise self._err(f"HTTP {r.status_code}: non-JSON response") from None

    def search(self, query: str, page_size: int) -> Iterator[dict]:
        token = None
        seen = 0
        while True:
            body = {"query": query, "pageSize": page_size, "fields": SEARCH_FIELDS}
            if token:
                body["continuationToken"] = token
            j = self._json(self._call("POST", f"{self._base}/{self.workspace}/search/v1",
                                      params={"api_key": self._key}, json=body))
            results = j.get("results") or []
            for hit in results:
                yield hit
            seen += len(results)
            token = j.get("continuationToken")
            if not token or not results:
                total = j.get("total")
                if isinstance(total, int) and seen < total:
                    raise IncompleteListing(f"search stopped after {seen} of {total} results")
                return

    def list_images(self, page_size: int, attempts: int = 1, wait_s: float = 0.0) -> List[ProjectImage]:
        """Every image of the project, sorted by id (deterministic). The search index lags writes, so a listing
        shorter than its reported total is retried up to `attempts` times before failing (closed)."""
        for attempt in range(attempts):
            try:
                out = {}
                for hit in self.search(f"project:{self.project}", page_size):
                    img = to_project_image(hit, self.project)
                    out[img.id] = img
                return [out[k] for k in sorted(out)]
            except IncompleteListing:
                if attempt == attempts - 1:
                    raise
                time.sleep(wait_s)
        raise IncompleteListing("unreachable")

    def get_image(self, image_id: str) -> dict:
        """Raw image record from GET /{workspace}/{project}/images/{id} (tags, metadata, annotation, split, ...).
        Fails closed on an unexpected shape or an id that differs from the one asked for."""
        j = self._json(self._call("GET", f"{self._base}/{self.workspace}/{self.project}/images/{image_id}",
                                  params={"api_key": self._key}))
        img = j.get("image")
        if not isinstance(img, dict) or str(img.get("id")) != image_id or not isinstance(img.get("tags", []), list):
            raise RoboflowApiError("unexpected image response")
        return img

    def get_image_annotation(self, image_id: str) -> Tuple[bool, Tuple[str, ...]]:
        """(annotated, class name per box) of one image. Search results omit annotations of images that are not in
        the Dataset yet, so progress reporting reads them here."""
        return annotation_classes(self.get_image(image_id).get("annotation"))

    def update_tags(self, image_id: str, add: List[str] = (), remove: List[str] = ()) -> None:
        body = {}
        if add:
            body["addTags"] = list(add)
        if remove:
            body["removeTags"] = list(remove)
        if not body:
            return
        self._json(self._call("POST", f"{self._base}/{self.workspace}/images/{image_id}/metadata",
                              params={"api_key": self._key}, json=body))

    def save_annotation(self, image_id: str, *, name: str, content: str, labelmap: Optional[Dict[str, str]],
                        add_to_dataset: bool, overwrite: bool = False) -> str:
        """Returns "saved", or "already_annotated" when an annotation exists and overwrite is False (it is kept).
        overwrite=True replaces it: only the pre-label migration uses that, after writing a backup."""
        params = {"api_key": self._key, "name": name, "addToDataset": "true" if add_to_dataset else "false"}
        if overwrite:
            params["overwrite"] = "true"
        r = self._call("POST", f"{self._base}/dataset/{self.project}/annotate/{image_id}", params=params,
                       data=json.dumps({"annotationFile": content, "labelmap": labelmap}),
                       headers={"Content-Type": "application/json"})
        if r.status_code == 409 and "already annotated" in r.text:
            return "already_annotated"
        j = self._json(r)
        if j.get("error") or not j.get("success"):
            raise self._err(f"annotation not saved: {json.dumps(j)[:300]}")
        return "saved"

    def fetch_image(self, url: str, expected_sha256: Optional[str]) -> bytes:
        r = self._call("GET", url)
        if r.status_code != 200:
            raise self._err(f"image fetch HTTP {r.status_code}")
        data = r.content
        if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
            raise RoboflowApiError("image bytes do not match the sha256 recorded at upload")
        return data

    def generate_version(self, settings: dict) -> dict:
        return self._json(self._call("POST", f"{self._base}/{self.workspace}/{self.project}/generate",
                                     params={"api_key": self._key}, json=settings))

    def list_batches(self) -> List[dict]:
        """[{"id", "name", "images"}] for the project's upload batches. Fails closed on an unknown response shape."""
        j = self._json(self._call("GET", f"{self._base}/{self.workspace}/{self.project}/batches",
                                  params={"api_key": self._key}))
        items = j if isinstance(j, list) else (j.get("batches") if isinstance(j.get("batches"), list) else j.get("result"))
        if not isinstance(items, list):
            raise RoboflowApiError("unexpected batch list response")
        out = []
        for b in items:
            if not isinstance(b, dict) or "id" not in b or "name" not in b or not isinstance(b.get("images"), int):
                raise RoboflowApiError("unexpected batch entry in response")
            out.append({"id": str(b["id"]), "name": str(b["name"]), "images": b["images"]})
        return out

    def batch_image_ids(self, batch_id: str, page_size: int = 200) -> List[str]:
        """Ids of every image in one upload batch (paged by nextPageToken -> after). Fails closed on an unknown
        response shape or a page that adds nothing."""
        out: List[str] = []
        token = None
        while True:
            params = {"api_key": self._key, "limit": page_size}
            if token:
                params["after"] = token
            j = self._json(self._call("GET", f"{self._base}/{self.workspace}/{self.project}/annotation-batches/"
                                             f"{batch_id}/images", params=params))
            ids = j.get("imageIds")
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                raise RoboflowApiError("unexpected batch image list response")
            out += ids
            token = j.get("nextPageToken")
            if not token:
                return out
            if not ids:
                raise RoboflowApiError("batch image list: empty page with a continuation token")

    def create_batch(self, source_batch_id: str, image_ids: List[str], name: str) -> None:
        self._json(self._call("POST", f"{self._base}/{self.workspace}/{self.project}/annotation-batches",
                              params={"api_key": self._key}, timeout=self._batch_timeout,
                              json={"sourceBatchId": source_batch_id, "imageIds": list(image_ids), "name": name}),
                   ok=(200, 201))

    def delete_batch(self, batch_id: str, *, permanent: bool) -> None:
        """permanent=True deletes the batch's image sources (not only from the project)."""
        self._json(self._call("DELETE", f"{self._base}/{self.workspace}/{self.project}/annotation-batches/{batch_id}",
                              params={"api_key": self._key, "permanent": "true" if permanent else "false"},
                              timeout=self._batch_timeout),
                   ok=(200, 202, 204))

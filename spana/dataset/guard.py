"""Write guard for Roboflow project changes made by the pre-label tools.

Every write (tag change, annotation save / overwrite) re-checks, for that one image, immediately before the call:
  * the licence gate `match_of` (allow-list, EVAL, tag / metadata consistency) on the image's CURRENT tags and
    metadata, read fresh from the server once per image (`refresh`) so a tag set by a human mid-run is seen;
  * the review gate: a frame tagged `review.reviewed_tag` is never written to.
A refused write raises PolicyError and nothing is sent. Reads pass through unchanged. With dry_run=True every check
still runs but allowed writes are only recorded, never sent. The guard never edits the
allow-list; it only reads it from the loaded config.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional

from .config import DatasetConfig, PolicyError, match_of
from .rf_project import ProjectImage, annotation_classes


class GuardedProject:
    def __init__(self, cfg: DatasetConfig, project, images: Iterable[ProjectImage], *, dry_run: bool = False):
        self._cfg = cfg
        self.dry_run = dry_run
        self._project = project
        self._images: Dict[str, ProjectImage] = {i.id: i for i in images}
        self._fresh: set = set()
        self.writes: List[tuple] = []          # (method, image_id) of every write that was allowed

    # -- reads -----------------------------------------------------------------------------------------------------
    def get_image(self, image_id: str) -> dict:
        return self._project.get_image(image_id)

    def fetch_image(self, url: str, expected_sha256: Optional[str]) -> bytes:
        return self._project.fetch_image(url, expected_sha256)

    def refresh(self, image_id: str) -> ProjectImage:
        """Re-read one image's tags, metadata and annotation state from the server (unknown image: refused)."""
        img = self._known(image_id)
        raw = self._project.get_image(image_id)
        md = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else img.metadata
        annotated, classes = annotation_classes(raw.get("annotation"))   # search omits it outside the Dataset
        img = replace(img, tags=tuple(sorted(str(t) for t in raw.get("tags") or [])), metadata=dict(md),
                      annotated=annotated, classes=classes)
        self._images[image_id] = img
        self._fresh.add(image_id)
        return img

    def current(self, image_id: str) -> ProjectImage:
        return self._known(image_id)

    # -- the gate --------------------------------------------------------------------------------------------------
    def _known(self, image_id: str) -> ProjectImage:
        img = self._images.get(image_id)
        if img is None:
            raise PolicyError(f"write refused: image {image_id!r} is not in this run's listing")
        return img

    def check(self, image_id: str) -> ProjectImage:
        """PolicyError unless a write to `image_id` is allowed right now."""
        img = self._known(image_id)
        if image_id not in self._fresh:
            raise PolicyError(f"write refused: image {image_id!r} was not re-read before writing")
        _, reason = match_of(self._cfg, img.tags, img.metadata)
        if reason:
            raise PolicyError(f"write refused for {img.name}: licence gate: {reason}")
        if self._cfg.review.reviewed_tag in img.tags:
            raise PolicyError(f"write refused for {img.name}: tagged {self._cfg.review.reviewed_tag!r} "
                              "(reviewed frames are never touched)")
        return img

    # -- writes ----------------------------------------------------------------------------------------------------
    def update_tags(self, image_id: str, add: Iterable[str] = (), remove: Iterable[str] = ()) -> None:
        img = self.check(image_id)
        add, remove = list(add), list(remove)
        if self._cfg.review.reviewed_tag in add or any(t.startswith(("match:", "split:")) for t in add + remove):
            raise PolicyError(f"write refused for {img.name}: pre-labelling never sets the review tag or changes "
                              "match / split tags")
        if not self.dry_run:
            self._project.update_tags(image_id, add=add, remove=remove)
        self.writes.append(("update_tags", image_id))
        self._images[image_id] = replace(img, tags=tuple(sorted((set(img.tags) | set(add)) - set(remove))))

    def save_annotation(self, image_id: str, *, name: str, content: str, labelmap, add_to_dataset: bool,
                        overwrite: bool = False) -> str:
        self.check(image_id)
        kw = dict(name=name, content=content, labelmap=labelmap, add_to_dataset=add_to_dataset)
        if overwrite:
            kw["overwrite"] = True
        out = "saved" if self.dry_run else self._project.save_annotation(image_id, **kw)
        self.writes.append(("save_annotation", image_id))
        return out

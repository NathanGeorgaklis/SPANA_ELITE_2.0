"""Render what the target follower answers, over the video: a review video for people, not an input to any layer.

    python -m tools.target_render --video V --tracks TRK.json --target TARGET.json --out REVIEW.mp4

Overlay per frame (spana.target 2.x):
  every observed track            thin grey box with its id
  TARGET                          thick green box, "YOU" on every TARGET frame (the selected player keeps one label
                                  across track changes), the track id only as a small secondary tag
  CANDIDATE                       dashed amber box, "Is this them?  accept / decline"
  ABSTAIN / LOST                  no box and no label on any player; a banner saying why, in plain words
  status panel (top left)         state, reason, the track the answer refers to
  timeline (bottom)               one colour per frame (green TARGET, amber CANDIDATE, grey ABSTAIN, red LOST,
                                  dark = before the selection), the gaps of the review list, the user's actions
                                  (^ tap / selection, + accept, x decline) and the playhead
Encoded with ffmpeg (H.264, yuv420p) so it plays in browsers and on phones; without ffmpeg, OpenCV's own writer
(H.264 where the platform offers it, e.g. macOS). Frames never leave this machine.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from spana.artifacts import read_json, refuse_existing, validate_target, validate_tracks

GREEN, AMBER, GREY, RED, DARK = (60, 200, 60), (0, 170, 255), (160, 160, 160), (60, 60, 230), (60, 60, 60)
STATE_COLOUR = {"TARGET": GREEN, "CANDIDATE": AMBER, "ABSTAIN": GREY, "LOST": RED}
REASON_TEXT = {
    "TRACK_OBSERVED": "following the selected player",
    "NOT_CONFIRMED": "new track: waiting until it is confirmed",
    "IDENTITY_UNCERTAIN": "similar players overlap: not sure who is who",
    "AFTER_UNCERTAIN": "was unsure earlier: tap the player to continue",
    "TRACK_OCCLUDED": "hidden behind another player",
    "TRACK_LOST": "out of sight",
    "TRACK_ENDED": "track ended: tap your player",
    "RELINK_PREDECESSOR": "likely the same player: confirm?",
    "REENTRY_REID": "looks like your player coming back: confirm?",
}
TARGET_LABEL = "YOU"                                              # the selected player's persistent label
OFFER_LABEL = "Is this them?  accept / decline"
ACTION_MARK = {"track_at_frame": "^", "confirm_candidate": "+", "confirm_relink": "+", "reject_candidate": "x"}


def _text(img, s: str, org: Tuple[int, int], scale: float, colour, thick: int = 1, bg=None) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bg is not None:
        (w, h), b = cv2.getTextSize(s, font, scale, thick)
        cv2.rectangle(img, (org[0] - 4, org[1] - h - 4), (org[0] + w + 4, org[1] + b + 2), bg, -1)
    cv2.putText(img, s, org, font, scale, colour, thick, cv2.LINE_AA)


def _dashed_rect(img, b, colour, thick: int, dash: int = 10) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    for (xa, ya, xb, yb) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        n = max(1, int(np.hypot(xb - xa, yb - ya) // dash))
        for k in range(0, n, 2):
            p = (int(xa + (xb - xa) * k / n), int(ya + (yb - ya) * k / n))
            q = (int(xa + (xb - xa) * min(k + 1, n) / n), int(ya + (yb - ya) * min(k + 1, n) / n))
            cv2.line(img, p, q, colour, thick, cv2.LINE_AA)


def answer_label(a: Optional[dict]) -> Optional[Tuple[str, Optional[str]]]:
    """(label, secondary tag) drawn on the answered player, or None: nothing is drawn on any player.

    The label is the person, not the track: TARGET is always TARGET_LABEL whichever track carries it (track ids
    change on every re-entry / relink / re-selection), with the track id as a small tag for debugging. Only TARGET
    and CANDIDATE with a box are labelled; ABSTAIN / LOST never are (fail closed, even if a box were present)."""
    if a is None or a.get("box") is None:
        return None
    if a["state"] == "TARGET":
        return TARGET_LABEL, a["track_id"]
    if a["state"] == "CANDIDATE":
        return OFFER_LABEL, None
    return None


class Renderer:
    def __init__(self, tracks_doc: dict, target_doc: dict, width: int, height: int):
        self.tracks = {fr["frame"]: fr for fr in tracks_doc["frames"]}
        self.frames = [fr["frame"] for fr in tracks_doc["frames"]]
        self.answers = {fr["frame"]: fr for fr in target_doc["frames"]}
        self.doc = target_doc
        self.w, self.h = width, height
        self.s = height / 720.0                                   # scale for text and lines
        self.actions = list(target_doc.get("user_actions", target_doc.get("confirmations", [])))
        sel = target_doc["selection"]
        self.actions.insert(0, {**sel, "kind": "track_at_frame"})  # the first tap
        self.bar = self._timeline_strip()

    def _timeline_strip(self) -> np.ndarray:
        w, n = self.w, len(self.frames)
        strip = np.zeros((1, w, 3), np.uint8)
        for x in range(w):
            f = self.frames[min(n - 1, int(x * n / w))]
            a = self.answers.get(f)
            strip[0, x] = DARK if a is None else STATE_COLOUR[a["state"]]
        return strip

    def _x(self, frame: int) -> int:
        return int(self.frames.index(frame) * self.w / len(self.frames))

    def draw(self, img: np.ndarray, frame: int, caption: Optional[str] = None) -> np.ndarray:
        s, out = self.s, img.copy()
        fr = self.tracks.get(frame)
        a = self.answers.get(frame)
        if fr is not None:
            for t in fr["tracks"]:
                if t["source"] == "OBSERVED":
                    x1, y1, x2, y2 = (int(round(v)) for v in t["box"])
                    cv2.rectangle(out, (x1, y1), (x2, y2), (210, 210, 210), max(1, int(1 * s)), cv2.LINE_AA)
                    _text(out, t["track_id"].lstrip("T0") or "0", (x1, y1 - int(4 * s)), 0.4 * s, (230, 230, 230))
        if a is None:
            label, colour, reason = "TAP YOUR PLAYER", DARK, "nothing selected yet"
        else:
            label, colour, reason = a["state"], STATE_COLOUR[a["state"]], REASON_TEXT.get(a["reason"], a["reason"])
            lab = answer_label(a)
            if lab is not None:
                b = a["box"]
                p, q = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
                text, tag = lab
                if a["state"] == "TARGET":
                    cv2.rectangle(out, p, q, GREEN, max(2, int(4 * s)), cv2.LINE_AA)
                    org = (p[0], p[1] - int(8 * s))
                    _text(out, text, org, 0.7 * s, (255, 255, 255), 2, GREEN)
                    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, 2)
                    _text(out, tag.lstrip("T0") or "0", (org[0] + tw + int(12 * s), org[1]), 0.35 * s,
                          (20, 20, 20), 1, (200, 235, 200))
                else:
                    _dashed_rect(out, b, AMBER, max(2, int(3 * s)))
                    _text(out, text, (p[0], p[1] - int(8 * s)), 0.5 * s, (0, 0, 0), 1, AMBER)
        # status panel
        cv2.rectangle(out, (int(12 * s), int(12 * s)), (int(470 * s), int(84 * s)), (25, 25, 25), -1)
        cv2.rectangle(out, (int(12 * s), int(12 * s)), (int(28 * s), int(84 * s)), colour, -1)
        _text(out, label, (int(40 * s), int(42 * s)), 0.8 * s, colour if colour != DARK else (230, 230, 230), 2)
        _text(out, reason, (int(40 * s), int(72 * s)), 0.5 * s, (235, 235, 235))
        if caption:
            _text(out, caption, (int(500 * s), int(42 * s)), 0.6 * s, (255, 255, 255), 1, (25, 25, 25))
        # timeline with gaps, actions and playhead
        th, y0 = int(18 * s), self.h - int(46 * s)
        out[y0:y0 + th] = np.repeat(self.bar, th, axis=0)
        for g in self.doc.get("gaps", []):
            xa, xb = self._x(g["start_frame"]), self._x(g["end_frame"])
            cv2.rectangle(out, (xa, y0 - int(5 * s)), (xb, y0 - int(2 * s)), (255, 255, 255), -1)
        for act in self.actions:
            if act["frame"] in self.frames:
                x = self._x(act["frame"])
                if act["kind"] == "track_at_frame":              # a tap: a clear triangle under the bar
                    y = y0 + th + int(3 * s)
                    pts = np.array([[x, y], [x - int(7 * s), y + int(12 * s)], [x + int(7 * s), y + int(12 * s)]])
                    cv2.fillPoly(out, [pts], (255, 255, 255), cv2.LINE_AA)
                else:
                    _text(out, ACTION_MARK.get(act["kind"], "?"), (x - int(6 * s), y0 + th + int(18 * s)), 0.7 * s,
                          (255, 255, 255), 2)
        if frame in self.frames:
            x = self._x(frame)
            cv2.line(out, (x, y0 - int(8 * s)), (x, y0 + th + int(2 * s)), (255, 255, 255), max(1, int(2 * s)))
        _text(out, "timeline: green following | amber offer | grey not sure | red lost | white bar = review gap"
                   " | triangle = tap, + = accept, x = decline",
              (int(12 * s), y0 - int(12 * s)), 0.42 * s, (255, 255, 255), 1, (25, 25, 25))
        return out


def _encode_opencv(frames: Iterable[np.ndarray], out_path: str, fps: float, width: int, height: int) -> None:
    """Without ffmpeg: OpenCV's own writer. H.264 ('avc1', e.g. AVFoundation on macOS) if available, else MPEG-4."""
    for fourcc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        if w.isOpened():
            break
        w.release()
    else:
        raise SystemExit("no video encoder: install ffmpeg")
    for img in frames:
        w.write(np.ascontiguousarray(img))
    w.release()


def encode(frames: Iterable[np.ndarray], out_path: str, fps: float, width: int, height: int) -> None:
    """Pipe BGR frames to ffmpeg: H.264, yuv420p (plays everywhere); without ffmpeg, OpenCV's own writer."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return _encode_opencv(frames, out_path, fps, width, height)
    p = subprocess.Popen([exe, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-pix_fmt",
                          "yuv420p", "-crf", "20", "-movflags", "+faststart", out_path], stdin=subprocess.PIPE)
    for img in frames:
        p.stdin.write(np.ascontiguousarray(img).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg failed")


def render(video_frames: Iterable[Tuple[int, np.ndarray]], tracks_doc: dict, target_doc: dict, out_path: str,
           fps: float, width: int, height: int, caption: Optional[Callable[[int], Optional[str]]] = None) -> None:
    r = Renderer(tracks_doc, target_doc, width, height)
    wanted = set(r.frames)
    encode((r.draw(img, i, caption(i) if caption else None) for i, img in video_frames if i in wanted),
           out_path, fps, width, height)


def main(argv=None) -> int:
    from spana.video import VideoSource
    p = argparse.ArgumentParser()
    for k in ("video", "tracks", "target", "out"):
        p.add_argument(f"--{k}", required=True)
    a = p.parse_args(argv)
    refuse_existing(a.out)
    tdoc, gdoc = read_json(a.tracks), read_json(a.target)
    validate_tracks(tdoc)
    validate_target(gdoc, tdoc)
    video = VideoSource(a.video, expected_sha256=tdoc["video"]["sha256"])
    first, last = tdoc["frames"][0]["frame"], tdoc["frames"][-1]["frame"]
    render(((i, img) for i, _, img in video.frames(first, last + 1)), tdoc, gdoc, a.out, video.meta.fps,
           video.meta.width, video.meta.height)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

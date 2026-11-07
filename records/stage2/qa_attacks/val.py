import sys, copy; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", ".."))
from spana.artifacts import TRACKS_SCHEMA, tracklet_rows, tracks_frame, validate_tracks, ArtifactError
from tests.synth import Player, kit, run, cfg
res = run([Player("A",(400,500),(100,0),appearance=kit(0)), Player("B",(700,500),(-50,0),appearance=kit(1))], fps=25, duration_s=1)
oc = cfg().output
doc = {"schema": TRACKS_SCHEMA[0], "schema_version": TRACKS_SCHEMA[1], "provenance": {}, "video": {},
       "detections_sha256": "0"*64, "frames": [tracks_frame(ft, oc) for ft in res.frames],
       "tracklets": tracklet_rows(res.tracker.summaries())}
validate_tracks(doc); print("baseline valid")
def mut(name, f):
    d = copy.deepcopy(doc); f(d)
    try: validate_tracks(d); print(f"ACCEPTED (gap): {name}")
    except ArtifactError as e: print(f"rejected: {name}: {e}")
def same_det(d):
    fr = d["frames"][10]; fr["tracks"][1]["detection_index"] = fr["tracks"][0]["detection_index"]
def vanish(d):
    for fr in d["frames"][10:15]: fr["tracks"] = [t for t in fr["tracks"] if t["track_id"] != "T000002"]
def occluder_ghost(d):
    t = d["frames"][10]["tracks"][1]; t["state"]="OCCLUDED"; t["source"]="PREDICTED"; t["detection_index"]=None; t["detection_score"]=None; t["occluder"]="T999999"; t["missed_frames"]=1; t["time_since_observed_s"]=0.04; t["contested"]=False
def det_oob(d): d["frames"][10]["tracks"][0]["detection_index"] = 999
def cut_no_epoch(d): d["frames"][10]["camera"]["status"] = "CUT"
def pred_future(d): d["frames"][10]["tracks"][0]["possible_predecessors"] = ["T424242"]
def missed_age(d): d["frames"][10]["tracks"][0]["age_frames"] = 1000
def conf_obs_unconfirmed_lost(d): d["frames"][10]["tracks"][0]["observed_frames"] = -5
mut("two tracks share one detection_index", same_det)
mut("track vanishes 5 frames without ended, reappears", vanish)
mut("occluder id not a live track", occluder_ghost)
mut("detection_index out of range", det_oob)
mut("CUT frame but tracks keep epoch/ids", cut_no_epoch)
mut("possible_predecessor never emitted", pred_future)
mut("age_frames jumps 10->1000", missed_age)
mut("observed_frames negative", conf_obs_unconfirmed_lost)

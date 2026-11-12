# Review checklist: pre-labelled frames (Roboflow `spana-elite`)

Pre-labels come from a general COCO detector. `player` boxes are COCO `person`, and `ball` boxes are COCO
`sports ball` above a high threshold. Treat every box as a suggestion. Classes: player, goalkeeper, referee, ball.

For each frame tagged `prelabel:v1`:

1. **Relabel people.** Change a `player` box to `goalkeeper` or `referee` where that is who it is. Assistant
   referees are `referee`.
2. **Ball.** Add the ball if it is visible and not boxed. If you are unsure (blurred, half-hidden, far away), still
   box it when you can see it. Delete any `ball` box that is not the match ball.
3. **Delete off-pitch people.** Delete boxes on spectators, substitutes, coaches, staff, ball kids and photographers.
4. **Box hidden players.** Box players partly hidden by others. Draw the box around the visible part of the body.
5. **Missed players.** Add players the model missed, such as small ones far away or ones at the image edge.
6. **Box fit.** Each box should cover head to feet, including the visible stretched-out limbs.
7. **Mark the frame reviewed.** When the frame is complete, add the tag `reviewed`. Frames tagged
   `excluded:near-duplicate` need no review; they never enter a version.

Track progress with `python -m tools.roboflow_progress`. A version cannot be generated while any pre-labelled frame
is unreviewed (`tools/roboflow_version.py`).

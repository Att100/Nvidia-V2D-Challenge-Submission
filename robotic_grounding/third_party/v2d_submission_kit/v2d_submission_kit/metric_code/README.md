# Metric scoring code

These are code-only copies of the production Kaggle metrics. The scoring functions and
validation logic are included; host reference values, object meshes, the MHR model asset,
and object-frame AUC point samples are omitted. They are for inspecting the scoring
implementation. Held-out Track 1 and Track 3 scores are computed on Kaggle.

- `track_1/`: CD-H, CD-O, ACC-H, ACC-O, PEN.
- `track_2/`: AUC, SP-SR, MP-SR, MPPE. The three tiers use the same formulas.
- `track_3/`: AUC, SP-SR, MP-SR, RPE, MPPE, CD-O.

The Track 1 MHR model code is derived from Meta's MHR under Apache 2.0. Its license and
notice are in `track_1/LICENSE.txt` and `track_1/NOTICE.md`.

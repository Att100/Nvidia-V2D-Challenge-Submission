# Challenge Result Policy

Generated deliverables are intentionally not included in this cleaned project.
Historical outputs and manually adjusted sequences from development are not part
of the final system. Final results must be regenerated through `run.py` and the
documented automatic stages.

The challenge episode identifiers used during development were:

```text
000001  000003  000005  000006  000008
000010  000014  000015  000016  000019
000020  000025  000028  000030  000032
000033  000035  000036  000037  000038
```

A newly generated deliverable is self-contained and has:

- synchronized native RGB reference video and rectified-left video;
- metric rectified-left depth;
- rectified-stereo calibration and rectified-left camera trajectory;
- scaled self-reconstructed object/support GLBs and per-frame 6DoF poses;
- official MANO parameters, separate scalar scale, topology, and validity masks;
- exact array schema, manifest, validation report, rendered validation video, and
  a standalone `render_hand_object.py` reader/renderer.

PASS means arrays are synchronized, transforms round-trip, MANO parameters rebuild
the exported hand geometry, object camera/world poses agree, and the standalone
renderer completes. It does not provide a ground-truth accuracy score.

## Producing and exporting results

Run the pipeline with an explicit `output.root`, then use
`tools/export_sequence_deliverable.py`. The exported standalone renderer does not
import pipeline source code.

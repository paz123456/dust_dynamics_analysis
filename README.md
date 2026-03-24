# Dust Dynamics Analysis

Utilities for running **automated TrackMate analysis in Fiji** and post-processing particle trajectory outputs in Python.

## Repository layout

- `fiji_automation/`
  - `automate_trackmate.py`: Fiji/Jython script executed in headless mode for a frame range.
  - `run_automated_trackmate.py`: Python launcher that batches frame ranges and invokes Fiji.
- `track_mate_visualisation/`
  - Python package for loading, filtering, and plotting TrackMate trajectory data.
 
## Fiji/Trackmate
Fiji can be downloaded here: https://imagej.net/software/fiji/downloads 
Trackmate needs to be installed as extra plugin (https://imagej.net/plugins/trackmate/)
Fiji/trackmate workflow:
- Import image sequence (tifs) into fiji
- start trackmate plugin
- select spot detection algorithm
- test different spot sizes/prominences/quality factors (in my case DogDetector worked best)
- test different linking algorithms (depend on the motion which is observed. With the parabolic dust i think the kalman linker is best suited)
- use the ideal algorithm settings for the automated version that automatically imports sequences of images and does the full Trackmate analysis 


## Automated TrackMate workflow

### 1) Prepare your image sequence

The Fiji script expects a sequence directory with files named like:

- `out_0001.tif`
- `out_0002.tif`
- ...

(Any integer suffix is supported as long as the pattern is `out_<number>.tif`.)

### 2) Run the batch launcher

From the repository root:

```bash
python fiji_automation/run_automated_trackmate.py \
  --fiji "/path/to/Fiji.app/Contents/MacOS/fiji-macosx" \
  --script "/path/to/repo/fiji_automation/automate_trackmate.py" \
  --input-root "/path/to/experiment_root" \
  --sequence-subdir tifs \
  --output-subdir results \
  --step 2000 \
  --xmx 18g \
  --radius 6.0 \
  --quality-thresh 271.79 \
  --link-dist 45.0 \
  --gap-dist 45.0 \
  --max-frame-gap 2
```

### 3) Outputs

For each processed frame window, the Fiji script exports:

- `spots_<start>_to_<end>.csv`
- `edges_<start>_to_<end>.csv`
- `tracks_<start>_to_<end>.csv`
- `run_<start>_to_<end>.log`

inside each experiment's output directory.

## Notes

- The batch launcher now uses explicit CLI arguments instead of hard-coded machine-specific paths.
- Use `--dry-run` to inspect generated commands without executing Fiji.
- Use `--strict` to fail fast if any experiment directory fails.

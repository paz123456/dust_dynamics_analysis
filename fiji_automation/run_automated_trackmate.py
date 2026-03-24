# run_automated_trackmate.py
import os, re, subprocess
import glob
from pathlib import Path


os.chdir("/Users/philippziegler/Documents/Fiji/")

FIJI = "./Fiji.app/Contents/MacOS/fiji-macos*"
SCRIPT = Path(
    "/Users/philippziegler/Documents/dust_lofting_project/fiji_utils/fiji_automation/automate_trackmate.py"
)
STEP = 2000
XMX = "18g"

cmd = f"{FIJI} -Xmx{XMX} --headless --run {SCRIPT}"

path = Path("/Volumes/Festplatte3/videos_silica_dust/2kV/")


# RADIUS = 12.5
ALLOW_TRACK_SPLITTING = 1


# RADIUS = 3.0
# QUALITY_THRESH = 138.0
# ALLOW_TRACK_SPLITTING = 1
LINK_DIST = 45.0
GAP_DIST = 45.0

MAX_FRAME_GAP = 2


RADIUS = 6.0
QUALITY_THRESH = 271.79


paths = glob.glob(f"{path}/*")

wdirs = [
    i
    for i in paths
    if "._" not in i
    and not i.endswith("py")
    and not i.endswith("txt")
    and not "oscillo" in i
    and not "not_complete" in i
]
print(wdirs)
for wdir in wdirs:
    out = path / wdir / "results"
    seq = path / wdir / "tifs"

    frames = [
        i
        for i in os.listdir(seq)
        if "out" in i and i.endswith(".tif") and not "._" in i
    ]
    n = len(frames)
    print(f"Found {n} frames in {seq}")

    for i in range(0, n, STEP):
        j = min(i + STEP, n)
        params = f"seqDir='{seq}',outDir='{out}',start={i},end={j},RADIUS={RADIUS},QUALITY_THRESH={QUALITY_THRESH},ALLOW_TRACK_SPLITTING={ALLOW_TRACK_SPLITTING},LINK_DIST={LINK_DIST},GAP_DIST={GAP_DIST},MAX_FRAME_GAP={MAX_FRAME_GAP}"
        updated_cmd = f'{cmd} "{params}"'
        os.system(updated_cmd)

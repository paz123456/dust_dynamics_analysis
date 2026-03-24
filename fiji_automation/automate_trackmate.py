# @ String (label="Sequence directory") seqDir
# @ String (label="Output directory")  outDir
# @ Integer (label="Start  (inclusive, e.g. 13615)") start
# @ Integer (label="End  (inclusive, e.g. 15614)") end
# @ Float (label="Radius of dust in pixel", value=5.) RADIUS
# @ Float (label="Quality threshold of dust detector", value=235.) QUALITY_THRESH
# @ Integer (label="Allow track splitting", value=False) ALLOW_TRACK_SPLITTING
# @ Float (label="linking distance", value=35.) LINK_DIST
# @ Float (label="gap linking distance", value=35.) GAP_DIST
# @ Integer (label="gap linking missing frames", value=2) MAX_FRAME_GAP

from java.io import File
from java.io import File as JFile
import os, sys, re
from ij import IJ, ImagePlus, ImageStack
from ij.io import Opener
from ij.measure import ResultsTable
from fiji.plugin.trackmate import Model, Settings, TrackMate, Logger
from fiji.plugin.trackmate.io import TmXmlWriter
from fiji.plugin.trackmate.detection import DogDetectorFactory as detector_factory
from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory
from fiji.plugin.trackmate.tracking.kalman import KalmanTrackerFactory
from fiji.plugin.trackmate.tracking.jaqaman import LAPUtils
from fiji.plugin.trackmate.io import CSVExporter
from fiji.plugin.trackmate.visualization.table import TrackTableView
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO


def write_run_log(outDir, seqDir, start, end, imp, settings, model):
    from java.util import Date
    from java.lang import System
    import os

    # handy getters with fallbacks
    try:
        su = settings.getSpaceUnits()
    except:
        su = "px"
    try:
        tu = settings.getTimeUnits()
    except:
        tu = "frame"

    lines = []
    lines.append("=== TrackMate run log ===")
    lines.append("timestamp: %s" % str(Date()))
    lines.append(
        "java: %s (%s)"
        % (System.getProperty("java.version"), System.getProperty("java.vendor"))
    )
    lines.append(
        "os: %s %s" % (System.getProperty("os.name"), System.getProperty("os.version"))
    )
    lines.append("")
    lines.append("input.seqDir: %s" % seqDir)
    lines.append("input.range:  %s..%s" % (start, end))
    lines.append("")
    lines.append("image.name:   %s" % imp.getTitle())
    lines.append(
        "image.size:   %dx%d (C=%d, Z=%d, T=%d)"
        % (
            imp.getWidth(),
            imp.getHeight(),
            imp.getNChannels(),
            imp.getNSlices(),
            imp.getNFrames(),
        )
    )
    lines.append(
        "calibration:  dx=%.6g dy=%.6g dz=%.6g %s, dt=%.6g %s"
        % (settings.dx, settings.dy, settings.dz, su, settings.dt, tu)
    )
    lines.append("")
    # detector
    lines.append(
        "detector.factory: %s" % settings.detectorFactory.getClass().getSimpleName()
    )
    ds = settings.detectorSettings
    keys = list(dict(ds))
    keys.sort()
    for k in keys:
        lines.append("detector.%s: %s" % (k, ds.get(k)))

    # tracker
    lines.append(
        "tracker.factory:  %s" % settings.trackerFactory.getClass().getSimpleName()
    )
    ts = settings.trackerSettings
    keys = list(dict(ts))
    keys.sort()
    for k in keys:
        lines.append("tracker.%s: %s" % (k, ts.get(k)))

    # analyzers actually registered
    try:
        lines.append("")
        lines.append(
            "spot_analyzers:  "
            + ", ".join(
                [
                    a.getClass().getSimpleName()
                    for a in settings.getSpotAnalyzerFactories()
                ]
            )
        )
    except:
        pass
    try:
        lines.append(
            "edge_analyzers:  "
            + ", ".join(
                [a.getClass().getSimpleName() for a in settings.getEdgeAnalyzers()]
            )
        )
    except:
        pass
    try:
        lines.append(
            "track_analyzers: "
            + ", ".join(
                [a.getClass().getSimpleName() for a in settings.getTrackAnalyzers()]
            )
        )
    except:
        pass
    # quick counts
    try:
        sc = model.getSpots()
        tm = model.getTrackModel()
        n_spots = sc.getNSpots(False)  # False => include filtered-out
        n_tracks = tm.nTracks(False) if hasattr(tm, "nTracks") else tm.getNTracks(False)
        lines.append("")
        lines.append("counts.spots_total:  %d" % n_spots)
        lines.append("counts.tracks_total: %d" % n_tracks)
    except:
        pass

    log_path = os.path.join(outDir, "run_%s_to_%s.log" % (start, end))
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    with open(log_path, "w") as fh:
        fh.write("\n".join(lines))
    IJ.log("Wrote log: " + log_path)


reload(sys)
sys.setdefaultencoding("utf-8")

SAVE_XML = True
# RADIUS = 3. # normally 5.
# QUALITY_THRESH = 106. # normally 235 # for non exploding experiments


DO_SUBPIXEL_LOCALIZATION = True
NORMALIZE = False
# RADIUS_Z = 0.0

DO_MEDIAN_FILTERING = False
TARGET_CH = 1

# LAP tracker
if MAX_FRAME_GAP > 0:
    ALLOW_GAP = True
else:
    ALLOW_GAP = False
# Calibration (pixels / frames)
PIXW = 1.0
PIXH = 1.0
PIXD = 1.0
DT = 1.0
SPACE_UNITS = "px"
TIME_UNITS = "frame"


tracker_factory = KalmanTrackerFactory


def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)


def open_sequence_range_as_T(dirpath, start_tok, end_tok):
    rx = re.compile(r"^out_(\d+)\.tif$", re.I)
    files = sorted(
        [
            f
            for f in os.listdir(dirpath)
            if rx.match(f) and start_tok <= int(rx.match(f).group(1)) <= end_tok
        ],
        key=lambda f: int(rx.match(f).group(1)),
    )
    if not files:
        raise IOError("No frames found in %s for range %s..%s" % (dirpath, start_tok, end_tok))

    op = Opener()
    imp0 = op.openImage(os.path.join(dirpath, files[0]))
    if imp0 is None:
        raise IOError("Failed to open first frame: %s" % files[0])
    w, h = imp0.getWidth(), imp0.getHeight()
    stack = ImageStack(w, h)
    stack.addSlice(imp0.getProcessor())
    imp0.close()
    for f in files[1:]:
        imp = op.openImage(os.path.join(dirpath, f))
        if imp is None:
            raise IOError("Failed to open frame: %s" % f)
        stack.addSlice(imp.getProcessor())
        imp.close()
    imp = ImagePlus("range_%d_%d" % (start_tok, end_tok), stack)
    imp.setDimensions(1, 1, stack.getSize())  # channels=1, slices=1, frames=T
    return imp


# --- load only the requested range ---
imp = open_sequence_range_as_T(seqDir, int(start), int(end))
cal = imp.getCalibration()
cal.pixelWidth, cal.pixelHeight, cal.pixelDepth = PIXW, PIXH, PIXD
cal.setUnit(SPACE_UNITS)


# --- TrackMate pipeline ---
model = Model()
model.setLogger(Logger.IJ_LOGGER)
settings = Settings(imp)
settings.dt = DT
settings.dx = PIXW
settings.dy = PIXH
settings.dz = PIXD

# dog detector with your params
settings.detectorFactory = detector_factory()
settings.detectorSettings = {
    "DO_SUBPIXEL_LOCALIZATION": DO_SUBPIXEL_LOCALIZATION,
    "RADIUS": RADIUS,
    "TARGET_CHANNEL": TARGET_CH,
    "THRESHOLD": QUALITY_THRESH,
    "DO_MEDIAN_FILTERING": DO_MEDIAN_FILTERING,
}

# LAP tracker with your params
settings.trackerFactory = tracker_factory()
settings.trackerSettings = (
    settings.trackerFactory.getDefaultSettings()
)  # almost good enough
# settings.trackerSettings["ALLOW_TRACK_SPLITTING"] = bool(ALLOW_TRACK_SPLITTING)
# settings.trackerSettings["ALLOW_TRACK_MERGING"] = False
# settings.trackerSettings["LINKING_MAX_DISTANCE"] = float(LINK_DIST)
# settings.trackerSettings["GAP_CLOSING_MAX_DISTANCE"] = float(GAP_DIST)
# settings.trackerSettings["MAX_FRAME_GAP"] = int(MAX_FRAME_GAP)
# settings.trackerSettings["ALLOW_GAP_CLOSING"] = bool(ALLOW_GAP)
settings.trackerSettings["MAX_FRAME_GAP"] = MAX_FRAME_GAP
settings.trackerSettings["KALMAN_SEARCH_RADIUS"] = LINK_DIST
settings.trackerSettings["LINKING_MAX_DISTANCE"] = GAP_DIST
settings.addAllAnalyzers()

tm = TrackMate(model, settings)
tm.computeSpotFeatures(True)
tm.computeTrackFeatures(True)

ok = tm.checkInput()
if not ok:
    print(str(tm.getErrorMessage()))

ok = tm.process()
if not ok:
    print(str(tm.getErrorMessage()))

ensure_dir(outDir)

ds = DisplaySettingsIO.readUserDefault()
TrackTableView.createSpotTable(model, ds).exportToCsv(
    JFile(os.path.join(outDir, "spots_%s_to_%s.csv" % (start, end)))
)
TrackTableView.createEdgeTable(model, ds).exportToCsv(
    JFile(os.path.join(outDir, "edges_%s_to_%s.csv" % (start, end)))
)
TrackTableView.createTrackTable(model, ds).exportToCsv(
    JFile(os.path.join(outDir, "tracks_%s_to_%s.csv" % (start, end)))
)


write_run_log(outDir, seqDir, start, end, imp, settings, model)

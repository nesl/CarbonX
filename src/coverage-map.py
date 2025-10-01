import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import os, json5

CONFIG_FILE_DIR = "../config/"
config_file = f"./{CONFIG_FILE_DIR}/config.json"
def _get_config():
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json5.load(f)
    else:
        print("Config file does not exist")
        exit(0)
    return config
SUPPORTED_CODES = set(_get_config()["SUPPORTED_REGIONS_FORECASTING"])

def rings_to_path(rings):
    """Convert rings to a Matplotlib Path that includes holes."""
    vertices, codes = [], []
    for ring in rings or []:
        if not ring or len(ring) < 3:
            continue
        if ring[0] != ring[-1]:
            ring = ring + [ring[0]]
        verts = [(float(x), float(y)) for x, y in ring]
        codes_ring = [MplPath.MOVETO] + [MplPath.LINETO]*(len(verts)-2) + [MplPath.CLOSEPOLY]
        vertices.extend(verts)
        codes.extend(codes_ring)
    return MplPath(vertices, codes) if vertices else None

def feature_to_patches(feature):
    """Turn Polygon/MultiPolygon into PathPatches (with holes)."""
    geom = feature.get("geometry", {})
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    patches = []
    if gtype == "Polygon":
        path = rings_to_path(coords)
        if path: patches.append(PathPatch(path))
    elif gtype == "MultiPolygon":
        for poly in coords or []:
            path = rings_to_path(poly)
            if path: patches.append(PathPatch(path))
    return patches

def get_code_from_props(props):
    """
    Preferred matching:
      1) zoneName  (e.g., 'US-CAL-CISO', 'PJM', 'CA-AB', etc.)
      2) countryKey   (e.g., 'US', 'CA', 'DE', etc.)
    Returns the first value that exists in SUPPORTED_CODES (case-insensitive).
    """
    z = props.get("zoneName")
    if isinstance(z, str) and z.strip():
        z_up = z.strip().upper()
        if z_up in SUPPORTED_CODES:
            return z_up

    c = props.get("countryKey")
    if isinstance(c, str) and c.strip():
        c_up = c.strip().upper()
        if c_up in SUPPORTED_CODES:
            return c_up

    return None  # not supported

def main():
    ap = argparse.ArgumentParser(description="Render electricity zone coverage map (PNG/SVG).")
    ap.add_argument("--in", dest="infile", required=True, help="Path to supported-grids.geojson")
    ap.add_argument("--out", dest="outfile", required=True, help="Output PNG (e.g., coverage-map.png)")
    ap.add_argument("--svg", dest="svgfile", default=None, help="Optional SVG output path")
    ap.add_argument("--title", default="Coverage")
    ap.add_argument("--width", type=float, default=14, help="Figure width (inches)")
    ap.add_argument("--height", type=float, default=7, help="Figure height (inches)")
    args = ap.parse_args()

    geo = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    features = geo.get("features", [])
    if not features:
        raise SystemExit("No features found in the GeoJSON.")

    supported_patches, other_patches = [], []
    minx = miny = float("inf")
    maxx = maxy = float("-inf")

    for feat in features:
        props = feat.get("properties", {}) or {}
        code = get_code_from_props(props)  # None if not supported by either zonename or country
        patches = feature_to_patches(feat)
        if not patches:
            continue

        target = supported_patches if code is not None else other_patches
        target.extend(patches)

        # update bounds
        for p in patches:
            xs, ys = zip(*p.get_path().vertices)
            minx, maxx = min(minx, min(xs)), max(maxx, max(xs))
            miny, maxy = min(miny, min(ys)), max(maxy, max(ys))

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=200)
    ax.set_axis_off()

    if other_patches:
        ax.add_collection(PatchCollection(other_patches, facecolor="#d9d9d9", edgecolor="#999999", linewidths=0.2))
    if supported_patches:
        ax.add_collection(PatchCollection(supported_patches, facecolor="#0f5c08", edgecolor="#131414", linewidths=0.35))

    if minx < maxx and miny < maxy:
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    ax.set_title(args.title, fontsize=14, pad=10)

    import matplotlib.patches as mpatches
    leg = ax.legend(
        handles=[mpatches.Patch(color="#0f5c08", label="Supported"),
                 mpatches.Patch(color="#d9d9d9", label="Other grids")],
        loc="lower left", frameon=True
    )
    for t in leg.get_texts(): t.set_fontsize(9)

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    if args.svgfile:
        Path(args.svgfile).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.svgfile, bbox_inches="tight")

if __name__ == "__main__":
    main()
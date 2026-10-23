#!/usr/bin/env python3
"""Niivue 3D group figures for the manual 6-subject HBCD_diff_morph run.

Port of HBCD_time_warp niivue_fig6_toy (make_fig6_3d_group_v2 + statics +
anim_real), combined into one script. Differences from the source:
  - tube COLOR = group-mean nodewise log det J (per metric set);
  - NO radius/width modulation and NO directional-decomposition metric
    (HBCD_diff_morph has none); legends/tooltips show log-J + SD only;
  - dual sets: non_demeaned (derivatives_2) and image_demeaned
    (derivatives_2_de_meaned, written by plot_logj_figures_8.py);
  - interactive HTML carries both color-data domains (anatomical T2w-image
    logJ default + tractography tract-mask logJ) with an in-page toggle.

Outputs per metric set, under --out-dir:
  fig3d_group_{set}.html          interactive (hover tooltips, bundle toggles)
  fig3d_group_{set}_static.png    bilateral grid, 3 views, numpy rasterizer
  fig3d_{set}_orbit.mp4 / .gif    360-degree rotation movie
  group_level_nodewise/group_nodewise_{set}.csv   cached group per-node stats

Tube centre-lines and the glass brain come from --geom-subject only (default
sub-7857079557); group means use all subjects that have the metric column.
"""

import argparse
import base64
import csv as csv_mod
import gc
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps

from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points

N_NODES = 100
N_RING = 16
N_SL_FOR_MEAN = 50
DECIMATION_REDUCTION = 0.96
SMOOTH_ITERATIONS = 15
CAP_FRACTION = 0.55
CAP_RINGS = 3
CMAP = colormaps["coolwarm"]
ALPHA_FLOOR = 0.15
PICK_PAD_MM = 0.4

CELL_STATIC = 480
CELL_ANIM = 512
SS = 2
BG = np.array([0.04, 0.04, 0.05])
RHO = 0.94
FPS = 24

VIEWS = [
    ("axial", np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
    ("sagittal", np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    ("coronal", np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
]

SET_SPECS = {
    "non_demeaned": {
        "anatomical": ("mean_logJ", "Anatomical logJ (non-demeaned)"),
        "tractography": ("mean_logJ", "Tractography logJ (non-demeaned)"),
    },
    "image_demeaned": {
        "anatomical": ("mean_t2w_image_demeaned_logJ", "T2w-image-demeaned logJ"),
        "tractography": ("mean_tract_mask_demeaned_logJ", "Tract-mask-demeaned logJ"),
    },
}
DOMAINS = ["anatomical", "tractography"]
DEFAULT_DOMAIN = "anatomical"
DOMAIN_UI = {"anatomical": "anatomical (T2w image)", "tractography": "tractography (tract mask)"}
STATIC_LABELS = {"non_demeaned": "non-demeaned", "image_demeaned": "T2w-image-demeaned"}

X3D_COLOR3 = "if(e.length===d*3){let k=0,U=0;for(let Y=0;Y<d;Y++){for(let K=0;K<3;K++)L[U]=Math.round(e[k]*255),k++,U++;U++}}"
X3D_COLOR4 = "if(e.length===d*4){let k=0,U=0;for(let Y=0;Y<d;Y++){for(let K=0;K<4;K++)L[U]=Math.round(e[k]*255),k++,U++}}else " + X3D_COLOR3
BLEND3 = "l>=1?(I.disable(I.BLEND),I.depthMask(!0)):(I.enable(I.BLEND),I.depthMask(!1))"
BLEND4 = "l>=1?(I.disable(I.BLEND),I.depthMask(!0)):l<.99?(I.enable(I.BLEND),I.depthMask(!1)):(I.enable(I.BLEND),I.depthMask(!0))"
MVP3 = "bi({gl:I,shader:S,mvpMtx:Q,normMtx:i,opacity:l})"
MVP4 = "window.__nvMvp=Float32Array.from(Q)," + MVP3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--t2w-deriv-root", required=True, help="Anatomical pipeline root, e.g. derivatives_2/T2w_diff_morph.")
    p.add_argument("--demeaned-t2w-root", required=True, help="Demeaned anatomical root, e.g. derivatives_2_de_meaned/T2w_diff_morph.")
    p.add_argument("--tract2tract-root", default=None, help="Tractography root (default: sibling tract2tract_diff_morph of --t2w-deriv-root).")
    p.add_argument("--demeaned-tract2tract-root", default=None, help="Demeaned tractography root (default: sibling tract2tract_diff_morph of --demeaned-t2w-root).")
    p.add_argument("--preproc-t2w-root", required=True, help="preproc_T2w root for the glass-brain mask.")
    p.add_argument("--out-dir", required=True, help="Output directory for HTML/PNG/MP4/GIF artifacts.")
    p.add_argument("--geom-subject", default="sub-7857079557", help="Subject whose warped tracts / mask provide geometry.")
    p.add_argument("--subjects", nargs="+", default=None, help="Group members (default: all sub-* under --t2w-deriv-root).")
    p.add_argument("--metric-sets", nargs="+", default=["non_demeaned", "image_demeaned"], choices=sorted(SET_SPECS))
    p.add_argument("--stages", nargs="+", default=["html", "statics", "anim"], choices=["html", "statics", "anim"])
    p.add_argument("--tube-radius-mm", type=float, default=1.5, help="Constant tube radius (no width modulation).")
    p.add_argument("--tract-families", nargs="+", default=["CST", "FX", "IFOF", "UF"])
    p.add_argument("--frames", type=int, default=162, help="Orbit frames.")
    p.add_argument("--niivue-js", default=None, help="Path to niivue.umd.js (default: next to this script).")
    return p.parse_args()


def make_x3d(verts, faces, rgb, alpha=None):
    ci = " ".join(f"{a} {b} {c} -1" for a, b, c in faces)
    pt = " ".join(f"{x:.2f} {y:.2f} {z:.2f}" for x, y, z in verts)
    if alpha is None:
        col = " ".join(f"{r:.3f} {g:.3f} {b:.3f}" for r, g, b in rgb)
    else:
        assert len(alpha) == len(rgb)
        col = " ".join(f"{r:.3f} {g:.3f} {b:.3f} {a:.3f}" for (r, g, b), a in zip(rgb, alpha))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<X3D profile="Interchange" version="3.3">\n'
        "<Scene><Shape>"
        '<Appearance><Material diffuseColor="1 1 1" transparency="0"></Material></Appearance>'
        f'<IndexedFaceSet solid="false" colorPerVertex="true" coordIndex="{ci}">'
        f'<Coordinate point="{pt}"></Coordinate>'
        f'<Color color="{col}"></Color>'
        "</IndexedFaceSet></Shape></Scene>\n</X3D>\n"
    )


def patched_niivue_js(js_path):
    """niivue.umd.js with four build-time edits (source file untouched):
    1. X3D reader also accepts 4-component (RGBA) per-vertex <Color>.
    2. mesh fragment shaders multiply uniform opacity by per-vertex alpha.
    3. meshes with uniform opacity in [0.99, 1) blend but still write depth.
    4. the live mesh view-projection matrix is exposed for hover ray-picking."""
    js = Path(js_path).read_text()
    assert js.count(X3D_COLOR3) == 1, "X3D color pattern not found in niivue.umd.js"
    js = js.replace(X3D_COLOR3, X3D_COLOR4)
    assert js.count(BLEND3) == 1, "mesh blend-depth pattern not found"
    js = js.replace(BLEND3, BLEND4)
    assert js.count(MVP3) == 1, "mesh draw mvpMtx callsite not found"
    js = js.replace(MVP3, MVP4)
    n = 0
    parts = []
    for chunk in js.split("#version 300 es"):
        if "in vec4 vClr;" in chunk and "out vec4 color;" in chunk:
            head, sep, tail = chunk.partition(", opacity)")
            pieces = []
            while sep:
                n += 1
                pieces.append(head + ", vClr.a * opacity)")
                head, sep, tail = tail.partition(", opacity)")
            pieces.append(head)
            parts.append("".join(pieces))
        else:
            parts.append(chunk)
    assert n >= 2, f"expected mesh-shader opacity writes, found {n}"
    js = "#version 300 es".join(parts)
    assert "background.rgb, opacity)" in js, "volume background shader must stay unpatched"
    print(f"patched niivue lib: RGBA X3D colors + {n} vertex-alpha-aware fragment outputs")
    return js


def brain_mesh(mask_path):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    img = nib.load(str(mask_path))
    data = (np.asanyarray(img.dataobj) > 0).astype(np.uint8)
    vtimg = vtk.vtkImageData()
    vtimg.SetDimensions(data.shape)
    vtimg.SetSpacing(1, 1, 1)
    vtimg.GetPointData().SetScalars(numpy_to_vtk(data.ravel(order="F"), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
    mc = vtk.vtkImageMarchingCubes()
    mc.SetInputData(vtimg)
    mc.SetNumberOfContours(1)
    mc.SetValue(0, 0.5)
    mc.Update()
    dec = vtk.vtkDecimatePro()
    dec.SetInputData(mc.GetOutput())
    dec.SetTargetReduction(DECIMATION_REDUCTION)
    dec.SetBoundaryVertexDeletion(0)
    dec.SetFeatureAngle(90)
    dec.Update()
    sm = vtk.vtkWindowedSincPolyDataFilter()
    sm.SetInputConnection(dec.GetOutputPort())
    sm.SetNumberOfIterations(SMOOTH_ITERATIONS)
    sm.BoundarySmoothingOn()
    sm.Update()
    out = sm.GetOutput()
    pts = vtk_to_numpy(out.GetPoints().GetData()).astype(float)
    pts = (img.affine[:3, :3] @ pts.T).T + img.affine[:3, 3]
    polys = vtk_to_numpy(out.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return pts, polys


def mean_curve(trk_path):
    sft = load_tractogram(str(trk_path), "same", bbox_valid_check=False)
    streamlines = sft.streamlines
    idx = np.linspace(0, len(streamlines) - 1, min(N_SL_FOR_MEAN, len(streamlines))).astype(int)
    subs = [np.asarray(streamlines[i], dtype=np.float64) for i in idx if len(streamlines[i]) >= 5]
    curve = np.stack(set_number_of_points(subs, N_NODES)).mean(axis=0)
    del streamlines, sft, subs
    gc.collect()
    return curve


def frames(curve):
    t = np.gradient(curve, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, t[0]))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    n0 = np.cross(t[0], helper)
    normals = np.empty_like(curve)
    normals[0] = n0 / np.linalg.norm(n0)
    for i in range(1, len(curve)):
        p = normals[i - 1] - np.dot(normals[i - 1], t[i]) * t[i]
        normals[i] = p / np.linalg.norm(p)
    return normals, np.cross(t, normals)


def spherical_cap(unit_n, unit_b, axis, r_end):
    h = CAP_FRACTION * r_end
    sphere_r = (r_end**2 + h**2) / (2 * h)
    theta0 = float(np.arccos((sphere_r - h) / sphere_r))
    cap_v, cap_f, prev = [], [], None
    phi = np.linspace(0.0, 2.0 * np.pi, N_RING, endpoint=False)
    for k in range(1, CAP_RINGS + 1):
        theta = theta0 * (1.0 - k / (CAP_RINGS + 1))
        ring_r = sphere_r * np.sin(theta)
        z = h - sphere_r * (1.0 - np.cos(theta))
        ring = unit_n * (ring_r * np.cos(phi)[:, None]) + unit_b * (ring_r * np.sin(phi)[:, None]) + axis * z
        v_off = k * N_RING - N_RING
        cap_v.append(ring)
        if prev is None:
            for j in range(N_RING):
                jn = (j + 1) % N_RING
                cap_f += [(-1 - j, -1 - jn, v_off + j), (-1 - jn, v_off + jn, v_off + j)]
        else:
            for j in range(N_RING):
                jn = (j + 1) % N_RING
                cap_f += [(prev + j, prev + jn, v_off + j), (prev + jn, v_off + jn, v_off + j)]
        prev = v_off
    tip = CAP_RINGS * N_RING
    cap_v.append((axis * h).reshape(1, 3))
    for j in range(N_RING):
        jn = (j + 1) % N_RING
        cap_f += [(prev + j, prev + jn, tip)]
    return np.vstack(cap_v), cap_f


def tube(curve, radius):
    normals, binormals = frames(curve)
    radii = np.full(N_NODES, float(radius))
    phi = np.linspace(0.0, 2.0 * np.pi, N_RING, endpoint=False)
    ring = radii[:, None, None] * (
        np.cos(phi)[None, :, None] * normals[:, None, :] + np.sin(phi)[None, :, None] * binormals[:, None, :]
    )
    verts = (curve[:, None, :] + ring).reshape(-1, 3)
    node_idx = np.repeat(np.arange(N_NODES), N_RING)
    faces = []
    for i in range(N_NODES - 1):
        for j in range(N_RING):
            jn = (j + 1) % N_RING
            a, b, c, d = i * N_RING + j, i * N_RING + jn, (i + 1) * N_RING + jn, (i + 1) * N_RING + j
            faces += [(a, b, c), (a, c, d)]
    tang0 = curve[1] - curve[0]
    tang0 /= np.linalg.norm(tang0)
    tang1 = curve[-1] - curve[-2]
    tang1 /= np.linalg.norm(tang1)
    for i, center, out_dir in ((0, curve[0], -tang0), (N_NODES - 1, curve[-1], tang1)):
        base = i * N_RING
        v_off = len(verts)
        cap_v, cap_f = spherical_cap(normals[i], binormals[i], out_dir, radii[i])
        if i == 0:
            cap_f = [(a, c, b) for a, b, c in cap_f]
        verts = np.vstack([verts, center + cap_v])
        node_idx = np.concatenate([node_idx, np.full(len(cap_v), i)])

        def dec(x):
            return (base - x - 1) if x < 0 else (x + v_off)

        faces += [(dec(a), dec(b), dec(c)) for a, b, c in cap_f]
    return verts, np.asarray(faces), node_idx


def colorbar_css(n=25):
    f = np.linspace(0, 1, n)
    rgb = (np.asarray(CMAP(f))[:, :3] * 255).astype(int)
    return "linear-gradient(to right, " + ", ".join(f"rgb({r},{g},{b}) {p:.1f}%" for (r, g, b), p in zip(rgb, f * 100)) + ")"


HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Group tract morphometry in 3D (Niivue)</title>
<style>
  body { margin: 0; font-family: Arial, Helvetica, sans-serif; background: #111; color: #eee; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
  #head { padding: 6px 14px; flex: 0 0 auto; }
  #head h1 { margin: 0; font-size: 15px; }
  #head p { margin: 3px 0 0; font-size: 11px; color: #bbb; }
  #wrap { flex: 1 1 auto; min-height: 0; display: flex; }
  #stage { flex: 1 1 auto; min-width: 0; position: relative; background: #000; }
  #gl1 { position: absolute; width: 100%; height: 100%; display: block; }
  #side { flex: 0 0 172px; padding: 8px 10px; overflow-y: auto; font-size: 13px; }
  #side label { display: block; padding: 3px 0; cursor: pointer; white-space: nowrap; }
  #dom { margin-bottom: 10px; }
  #allbtns { margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }
  #allbtns button { background: #333; border: 1px solid #555; color: #eee; padding: 5px 8px; font-size: 12px; cursor: pointer; border-radius: 4px; }
  #bar { display: flex; align-items: center; gap: 10px; padding: 6px 14px; font-size: 12px; flex-wrap: wrap; }
  #cbar { width: 220px; height: 14px; border: 1px solid #666; }
  #err { color: #f88; font-size: 12px; white-space: pre-wrap; padding: 0 14px; }
  #tip { position: absolute; display: none; pointer-events: none; background: rgba(15,15,15,0.92); border: 1px solid #666; border-radius: 5px; padding: 6px 9px; font-size: 12px; line-height: 1.5; color: #eee; z-index: 10; white-space: nowrap; }
  #tip b { color: #fff; }
  #tip .v { color: #9cf; }
</style></head><body>
<div id="head">
  <h1>$TITLE</h1>
  <p>Group-level findings (ses-V02&rarr;V03, $NSUB subjects, anatomical + tractography pipelines); anatomy &amp; tract paths from one anonymized reference subject. Drag to rotate, scroll to zoom.
     Color = group-mean <span id="capmetric">$METRIC</span> (coolwarm, symmetric around 0; switch color data at right); fade along length = between-subject SD of logJ (opaque where SD &le; <span id="capdlo">$SDLO</span>, most transparent where SD &ge; <span id="capdhi">$SDHI</span>, alpha floor 0.15); tube width constant; gray shell = T2w mask mesh. Hover over a tube for per-node values.</p>
</div>
<div id="wrap">
  <div id="stage"><canvas id="gl1" class="niivue-canvas"></canvas><div id="tip"></div></div>
  <div id="side"><div id="dom"><b>Color data</b>$DOMAIN_RADIOS</div><b>Bundles</b>$CHECKBOXES<div id="allbtns"><button id="showall">show all bundles</button><button id="hideall">hide all bundles</button></div></div>
</div>
<div id="bar">
  <span>logJ</span><span id="cbarmin"></span><span id="cbar" style="background:$CSS"></span><span id="cbarmax"></span>
  <span id="barlabel">$METRICLABEL (group mean)</span>
</div>
<div id="err"></div>
<script>$NIIVUE_JS</script>
<script>
window.onerror = (m, s, l) => { document.getElementById('err').textContent += 'ERR: ' + m + ' @' + l + '\\n'; };
const DATA = $DATA_JSON;
const nv = new niivue.Niivue({ dragAndDropEnabled: false, show3Dcrosshair: false });
async function boot() {
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  await nv.attachTo('gl1');
  const opts = [];
  for (const tr of DATA.tracts)
    opts.push({ url: 'data:application/octet-stream;base64,' + tr.b64, name: tr.name + '.x3d', opacity: 0.999 });
  for (const tr of DATA.tracts)
    opts.push({ url: 'data:application/octet-stream;base64,' + tr.b64_tract, name: tr.name + '__tracto.x3d', opacity: 0.999 });
  opts.push({ url: 'data:application/octet-stream;base64,' + DATA.brain_b64, name: 'brain.x3d', opacity: 0.15, rgba255: [210, 210, 212, 255] });
  await nv.loadMeshes(opts);
  const nTr = DATA.tracts.length;
  const tractMesh = { anatomical: {}, tractography: {} };
  for (let i = 0; i < nTr; i++) {
    tractMesh.anatomical[DATA.tracts[i].name] = nv.meshes[i].id;
    tractMesh.tractography[DATA.tracts[i].name] = nv.meshes[nTr + i].id;
  }
  const brainMesh = nv.meshes[nv.meshes.length - 1].id;
  const vis = {};
  let domain = DATA.domain || 'anatomical';
  for (const tr of DATA.tracts) {
    vis[tr.name] = true;
    nv.setMeshProperty(tractMesh.tractography[tr.name], 'visible', false);
  }
  for (const chk of document.querySelectorAll('#side input[type="checkbox"]')) {
    if (chk.dataset.tract === '__brain__') nv.setMeshProperty(brainMesh, 'visible', chk.checked);
    chk.addEventListener('change', () => {
      const id = chk.dataset.tract === '__brain__' ? brainMesh : tractMesh[domain][chk.dataset.tract];
      nv.setMeshProperty(id, 'visible', chk.checked);
      if (chk.dataset.tract !== '__brain__') vis[chk.dataset.tract] = chk.checked;
      nv.drawScene();
    });
  }
  nv.drawScene();
  const setAllTracts = (visible) => {
    for (const chk of document.querySelectorAll('#side input[data-tract]:not([data-tract="__brain__"])')) {
      chk.checked = visible;
      vis[chk.dataset.tract] = visible;
      nv.setMeshProperty(tractMesh[domain][chk.dataset.tract], 'visible', visible);
    }
    nv.drawScene();
  };
  document.getElementById('showall').addEventListener('click', () => setAllTracts(true));
  document.getElementById('hideall').addEventListener('click', () => setAllTracts(false));
  function updateDomainBar() {
    const D = DATA.domains[domain];
    document.getElementById('cbarmin').textContent = D.vmin.toFixed(2);
    document.getElementById('cbarmax').textContent = D.vmax.toFixed(2);
    document.getElementById('barlabel').textContent = D.label + ' (group mean)';
    document.getElementById('capmetric').textContent = D.label;
    document.getElementById('capdlo').textContent = D.sdlo.toFixed(3);
    document.getElementById('capdhi').textContent = D.sdhi.toFixed(3);
  }
  for (const rad of document.querySelectorAll('input[name="dom"]')) {
    rad.addEventListener('change', () => {
      if (!rad.checked || rad.value === domain) return;
      for (const tr of DATA.tracts) {
        nv.setMeshProperty(tractMesh[domain][tr.name], 'visible', false);
        nv.setMeshProperty(tractMesh[rad.value][tr.name], 'visible', !!vis[tr.name]);
      }
      domain = rad.value;
      tip.style.display = 'none';
      updateDomainBar();
      nv.drawScene();
    });
  }
  const tip = document.getElementById('tip');
  const canvas = document.getElementById('gl1');
  function invM4(m) {
    const a00=m[0],a01=m[1],a02=m[2],a03=m[3],a10=m[4],a11=m[5],a12=m[6],a13=m[7],
          a20=m[8],a21=m[9],a22=m[10],a23=m[11],a30=m[12],a31=m[13],a32=m[14],a33=m[15];
    const b00=a00*a11-a01*a10,b01=a00*a12-a02*a10,b02=a00*a13-a03*a10,b03=a01*a12-a02*a11,
          b04=a01*a13-a03*a11,b05=a02*a13-a03*a12,b06=a20*a31-a21*a30,b07=a20*a32-a22*a30,
          b08=a20*a33-a23*a30,b09=a21*a32-a22*a31,b10=a21*a33-a23*a31,b11=a22*a33-a23*a32;
    let det=b00*b11-b01*b10+b02*b09+b03*b08-b04*b07+b05*b06;
    if (!det) return null; det = 1.0/det;
    return new Float32Array([
      (a11*b11-a12*b10+a13*b09)*det,(a02*b10-a01*b11-a03*b09)*det,(a31*b05-a32*b04+a33*b03)*det,(a22*b04-a21*b05-a23*b03)*det,
      (a12*b08-a10*b11-a13*b07)*det,(a00*b11-a02*b08+a03*b07)*det,(a32*b02-a30*b05-a33*b01)*det,(a20*b05-a22*b02+a23*b01)*det,
      (a10*b10-a11*b08+a13*b06)*det,(a01*b08-a00*b10-a03*b06)*det,(a30*b04-a31*b02+a33*b00)*det,(a21*b02-a20*b04-a23*b00)*det,
      (a11*b07-a10*b09-a12*b06)*det,(a00*b09-a01*b07+a02*b06)*det,(a31*b01-a30*b03-a32*b00)*det,(a20*b03-a21*b01+a22*b00)*det]);
  }
  function unproj(i, x, y, z) {
    const w = i[3]*x + i[7]*y + i[11]*z + i[15];
    return [(i[0]*x + i[4]*y + i[8]*z + i[12])/w, (i[1]*x + i[5]*y + i[9]*z + i[13])/w, (i[2]*x + i[6]*y + i[10]*z + i[14])/w];
  }
  function fmt(v, d) { return (v >= 0 ? '+' : '') + v.toFixed(d); }
  function sig(v) { return v >= 0 ? '#f88' : '#8cf'; }
  function nodesFor(tr) { return domain === 'anatomical' ? tr.nodes : tr.nodes_tract; }
  function pickAt(nx, ny) {
    const mvp = window.__nvMvp;
    if (!mvp) return {err: 'mvp-missing'};
    const inv = invM4(mvp);
    if (!inv) return {err: 'matrix-not-invertible'};
    const near = unproj(inv, nx, ny, -1), far = unproj(inv, nx, ny, 1);
    let dx = far[0]-near[0], dy = far[1]-near[1], dz = far[2]-near[2];
    const dl = Math.hypot(dx, dy, dz);
    if (!(dl > 0)) return {err: 'degenerate-ray'};
    dx/=dl; dy/=dl; dz/=dl;
    let best = null;
    for (const tr of DATA.tracts) {
      if (!vis[tr.name]) continue;
      const nd = nodesFor(tr), c = nd.c, r = nd.r;
      for (let j = 0; j < nd.n_nodes; j++) {
        const px = c[j*3]-near[0], py = c[j*3+1]-near[1], pz = c[j*3+2]-near[2];
        const b = px*dx + py*dy + pz*dz;
        const cq = px*px + py*py + pz*pz - r[j]*r[j];
        const disc = b*b - cq;
        if (disc < 0) continue;
        const t = b - Math.sqrt(disc);
        const tt = t > 0 ? t : b + Math.sqrt(disc);
        if (tt > 0 && (best === null || tt < best.t)) best = {t: tt, tr, j};
      }
    }
    return {best};
  }
  function opacityFor(sd) {
    const D = DATA.domains[domain];
    return DATA.alphafloor + (1.0 - DATA.alphafloor) * Math.max(0, Math.min(1, (D.sdhi - sd) / (D.sdhi - D.sdlo)));
  }
  function showTip(best, cx, cy) {
    const nd = nodesFor(best.tr), j = best.j, D = DATA.domains[domain];
    tip.innerHTML = '<b>' + best.tr.name + '</b> &middot; ' + D.short + ' &middot; node ' + j + '/99 along tract<br>' +
      'logJ <span class="v" style="color:' + sig(nd.lj[j]) + '">' + fmt(nd.lj[j], 3) + '</span> (SD ' + nd.sd[j].toFixed(3) + ', n=' + nd.n + ')<br>' +
      'opacity ' + opacityFor(nd.sd[j]).toFixed(2);
    tip.style.display = 'block';
    const rect = canvas.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let lx = cx - rect.left + 14, ly = cy - rect.top + 14;
    if (lx + tw > rect.width) lx = cx - rect.left - tw - 10;
    if (ly + th > rect.height) ly = cy - rect.top - th - 10;
    tip.style.left = lx + 'px'; tip.style.top = ly + 'px';
  }
  let warnedNoMvp = false;
  function hoverAt(cx, cy) {
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return;
    const nx = ((cx - rect.left) / rect.width) * 2 - 1;
    const ny = 1 - ((cy - rect.top) / rect.height) * 2;
    const hit = pickAt(nx, ny);
    if (hit.err) {
      if (hit.err === 'mvp-missing' && !warnedNoMvp) {
        warnedNoMvp = true;
        document.getElementById('err').textContent += 'HOVER: view matrix not captured (stale or unpatched build?)\\n';
      }
      tip.style.display = 'none'; return;
    }
    if (!hit.best) { tip.style.display = 'none'; return; }
    showTip(hit.best, cx, cy);
  }
  canvas.addEventListener('mousemove', (e) => {
    if (e.buttons !== 0) { tip.style.display = 'none'; return; }
    hoverAt(e.clientX, e.clientY);
  });
  canvas.addEventListener('mousedown', () => { tip.style.display = 'none'; });
  canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
  canvas.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
  let touchStart = null;
  canvas.addEventListener('touchstart', (e) => {
    tip.style.display = 'none';
    touchStart = e.touches.length === 1
      ? {x: e.changedTouches[0].clientX, y: e.changedTouches[0].clientY, t: performance.now()} : null;
  }, {capture: true, passive: true});
  canvas.addEventListener('touchend', (e) => {
    if (touchStart) {
      const t = e.changedTouches[0];
      if (Math.hypot(t.clientX - touchStart.x, touchStart.y - touchStart.y) <= 12 &&
          performance.now() - touchStart.t <= 600) hoverAt(t.clientX, t.clientY);
    }
    touchStart = null;
  }, {capture: true, passive: true});
  canvas.addEventListener('touchcancel', () => { touchStart = null; }, {capture: true, passive: true});
  updateDomainBar();
  nv.setRenderAzimuthElevation(110, 15);
  nv.drawScene();
  setTimeout(() => nv.drawScene(), 300);
  setTimeout(() => nv.drawScene(), 1000);
}
boot().catch((e) => { document.getElementById('err').textContent += 'ERR: ' + (e && e.stack ? e.stack : e) + '\\n'; });
</script></body></html>
"""


def node_series(csv_path, col):
    out = np.full(N_NODES, np.nan)
    with open(csv_path) as fh:
        rows = list(csv_mod.DictReader(fh))
    if not rows or col not in rows[0]:
        return out
    nodes = np.array([float(r["node"]) for r in rows])
    order = np.argsort(nodes)
    nodes = nodes[order]
    vals = np.array([float(r[col]) if r.get(col) not in (None, "") else np.nan for r in rows], dtype=np.float64)[order]
    finite = np.isfinite(vals)
    if finite.sum() < 2:
        return out
    grid = np.linspace(0.0, nodes[-1], N_NODES)
    return np.interp(grid, nodes[finite], vals[finite])


def subject_node_csvs(root, subject, pipeline):
    if pipeline == "t2w":
        return sorted((Path(root) / subject / "morphometry" / "nodewise_logJ_summaries").glob("*_logJ_node_summary.csv"))
    return sorted((Path(root) / subject / "mask_syn").glob(f"*_{subject}_ses-V02_T2w/*_mask_syn_node_profile.csv"))


def group_nodewise(subjects, t2w_root, metric_col, cache_path, pipeline="t2w"):
    if Path(cache_path).exists() and not os.environ.get("REFRESH_GROUP_CACHE"):
        result = {}
        with open(cache_path) as fh:
            for r in csv_mod.DictReader(fh):
                d = result.setdefault(r["tract"], {"lj": [], "sd": [], "n": 0})
                d["lj"].append(float(r["mean_logj"]))
                d["sd"].append(float(r["sd_logj"]))
                d["n"] = int(r["n_subjects"])
        print(f"using cached group profiles {cache_path} (REFRESH_GROUP_CACHE=1 to recompute)")
        return {k: (np.array(v["lj"]), np.array(v["sd"]), v["n"]) for k, v in result.items()}

    result = {}
    if not subjects:
        raise RuntimeError("no subjects provided")
    per_subject = {}
    for s in subjects:
        p = Path(t2w_root) / s / ("morphometry/nodewise_logJ_summaries" if pipeline == "t2w" else "mask_syn")
        files = subject_node_csvs(t2w_root, s, pipeline)
        if not files:
            print(f"  {s}: no CSVs under {p}; skipped")
            continue
        series = {}
        for f in files:
            tract = f.name.split(f"_{s}_")[0]
            series[tract] = node_series(f, metric_col)
        per_subject[s] = series
    tracts = sorted({t for s in per_subject for t in per_subject[s]})
    for tract in tracts:
        lsum = np.zeros(N_NODES)
        lsq = np.zeros(N_NODES)
        lcnt = np.zeros(N_NODES)
        n_sub = 0
        for s in subjects:
            vals = per_subject.get(s, {}).get(tract)
            if vals is None:
                continue
            ok = np.isfinite(vals)
            lsum[ok] += vals[ok]
            lsq[ok] += vals[ok] ** 2
            lcnt[ok] += 1
            n_sub += 1
        if n_sub < 2 or (lcnt < 2).any():
            print(f"  WARNING {tract}: incomplete group coverage (n={n_sub}); skipped")
            continue
        lj_mean = lsum / lcnt
        var = (lsq - lcnt * lj_mean**2) / np.maximum(lcnt - 1, 1)
        result[tract] = (lj_mean, np.sqrt(np.maximum(var, 0.0)), n_sub)
        print(f"  {tract}: n={n_sub}")

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as fh:
        w = csv_mod.writer(fh)
        w.writerow(["tract", "node", "mean_logj", "sd_logj", "n_subjects"])
        for tract, (lj, sd, n) in result.items():
            for node in range(N_NODES):
                w.writerow([tract, node, f"{lj[node]:.6f}", f"{sd[node]:.6f}", n])
    print(f"cached group profiles -> {cache_path}")
    return result


def set_roots(args, set_name):
    if set_name == "non_demeaned":
        anat, tract = Path(args.t2w_deriv_root), args.tract2tract_root
        tract = Path(tract) if tract else anat.parent / "tract2tract_diff_morph"
    else:
        anat, tract = Path(args.demeaned_t2w_root), args.demeaned_tract2tract_root
        tract = Path(tract) if tract else anat.parent / "tract2tract_diff_morph"
    return {"anatomical": anat, "tractography": tract}


def domain_stats(drawn, groups):
    stats = {}
    for d in DOMAINS:
        vmax = float(np.ceil(max(np.max(np.abs(groups[d][t][0])) for t, _ in drawn) * 20) / 20)
        all_sd = np.concatenate([groups[d][t][1] for t, _ in drawn])
        sd_lo, sd_hi = (float(q) for q in np.percentile(all_sd, [1, 99]))
        if sd_hi - sd_lo < 1e-9:
            sd_lo, sd_hi = 0.0, float(max(sd_hi, 1e-6))
        stats[d] = {"vmax": vmax, "sdlo": sd_lo, "sdhi": sd_hi}
    return stats


def build_html(set_name, args, subjects, out_dir, js_path, brain_b64):
    roots = set_roots(args, set_name)

    groups = {}
    for d in DOMAINS:
        suffix = "" if d == DEFAULT_DOMAIN else f"_{d}"
        cache = Path(out_dir) / "group_level_nodewise" / f"group_nodewise_{set_name}{suffix}.csv"
        groups[d] = group_nodewise(subjects, roots[d], SET_SPECS[set_name][d][0], cache,
                                   pipeline="t2w" if d == "anatomical" else "tract")
        if not groups[d]:
            raise RuntimeError(f"no group data for set {set_name}, domain {d}")

    radius = args.tube_radius_mm
    drawn = []
    for tract in sorted(groups[DEFAULT_DOMAIN]):
        hits = sorted((Path(args.t2w_deriv_root) / args.geom_subject).rglob(f"{tract}_{args.geom_subject}_ses-V02_T2w_filtered_warped_to_ses-V03.trk"))
        if not hits:
            print(f"  {tract}: no geometry .trk for {args.geom_subject}; not drawn")
            continue
        drawn.append((tract, hits[0]))
    if not drawn:
        raise RuntimeError("no tracts drawable for geometry subject")
    missing = [t for t, _ in drawn if t not in groups["tractography"]]
    if missing:
        raise RuntimeError(f"tractography domain lacks these drawn tracts: {missing}")

    stats = domain_stats(drawn, groups)
    nsub = max(groups[DEFAULT_DOMAIN][t][2] for t, _ in drawn)

    tracts_payload = []
    for tract, trk in drawn:
        curve = mean_curve(trk)
        verts, faces, node_of_vertex = tube(curve, radius)
        curve_c = [round(v, 1) for v in curve.reshape(-1)]
        entry = {"name": tract}
        for d, key in (("anatomical", "b64"), ("tractography", "b64_tract")):
            logj, sd, n_group = groups[d][tract]
            vmax, sd_lo, sd_hi = stats[d]["vmax"], stats[d]["sdlo"], stats[d]["sdhi"]
            frac = np.clip((logj + vmax) / (2 * vmax), 0, 1)
            node_rgb = np.asarray(CMAP(frac))[:, :3]
            node_alpha = ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * np.clip((sd_hi - sd) / (sd_hi - sd_lo), 0.0, 1.0)
            entry[key] = base64.b64encode(
                make_x3d(verts, faces, node_rgb[node_of_vertex], alpha=np.round(node_alpha[node_of_vertex], 3)).encode()
            ).decode()
            entry["nodes" if d == DEFAULT_DOMAIN else "nodes_tract"] = {
                "c": curve_c,
                "r": [radius + PICK_PAD_MM] * N_NODES,
                "lj": [round(v, 4) for v in logj],
                "sd": [round(v, 4) for v in sd],
                "n": int(n_group),
                "n_nodes": N_NODES,
            }
        tracts_payload.append(entry)
        print("built tube", tract)
        del curve, verts, faces
        gc.collect()

    checkboxes = ['<label><input type="checkbox" data-tract="__brain__" checked>glass brain</label>']
    checkboxes += [f'<label><input type="checkbox" data-tract="{t}" checked>{t}</label>' for t, _ in drawn]
    radio_html = "".join(
        f'<label><input type="radio" name="dom" value="{d}"{" checked" if d == DEFAULT_DOMAIN else ""}>{DOMAIN_UI[d]}</label>'
        for d in DOMAINS
    )
    title = f"Tract morphometry in 3D (hover nodes): group-level {set_name.replace('_', ' ')} log det J ({len(drawn)} bundles, N={nsub}, anonymized anatomy)"
    metric_html = SET_SPECS[set_name][DEFAULT_DOMAIN][1]
    data = json.dumps({
        "tracts": tracts_payload,
        "brain_b64": brain_b64,
        "vmin": -stats[DEFAULT_DOMAIN]["vmax"],
        "vmax": stats[DEFAULT_DOMAIN]["vmax"],
        "sdlo": stats[DEFAULT_DOMAIN]["sdlo"],
        "sdhi": stats[DEFAULT_DOMAIN]["sdhi"],
        "alphafloor": ALPHA_FLOOR,
        "metric_set": set_name,
        "domain": DEFAULT_DOMAIN,
        "domains": {
            d: {
                "vmin": -stats[d]["vmax"],
                "vmax": stats[d]["vmax"],
                "sdlo": stats[d]["sdlo"],
                "sdhi": stats[d]["sdhi"],
                "label": SET_SPECS[set_name][d][1],
                "short": d,
            }
            for d in DOMAINS
        },
    })
    html = (
        HTML.replace("$TITLE", title)
        .replace("$NSUB", str(nsub))
        .replace("$CHECKBOXES", "\n".join(checkboxes))
        .replace("$DOMAIN_RADIOS", radio_html)
        .replace("$CSS", colorbar_css())
        .replace("$METRICLABEL", metric_html)
        .replace("$METRIC", metric_html)
        .replace("$SDLO", f"{stats[DEFAULT_DOMAIN]['sdlo']:.3f}")
        .replace("$SDHI", f"{stats[DEFAULT_DOMAIN]['sdhi']:.3f}")
        .replace("$NIIVUE_JS", patched_niivue_js(js_path))
        .replace("$DATA_JSON", data)
    )
    out = Path(out_dir) / f"fig3d_group_{set_name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def load_payload(html_path):
    html = Path(html_path).read_text(encoding="utf-8")
    for line in html.split("\n"):
        if line.startswith("const DATA = "):
            return json.loads(line[len("const DATA = "):].rstrip(";"))
    raise RuntimeError(f"DATA line not found in {html_path}")


def parse_x3d(b64):
    txt = base64.b64decode(b64).decode("utf-8")
    pts = re.search(r'<Coordinate point="([^"]*)"', txt, re.S).group(1).split()
    verts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    ci = [int(v) for v in re.search(r'coordIndex="([^"]*)"', txt, re.S).group(1).split()]
    tri = np.asarray([v for v in ci if v >= 0], dtype=np.int64).reshape(-1, 3)
    m = re.search(r'<Color rgba="([^"]*)"', txt, re.S) or re.search(r'<Color color="([^"]*)"', txt, re.S)
    if m is None:
        raise RuntimeError("no Color node found")
    vals = np.asarray(m.group(1).split(), dtype=np.float64)
    if vals.size == 4 * len(verts):
        c = vals.reshape(-1, 4)
        rgb, alpha = c[:, :3], c[:, 3]
    elif vals.size == 3 * len(verts):
        rgb = vals.reshape(-1, 3)
        alpha = np.ones(len(rgb))
    else:
        raise RuntimeError(f"unexpected color value count {vals.size} for {len(verts)} vertices")
    assert len(rgb) == len(verts)
    return verts, tri, rgb, alpha


def camera_basis(cdir, up):
    cdir = cdir / np.linalg.norm(cdir)
    fwd = -cdir
    r = np.cross(fwd, up)
    n = np.linalg.norm(r)
    if n < 1e-9:
        raise ValueError("camera direction parallel to up vector")
    r /= n
    u = np.cross(r, fwd)
    return cdir, r, u


def raster(img, verts, tri, rgb, alpha, cdir, r, u, center, scale, W, H):
    px = (verts - center) @ r
    py = (verts - center) @ u
    dep = (verts - center) @ cdir
    sx = W * 0.5 + scale * px
    sy = H * 0.5 - scale * py
    fn = np.cross(verts[tri[:, 1]] - verts[tri[:, 0]], verts[tri[:, 2]] - verts[tri[:, 0]])
    ln = np.linalg.norm(fn, axis=1)
    ok = ln > 1e-12
    fn[ok] /= ln[ok, None]
    vn = np.zeros_like(verts)
    np.add.at(vn, tri[:, 0], fn)
    np.add.at(vn, tri[:, 1], fn)
    np.add.at(vn, tri[:, 2], fn)
    vmag = np.linalg.norm(vn, axis=1)
    vn = vn / np.maximum(vmag, 1e-12)[:, None]
    light = 0.45 + 0.55 * np.abs(np.einsum("ij,j->i", vn, cdir))
    colv = np.clip(rgb * light[:, None], 0.0, 1.0)
    d = dep[tri].mean(axis=1)
    for fi in np.argsort(d):
        i, j, k = tri[fi]
        a_mean = (alpha[i] + alpha[j] + alpha[k]) / 3.0
        if a_mean <= 1e-3:
            continue
        x = np.array([sx[i], sx[j], sx[k]])
        y = np.array([sy[i], sy[j], sy[k]])
        x0 = max(int(np.floor(x.min())), 0)
        x1 = min(int(np.ceil(x.max())) + 1, W)
        y0 = max(int(np.floor(y.min())), 0)
        y1 = min(int(np.ceil(y.max())) + 1, H)
        if x0 >= x1 or y0 >= y1 or (x1 - x0) * (y1 - y0) > 120000:
            continue
        xx = np.arange(x0, x1) + 0.5
        yy = np.arange(y0, y1) + 0.5
        area = (x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0])
        if abs(area) < 1e-9:
            continue
        w1 = ((x[1] - x[0]) * (yy[:, None] - y[0]) - (y[1] - y[0]) * (xx[None, :] - x[0])) / area
        w2 = ((x[0] - x[2]) * (yy[:, None] - y[2]) - (y[0] - y[2]) * (xx[None, :] - x[2])) / area
        sel = (w1 >= 0.0) & (w2 >= 0.0) & (w1 + w2 <= 1.0)
        if not sel.any():
            continue
        w1s, w2s = w1[sel], w2[sel]
        w0s = 1.0 - w1s - w2s
        cpix = w0s[:, None] * colv[i] + w1s[:, None] * colv[j] + w2s[:, None] * colv[k]
        apix = np.clip(w0s * alpha[i] + w1s * alpha[j] + w2s * alpha[k], 0.0, 1.0)
        ry, rx = np.nonzero(sel)
        pix = img[y0 + ry, x0 + rx]
        img[y0 + ry, x0 + rx] = apix[:, None] * cpix + (1.0 - apix)[:, None] * pix


def render_view(meshes, cdir_in, up, cell_px):
    cdir, r, u = camera_basis(cdir_in, up)
    allv = np.concatenate([m[0] for m in meshes])
    center = 0.5 * (allv.min(axis=0) + allv.max(axis=0))
    pxv = (allv - center) @ r
    pyv = (allv - center) @ u
    half = max(abs(pxv).max(), abs(pyv).max(), 1e-6)
    H = W = cell_px * SS
    scale = 0.5 * RHO * W / half
    img = np.empty((H, W, 3), dtype=np.float64)
    img[:] = BG
    for verts, tri, rgb, alpha in meshes:
        raster(img, verts, tri, rgb, alpha, cdir, r, u, center, scale, W, H)
    return img.reshape(cell_px, SS, cell_px, SS, 3).mean(axis=(1, 3))


def png_b64(img):
    from matplotlib import pyplot as plt

    buf = io.BytesIO()
    plt.imsave(buf, np.clip(img, 0, 1), format="png", dpi=200)
    return base64.b64encode(buf.getvalue()).decode()


def verify_color(verts, tri, rgb, alpha, cdir_in, up):
    """Front-most near-opaque face must show its own color at its centroid."""
    cdir, r, u = camera_basis(cdir_in, up)
    center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    H = W = CELL_STATIC * SS
    scale = 0.5 * RHO * W / max(abs((verts - center) @ r).max(), abs((verts - center) @ u).max())
    img = np.zeros((H, W, 3))
    raster(img, verts, tri, rgb, alpha, cdir, r, u, center, scale, W, H)
    sx = W * 0.5 + scale * ((verts - center) @ r)
    sy = H * 0.5 - scale * ((verts - center) @ u)
    dep = (verts - center) @ cdir
    fn = np.cross(verts[tri[:, 1]] - verts[tri[:, 0]], verts[tri[:, 2]] - verts[tri[:, 0]])
    d = dep[tri].mean(axis=1)
    a = alpha[tri].mean(axis=1)
    for fi in np.argsort(d)[::-1]:
        if a[fi] < 0.93:
            continue
        cxp = int(sx[tri[fi]].mean())
        cyp = int(sy[tri[fi]].mean())
        if not (0 <= cxp < W and 0 <= cyp < H):
            continue
        i, j, k = tri[fi]
        fnn = fn[fi] / max(np.linalg.norm(fn[fi]), 1e-12)
        light = 0.45 + 0.55 * abs(fnn @ cdir)
        expected = np.clip(rgb[tri[fi]].mean(axis=0) * light, 0, 1)
        got = img[cyp, cxp]
        err = float(np.max(np.abs(got - expected)))
        assert err < 0.14, f"face color check failed: {got} vs {expected} (err {err:.3f})"
        return True
    raise AssertionError("no near-opaque front face to verify")


def compose_static_grid(shots, view_names, pairs, out_path, vmax, sd_lo, sd_hi, nsub, metric_label):
    import base64 as _b64

    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "white",
        "axes.linewidth": 0.0, "axes.grid": False, "xtick.color": "0.25", "ytick.color": "0.25",
        "text.color": "0.15", "axes.labelcolor": "0.20", "axes.titlecolor": "0.15",
        "font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 9, "savefig.facecolor": "white", "savefig.edgecolor": "white",
    })

    cell = 1.42
    gap = 0.07
    label_w = 0.60
    cbar_gap, cbar_w, cbar_lab = 0.14, 0.09, 0.60
    title_h, head_h, leg_h = 0.86, 0.24, 0.55
    top_m, bot_m = 0.06, 0.08
    W = label_w + 3 * cell + 2 * gap + cbar_gap + cbar_w + cbar_lab
    H = top_m + title_h + head_h + len(pairs) * cell + (len(pairs) - 1) * gap + leg_h + bot_m
    fig = Figure(figsize=(W, H))

    def rect(x, y, w, h):
        return [x / W, y / H, w / W, h / H]

    def minimal_colorbar(cbar):
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0, labelsize=6.5, colors="0.30")
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

    grid_top = H - (top_m + title_h + head_h)
    for ri, (fam, names) in enumerate(pairs):
        for vi in range(3):
            x = label_w + vi * (cell + gap)
            y = grid_top - (ri + 1) * cell - ri * gap
            ax = fig.add_axes(rect(x, y, cell, cell))
            arr = plt.imread(io.BytesIO(_b64.b64decode(shots[(fam, vi)])))
            ax.imshow(arr, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("0.85")
                spine.set_linewidth(0.6)
        yc = grid_top - ri * (cell + gap) - cell / 2
        fig.text((label_w - 0.06) / W, (yc + 0.08) / H, fam, ha="right", va="center", fontsize=9, fontweight="bold")
        fig.text((label_w - 0.06) / W, (yc - 0.12) / H, "L + R", ha="right", va="center", fontsize=7, color="0.35")

    for vi, vn in enumerate(view_names):
        x = label_w + vi * (cell + gap) + cell / 2
        fig.text(x / W, (grid_top + 0.05) / H, vn, ha="center", va="bottom", fontsize=8, color="0.30")

    texts = [
        fig.text(label_w / W, (H - top_m - 0.09) / H,
                 "Bilateral tract morphometry in glass brain", fontsize=8.5, fontweight="bold", ha="left", va="top"),
        fig.text(label_w / W, (H - top_m - 0.28) / H,
                 f"Anatomical {metric_label} log det $J$ group means, ses-V02\u2192ses-V03 (N={nsub}; anonymized anatomy)",
                 fontsize=7.6, ha="left", va="top", color="0.20"),
        fig.text(label_w / W, (H - top_m - 0.47) / H,
                 "color = mean log det $J$ \u00b7 fade = between-subject SD \u00b7 constant tube width",
                 fontsize=7, ha="left", va="top", color="0.35"),
    ]

    gx0 = label_w + 3 * cell + 2 * gap + cbar_gap
    gy0 = grid_top - len(pairs) * cell - (len(pairs) - 1) * gap
    cax = fig.add_axes(rect(gx0, gy0, cbar_w, grid_top - gy0))
    col_norm = Normalize(vmin=-vmax, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=col_norm)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"group mean {metric_label} log det $J$", fontsize=7.5, color="0.25", labelpad=8)
    minimal_colorbar(cbar)

    tax = fig.add_axes(rect(label_w, bot_m + 0.12, 2.30, 0.10))
    alphas = np.linspace(1.0, ALPHA_FLOOR, 256).reshape(1, -1)
    gray = np.ones((1, 256, 4)) * np.array([0.55, 0.55, 0.55, 1.0])
    gray[..., 3] = alphas
    tax.imshow(gray, aspect="auto", extent=[sd_lo, sd_hi, -0.5, 0.5])
    tax.set_yticks([])
    tax.set_xticks([sd_lo, sd_hi])
    tax.set_xticklabels([f"{sd_lo:.3f}", f"{sd_hi:.3f}"], fontsize=6.5, color="0.30")
    for spine in tax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.6")
        spine.set_linewidth(0.5)
    tax.tick_params(length=2, colors="0.4")
    tax.set_title("between-subject SD of logJ:  opaque (agreement) \u2192 faded ($\\alpha$ 1.0 \u2192 0.15)",
                  fontsize=6.8, color="0.25", pad=3)

    fig.canvas.draw()
    lim = fig.get_size_inches()[0] * fig.dpi - 3
    for t in texts:
        for _ in range(6):
            bb = t.get_window_extent()
            if bb.x1 <= lim:
                break
            t.set_fontsize(t.get_fontsize() * 0.93)
    fig.savefig(out_path, dpi=220)
    return out_path


def render_statics(set_name, html_path, out_dir):
    t0 = time.time()
    data = load_payload(html_path)
    metric_label = STATIC_LABELS[set_name]
    by_name = {t["name"]: t for t in data["tracts"]}
    families = []
    for fam in sorted({n.split("_")[0] for n in by_name}):
        pair = [f"{fam}_L", f"{fam}_R"]
        if all(p in by_name for p in pair):
            families.append((fam, pair))
    if not families:
        raise RuntimeError("no bilateral pairs found in payload")

    brain = parse_x3d(data["brain_b64"])
    brain = (brain[0], brain[1], np.full((len(brain[0]), 3), 210 / 255.0), np.full(len(brain[0]), 0.15))
    tract_cache = {t["name"]: parse_x3d(t["b64"]) for t in data["tracts"]}

    first_pair = families[0][1][0]
    vL, fL, cL, aL = tract_cache[first_pair]
    assert verify_color(vL, fL, cL, aL, *VIEWS[2][1:]), "color verification failed"
    print("per-pixel color fidelity check passed (front opaque face matches CMAP color)")

    names = [v[0] for v in VIEWS]
    shots = {}
    for fam, pair in families:
        meshes = [brain] + [tract_cache[n] for n in pair]
        bg_only = [brain]
        for vi, (vname, cdir, up) in enumerate(VIEWS):
            img = render_view(meshes, cdir, up, CELL_STATIC)
            img_bg = render_view(bg_only, cdir, up, CELL_STATIC)
            cov = float(np.mean(np.abs(img - img_bg).sum(axis=2) > 0.05))
            chroma_px = int(((img.max(axis=2) - img.min(axis=2)) > 0.12).sum())
            assert not np.isnan(img).any(), f"NaN pixels in {fam} {vname}"
            assert 0.002 < cov < 0.45, f"coverage {cov:.3f} outside sane range for {fam} {vname}"
            assert chroma_px >= 200, f"too few colored tract pixels for {fam} {vname}: {chroma_px}"
            if fam == families[0][0] and vname == "sagittal":
                ys, xs = np.where((img.max(axis=2) - img.min(axis=2)) > 0.12)
                wbox, hbox = xs.max() - xs.min(), ys.max() - ys.min()
                if hbox > 0 and wbox >= hbox * 0.95:
                    print(f"WARNING sagittal bbox w/h {wbox/hbox:.2f} (expected roughly vertical)")
            if fam == families[0][0] and vi == 0:
                img2 = render_view(meshes, cdir, up, CELL_STATIC)
                assert hashlib.md5(img2.tobytes()).hexdigest() == hashlib.md5(img.tobytes()).hexdigest(), "non-deterministic render"
            shots[(fam, vi)] = png_b64(img)
            print(f"{fam:5} {vname:9}: coverage {cov * 100:.2f}%  chroma px {chroma_px}  [{time.time() - t0:.0f}s]")

    nsub = max(t["nodes"]["n"] for t in data["tracts"])
    out = Path(out_dir) / f"fig3d_group_{set_name}_static.png"
    compose_static_grid(shots, names, families, out, float(data["vmax"]),
                        float(data["sdlo"]), float(data["sdhi"]), int(nsub), metric_label)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")


def rot_cameras(n, el_deg=15.0):
    cams = []
    for az in 360.0 * np.arange(n) / n:
        a = np.deg2rad(az)
        e = np.deg2rad(el_deg)
        cams.append(np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)]))
    return cams


def scene_render(meshes, cdir, up, fit, cell):
    cdir_n, r, u = camera_basis(cdir, up)
    center, half = fit
    H = W = cell * SS
    scale = 0.5 * RHO * W / half
    img = np.empty((H, W, 3))
    img[:] = BG
    offs, allv, alltri, allrgb, alla = 0, [], [], [], []
    for verts, tri, rgb, alpha in meshes:
        allv.append(verts)
        alltri.append(tri + offs)
        offs += len(verts)
        allrgb.append(rgb)
        alla.append(alpha)
    raster(img, np.concatenate(allv), np.concatenate(alltri),
           np.concatenate(allrgb), np.concatenate(alla), cdir_n, r, u, center, scale, W, H)
    return img.reshape(cell, SS, cell, SS, 3).mean(axis=(1, 3))


def fit_rotation(tubes, brain, cams):
    allv = np.concatenate([brain[0]] + [t[0] for t in tubes])
    center = 0.5 * (allv.min(axis=0) + allv.max(axis=0))
    half = 0.0
    for cdir in cams:
        _, r, u = camera_basis(cdir, np.array([0.0, 0.0, 1.0]))
        half = max(half, abs((allv - center) @ r).max(), abs((allv - center) @ u).max())
    return center, half * 1.005


def annotate(img, text, cell):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    path = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    font = ImageFont.load_default()
    try:
        for size in range(17, 7, -1):
            font = ImageFont.truetype(path, size)
            bb = d.textbbox((0, 0), text, font=font)
            if bb[2] - bb[0] <= cell - 24:
                break
    except OSError:
        pass
    bb = d.textbbox((0, 0), text, font=font)
    th = bb[3] - bb[1]
    d.rectangle([0, cell - th - 20, cell, cell], fill=(0, 0, 0, 150))
    d.text((cell / 2, cell - 11), text, font=font, fill=(255, 255, 255, 230), anchor="mb")
    return np.asarray(im.convert("RGB")).astype(np.float64) / 255.0


def save_frame(img, path):
    from PIL import Image

    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def encode(frames_dir, mp4_path, gif_path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(frames_dir / "f%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", str(mp4_path)], check=True)
    pal = Path(frames_dir).parent / "palette.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(frames_dir / "f%03d.png"), "-vf", "fps=12,scale=420:-1:flags=lanczos,palettegen",
                    str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", str(frames_dir / "f%03d.png"), "-i", str(pal),
                    "-lavfi", "fps=12,scale=420:-1:flags=lanczos [x]; [x][1:v] paletteuse",
                    str(gif_path)], check=True)


def render_anim(set_name, html_path, out_dir, n_frames):
    t0 = time.time()
    data = load_payload(html_path)
    brain = parse_x3d(data["brain_b64"])
    brain = (brain[0], brain[1], np.full((len(brain[0]), 3), 210 / 255.0), np.full(len(brain[0]), 0.15))
    tubes = [parse_x3d(t["b64"]) for t in data["tracts"]]
    nsub = max(t["nodes"]["n"] for t in data["tracts"])
    print(f"{len(tubes)} real-data tubes loaded; N={nsub} group")

    up = np.array([0.0, 0.0, 1.0])
    frames_dir = Path(out_dir) / f"anim_frames_{set_name}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cams = rot_cameras(n_frames)
    fit = fit_rotation(tubes, brain, cams)
    label = f"group-level {set_name.replace('_', ' ')} log det J morphometry (N={nsub}; color & transparency as in interactive figure)"

    areas = []
    for f, cam in enumerate(cams):
        img = scene_render([brain] + tubes, cam, up, fit, CELL_ANIM)
        areas.append(float(np.mean(np.abs(img - BG).sum(2) > 0.05)))
        if f == 0:
            h = hashlib.md5(scene_render([brain] + tubes, cam, up, fit, CELL_ANIM).tobytes()).hexdigest()
            assert h == hashlib.md5(img.tobytes()).hexdigest(), "non-deterministic render"
        save_frame(annotate(img, label, CELL_ANIM), frames_dir / f"f{f:03d}.png")
        if f % 30 == 0:
            print(f"frame {f}/{n_frames} [{time.time()-t0:.0f}s]")

    a = np.asarray(areas)
    jump = float(np.max(np.abs(np.diff(np.concatenate([a, a[:1]]))) / a.mean()))
    if jump >= 0.12:
        print(f"WARNING: silhouette step {jump*100:.1f}% of mean area despite frozen fit")

    mp4 = Path(out_dir) / f"fig3d_{set_name}_orbit.mp4"
    gif = Path(out_dir) / f"fig3d_{set_name}_orbit.gif"
    encode(frames_dir, mp4, gif)
    print(f"encoded {mp4.name}, {gif.name} in {time.time()-t0:.0f} s")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js_path = args.niivue_js or str(Path(__file__).resolve().parent / "niivue.umd.js")
    if not Path(js_path).is_file():
        raise SystemExit(f"niivue.umd.js not found: {js_path}")

    subjects = args.subjects
    if not subjects:
        subjects = sorted(p.name for p in Path(args.t2w_deriv_root).iterdir() if p.is_dir() and p.name.startswith("sub-"))
    print(f"group subjects: {subjects}")

    mask_path = (
        Path(args.preproc_t2w_root) / args.geom_subject / "ses-V03"
        / f"{args.geom_subject}_ses-V03_T2w_RAS_synthstrip-mask-bin_fillholes.nii.gz"
    )
    if not mask_path.is_file():
        raise SystemExit(f"glass-brain mask not found: {mask_path}")

    t0 = time.time()
    brain_verts, brain_polys = brain_mesh(mask_path)
    print(f"glass brain mesh: {len(brain_verts)} verts, {len(brain_polys)} faces [{time.time()-t0:.0f}s]")
    brain_b64 = base64.b64encode(
        make_x3d(brain_verts, brain_polys, np.tile(np.array([0.80, 0.80, 0.82]), (len(brain_verts), 1))).encode()
    ).decode()

    for set_name in args.metric_sets:
        html_path = None
        if "html" in args.stages:
            html_path = build_html(set_name, args, subjects, out_dir, js_path, brain_b64)
        if "statics" in args.stages:
            html_path = html_path or next(iter(sorted(out_dir.glob(f"fig3d_group_{set_name}.html"))), None)
            if html_path is None:
                raise SystemExit(f"{set_name}: run the html stage first (missing fig3d_group_{set_name}.html)")
            render_statics(set_name, html_path, out_dir)
        if "anim" in args.stages:
            html_path = html_path or next(iter(sorted(out_dir.glob(f"fig3d_group_{set_name}.html"))), None)
            if html_path is None:
                raise SystemExit(f"{set_name}: run the html stage first (missing fig3d_group_{set_name}.html)")
            render_anim(set_name, html_path, out_dir, args.frames)


if __name__ == "__main__":
    main()

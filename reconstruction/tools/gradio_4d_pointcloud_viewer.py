#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from plyfile import PlyData
import trimesh

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_SEQUENCE_ROOT = REPOSITORY_ROOT / "output/track3_run"
DEFAULT_PLAYBACK_FPS = 6.0
MAX_PLY_CACHE_ENTRIES = 8
PALETTE = np.asarray(
    [
        [239, 83, 80],
        [66, 165, 245],
        [102, 187, 106],
        [255, 202, 40],
        [171, 71, 188],
        [255, 112, 67],
        [38, 198, 218],
        [236, 64, 122],
    ],
    dtype=np.uint8,
)
DEMO_CSS = """
#viewer-shell { max-width: 1500px; margin: 0 auto; }
#pointcloud-viewer { border-radius: 14px; overflow: hidden; }
#source-video-player video, #sam3-tracking-player video {
  object-fit: contain; background: #07090d; pointer-events: none;
}
#source-video-player .controls, #sam3-tracking-player .controls { display: none !important; }
"""

_PLY_CACHE: OrderedDict[tuple[str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray | None]] = (
    OrderedDict()
)
_CACHE_LOCK = threading.Lock()
AXIS_COLORS = np.asarray([[255, 66, 66], [67, 227, 111], [67, 133, 255]], dtype=np.uint8)
FOUNDATIONPOSE_BBOX_COLOR = np.asarray([255, 183, 45], dtype=np.uint8)
CAMERA_FRUSTUM_COLOR = np.asarray([65, 220, 255], dtype=np.uint8)
HAND_COLORS = np.asarray([[255, 176, 72], [86, 232, 133]], dtype=np.uint8)
PLANE_AXIS_MODE = "Largest plane in frame 0 (recommended)"
WORLD_AXIS_MODE = "Pi3 world coordinates"
ORIGINAL_RGB_MODE = "Original RGB"
FOLDER_COLOR_MODE = "Color by sequence"
TRACK_COLOR_MODE = "Color by track ID"
POINT_CLOUD_REPRESENTATION = "Object point cloud"
TEXTURED_MESH_REPRESENTATION = "Aligned textured mesh"
POINT_CLOUD_AND_MESH_REPRESENTATION = "Object point cloud + aligned textured mesh"


POINTCLOUD_HTML = """
<div class="pc-shell">
  <canvas class="pc-canvas"></canvas>
  <div class="pc-axis-legend"><span class="pc-x">X</span><span class="pc-y">Y</span><span class="pc-z">Z</span><span class="pc-bbox">BBOX</span><span class="pc-camera">CAM</span></div>
  <div class="pc-help">Left drag: rotate · Shift/right drag: pan · Wheel: zoom · Double-click: reset</div>
  <div class="pc-message">Waiting for a scene</div>
</div>
"""

POINTCLOUD_CSS = """
.pc-shell { position: relative; width: 100%; height: 720px; overflow: hidden;
  border-radius: 14px; background: #07090d; touch-action: none; }
.pc-canvas { width: 100%; height: 100%; display: block; outline: none; }
.pc-help { position: absolute; left: 14px; bottom: 12px; padding: 6px 9px;
  color: #c7d0dc; background: rgba(8, 12, 18, .68); border-radius: 7px;
  font-size: 12px; pointer-events: none; }
.pc-message { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
  color: #dce6f2; background: rgba(8, 12, 18, .75); border-radius: 8px;
  padding: 8px 12px; pointer-events: none; }
.pc-axis-legend { position: absolute; right: 14px; top: 12px; display: flex; gap: 9px;
  padding: 6px 9px; background: rgba(8, 12, 18, .68); border-radius: 7px;
  font: 700 13px monospace; pointer-events: none; }
.pc-x { color: #ff4242; } .pc-y { color: #43e36f; } .pc-z { color: #4385ff; }
.pc-bbox { color: #ffb72d; }
.pc-camera { color: #41dcff; }
"""

POINTCLOUD_JS = r"""
const canvas = element.querySelector('.pc-canvas');
const message = element.querySelector('.pc-message');
const gl = canvas.getContext('webgl', {antialias: true, alpha: false});
if (!gl) {
  message.textContent = 'WebGL is not available in this browser';
  throw new Error('WebGL is unavailable');
}

const vertexSource = `
attribute vec3 aPosition;
attribute vec3 aColor;
uniform float uYaw;
uniform float uPitch;
uniform float uDistance;
uniform vec2 uPan;
uniform float uAspect;
uniform float uPointSize;
varying vec3 vColor;
void main() {
  // Pi3/Pi3X uses OpenCV camera axes (right, down, forward), whereas the
  // viewer looks down WebGL's -Z axis (right, up, backward). Flipping both
  // Y and Z is a proper rotation and preserves left/right handedness.
  vec3 p = vec3(aPosition.x, -aPosition.y, -aPosition.z);
  float cy = cos(uYaw), sy = sin(uYaw);
  p = vec3(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);
  float cp = cos(uPitch), sp = sin(uPitch);
  p = vec3(p.x, cp * p.y - sp * p.z, sp * p.y + cp * p.z);
  p.xy += uPan;
  float depth = max(0.02, uDistance - p.z);
  float tanHalfFov = 0.46630766;
  float nearPlane = 0.01, farPlane = 100.0;
  float clipZ = ((farPlane + nearPlane) / (farPlane - nearPlane)) * depth
              - (2.0 * farPlane * nearPlane / (farPlane - nearPlane));
  gl_Position = vec4(p.x / (uAspect * tanHalfFov), p.y / tanHalfFov, clipZ, depth);
  gl_PointSize = uPointSize * clamp(4.0 / depth, 0.75, 2.5);
  vColor = aColor;
}`;
const fragmentSource = `
precision mediump float;
varying vec3 vColor;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(vColor, 1.0);
}`;
const lineFragmentSource = `
precision mediump float;
varying vec3 vColor;
void main() { gl_FragColor = vec4(vColor, 1.0); }`;
const meshVertexSource = `
attribute vec3 aPosition;
attribute vec3 aNormal;
attribute vec2 aTexCoord;
uniform mat4 uModel;
uniform float uYaw;
uniform float uPitch;
uniform float uDistance;
uniform vec2 uPan;
uniform float uAspect;
varying vec2 vTexCoord;
varying vec3 vNormal;
void main() {
  vec3 p = (uModel * vec4(aPosition, 1.0)).xyz;
  vec3 n = normalize(mat3(uModel) * aNormal);
  p = vec3(p.x, -p.y, -p.z);
  n = vec3(n.x, -n.y, -n.z);
  float cy = cos(uYaw), sy = sin(uYaw);
  p = vec3(cy * p.x + sy * p.z, p.y, -sy * p.x + cy * p.z);
  n = vec3(cy * n.x + sy * n.z, n.y, -sy * n.x + cy * n.z);
  float cp = cos(uPitch), sp = sin(uPitch);
  p = vec3(p.x, cp * p.y - sp * p.z, sp * p.y + cp * p.z);
  n = vec3(n.x, cp * n.y - sp * n.z, sp * n.y + cp * n.z);
  p.xy += uPan;
  float depth = max(0.02, uDistance - p.z);
  float tanHalfFov = 0.46630766;
  float nearPlane = 0.01, farPlane = 100.0;
  float clipZ = ((farPlane + nearPlane) / (farPlane - nearPlane)) * depth
              - (2.0 * farPlane * nearPlane / (farPlane - nearPlane));
  gl_Position = vec4(p.x / (uAspect * tanHalfFov), p.y / tanHalfFov, clipZ, depth);
  vTexCoord = aTexCoord;
  vNormal = normalize(n);
}`;
const meshFragmentSource = `
precision mediump float;
varying vec2 vTexCoord;
varying vec3 vNormal;
uniform sampler2D uTexture;
uniform float uLightingStrength;
uniform float uOpacity;
void main() {
  vec4 texel = texture2D(uTexture, vTexCoord);
  vec3 n = normalize(vNormal);
  if (!gl_FrontFacing) n = -n;
  vec3 lightDirection = normalize(vec3(-0.45, 0.70, 0.55));
  float diffuse = max(dot(n, lightDirection), 0.0);
  float fill = 0.18 * max(dot(n, -lightDirection), 0.0);
  float hemisphere = 0.5 + 0.5 * n.y;
  float specular = 0.16 * pow(max(dot(reflect(-lightDirection, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0);
  float illumination = 0.28 + 0.58 * diffuse + fill + 0.14 * hemisphere;
  vec3 shaded = texel.rgb * illumination + vec3(specular);
  vec3 color = mix(texel.rgb, shaded, clamp(uLightingStrength, 0.0, 1.0));
  gl_FragColor = vec4(color, texel.a * clamp(uOpacity, 0.0, 1.0));
}`;

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}
function makeProgram(fragment, vertex = vertexSource) {
  const result = gl.createProgram();
  gl.attachShader(result, compileShader(gl.VERTEX_SHADER, vertex));
  gl.attachShader(result, compileShader(gl.FRAGMENT_SHADER, fragment));
  gl.linkProgram(result);
  if (!gl.getProgramParameter(result, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(result));
  return result;
}
function getLocations(targetProgram) {
  return {
    position: gl.getAttribLocation(targetProgram, 'aPosition'),
    color: gl.getAttribLocation(targetProgram, 'aColor'),
    yaw: gl.getUniformLocation(targetProgram, 'uYaw'),
    pitch: gl.getUniformLocation(targetProgram, 'uPitch'),
    distance: gl.getUniformLocation(targetProgram, 'uDistance'),
    pan: gl.getUniformLocation(targetProgram, 'uPan'),
    aspect: gl.getUniformLocation(targetProgram, 'uAspect'),
    pointSize: gl.getUniformLocation(targetProgram, 'uPointSize')
  };
}
const pointProgram = makeProgram(fragmentSource);
const lineProgram = makeProgram(lineFragmentSource);
const meshProgram = makeProgram(meshFragmentSource, meshVertexSource);
const pointLocations = getLocations(pointProgram);
const lineLocations = getLocations(lineProgram);
const meshLocations = {
  position: gl.getAttribLocation(meshProgram, 'aPosition'),
  normal: gl.getAttribLocation(meshProgram, 'aNormal'),
  texCoord: gl.getAttribLocation(meshProgram, 'aTexCoord'),
  model: gl.getUniformLocation(meshProgram, 'uModel'),
  texture: gl.getUniformLocation(meshProgram, 'uTexture'),
  lightingStrength: gl.getUniformLocation(meshProgram, 'uLightingStrength'),
  opacity: gl.getUniformLocation(meshProgram, 'uOpacity'),
  yaw: gl.getUniformLocation(meshProgram, 'uYaw'),
  pitch: gl.getUniformLocation(meshProgram, 'uPitch'),
  distance: gl.getUniformLocation(meshProgram, 'uDistance'),
  pan: gl.getUniformLocation(meshProgram, 'uPan'),
  aspect: gl.getUniformLocation(meshProgram, 'uAspect')
};
function makeCloud() {
  return {position: gl.createBuffer(), color: gl.createBuffer(), count: 0, visible: true};
}
const staticCloud = makeCloud();
const dynamicCloud = makeCloud();
const axisCloud = makeCloud();
const meshClouds = [];

function makeMeshCloud() {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
    new Uint8Array([210, 210, 210, 255])
  );
  return {
    position: gl.createBuffer(), normal: gl.createBuffer(), texCoord: gl.createBuffer(), texture,
    textureSource: '', count: 0, visible: false,
    lightingStrength: 0.45, opacity: 1.0,
    model: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
  };
}

function decodeBytes(encoded) {
  if (!encoded) return new Uint8Array(0);
  const raw = atob(encoded);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}
function uploadCloud(target, cloud) {
  const positionBytes = decodeBytes(cloud?.positions || '');
  const colorBytes = decodeBytes(cloud?.colors || '');
  const positions = new Float32Array(
    positionBytes.buffer, positionBytes.byteOffset, Math.floor(positionBytes.byteLength / 4)
  );
  gl.bindBuffer(gl.ARRAY_BUFFER, target.position);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, target.color);
  gl.bufferData(gl.ARRAY_BUFFER, colorBytes, gl.STATIC_DRAW);
  target.count = Number(cloud?.count || 0);
}
function uploadMesh(meshCloud, mesh) {
  if (mesh.positions !== undefined) {
    const positionBytes = decodeBytes(mesh.positions || '');
    const positions = new Float32Array(
      positionBytes.buffer, positionBytes.byteOffset, Math.floor(positionBytes.byteLength / 4)
    );
    gl.bindBuffer(gl.ARRAY_BUFFER, meshCloud.position);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    const normalBytes = decodeBytes(mesh.normals || '');
    let normals = new Float32Array(
      normalBytes.buffer, normalBytes.byteOffset, Math.floor(normalBytes.byteLength / 4)
    );
    if (normals.length !== positions.length) {
      normals = new Float32Array(positions.length);
      for (let index = 2; index < normals.length; index += 3) normals[index] = 1.0;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, meshCloud.normal);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);
    const texCoordBytes = decodeBytes(mesh.texcoords || '');
    const texCoords = new Float32Array(
      texCoordBytes.buffer, texCoordBytes.byteOffset, Math.floor(texCoordBytes.byteLength / 4)
    );
    gl.bindBuffer(gl.ARRAY_BUFFER, meshCloud.texCoord);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
    meshCloud.count = Number(mesh.count || 0);
  }
  if (mesh.lighting_strength !== undefined) {
    meshCloud.lightingStrength = Math.min(1.0, Math.max(0.0, Number(mesh.lighting_strength)));
  }
  if (mesh.opacity !== undefined) {
    meshCloud.opacity = Math.min(1.0, Math.max(0.0, Number(mesh.opacity)));
  }
  if (Array.isArray(mesh.model) && mesh.model.length === 16) {
    meshCloud.model.set(mesh.model.map(Number));
  }
  meshCloud.visible = Boolean(mesh.visible) && meshCloud.count > 0;
  if (mesh.texture && mesh.texture !== meshCloud.textureSource) {
    meshCloud.textureSource = mesh.texture;
    const textureImage = new Image();
    textureImage.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, meshCloud.texture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, textureImage);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    };
    textureImage.src = mesh.texture;
  }
}
function uploadMeshes(meshes) {
  const values = Array.isArray(meshes) ? meshes : [];
  while (meshClouds.length < values.length) meshClouds.push(makeMeshCloud());
  for (let index = 0; index < meshClouds.length; index++) {
    if (index < values.length && values[index]) uploadMesh(meshClouds[index], values[index]);
    else meshClouds[index].visible = false;
  }
}

let yaw = 0.0, pitch = 0.0, distance = 4.0, panX = 0.0, panY = 0.0;
function resetCamera() { yaw = 0.0; pitch = 0.0; distance = 4.0; panX = 0.0; panY = 0.0; }

function seekSynchronizedVideo(selector, target, retry = 0) {
  if (target === null || target === undefined || !Number.isFinite(Number(target))) return;
  const video = document.querySelector(`${selector} video`);
  if (!video) {
    if (retry < 40) setTimeout(() => seekSynchronizedVideo(selector, target, retry + 1), 50);
    return;
  }
  video.__pointcloudTargetTime = Number(target);
  if (!video.__pointcloudSeekPump) {
    video.__pointcloudSeekPump = () => {
      if (video.seeking || video.readyState < 1) return;
      const duration = Number.isFinite(video.duration) ? video.duration : Infinity;
      const requested = Number(video.__pointcloudTargetTime);
      const clamped = Math.max(0, Math.min(requested, Math.max(0, duration - 0.001)));
      if (!video.paused) video.pause();
      if (Math.abs(video.currentTime - clamped) > 0.002) video.currentTime = clamped;
    };
    video.addEventListener('seeked', video.__pointcloudSeekPump);
    video.addEventListener('loadedmetadata', video.__pointcloudSeekPump);
  }
  // Do not cancel an in-flight seek on every timer tick. The seeked listener
  // consumes the latest target immediately after the current decoder request.
  video.__pointcloudSeekPump();
}

let lastPayload = '';
function applyPayload() {
  const raw = props.value || '';
  if (!raw || raw === lastPayload) return;
  lastPayload = raw;
  try {
    const payload = JSON.parse(raw);
    if (payload.type === 'init') {
      uploadCloud(staticCloud, payload.static);
      uploadCloud(dynamicCloud, payload.dynamic);
      uploadCloud(axisCloud, payload.axes);
      uploadMeshes(payload.meshes || (payload.mesh ? [payload.mesh] : []));
      resetCamera();
    } else if (payload.type === 'frame') {
      // The old dynamic buffer stays visible until this synchronous upload completes.
      uploadCloud(dynamicCloud, payload.dynamic);
      uploadCloud(axisCloud, payload.axes);
      uploadMeshes(payload.meshes || (payload.mesh ? [payload.mesh] : []));
    }
    staticCloud.visible = payload.show_static !== false;
    seekSynchronizedVideo('#source-video-player', payload.source_video_position);
    seekSynchronizedVideo('#sam3-tracking-player', payload.tracking_video_position);
    message.style.display = 'none';
  } catch (error) {
    console.error(error);
    message.textContent = `Failed to parse scene data: ${error.message}`;
    message.style.display = 'block';
  }
}
watch('value', applyPayload);
applyPayload();

function bindCloud(cloud, pointSize) {
  if (!cloud.visible || !cloud.count) return;
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.position);
  gl.enableVertexAttribArray(pointLocations.position);
  gl.vertexAttribPointer(pointLocations.position, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.color);
  gl.enableVertexAttribArray(pointLocations.color);
  gl.vertexAttribPointer(pointLocations.color, 3, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.uniform1f(pointLocations.pointSize, pointSize);
  gl.drawArrays(gl.POINTS, 0, cloud.count);
}
function setViewUniforms(locations) {
  gl.uniform1f(locations.yaw, yaw);
  gl.uniform1f(locations.pitch, pitch);
  gl.uniform1f(locations.distance, distance);
  gl.uniform2f(locations.pan, panX, panY);
  gl.uniform1f(locations.aspect, canvas.width / canvas.height);
  if (locations.pointSize !== undefined && locations.pointSize !== null) {
    gl.uniform1f(locations.pointSize, 1.0);
  }
}
function bindAxes() {
  if (!axisCloud.count) return;
  gl.useProgram(lineProgram);
  setViewUniforms(lineLocations);
  gl.bindBuffer(gl.ARRAY_BUFFER, axisCloud.position);
  gl.enableVertexAttribArray(lineLocations.position);
  gl.vertexAttribPointer(lineLocations.position, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, axisCloud.color);
  gl.enableVertexAttribArray(lineLocations.color);
  gl.vertexAttribPointer(lineLocations.color, 3, gl.UNSIGNED_BYTE, true, 0, 0);
  gl.lineWidth(3.0);
  gl.drawArrays(gl.LINES, 0, axisCloud.count);
}
function bindMesh(meshCloud) {
  if (!meshCloud.visible || !meshCloud.count) return;
  gl.useProgram(meshProgram);
  setViewUniforms(meshLocations);
  gl.uniformMatrix4fv(meshLocations.model, false, meshCloud.model);
  gl.uniform1f(meshLocations.lightingStrength, meshCloud.lightingStrength);
  gl.uniform1f(meshLocations.opacity, meshCloud.opacity);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, meshCloud.texture);
  gl.uniform1i(meshLocations.texture, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, meshCloud.position);
  gl.enableVertexAttribArray(meshLocations.position);
  gl.vertexAttribPointer(meshLocations.position, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, meshCloud.normal);
  gl.enableVertexAttribArray(meshLocations.normal);
  gl.vertexAttribPointer(meshLocations.normal, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, meshCloud.texCoord);
  gl.enableVertexAttribArray(meshLocations.texCoord);
  gl.vertexAttribPointer(meshLocations.texCoord, 2, gl.FLOAT, false, 0, 0);
  if (meshCloud.opacity < 0.999) {
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  } else {
    gl.disable(gl.BLEND);
  }
  gl.enable(gl.CULL_FACE);
  gl.cullFace(gl.BACK);
  gl.drawArrays(gl.TRIANGLES, 0, meshCloud.count);
  gl.disable(gl.CULL_FACE);
  gl.disable(gl.BLEND);
}
function bindMeshes() {
  for (const meshCloud of meshClouds) bindMesh(meshCloud);
}
function resize() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  gl.viewport(0, 0, width, height);
}
function draw() {
  if (!element.isConnected) return;
  resize();
  gl.clearColor(0.025, 0.03, 0.04, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.useProgram(pointProgram);
  setViewUniforms(pointLocations);
  bindCloud(staticCloud, 1.8);
  bindCloud(dynamicCloud, 3.0);
  bindMeshes();
  bindAxes();
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

let dragging = false, panMode = false, lastX = 0, lastY = 0;
canvas.addEventListener('contextmenu', event => event.preventDefault());
canvas.addEventListener('pointerdown', event => {
  dragging = true; panMode = event.shiftKey || event.button === 2;
  lastX = event.clientX; lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', event => {
  if (!dragging) return;
  const dx = event.clientX - lastX, dy = event.clientY - lastY;
  lastX = event.clientX; lastY = event.clientY;
  if (panMode) {
    panX += dx * 0.0025 * distance;
    panY -= dy * 0.0025 * distance;
  } else {
    yaw += dx * 0.008;
    pitch = Math.max(-1.5, Math.min(1.5, pitch + dy * 0.008));
  }
});
canvas.addEventListener('pointerup', event => {
  dragging = false; canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener('pointercancel', () => { dragging = false; });
canvas.addEventListener('wheel', event => {
  event.preventDefault();
  distance = Math.max(1.15, Math.min(30.0, distance * Math.exp(event.deltaY * 0.001)));
}, {passive: false});
canvas.addEventListener('dblclick', resetCamera);
"""


class PointCloudCanvas(gr.HTML):
    """Persistent WebGL point-cloud canvas whose buffers update without remounting."""

    def __init__(self, value: str = "{}", **kwargs: Any):
        super().__init__(
            value=value,
            html_template=POINTCLOUD_HTML,
            css_template=POINTCLOUD_CSS,
            js_on_load=POINTCLOUD_JS,
            apply_default_css=False,
            min_height=720,
            **kwargs,
        )


def natural_sort_key(path: Path) -> list[str | int]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def parse_dynamic_directories(value: str) -> list[Path]:
    entries = [entry.strip() for entry in value.replace(";", "\n").splitlines()]
    return [Path(entry).expanduser().resolve() for entry in entries if entry]


def discover_dynamic_sequences(value: str) -> list[list[Path]]:
    directories = parse_dynamic_directories(value)
    if not directories:
        raise gr.Error("Enter at least one dynamic PLY directory.")
    sequences: list[list[Path]] = []
    for directory in directories:
        if not directory.is_dir():
            raise gr.Error(f"Dynamic point-cloud directory does not exist: {directory}")
        files = sorted(directory.glob("*.ply"), key=natural_sort_key)
        objects_path = directory / "objects.json"
        if not files and objects_path.is_file():
            try:
                objects = json.loads(objects_path.read_text(encoding="utf-8"))["objects"]
                for item in objects:
                    object_dir = directory / str(item["directory"])
                    object_files = sorted(object_dir.glob("*.ply"), key=natural_sort_key)
                    if not object_files:
                        raise gr.Error(f"No PLY files found in object sequence: {object_dir}")
                    sequences.append(object_files)
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise gr.Error(f"Failed to parse object sequence index {objects_path}: {error}") from error
            continue
        if not files:
            raise gr.Error(f"No PLY files found in directory: {directory}")
        sequences.append(files)
    return sequences


@dataclass(frozen=True)
class SequenceViewerInputs:
    root: Path
    static_path: Path
    dynamic_root: Path
    camera_path: Path
    intrinsics_path: Path | None
    foundationpose_pose_paths: tuple[Path, ...]
    foundationpose_mesh_paths: tuple[Path, ...]
    alignment_report_paths: tuple[Path, ...]
    mesh_sequence_indices: tuple[int, ...]
    wilor_hand_motion_path: Path | None
    source_video_path: Path | None
    tracking_overlay_path: Path | None
    experiment_id: str
    experiment_label: str


@dataclass(frozen=True)
class ViewerExperiment:
    experiment_id: str
    label: str
    hand_motion: str | None
    objects: tuple[dict[str, Any], ...]


def discover_viewer_experiments(value: str) -> tuple[list[ViewerExperiment], str]:
    """Load complete hand/object experiment definitions for a sequence.

    Each experiment deliberately binds the hand motion, object pose, and object
    mesh together.  This prevents the UI from constructing hand/object
    combinations that were never evaluated.
    """
    root = Path(value).expanduser().resolve()
    manifest_path = root / "experiments/manifest.json"
    experiments: list[ViewerExperiment] = []
    default_id = "baseline"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            default_id = str(manifest.get("default_experiment_id", default_id))
            records = manifest.get("experiments", [])
            if not isinstance(records, list):
                raise TypeError("experiments must be a list")
            for record in records:
                if not isinstance(record, dict):
                    raise TypeError("each experiment must be an object")
                experiment_id = str(record["id"])
                config = record
                config_value = record.get("manifest")
                if config_value:
                    config_path = Path(str(config_value)).expanduser()
                    if not config_path.is_absolute():
                        config_path = root / config_path
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                experiments.append(
                    ViewerExperiment(
                        experiment_id=experiment_id,
                        label=str(record.get("label", config.get("label", experiment_id))),
                        hand_motion=(
                            str(config["hand_motion"])
                            if config.get("hand_motion")
                            else None
                        ),
                        objects=tuple(config.get("objects", [])),
                    )
                )
        except (KeyError, TypeError, json.JSONDecodeError, OSError) as error:
            raise gr.Error(f"Failed to parse experiment manifest {manifest_path}: {error}") from error
    if not experiments:
        experiments = [
            ViewerExperiment(
                experiment_id="baseline",
                label="Baseline",
                hand_motion="hands/wilor/hand_motion.npz",
                objects=(),
            )
        ]
        default_id = "baseline"
    ids = {item.experiment_id for item in experiments}
    if default_id not in ids:
        default_id = experiments[0].experiment_id
    return experiments, default_id


def experiment_dropdown_update(value: str) -> dict[str, Any]:
    """Refresh the experiment dropdown after the sequence root changes."""
    try:
        experiments, default_id = discover_viewer_experiments(value)
        choices = [(item.label, item.experiment_id) for item in experiments]
        return gr.update(choices=choices, value=default_id)
    except Exception:
        return gr.update(choices=[("Baseline", "baseline")], value="baseline")


def _experiment_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _resolve_manifest_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    return path if path.is_file() else None


def discover_sequence_viewer_inputs(
    value: str,
    experiment_id: str | None = None,
) -> SequenceViewerInputs:
    """Resolve all viewer artifacts from one canonical sequence output directory."""
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise gr.Error(f"Sequence output directory does not exist: {root}")
    pointcloud_root = root / "4d"
    static_path = pointcloud_root / "static/static.ply"
    dynamic_root = pointcloud_root / "dynamic"
    camera_path = pointcloud_root / "cameras.npz"
    for path in (static_path, dynamic_root, camera_path):
        if not path.exists():
            raise gr.Error(f"Sequence output is incomplete; missing: {path}")

    objects_path = dynamic_root / "objects.json"
    if objects_path.is_file():
        try:
            dynamic_objects = json.loads(objects_path.read_text(encoding="utf-8"))["objects"]
            sequence_names = [str(item["directory"]) for item in dynamic_objects]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise gr.Error(f"Failed to parse dynamic object index {objects_path}: {error}") from error
    else:
        sequence_names = [
            path.name for path in sorted(dynamic_root.iterdir(), key=natural_sort_key) if path.is_dir()
        ]
    if not sequence_names:
        raise gr.Error(f"No dynamic sequences were found under: {dynamic_root}")

    experiments, default_experiment_id = discover_viewer_experiments(str(root))
    selected_id = experiment_id or default_experiment_id
    selected = next(
        (item for item in experiments if item.experiment_id == selected_id), None
    )
    if selected is None:
        available = ", ".join(item.experiment_id for item in experiments)
        raise gr.Error(f"Unknown experiment '{selected_id}'. Available: {available}")
    object_configs = {
        str(record.get("sequence", "")): record
        for record in selected.objects
        if isinstance(record, dict)
    }

    pose_paths: list[Path] = []
    mesh_paths: list[Path] = []
    alignment_paths: list[Path] = []
    sequence_indices: list[int] = []
    intrinsics_path: Path | None = None
    foundationpose_root = root / "foundationpose"
    for sequence_index, sequence_name in enumerate(sequence_names):
        object_root = foundationpose_root / sequence_name
        config = object_configs.get(sequence_name)
        pose_path = _experiment_path(root, config.get("pose")) if config else None
        mesh_path_override = _experiment_path(root, config.get("mesh")) if config else None
        alignment_override = (
            _experiment_path(root, config.get("alignment_report")) if config else None
        )
        if pose_path is None:
            pose_path = object_root / "tracking/poses.npz"
        metric_meshes = sorted((object_root / "scale").glob("*_metric_centered.glb"))
        if not pose_path.is_file() or (mesh_path_override is None and not metric_meshes):
            continue
        pose_paths.append(pose_path)
        mesh_paths.append(mesh_path_override or metric_meshes[0])
        alignment_path = alignment_override or object_root / "scale/scale_alignment_report.json"
        alignment_paths.append(alignment_path if alignment_path.is_file() else Path("-"))
        sequence_indices.append(sequence_index)
        candidate_intrinsics = object_root / "dataset/camera_K.npz"
        if intrinsics_path is None and candidate_intrinsics.is_file():
            intrinsics_path = candidate_intrinsics

    wilor_candidate = _experiment_path(root, selected.hand_motion)
    wilor_hand_motion_path = (
        wilor_candidate if wilor_candidate is not None and wilor_candidate.is_file() else None
    )

    source_video_path: Path | None = None
    manifest_path = root / "sequence_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_video_path = _resolve_manifest_path(root, manifest.get("video"))
        except (json.JSONDecodeError, TypeError):
            source_video_path = None
    if source_video_path is None:
        pointcloud_manifest_path = pointcloud_root / "manifest.json"
        if pointcloud_manifest_path.is_file():
            try:
                manifest = json.loads(pointcloud_manifest_path.read_text(encoding="utf-8"))
                source_video_path = _resolve_manifest_path(root, manifest.get("video"))
            except (json.JSONDecodeError, TypeError):
                source_video_path = None

    tracking_overlay_path = next(
        (
            path.resolve()
            for path in (
                pointcloud_root / "tracking/overlay.mp4",
                root / "discovery/tracking/overlay.mp4",
            )
            if path.is_file()
        ),
        None,
    )

    return SequenceViewerInputs(
        root=root,
        static_path=static_path,
        dynamic_root=dynamic_root,
        camera_path=camera_path,
        intrinsics_path=intrinsics_path,
        foundationpose_pose_paths=tuple(pose_paths),
        foundationpose_mesh_paths=tuple(mesh_paths),
        alignment_report_paths=tuple(alignment_paths),
        mesh_sequence_indices=tuple(sequence_indices),
        wilor_hand_motion_path=wilor_hand_motion_path,
        source_video_path=source_video_path,
        tracking_overlay_path=tracking_overlay_path,
        experiment_id=selected.experiment_id,
        experiment_label=selected.label,
    )


def parse_pose_assignments(value: str, sequence_count: int) -> dict[int, Path]:
    """Parse either ordered paths or explicit `sequence_index=path` assignments."""
    entries = [entry.strip() for entry in value.replace(";", "\n").splitlines() if entry.strip()]
    assignments: dict[int, Path] = {}
    next_index = 0
    for entry in entries:
        if "=" in entry:
            index_text, path_text = entry.split("=", 1)
            try:
                index = int(index_text.strip())
            except ValueError as error:
                raise gr.Error(f"Pose assignment index must be an integer: {entry}") from error
        else:
            while next_index in assignments:
                next_index += 1
            index, path_text = next_index, entry
            next_index += 1
        if not 0 <= index < sequence_count:
            raise gr.Error(
                f"Pose assignment index {index} is outside the {sequence_count} dynamic sequences."
            )
        path_text = path_text.strip()
        if path_text and path_text != "-":
            assignments[index] = Path(path_text).expanduser().resolve()
    return assignments


def auto_discover_pose_path(sequence_directory: Path) -> Path | None:
    candidates = [
        sequence_directory / "pose_sequence_optimized.npz",
        sequence_directory / "pose_sequence.npz",
        sequence_directory.parent / "pose_sequence_optimized.npz",
        sequence_directory.parent / "pose_sequence.npz",
    ]
    if sequence_directory.parent.name == "dynamic":
        result_dir = sequence_directory.parent.parent
        tracking_dir = result_dir / "object_tracking" / sequence_directory.name
        candidates.extend(
            [
                tracking_dir / "sliding/pose_sequence_optimized.npz",
                tracking_dir / "pose_sequence.npz",
            ]
        )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _deterministic_vector_sign(vector: np.ndarray) -> np.ndarray:
    """Resolve PCA's sign ambiguity using the dominant world-coordinate component."""
    vector = np.asarray(vector, dtype=np.float64)
    dominant = int(np.argmax(np.abs(vector)))
    return -vector if vector[dominant] < 0 else vector


def estimate_largest_plane_basis(
    points: np.ndarray,
    *,
    max_points: int = 8_000,
    ransac_iterations: int = 384,
) -> tuple[np.ndarray, float]:
    """Estimate a right-handed XYZ basis whose XY plane is the largest planar patch.

    X follows the patch's longest PCA direction, Y is its other in-plane direction,
    and Z is the plane normal. The fixed RNG and sign convention make reloads stable.
    """
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return np.eye(3, dtype=np.float32), 0.0
    if len(points) > max_points:
        sample_indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        sample = points[sample_indices]
    else:
        sample = points
    low, high = np.percentile(sample, [2.0, 98.0], axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if not np.isfinite(diagonal) or diagonal <= 1e-8:
        return np.eye(3, dtype=np.float32), 0.0
    distance_threshold = max(0.015 * diagonal, 1e-6)
    rng = np.random.default_rng(20260823)
    best_mask: np.ndarray | None = None
    best_count = 0
    for _ in range(ransac_iterations):
        triangle = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-10 * diagonal * diagonal:
            continue
        normal /= norm
        mask = np.abs((sample - triangle[0]) @ normal) <= distance_threshold
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
    if best_mask is None or best_count < 3:
        best_mask = np.ones(len(sample), dtype=bool)

    # Refine the normal and inliers twice, then obtain stable in-plane principal axes.
    inliers = sample[best_mask]
    for _ in range(2):
        center = inliers.mean(axis=0)
        covariance = (inliers - center).T @ (inliers - center) / max(len(inliers), 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        refined_mask = np.abs((sample - center) @ normal) <= distance_threshold
        if int(refined_mask.sum()) < 3:
            break
        best_mask = refined_mask
        inliers = sample[best_mask]

    center = inliers.mean(axis=0)
    covariance = (inliers - center).T @ (inliers - center) / max(len(inliers), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    x_axis = _deterministic_vector_sign(eigenvectors[:, order[0]])
    z_axis = _deterministic_vector_sign(eigenvectors[:, order[-1]])
    x_axis -= z_axis * float(np.dot(x_axis, z_axis))
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
    basis = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
    return basis, float(best_mask.mean())


def load_pose_sequences(
    sequences: list[list[Path]], pose_paths_value: str
) -> tuple[list[dict[str, Any] | None], list[str]]:
    assignments = parse_pose_assignments(pose_paths_value, len(sequences))
    configs: list[dict[str, Any] | None] = []
    messages: list[str] = []
    for sequence_index, sequence in enumerate(sequences):
        sequence_dir = sequence[0].parent
        pose_path = assignments.get(sequence_index) or auto_discover_pose_path(sequence_dir)
        if pose_path is None:
            configs.append(None)
            continue
        if not pose_path.is_file():
            raise gr.Error(f"Pose file does not exist: {pose_path}")
        with np.load(pose_path) as archive:
            if "poses" not in archive:
                raise gr.Error(f"Pose NPZ is missing the 'poses' field: {pose_path}")
            poses = np.asarray(archive["poses"], dtype=np.float32)
            valid = (
                np.asarray(archive["pose_valid"], dtype=bool)
                if "pose_valid" in archive
                else np.ones(len(poses), dtype=bool)
            )
            canonical_frame = int(archive["query_frame"]) if "query_frame" in archive else 0
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise gr.Error(f"Poses must have shape Fx4x4: {pose_path} -> {poses.shape}")
        if valid.shape != (len(poses),):
            raise gr.Error(f"pose_valid must have shape F: {pose_path}")
        if not np.isfinite(poses).all():
            raise gr.Error(f"Poses contain non-finite values: {pose_path}")
        canonical_frame = min(max(canonical_frame, 0), len(sequence) - 1)
        canonical_points, _, _ = _read_ply(sequence[canonical_frame])
        if not len(canonical_points):
            raise gr.Error(f"Canonical PLY contains no points: {sequence[canonical_frame]}")
        low, high = np.percentile(canonical_points, [2.0, 98.0], axis=0)
        center = canonical_points.mean(axis=0).astype(np.float32)
        automatic_length = max(float(np.linalg.norm(high - low)) * 0.35, 1e-4)
        first_frame_points, _, _ = _read_ply(sequence[0])
        plane_basis_frame_zero, plane_inlier_ratio = estimate_largest_plane_basis(
            first_frame_points
        )
        # poses[0] maps the canonical/query coordinate system into frame zero.
        # Store the plane basis in canonical coordinates so every pose can propagate it.
        canonical_plane_basis = poses[0, :3, :3].T @ plane_basis_frame_zero
        configs.append(
            {
                "path": str(pose_path),
                "poses": poses,
                "valid": valid,
                "canonical_center": center,
                "canonical_plane_basis": canonical_plane_basis.astype(np.float32),
                "plane_inlier_ratio": plane_inlier_ratio,
                "automatic_length": automatic_length,
                "sequence_index": sequence_index,
            }
        )
        messages.append(f"sequence {sequence_index}: {pose_path.name}")
    return configs, messages


def _load_single_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False, force="scene")
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    geometries = tuple(scene.dump())
    if not geometries:
        raise gr.Error(f"Mesh contains no geometry: {path}")
    mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise gr.Error(f"A triangular mesh is required: {path}")
    if not np.isfinite(mesh.vertices).all():
        raise gr.Error(f"Mesh contains non-finite vertices: {path}")
    return mesh


def load_camera_sequence(
    camera_path_value: str,
    intrinsics_path_value: str,
) -> tuple[dict[str, Any] | None, str]:
    camera_value = camera_path_value.strip()
    if not camera_value:
        if intrinsics_path_value.strip():
            raise gr.Error("Camera intrinsics were provided without a camera pose file.")
        return None, "not loaded"
    camera_path = Path(camera_value).expanduser().resolve()
    if not camera_path.is_file():
        raise gr.Error(f"Camera pose file does not exist: {camera_path}")
    with np.load(camera_path) as archive:
        if "camera_to_world" in archive:
            camera_to_world = np.asarray(archive["camera_to_world"], dtype=np.float64)
        elif "camera_poses" in archive:
            camera_to_world = np.asarray(archive["camera_poses"], dtype=np.float64)
        else:
            raise gr.Error(
                f"Camera pose NPZ is missing camera_to_world/camera_poses: {camera_path}"
            )
        source_indices = (
            np.asarray(archive["source_indices"], dtype=np.int64)
            if "source_indices" in archive
            else np.arange(len(camera_to_world), dtype=np.int64)
        )
    if camera_to_world.ndim != 3 or camera_to_world.shape[1:] != (4, 4):
        raise gr.Error(f"Camera poses must have shape Fx4x4: {camera_to_world.shape}")
    if not len(camera_to_world) or not np.isfinite(camera_to_world).all():
        raise gr.Error("Camera pose sequence is empty or contains non-finite values.")
    if source_indices.ndim != 1 or len(source_indices) < len(camera_to_world):
        raise gr.Error(
            "Camera source_indices must be a 1D array with one entry per camera frame."
        )
    if np.any(source_indices[: len(camera_to_world)] < 0):
        raise gr.Error("Camera source_indices cannot contain negative frame indices.")

    intrinsics: np.ndarray | None = None
    intrinsics_value = intrinsics_path_value.strip()
    if intrinsics_value:
        intrinsics_path = Path(intrinsics_value).expanduser().resolve()
        if not intrinsics_path.is_file():
            raise gr.Error(f"Camera intrinsics file does not exist: {intrinsics_path}")
        with np.load(intrinsics_path) as archive:
            if "intrinsics" in archive:
                intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64)
            elif "median_intrinsics" in archive:
                intrinsics = np.asarray(archive["median_intrinsics"], dtype=np.float64)
            else:
                raise gr.Error(
                    f"Intrinsics NPZ is missing intrinsics/median_intrinsics: {intrinsics_path}"
                )
        if intrinsics.shape != (3, 3) and (
            intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3)
        ):
            raise gr.Error(f"Intrinsics must have shape 3x3 or Fx3x3: {intrinsics.shape}")
        if not np.isfinite(intrinsics).all():
            raise gr.Error("Camera intrinsics contain non-finite values.")
    return (
        {
            "path": str(camera_path),
            "camera_to_world": camera_to_world.astype(np.float32),
            "intrinsics": None if intrinsics is None else intrinsics.astype(np.float32),
            "frame_count": len(camera_to_world),
            "source_indices": source_indices[: len(camera_to_world)],
        },
        f"{len(camera_to_world)} frames",
    )


def load_wilor_hand_motion(path_value: str) -> tuple[dict[str, Any] | None, str]:
    """Load the fixed-slot left/right WiLoR hand sequence exported by this project."""
    value = path_value.strip()
    if not value:
        return None, "not loaded"
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise gr.Error(f"WiLoR hand-motion file does not exist: {path}")
    with np.load(path) as archive:
        required = {"vertices_world", "joints_world", "valid", "faces_left", "faces_right"}
        missing = required.difference(archive.files)
        if missing:
            raise gr.Error(f"WiLoR NPZ is missing keys: {', '.join(sorted(missing))}")
        vertices = np.asarray(archive["vertices_world"], dtype=np.float32)
        joints = np.asarray(archive["joints_world"], dtype=np.float32)
        valid = np.asarray(archive["valid"], dtype=bool)
        faces = (
            np.asarray(archive["faces_left"], dtype=np.int32),
            np.asarray(archive["faces_right"], dtype=np.int32),
        )
        edges = (
            np.asarray(archive["joint_edges"], dtype=np.int32)
            if "joint_edges" in archive
            else np.asarray(
                [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                ],
                dtype=np.int32,
            )
        )
        labels = (
            [str(item) for item in archive["hand_labels"].tolist()]
            if "hand_labels" in archive
            else ["left", "right"]
        )
        scene_scale = float(archive["scene_scale"]) if "scene_scale" in archive else np.nan
    if vertices.ndim != 4 or vertices.shape[1:] != (2, 778, 3):
        raise gr.Error(f"WiLoR vertices must have shape Fx2x778x3: {vertices.shape}")
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise gr.Error(f"WiLoR joints must have shape Fx2x21x3: {joints.shape}")
    if valid.shape != vertices.shape[:2] or len(joints) != len(vertices):
        raise gr.Error("WiLoR validity flags do not match the geometry sequence")
    for hand_faces in faces:
        if hand_faces.ndim != 2 or hand_faces.shape[1] != 3:
            raise gr.Error(f"WiLoR faces must have shape Tx3: {hand_faces.shape}")
    return (
        {
            "path": str(path),
            "vertices_world": vertices,
            "joints_world": joints,
            "valid": valid,
            "faces": faces,
            "edges": edges,
            "labels": labels,
            "textures": tuple(_solid_texture_data_url(color) for color in HAND_COLORS),
            "texcoords": tuple(
                base64.b64encode(
                    np.zeros((len(hand_faces) * 3, 2), dtype="<f4").tobytes()
                ).decode("ascii")
                for hand_faces in faces
            ),
            "frame_count": len(vertices),
            "scene_scale": scene_scale,
        },
        f"{len(vertices)} frames · left {int(valid[:, 0].sum())} / right {int(valid[:, 1].sum())}",
    )


def propagate_first_frame_alignment(
    object_in_world: np.ndarray,
    mesh_to_world_frame_zero: np.ndarray,
) -> np.ndarray:
    """Anchor the mesh in frame zero and propagate FoundationPose relative motion."""
    object_in_world = np.asarray(object_in_world, dtype=np.float64)
    frame_zero = np.asarray(mesh_to_world_frame_zero, dtype=np.float64)
    if object_in_world.ndim != 3 or object_in_world.shape[1:] != (4, 4):
        raise ValueError(f"Expected Fx4x4 object poses, got {object_in_world.shape}")
    if frame_zero.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 frame-zero alignment, got {frame_zero.shape}")
    relative_motion = object_in_world @ np.linalg.inv(object_in_world[0])
    return relative_motion @ frame_zero


def _load_first_frame_mesh_alignment(path_value: str) -> tuple[np.ndarray | None, str]:
    value = path_value.strip()
    if not value:
        return None, "FoundationPose absolute pose"
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise gr.Error(f"First-frame mesh alignment file does not exist: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("use_as_foundationpose_anchor", True) is False:
            return None, f"{path.name} (diagnostic-only alignment ignored)"
        transform = np.asarray(report["metric_centered_to_frame0_world"], dtype=np.float64)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise gr.Error(
            f"Failed to read metric_centered_to_frame0_world from {path}: {error}"
        ) from error
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise gr.Error(f"First-frame mesh alignment must be a finite 4x4 matrix: {path}")
    return transform, path.name


def _mesh_texture_data_url(mesh: trimesh.Trimesh, max_size: int = 1024) -> str:
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    if image is None:
        image = Image.new("RGB", (1, 1), (210, 210, 210))
    texture = image.copy().convert("RGBA")
    texture.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    texture.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _solid_texture_data_url(color: np.ndarray) -> str:
    pixel = Image.new("RGBA", (1, 1), tuple(int(value) for value in color) + (255,))
    buffer = io.BytesIO()
    pixel.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def expanded_smooth_mesh_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return one area-weighted smooth normal per expanded triangle vertex."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Mesh vertices must have shape Vx3: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Mesh faces must have shape Tx3: {faces.shape}")
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    vertex_normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-12
    vertex_normals[valid] /= lengths[valid]
    vertex_normals[~valid] = np.asarray([0.0, 0.0, 1.0])
    return np.ascontiguousarray(vertex_normals[faces].reshape(-1, 3), dtype="<f4")


def encode_textured_mesh(mesh: trimesh.Trimesh) -> dict[str, Any]:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or np.asarray(uv).shape != (len(mesh.vertices), 2):
        raise gr.Error("The selected mesh does not contain per-vertex texture coordinates.")
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangle_vertices = np.ascontiguousarray(
        np.asarray(mesh.vertices, dtype=np.float32)[faces].reshape(-1, 3), dtype="<f4"
    )
    triangle_uv = np.ascontiguousarray(
        np.asarray(uv, dtype=np.float32)[faces].reshape(-1, 2), dtype="<f4"
    )
    triangle_normals = expanded_smooth_mesh_normals(mesh.vertices, faces)
    return {
        "count": len(triangle_vertices),
        "triangle_count": len(faces),
        "positions": base64.b64encode(triangle_vertices.tobytes()).decode("ascii"),
        "normals": base64.b64encode(triangle_normals.tobytes()).decode("ascii"),
        "texcoords": base64.b64encode(triangle_uv.tobytes()).decode("ascii"),
        "texture": _mesh_texture_data_url(mesh),
        "lighting_strength": 0.55,
        "opacity": 1.0,
    }


def load_foundationpose_overlay(
    pose_path_value: str,
    mesh_path_value: str,
    first_frame_alignment_path_value: str,
    camera: dict[str, Any] | None,
    *,
    label: str | None = None,
    dynamic_sequence_index: int | None = None,
) -> tuple[dict[str, Any] | None, str]:
    pose_value, mesh_value = pose_path_value.strip(), mesh_path_value.strip()
    if not pose_value and not mesh_value:
        return None, "not loaded"
    if not pose_value or not mesh_value:
        raise gr.Error("Each FoundationPose object requires both poses and a metric mesh.")
    pose_path = Path(pose_value).expanduser().resolve()
    mesh_path = Path(mesh_value).expanduser().resolve()
    for path in (pose_path, mesh_path):
        if not path.is_file():
            raise gr.Error(f"FoundationPose input does not exist: {path}")

    with np.load(pose_path) as archive:
        if "object_to_world" in archive:
            object_in_world_archive = np.asarray(
                archive["object_to_world"], dtype=np.float64
            )
        elif "ob_in_world" in archive:
            object_in_world_archive = np.asarray(
                archive["ob_in_world"], dtype=np.float64
            )
        else:
            object_in_world_archive = None
        if "object_to_camera" in archive:
            object_in_camera = np.asarray(
                archive["object_to_camera"], dtype=np.float64
            )
        elif "ob_in_cam" in archive:
            object_in_camera = np.asarray(archive["ob_in_cam"], dtype=np.float64)
        elif "poses" in archive:
            object_in_camera = np.asarray(archive["poses"], dtype=np.float64)
        elif object_in_world_archive is not None:
            object_in_camera = None
        else:
            raise gr.Error(
                "FoundationPose NPZ is missing object_to_world/object_to_camera/"
                f"ob_in_world/ob_in_cam/poses: {pose_path}"
            )
        if "state" in archive:
            pose_valid_archive = np.asarray(archive["state"], dtype=np.uint8) > 0
        elif "pose_valid" in archive:
            pose_valid_archive = np.asarray(archive["pose_valid"], dtype=bool)
        else:
            pose_valid_archive = None
    if object_in_world_archive is not None:
        object_in_world = object_in_world_archive
        frame_count = len(object_in_world)
    else:
        if object_in_camera is None or object_in_camera.ndim != 3 or object_in_camera.shape[1:] != (4, 4):
            shape = None if object_in_camera is None else object_in_camera.shape
            raise gr.Error(f"FoundationPose poses must have shape Fx4x4: {shape}")
        if camera is None:
            raise gr.Error(
                f"{pose_path.name} contains camera-space poses, so Pi3X camera poses are required."
            )
        camera_to_world = np.asarray(camera["camera_to_world"], dtype=np.float64)
        frame_count = min(len(object_in_camera), len(camera_to_world))
        object_in_world = camera_to_world[:frame_count] @ object_in_camera[:frame_count]
    if object_in_world.ndim != 3 or object_in_world.shape[1:] != (4, 4):
        raise gr.Error(f"FoundationPose world poses must have shape Fx4x4: {object_in_world.shape}")
    if frame_count == 0:
        raise gr.Error("FoundationPose or camera pose sequence is empty.")
    finite_pose = np.isfinite(object_in_world).all(axis=(1, 2))
    if pose_valid_archive is None:
        pose_valid = finite_pose
    else:
        pose_valid = np.asarray(pose_valid_archive[:frame_count], dtype=bool)
        if pose_valid.shape != (frame_count,):
            raise gr.Error(
                f"FoundationPose validity must have shape ({frame_count},): {pose_path}"
            )
        pose_valid &= finite_pose
    if not pose_valid.any():
        raise gr.Error(f"FoundationPose track has no valid poses: {pose_path}")
    # Full-timeline tracks intentionally contain NaNs before initialization.
    # Keep those frames invisible and sanitize their matrices so no NaN reaches
    # the WebGL payload.
    object_in_world = object_in_world.copy()
    object_in_world[~pose_valid] = np.eye(4, dtype=np.float64)

    mesh = _load_single_mesh(mesh_path)
    frame_zero_alignment, alignment_label = _load_first_frame_mesh_alignment(
        first_frame_alignment_path_value
    )
    mesh_to_world = (
        object_in_world
        if frame_zero_alignment is None
        else propagate_first_frame_alignment(object_in_world, frame_zero_alignment)
    )
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    # FoundationPose tracks the metric-centered mesh frame. Apply the same aligned
    # mesh pose to its oriented bounding box so mesh, axes, and bbox remain coherent.
    bbox_center_in_world = mesh_to_world @ np.linalg.inv(to_origin)
    automatic_length = max(float(np.linalg.norm(extents)) * 0.35, 1e-4)
    mesh_payload = encode_textured_mesh(mesh)
    object_label = label or (pose_path.parent.parent.name if pose_path.parent.name == "tracking" else pose_path.parent.name)
    return (
        {
            "label": object_label,
            "dynamic_sequence_index": dynamic_sequence_index,
            "pose_path": str(pose_path),
            "mesh_path": str(mesh_path),
            "mesh_to_world": mesh_to_world.astype(np.float32),
            "bbox_center_in_world": bbox_center_in_world.astype(np.float32),
            "bbox_extents": np.asarray(extents, dtype=np.float32),
            "automatic_length": automatic_length,
            "frame_count": frame_count,
            "pose_valid": pose_valid,
            "mesh_payload": mesh_payload,
            "alignment_label": alignment_label,
        },
        f"{object_label}: {int(pose_valid.sum())}/{frame_count} valid poses · "
        f"{mesh_payload['triangle_count']:,} triangles · "
        f"frame-0 alignment: {alignment_label}",
    )


def parse_foundationpose_inputs(
    pose_paths_value: str,
    mesh_paths_value: str,
    alignment_paths_value: str,
    dynamic_sequence_indices_value: str,
    sequence_count: int,
) -> list[dict[str, Any]]:
    """Parse line-aligned FoundationPose object inputs."""
    pose_entries = [item.strip() for item in pose_paths_value.replace(";", "\n").splitlines() if item.strip()]
    mesh_entries = [item.strip() for item in mesh_paths_value.replace(";", "\n").splitlines() if item.strip()]
    if not pose_entries and not mesh_entries:
        return []
    if len(pose_entries) != len(mesh_entries):
        raise gr.Error(
            "FoundationPose pose and mesh fields must contain the same number of non-empty lines."
        )
    object_count = len(pose_entries)

    raw_alignments = [item.strip() for item in alignment_paths_value.replace(";", "\n").splitlines() if item.strip()]
    if raw_alignments and len(raw_alignments) != object_count:
        raise gr.Error(
            "Frame-0 alignment reports must be empty or contain one line per FoundationPose object; use '-' to skip one object."
        )
    alignments = ["" if item == "-" else item for item in raw_alignments]
    if not alignments:
        alignments = [""] * object_count

    raw_indices = [
        item.strip()
        for item in re.split(r"[,;\s]+", dynamic_sequence_indices_value.strip())
        if item.strip()
    ]
    if raw_indices and len(raw_indices) != object_count:
        raise gr.Error(
            "Dynamic sequence indices must be empty or contain one value per FoundationPose object; use '-' to keep its point cloud."
        )
    if not raw_indices:
        raw_indices = ["-"] * object_count

    records: list[dict[str, Any]] = []
    for object_index, (pose_path, mesh_path, alignment_path, index_text) in enumerate(
        zip(pose_entries, mesh_entries, alignments, raw_indices)
    ):
        dynamic_sequence_index: int | None = None
        if index_text != "-":
            try:
                dynamic_sequence_index = int(index_text)
            except ValueError as error:
                raise gr.Error(f"Dynamic sequence index must be an integer or '-': {index_text}") from error
            if not 0 <= dynamic_sequence_index < sequence_count:
                raise gr.Error(
                    f"Dynamic sequence index {dynamic_sequence_index} is outside the {sequence_count} loaded sequences."
                )
        records.append(
            {
                "object_index": object_index,
                "pose_path": pose_path,
                "mesh_path": mesh_path,
                "alignment_path": alignment_path,
                "dynamic_sequence_index": dynamic_sequence_index,
            }
        )
    return records


def load_foundationpose_overlays(
    pose_paths_value: str,
    mesh_paths_value: str,
    alignment_paths_value: str,
    dynamic_sequence_indices_value: str,
    camera: dict[str, Any] | None,
    sequence_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    configs: list[dict[str, Any]] = []
    messages: list[str] = []
    for record in parse_foundationpose_inputs(
        pose_paths_value,
        mesh_paths_value,
        alignment_paths_value,
        dynamic_sequence_indices_value,
        sequence_count,
    ):
        config, message = load_foundationpose_overlay(
            record["pose_path"],
            record["mesh_path"],
            record["alignment_path"],
            camera,
            dynamic_sequence_index=record["dynamic_sequence_index"],
        )
        if config is not None:
            configs.append(config)
            messages.append(message)
    return configs, messages


def rotation_to_rpy_degrees(rotation: np.ndarray) -> np.ndarray:
    """Return roll/pitch/yaw for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    rotation = np.asarray(rotation, dtype=np.float64)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-7:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def pose_axis_vertices(
    pose: np.ndarray,
    canonical_center: np.ndarray,
    axis_length: float,
    canonical_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build GL_LINES shafts and four-sided arrowheads for one SE(3) pose."""
    rotation = pose[:3, :3].astype(np.float64)
    basis = (
        np.eye(3, dtype=np.float64)
        if canonical_basis is None
        else np.asarray(canonical_basis, dtype=np.float64)
    )
    origin = rotation @ canonical_center.astype(np.float64) + pose[:3, 3]
    vertices: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for axis_id in range(3):
        direction = rotation @ basis[:, axis_id]
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        helper = np.asarray([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.asarray([0.0, 1.0, 0.0])
        side_a = np.cross(direction, helper)
        side_a /= max(float(np.linalg.norm(side_a)), 1e-12)
        side_b = np.cross(direction, side_a)
        tip = origin + axis_length * direction
        head_base = tip - 0.22 * axis_length * direction
        head_radius = 0.09 * axis_length
        segments = [(origin, tip)]
        for side in (side_a, -side_a, side_b, -side_b):
            segments.append((tip, head_base + head_radius * side))
        for start, end in segments:
            vertices.extend([start, end])
            colors.extend([AXIS_COLORS[axis_id], AXIS_COLORS[axis_id]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(colors, dtype=np.uint8)


def posed_bbox_vertices(
    center_pose: np.ndarray,
    extents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(extents, dtype=np.float64) * 0.5
    corners = np.asarray(
        [
            [x, y, z]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])
        ],
        dtype=np.float64,
    )
    world = corners @ center_pose[:3, :3].T + center_pose[:3, 3]
    # Corner index bits are (x, y, z); connect pairs that differ in one bit.
    edges = [
        (first, second)
        for first in range(8)
        for second in range(first + 1, 8)
        if (first ^ second) in (1, 2, 4)
    ]
    vertices = np.asarray(
        [world[index] for edge in edges for index in edge], dtype=np.float32
    )
    colors = np.broadcast_to(FOUNDATIONPOSE_BBOX_COLOR, vertices.shape).copy()
    return vertices, colors


def _rigid_camera_pose(camera_to_world: np.ndarray) -> np.ndarray:
    """Remove any Sim(3) scale/shear before drawing a camera orientation."""
    pose = np.asarray(camera_to_world, dtype=np.float64).copy()
    left, _, right_t = np.linalg.svd(pose[:3, :3])
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    pose[:3, :3] = rotation
    pose[3] = [0.0, 0.0, 0.0, 1.0]
    return pose


def camera_frustum_vertices(
    camera_to_world: np.ndarray,
    length: float,
    intrinsics: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an OpenCV +Z camera frustum as GL_LINES."""
    depth = max(float(length), 1e-6)
    if intrinsics is not None:
        matrix = np.asarray(intrinsics, dtype=np.float64)
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        if fx > 0.0 and fy > 0.0 and cx > 0.0 and cy > 0.0:
            half_width = depth * cx / fx
            half_height = depth * cy / fy
        else:
            half_width, half_height = depth * 0.75, depth * 0.42
    else:
        half_width, half_height = depth * 0.75, depth * 0.42
    camera_vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
            [half_width, half_height, depth],
            [-half_width, half_height, depth],
        ],
        dtype=np.float64,
    )
    pose = _rigid_camera_pose(camera_to_world)
    world = camera_vertices @ pose[:3, :3].T + pose[:3, 3]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    vertices = np.asarray([world[index] for edge in edges for index in edge], dtype=np.float32)
    colors = np.broadcast_to(CAMERA_FRUSTUM_COLOR, vertices.shape).copy()
    return vertices, colors


def _camera_frame_data(
    state: dict[str, Any],
    frame_id: int,
    show_camera_frustum: bool,
    frustum_length_value: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    config = state.get("camera")
    if not show_camera_frustum:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            "hidden",
        )
    if config is None or frame_id >= int(config["frame_count"]):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            "unavailable",
        )
    requested_length = float(frustum_length_value)
    length = requested_length if requested_length > 0.0 else float(state["camera_auto_length"])
    intrinsics = config.get("intrinsics")
    if intrinsics is not None and np.asarray(intrinsics).ndim == 3:
        intrinsics = intrinsics[min(frame_id, len(intrinsics) - 1)]
    vertices, colors = camera_frustum_vertices(
        config["camera_to_world"][frame_id], length, intrinsics
    )
    translation = np.asarray(config["camera_to_world"][frame_id])[:3, 3]
    return (
        vertices,
        colors,
        f"frame {frame_id} at ({translation[0]:+.3f}, {translation[1]:+.3f}, "
        f"{translation[2]:+.3f})",
    )


def _normalized_mesh_model(
    mesh_to_world: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> list[float]:
    normalization = np.eye(4, dtype=np.float64)
    normalization[:3, :3] *= float(scale)
    normalization[:3, 3] = -np.asarray(center, dtype=np.float64) * float(scale)
    model = normalization @ np.asarray(mesh_to_world, dtype=np.float64)
    # WebGL consumes matrices in column-major order when transpose is false.
    return model.astype(np.float32).reshape(-1, order="F").tolist()


def _wilor_hand_mesh_payloads(
    state: dict[str, Any],
    frame_id: int,
    show_hand_meshes: bool,
) -> tuple[list[dict[str, Any]], list[str], int]:
    config = state.get("wilor_hands")
    if config is None:
        return [], [], 0
    payloads: list[dict[str, Any]] = []
    visible_labels: list[str] = []
    triangle_count = 0
    center = np.asarray(state["center"], dtype=np.float32)
    scale = float(state["scale"])
    identity = np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist()
    for hand_id in range(2):
        visible = (
            show_hand_meshes
            and frame_id < int(config["frame_count"])
            and bool(config["valid"][frame_id, hand_id])
        )
        payload: dict[str, Any] = {"visible": visible, "model": identity}
        if visible:
            vertices = np.asarray(config["vertices_world"][frame_id, hand_id], dtype=np.float32)
            faces = np.asarray(config["faces"][hand_id], dtype=np.int64)
            triangles = np.ascontiguousarray(
                ((vertices - center[None]) * scale)[faces].reshape(-1, 3),
                dtype="<f4",
            )
            triangle_normals = expanded_smooth_mesh_normals(vertices, faces)
            payload.update(
                {
                    "count": len(triangles),
                    "triangle_count": len(faces),
                    "positions": base64.b64encode(triangles.tobytes()).decode("ascii"),
                    "normals": base64.b64encode(triangle_normals.tobytes()).decode("ascii"),
                    "texcoords": config["texcoords"][hand_id],
                    "texture": config["textures"][hand_id],
                    "lighting_strength": 1.0,
                    "opacity": 0.82,
                }
            )
            visible_labels.append(str(config["labels"][hand_id]))
            triangle_count += len(faces)
        payloads.append(payload)
    return payloads, visible_labels, triangle_count


def _wilor_hand_skeleton_frame_data(
    state: dict[str, Any],
    frame_id: int,
    show_hand_joints: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    config = state.get("wilor_hands")
    if config is None:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            "not loaded",
        )
    labels = []
    vertices_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    if frame_id >= int(config["frame_count"]):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            "unavailable",
        )
    edges = np.asarray(config["edges"], dtype=np.int64)
    for hand_id in range(2):
        if not bool(config["valid"][frame_id, hand_id]):
            continue
        label = str(config["labels"][hand_id])
        labels.append(label)
        if show_hand_joints:
            joints = np.asarray(config["joints_world"][frame_id, hand_id], dtype=np.float32)
            lines = joints[edges].reshape(-1, 3)
            vertices_parts.append(lines)
            color_parts.append(np.broadcast_to(HAND_COLORS[hand_id], lines.shape).copy())
    label = ", ".join(labels) if labels else "no detection"
    if not vertices_parts:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            label + (" (skeleton hidden)" if labels else ""),
        )
    return np.concatenate(vertices_parts), np.concatenate(color_parts), label


def _meshes_frame_payload(
    state: dict[str, Any],
    frame_id: int,
    object_representation: str,
    *,
    include_geometry: bool,
    show_hand_meshes: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    payloads: list[dict[str, Any]] = []
    visible_labels: list[str] = []
    total_triangles = 0
    for config in state.get("foundationpose_objects", []):
        visible = (
            object_representation
            in {
                TEXTURED_MESH_REPRESENTATION,
                POINT_CLOUD_AND_MESH_REPRESENTATION,
            }
            and frame_id < int(config["frame_count"])
            and bool(config["pose_valid"][frame_id])
        )
        payload: dict[str, Any] = {
            "visible": visible,
            "model": np.eye(4, dtype=np.float32).reshape(-1, order="F").tolist(),
        }
        pose_index = min(frame_id, int(config["frame_count"]) - 1)
        payload["model"] = _normalized_mesh_model(
            config["mesh_to_world"][pose_index],
            np.asarray(state["center"], dtype=np.float32),
            float(state["scale"]),
        )
        if include_geometry:
            payload.update(config["mesh_payload"])
        if visible:
            # In overlay mode the point cloud is drawn first. A translucent mesh
            # keeps those samples visible instead of depth-occluding them.
            payload["opacity"] = (
                0.55
                if object_representation == POINT_CLOUD_AND_MESH_REPRESENTATION
                else 1.0
            )
        payloads.append(payload)
        if visible:
            visible_labels.append(str(config["label"]))
            total_triangles += int(config["mesh_payload"]["triangle_count"])
    hand_payloads, hand_labels, hand_triangles = _wilor_hand_mesh_payloads(
        state, frame_id, show_hand_meshes
    )
    payloads.extend(hand_payloads)
    visible_labels.extend(f"{label} hand" for label in hand_labels)
    total_triangles += hand_triangles
    if visible_labels:
        mesh_kind = "meshes" if hand_labels else "textured meshes"
        return (
            payloads,
            f"{len(visible_labels)} {mesh_kind} ({total_triangles:,} triangles): "
            + ", ".join(visible_labels),
        )
    if object_representation in {
        TEXTURED_MESH_REPRESENTATION,
        POINT_CLOUD_AND_MESH_REPRESENTATION,
    }:
        return payloads, "textured meshes unavailable"
    return payloads, "object point clouds"


def _foundationpose_frame_data(
    state: dict[str, Any],
    frame_id: int,
    show_axes: bool,
    show_bbox: bool,
    axis_length_value: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    configs = state.get("foundationpose_objects", [])
    if not configs:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            "not shown",
        )
    parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    labels: list[str] = []
    for config in configs:
        if (
            frame_id >= int(config["frame_count"])
            or not bool(config["pose_valid"][frame_id])
        ):
            labels.append(f"{config['label']}: unavailable")
            continue
        center_pose = np.asarray(config["bbox_center_in_world"][frame_id], dtype=np.float64)
        if show_axes:
            requested_length = float(axis_length_value)
            length = requested_length if requested_length > 0 else float(config["automatic_length"])
            vertices, colors = pose_axis_vertices(
                center_pose,
                np.zeros(3, dtype=np.float32),
                length,
            )
            parts.append(vertices)
            color_parts.append(colors)
        if show_bbox:
            vertices, colors = posed_bbox_vertices(center_pose, config["bbox_extents"])
            parts.append(vertices)
            color_parts.append(colors)

        translation = center_pose[:3, 3]
        rpy = rotation_to_rpy_degrees(center_pose[:3, :3])
        labels.append(
            f"{config['label']}: t=({translation[0]:+.3f}, {translation[1]:+.3f}, "
            f"{translation[2]:+.3f}) · RPY=({rpy[0]:+.1f}°, {rpy[1]:+.1f}°, {rpy[2]:+.1f}°)"
        )
    label = " | ".join(labels)
    if not parts:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            label + (" (overlay disabled)" if label else "not shown"),
        )
    return np.concatenate(parts), np.concatenate(color_parts), label


def _read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".ply":
        raise gr.Error(f"PLY file does not exist or has the wrong extension: {path}")
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _CACHE_LOCK:
        cached = _PLY_CACHE.get(key)
        if cached is not None:
            _PLY_CACHE.move_to_end(key)
            return cached

    vertices = PlyData.read(str(path))["vertex"].data
    names = set(vertices.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise gr.Error(f"PLY is missing x/y/z vertex fields: {path}")
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(
        np.float32, copy=False
    )
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]])
        if np.issubdtype(colors.dtype, np.floating) and colors.size and float(colors.max()) <= 1.0:
            colors = colors * 255
        colors = np.clip(colors, 0, 255).astype(np.uint8, copy=False)
    else:
        colors = np.full((len(points), 3), 210, dtype=np.uint8)
    id_field = "track_id" if "track_id" in names else "object_track_id"
    track_ids = vertices[id_field].astype(np.int32, copy=False) if id_field in names else None
    finite = np.isfinite(points).all(axis=1)
    result = (points[finite], colors[finite], track_ids[finite] if track_ids is not None else None)

    with _CACHE_LOCK:
        _PLY_CACHE[key] = result
        _PLY_CACHE.move_to_end(key)
        while len(_PLY_CACHE) > MAX_PLY_CACHE_ENTRIES:
            _PLY_CACHE.popitem(last=False)
    return result


def _probe_video_metadata(path_value: str) -> tuple[float, int]:
    """Return FPS and frame count without decoding the complete video."""
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise gr.Error(f"Video file does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise gr.Error(f"Failed to open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if not np.isfinite(fps) or fps <= 0:
        raise gr.Error(f"Video reports an invalid FPS: {path}")
    return fps, max(0, frame_count)


def _prepare_browser_video(source: Path, cache_dir: Path, role: str) -> Path:
    """Create a cached H.264 MP4 with frequent keyframes for framewise seeking."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise gr.Error(f"Video file does not exist: {source}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise gr.Error("ffmpeg is required to prepare browser-compatible viewer videos")
    source_fps, source_frame_count = _probe_video_metadata(str(source))
    stat = source.stat()
    signature = hashlib.sha1(
        (
            f"seek-h264-v2|{source}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{source_fps:.9f}|{source_frame_count}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    safe_role = re.sub(r"[^a-zA-Z0-9_-]+", "_", role).strip("_") or "video"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{safe_role}_{signature}.mp4"
    if output.is_file():
        try:
            cached_fps, cached_frame_count = _probe_video_metadata(str(output))
            if (
                abs(cached_fps - source_fps) <= max(0.01, 0.001 * source_fps)
                and abs(cached_frame_count - source_frame_count) <= 1
            ):
                return output
        except gr.Error:
            pass

    temporary = output.with_name(f"{output.stem}.tmp.mp4")
    keyframe_interval = max(1, int(round(source_fps * 0.25)))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "fastdecode",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(keyframe_interval),
        "-keyint_min",
        str(keyframe_interval),
        "-sc_threshold",
        "0",
        "-bf",
        "0",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        cached_fps, cached_frame_count = _probe_video_metadata(str(temporary))
        if abs(cached_fps - source_fps) > max(0.01, 0.001 * source_fps):
            raise gr.Error(
                f"Browser proxy FPS changed from {source_fps:.6g} to {cached_fps:.6g}: {source}"
            )
        if abs(cached_frame_count - source_frame_count) > 1:
            raise gr.Error(
                "Browser proxy frame count changed from "
                f"{source_frame_count} to {cached_frame_count}: {source}"
            )
        temporary.replace(output)
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or str(error)).strip()
        raise gr.Error(f"Failed to encode browser video {source}: {details}") from error
    return output


def synchronized_video_positions(
    frame_value: float,
    state: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    """Return source/tracking timestamps for one reconstruction frame."""
    if not state:
        return None, None
    frame_count = max(1, int(state["frame_count"]))
    frame_id = min(max(int(round(frame_value)), 0), frame_count - 1)
    source_indices = state.get("source_indices", [])
    source_frame_id = (
        int(source_indices[frame_id]) if frame_id < len(source_indices) else frame_id
    )
    source_fps = float(state.get("source_video_fps", 0.0))
    tracking_fps = float(state.get("tracking_video_fps", 0.0))
    source_position = (source_frame_id + 0.5) / source_fps if source_fps > 0 else None
    tracking_position = (frame_id + 0.5) / tracking_fps if tracking_fps > 0 else None
    return source_position, tracking_position


def _sample_points_and_colors(
    points: np.ndarray, colors: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= count:
        return points, colors
    indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    return points[indices], colors[indices]


def _color_dynamic(
    original: np.ndarray,
    track_ids: np.ndarray | None,
    source_index: int,
    mode: str,
) -> np.ndarray:
    if mode == FOLDER_COLOR_MODE:
        return np.broadcast_to(PALETTE[source_index % len(PALETTE)], original.shape).copy()
    if mode == TRACK_COLOR_MODE and track_ids is not None:
        colors = np.empty_like(original)
        for track_id in np.unique(track_ids):
            colors[track_ids == track_id] = PALETTE[int(track_id) % len(PALETTE)]
        return colors
    return original


def _encode_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> dict[str, str | int]:
    normalized = np.ascontiguousarray((points - center[None]) * scale, dtype="<f4")
    colors = np.ascontiguousarray(colors, dtype=np.uint8)
    return {
        "count": len(normalized),
        "positions": base64.b64encode(normalized.tobytes()).decode("ascii"),
        "colors": base64.b64encode(colors.tobytes()).decode("ascii"),
    }


def _pose_axes_frame_data(
    state: dict[str, Any],
    frame_id: int,
    show_pose_axes: bool,
    axis_basis_mode: str,
    axis_length_value: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not show_pose_axes:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), []
    vertices_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    labels: list[str] = []
    requested_length = float(axis_length_value)
    for config in state.get("pose_sequences", []):
        if config is None or frame_id >= len(config["poses"]) or not config["valid"][frame_id]:
            continue
        length = requested_length if requested_length > 0 else float(config["automatic_length"])
        canonical_basis = (
            config["canonical_plane_basis"] if axis_basis_mode == PLANE_AXIS_MODE else None
        )
        vertices, colors = pose_axis_vertices(
            config["poses"][frame_id],
            config["canonical_center"],
            length,
            canonical_basis,
        )
        vertices_parts.append(vertices)
        color_parts.append(colors)
        if axis_basis_mode == PLANE_AXIS_MODE:
            labels.append(
                f"sequence {config['sequence_index']} (frame-0 plane inliers: "
                f"{100.0 * float(config['plane_inlier_ratio']):.0f}%)"
            )
        else:
            labels.append(f"sequence {config['sequence_index']} (Pi3 world coordinates)")
    if not vertices_parts:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), labels
    return np.concatenate(vertices_parts), np.concatenate(color_parts), labels


def _dynamic_frame_data(
    state: dict[str, Any],
    frame_id: int,
    color_mode: str,
    object_representation: str,
    hide_arm_pointcloud: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dynamic_points_parts: list[np.ndarray] = []
    dynamic_colors_parts: list[np.ndarray] = []
    used_files: list[str] = []
    replaced_sequence_indices = {
        int(config["dynamic_sequence_index"])
        for config in state.get("foundationpose_objects", [])
        if object_representation == TEXTURED_MESH_REPRESENTATION
        and config.get("dynamic_sequence_index") is not None
        and frame_id < int(config["frame_count"])
        and bool(config["pose_valid"][frame_id])
    }
    hidden_arm_indices = (
        {int(value) for value in state.get("arm_sequence_indices", [])}
        if hide_arm_pointcloud
        else set()
    )
    for source_index, files in enumerate(state["sequences"]):
        if source_index in replaced_sequence_indices or source_index in hidden_arm_indices:
            continue
        if frame_id >= len(files):
            continue
        path = files[frame_id]
        points, colors, track_ids = _read_ply(path)
        dynamic_points_parts.append(points)
        dynamic_colors_parts.append(_color_dynamic(colors, track_ids, source_index, color_mode))
        used_files.append(f"sequence {source_index}: {Path(path).name}")
    if not dynamic_points_parts:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            used_files,
        )
    return (
        np.concatenate(dynamic_points_parts, axis=0),
        np.concatenate(dynamic_colors_parts, axis=0),
        used_files,
    )


def _infer_arm_sequence_indices(sequences_paths: list[list[Path]]) -> tuple[int, ...]:
    """Identify arm point-cloud sequences from their index label or directory name."""
    labels_by_directory: dict[str, str] = {}
    object_indexes = {
        sequence[0].parent.parent / "objects.json"
        for sequence in sequences_paths
        if sequence
    }
    for objects_path in object_indexes:
        if not objects_path.is_file():
            continue
        try:
            records = json.loads(objects_path.read_text(encoding="utf-8")).get("objects", [])
            for record in records:
                labels_by_directory[str(record["directory"])] = str(record.get("label", ""))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue

    def contains_arm_token(value: str) -> bool:
        return re.search(r"(^|[^a-z0-9])arms?([^a-z0-9]|$)", value.lower()) is not None

    indices: list[int] = []
    for sequence_index, sequence in enumerate(sequences_paths):
        if not sequence:
            continue
        directory_name = sequence[0].parent.name
        if contains_arm_token(labels_by_directory.get(directory_name, "")) or contains_arm_token(
            directory_name
        ):
            indices.append(sequence_index)
    return tuple(indices)


def render_frame(
    frame_value: float,
    state: dict[str, Any] | None,
    max_points_value: float,
    color_mode: str,
    object_representation: str,
    show_pose_axes: bool,
    axis_basis_mode: str,
    axis_length_value: float,
    show_foundationpose_axes: bool,
    show_foundationpose_bbox: bool,
    foundationpose_axis_length_value: float,
    show_camera_frustum: bool,
    camera_frustum_length_value: float,
    show_wilor_meshes: bool,
    show_wilor_joints: bool,
    hide_arm_pointcloud: bool = False,
    hide_static_scene: bool = False,
) -> tuple[str, str]:
    if not state:
        return "{}", "Click **Load scene** first."
    frame_id = int(round(frame_value))
    frame_count = int(state["frame_count"])
    frame_id = min(max(frame_id, 0), frame_count - 1)
    maximum = max(1_000, int(max_points_value))
    dynamic_points, dynamic_colors, used_files = _dynamic_frame_data(
        state,
        frame_id,
        color_mode,
        object_representation,
        hide_arm_pointcloud,
    )
    visible_static_count = 0 if hide_static_scene else int(state["static_display_count"])
    dynamic_budget = max(0, maximum - visible_static_count)
    shown_dynamic_points, shown_dynamic_colors = _sample_points_and_colors(
        dynamic_points, dynamic_colors, dynamic_budget
    )
    axis_points, axis_colors, axis_labels = _pose_axes_frame_data(
        state, frame_id, show_pose_axes, axis_basis_mode, axis_length_value
    )
    foundation_points, foundation_colors, foundation_label = _foundationpose_frame_data(
        state,
        frame_id,
        show_foundationpose_axes,
        show_foundationpose_bbox,
        foundationpose_axis_length_value,
    )
    if len(foundation_points):
        axis_points = np.concatenate([axis_points, foundation_points], axis=0)
        axis_colors = np.concatenate([axis_colors, foundation_colors], axis=0)
    camera_points, camera_colors, camera_label = _camera_frame_data(
        state,
        frame_id,
        show_camera_frustum,
        camera_frustum_length_value,
    )
    if len(camera_points):
        axis_points = np.concatenate([axis_points, camera_points], axis=0)
        axis_colors = np.concatenate([axis_colors, camera_colors], axis=0)
    hand_points, hand_colors, hand_label = _wilor_hand_skeleton_frame_data(
        state,
        frame_id,
        show_wilor_joints,
    )
    if len(hand_points):
        axis_points = np.concatenate([axis_points, hand_points], axis=0)
        axis_colors = np.concatenate([axis_colors, hand_colors], axis=0)
    mesh_payloads, mesh_label = _meshes_frame_payload(
        state,
        frame_id,
        object_representation,
        include_geometry=False,
        show_hand_meshes=show_wilor_meshes,
    )
    source_position, tracking_position = synchronized_video_positions(frame_id, state)
    payload = json.dumps(
        {
            "type": "frame",
            "frame_id": frame_id,
            "show_static": not hide_static_scene,
            "source_video_position": source_position,
            "tracking_video_position": tracking_position,
            "dynamic": _encode_cloud(
                shown_dynamic_points,
                shown_dynamic_colors,
                np.asarray(state["center"], dtype=np.float32),
                float(state["scale"]),
            ),
            "axes": _encode_cloud(
                axis_points,
                axis_colors,
                np.asarray(state["center"], dtype=np.float32),
                float(state["scale"]),
            ),
            "meshes": mesh_payloads,
        },
        separators=(",", ":"),
    )
    static_display_count = visible_static_count
    arm_sequence_count = len(state.get("arm_sequence_indices", []))
    arm_pointcloud_label = (
        f"hidden ({arm_sequence_count} sequences)"
        if hide_arm_pointcloud
        else f"visible ({arm_sequence_count} sequences)"
    )
    info = (
        f"**Frame {frame_id + 1}/{frame_count}** · "
        f"{static_display_count + len(shown_dynamic_points):,} displayed points "
        f"(static {static_display_count:,} / dynamic {len(shown_dynamic_points):,})  \n"
        f"Dynamic inputs: {', '.join(used_files) if used_files else 'none'}  \n"
        f"Static scene: {'hidden' if hide_static_scene else 'visible'} · "
        f"Arm point clouds: {arm_pointcloud_label}  \n"
        f"Object representation: {mesh_label}  \n"
        f"Point-cloud registration axes: {', '.join(axis_labels) if axis_labels else 'hidden'}  \n"
        f"FoundationPose 6DoF: {foundation_label}  \n"
        f"Camera frustum: {camera_label}  \n"
        f"WiLoR hands: {hand_label}"
    )
    return payload, info


def load_scene(
    static_path_value: str,
    dynamic_directories_value: str,
    max_points_value: float,
    color_mode: str,
    pose_paths_value: str,
    foundationpose_pose_paths_value: str,
    camera_path_value: str,
    camera_intrinsics_path_value: str,
    foundationpose_mesh_paths_value: str,
    first_frame_alignment_paths_value: str,
    object_representation: str,
    mesh_sequence_indices_value: str,
    show_pose_axes: bool,
    axis_basis_mode: str,
    axis_length_value: float,
    show_foundationpose_axes: bool,
    show_foundationpose_bbox: bool,
    foundationpose_axis_length_value: float,
    show_camera_frustum: bool,
    camera_frustum_length_value: float,
    wilor_hand_motion_path_value: str,
    show_wilor_meshes: bool,
    show_wilor_joints: bool,
    hide_arm_pointcloud: bool = False,
    hide_static_scene: bool = False,
    source_video_path_value: str = "",
    tracking_overlay_path_value: str = "",
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    dict[str, Any],
    dict[str, Any],
]:
    static_path = Path(static_path_value).expanduser().resolve()
    static_points, static_colors, _ = _read_ply(static_path)
    sequences_paths = discover_dynamic_sequences(dynamic_directories_value)
    pose_sequences, pose_messages = load_pose_sequences(sequences_paths, pose_paths_value)
    camera, camera_message = load_camera_sequence(
        camera_path_value,
        camera_intrinsics_path_value,
    )
    foundationpose_objects, foundationpose_messages = load_foundationpose_overlays(
        foundationpose_pose_paths_value,
        foundationpose_mesh_paths_value,
        first_frame_alignment_paths_value,
        mesh_sequence_indices_value,
        camera,
        len(sequences_paths),
    )
    wilor_hands, wilor_message = load_wilor_hand_motion(wilor_hand_motion_path_value)
    source_video_path = source_video_path_value.strip()
    tracking_overlay_path = tracking_overlay_path_value.strip()
    source_video_fps, source_video_frame_count = (
        _probe_video_metadata(source_video_path) if source_video_path else (0.0, 0)
    )
    tracking_video_fps, tracking_video_frame_count = (
        _probe_video_metadata(tracking_overlay_path) if tracking_overlay_path else (0.0, 0)
    )
    sequences = [[str(path) for path in sequence] for sequence in sequences_paths]
    arm_sequence_indices = _infer_arm_sequence_indices(sequences_paths)
    frame_count = max(len(sequence) for sequence in sequences)
    if wilor_hands is not None:
        frame_count = max(frame_count, int(wilor_hands["frame_count"]))
    maximum = max(1_000, int(max_points_value))
    static_budget = min(len(static_points), max(1, int(maximum * 0.65)))
    shown_static_points, shown_static_colors = _sample_points_and_colors(
        static_points, static_colors, static_budget
    )
    if len(shown_static_points) < 3:
        raise gr.Error("The static PLY must contain at least three finite points.")
    bounds_sample = shown_static_points
    low, high = np.percentile(bounds_sample, [1.0, 99.0], axis=0)
    center = ((low + high) / 2).astype(np.float32)
    extent = float(np.max(high - low))
    scale = 2.0 / extent if np.isfinite(extent) and extent > 1e-8 else 1.0
    state: dict[str, Any] = {
        "static_path": str(static_path),
        "sequences": sequences,
        "arm_sequence_indices": list(arm_sequence_indices),
        "frame_count": frame_count,
        "static_display_count": len(shown_static_points),
        "center": center.tolist(),
        "scale": scale,
        "pose_sequences": pose_sequences,
        "camera": camera,
        "camera_auto_length": max(extent * 0.07, 1e-3),
        "foundationpose_objects": foundationpose_objects,
        "wilor_hands": wilor_hands,
        "source_video_path": source_video_path,
        "tracking_overlay_path": tracking_overlay_path,
        "source_video_fps": source_video_fps,
        "source_video_frame_count": source_video_frame_count,
        "tracking_video_fps": tracking_video_fps,
        "tracking_video_frame_count": tracking_video_frame_count,
        "source_indices": (
            camera["source_indices"].astype(np.int64).tolist()
            if camera is not None
            else list(range(frame_count))
        ),
    }
    frame_payload, info = render_frame(
        0,
        state,
        max_points_value,
        color_mode,
        object_representation,
        show_pose_axes,
        axis_basis_mode,
        axis_length_value,
        show_foundationpose_axes,
        show_foundationpose_bbox,
        foundationpose_axis_length_value,
        show_camera_frustum,
        camera_frustum_length_value,
        show_wilor_meshes,
        show_wilor_joints,
        hide_arm_pointcloud,
        hide_static_scene,
    )
    frame_data = json.loads(frame_payload)
    initial_meshes, _ = _meshes_frame_payload(
        state,
        0,
        object_representation,
        include_geometry=True,
        show_hand_meshes=show_wilor_meshes,
    )
    initial_payload = json.dumps(
        {
            "type": "init",
            "frame_id": 0,
            "show_static": not hide_static_scene,
            "source_video_position": frame_data.get("source_video_position"),
            "tracking_video_position": frame_data.get("tracking_video_position"),
            "static": _encode_cloud(shown_static_points, shown_static_colors, center, scale),
            "dynamic": frame_data["dynamic"],
            "axes": frame_data["axes"],
            "meshes": initial_meshes,
        },
        separators=(",", ":"),
    )
    status = (
        f"Loaded one static cloud and {len(sequences)} dynamic sequences "
        f"({len(arm_sequence_indices)} arm sequences) "
        f"({frame_count} frames). Found {len(pose_messages)} point-cloud pose sequences. "
        f"Camera: {camera_message}. FoundationPose objects: "
        f"{'; '.join(foundationpose_messages) if foundationpose_messages else 'not loaded'}. "
        f"WiLoR hands: {wilor_message}. "
        f"Source video: {'loaded' if source_video_path else 'not found'}. "
        f"SAM3 tracking overlay: "
        f"{'loaded' if tracking_overlay_path else 'not found'}. "
        "Release the frame slider to render a selected frame."
    )
    source_position = frame_data.get("source_video_position") or 0.0
    tracking_position = frame_data.get("tracking_video_position") or 0.0
    source_player = gr.update(
        value=source_video_path or None,
        playback_position=source_position,
    )
    tracking_player = gr.update(
        value=tracking_overlay_path or None,
        playback_position=tracking_position,
    )
    return state, gr.update(
        minimum=0, maximum=max(1, frame_count - 1), value=0, step=1, interactive=frame_count > 1
    ), initial_payload, (status + "\n\n" + info), source_player, tracking_player


def load_sequence_scene(
    sequence_root_value: str,
    experiment_id: str,
    max_points_value: float,
    color_mode: str,
    object_representation: str,
    show_pose_axes: bool,
    axis_basis_mode: str,
    axis_length_value: float,
    show_foundationpose_axes: bool,
    show_foundationpose_bbox: bool,
    foundationpose_axis_length_value: float,
    show_camera_frustum: bool,
    camera_frustum_length_value: float,
    show_wilor_meshes: bool,
    show_wilor_joints: bool,
    hide_arm_pointcloud: bool = False,
    hide_static_scene: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    str,
    dict[str, Any],
    dict[str, Any],
]:
    inputs = discover_sequence_viewer_inputs(sequence_root_value, experiment_id)
    browser_media_dir = inputs.root / "viewer_media"
    source_video_path = (
        _prepare_browser_video(inputs.source_video_path, browser_media_dir, "source")
        if inputs.source_video_path is not None
        else None
    )
    tracking_overlay_path = (
        _prepare_browser_video(inputs.tracking_overlay_path, browser_media_dir, "sam3_tracking")
        if inputs.tracking_overlay_path is not None
        else None
    )
    alignment_values = "\n".join(
        "-" if path == Path("-") else str(path) for path in inputs.alignment_report_paths
    )
    result = load_scene(
        str(inputs.static_path),
        str(inputs.dynamic_root),
        max_points_value,
        color_mode,
        "",
        "\n".join(map(str, inputs.foundationpose_pose_paths)),
        str(inputs.camera_path),
        "" if inputs.intrinsics_path is None else str(inputs.intrinsics_path),
        "\n".join(map(str, inputs.foundationpose_mesh_paths)),
        alignment_values,
        object_representation,
        "\n".join(map(str, inputs.mesh_sequence_indices)),
        show_pose_axes,
        axis_basis_mode,
        axis_length_value,
        show_foundationpose_axes,
        show_foundationpose_bbox,
        foundationpose_axis_length_value,
        show_camera_frustum,
        camera_frustum_length_value,
        "" if inputs.wilor_hand_motion_path is None else str(inputs.wilor_hand_motion_path),
        show_wilor_meshes,
        show_wilor_joints,
        hide_arm_pointcloud,
        hide_static_scene,
        "" if source_video_path is None else str(source_video_path),
        "" if tracking_overlay_path is None else str(tracking_overlay_path),
    )
    state, slider, payload, status, source_player, tracking_player = result
    discovered = (
        f"Sequence: `{inputs.root}`  \n"
        f"Auto-discovered {len(state['sequences'])} dynamic sequences and "
        f"{len(inputs.foundationpose_pose_paths)} FoundationPose objects. "
        f"Experiment: **{inputs.experiment_label}** (`{inputs.experiment_id}`). "
        f"Hand motion: "
        f"{'loaded' if inputs.wilor_hand_motion_path else 'not found'}. "
        f"Object pose/mesh pairs: {len(inputs.foundationpose_pose_paths)}. "
        f"Source video: {'seek-optimized H.264' if source_video_path else 'not found'}. "
        f"SAM3 overlay: {'seek-optimized H.264' if tracking_overlay_path else 'not found'}.  \n"
    )
    return state, slider, payload, discovered + status, source_player, tracking_player


def toggle_playback(is_playing: bool) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    next_state = not bool(is_playing)
    return (
        next_state,
        gr.update(value="Pause" if next_state else "Play"),
        gr.update(active=next_state),
    )


def update_playback_rate(fps_value: float) -> dict[str, Any]:
    fps = min(24.0, max(1.0, float(fps_value)))
    return gr.update(value=1.0 / fps)


def advance_frame(
    frame_value: float,
    state: dict[str, Any] | None,
    max_points_value: float,
    color_mode: str,
    object_representation: str,
    show_pose_axes: bool,
    axis_basis_mode: str,
    axis_length_value: float,
    show_foundationpose_axes: bool,
    show_foundationpose_bbox: bool,
    foundationpose_axis_length_value: float,
    show_camera_frustum: bool,
    camera_frustum_length_value: float,
    show_wilor_meshes: bool,
    show_wilor_joints: bool,
    hide_arm_pointcloud: bool = False,
    hide_static_scene: bool = False,
) -> tuple[
    dict[str, Any],
    str | None,
    str,
]:
    if not state:
        return (
            gr.update(value=0),
            None,
            "Click **Load scene** first.",
        )
    next_frame = (int(round(frame_value)) + 1) % int(state["frame_count"])
    model_path, info = render_frame(
        next_frame,
        state,
        max_points_value,
        color_mode,
        object_representation,
        show_pose_axes,
        axis_basis_mode,
        axis_length_value,
        show_foundationpose_axes,
        show_foundationpose_bbox,
        foundationpose_axis_length_value,
        show_camera_frustum,
        camera_frustum_length_value,
        show_wilor_meshes,
        show_wilor_joints,
        hide_arm_pointcloud,
        hide_static_scene,
    )
    return gr.update(value=next_frame), model_path, info


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="4D Point Cloud Viewer") as demo:
        state = gr.State(value=None)
        is_playing = gr.State(value=False)
        playback_timer = gr.Timer(value=1.0 / DEFAULT_PLAYBACK_FPS, active=False)
        with gr.Column(elem_id="viewer-shell"):
            gr.Markdown(
                "# 4D Point Cloud Viewer\n"
                "View a static scene plus one or more per-frame dynamic PLY sequences. "
                "Enter one canonical `outputs/<video_stem>` sequence directory; all artifacts "
                "are discovered automatically. "
                "Tracked objects can be shown as reconstructed point clouds or as "
                "multiple textured meshes propagated by their FoundationPose 6DoF sequences. "
                "If `hands/wilor/hand_motion.npz` exists, the recovered MANO meshes and "
                "joint skeletons are loaded in the same Pi3X world coordinates. "
                "Axes are **X red / Y green / Z blue**, the FoundationPose bbox is "
                "**orange**, and the current camera frustum is **cyan**."
            )
            with gr.Row():
                sequence_root = gr.Textbox(
                    label="Sequence output directory",
                    value=str(DEFAULT_SEQUENCE_ROOT),
                    info=(
                        "Canonical layout: outputs/<video_stem> with 4d/, meshes/, "
                        "foundationpose/, discovery/, and hands/ subdirectories"
                    ),
                    scale=8,
                )
                load_button = gr.Button("Load scene", variant="primary", scale=1)
            initial_experiments, initial_experiment_id = (
                discover_viewer_experiments(str(DEFAULT_SEQUENCE_ROOT))
                if DEFAULT_SEQUENCE_ROOT.is_dir()
                else ([], "baseline")
            )
            experiment = gr.Dropdown(
                choices=(
                    [(item.label, item.experiment_id) for item in initial_experiments]
                    or [("Baseline", "baseline")]
                ),
                value=initial_experiment_id,
                label="Experiment",
                info=(
                    "A complete, evaluated hand + object pose + object mesh configuration. "
                    "Choices are read from experiments/manifest.json."
                ),
            )
            gr.Markdown(
                "The viewer automatically loads point clouds, Pi3X cameras, textured meshes, "
                "FoundationPose poses, WiLoR hands, the source video, and the SAM3 overlay."
            )

            with gr.Accordion("Viewer options", open=True):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Point clouds and objects")
                        max_points = gr.Number(
                            label="Maximum displayed points",
                            value=300_000,
                            minimum=1_000,
                            precision=0,
                        )
                        color_mode = gr.Radio(
                            [ORIGINAL_RGB_MODE, FOLDER_COLOR_MODE, TRACK_COLOR_MODE],
                            value=ORIGINAL_RGB_MODE,
                            label="Dynamic point coloring",
                        )
                        object_representation = gr.Radio(
                            [
                                POINT_CLOUD_REPRESENTATION,
                                TEXTURED_MESH_REPRESENTATION,
                                POINT_CLOUD_AND_MESH_REPRESENTATION,
                            ],
                            value=POINT_CLOUD_REPRESENTATION,
                            label="Tracked object representation",
                            info=(
                                "Overlay mode keeps the object PLY visible and renders "
                                "its aligned textured mesh at 55% opacity."
                            ),
                        )
                        hide_arm_pointcloud = gr.Checkbox(
                            value=False,
                            label="Hide arm point clouds",
                            info="Keeps WiLoR MANO meshes and joints independently controllable",
                        )
                        hide_static_scene = gr.Checkbox(
                            value=False,
                            label="Hide static scene",
                            info="Leaves the static cloud loaded so it can be restored instantly",
                        )

                    with gr.Column():
                        gr.Markdown("#### Pose display")
                        show_pose_axes = gr.Checkbox(
                            value=False, label="Show point-cloud registration axes"
                        )
                        axis_basis_mode = gr.Radio(
                            [PLANE_AXIS_MODE, WORLD_AXIS_MODE],
                            value=PLANE_AXIS_MODE,
                            label="Registration-axis basis",
                            info="Largest-plane mode uses frame-0 planar X/Y and its normal as Z",
                        )
                        axis_length = gr.Number(
                            label="Registration axis length (world units; 0=auto)",
                            value=0.0,
                            minimum=0.0,
                        )
                        show_foundationpose_axes = gr.Checkbox(
                            value=True, label="Show FoundationPose 6DoF axes"
                        )
                        show_foundationpose_bbox = gr.Checkbox(
                            value=True, label="Show FoundationPose 3D bbox"
                        )
                        foundationpose_axis_length = gr.Number(
                            label="FoundationPose axis length (world units; 0=auto)",
                            value=0.0,
                            minimum=0.0,
                        )
                    with gr.Column():
                        gr.Markdown("#### Camera and hands")
                        show_camera_frustum = gr.Checkbox(
                            value=False, label="Show current camera frustum"
                        )
                        camera_frustum_length = gr.Number(
                            label="Camera frustum length (world units; 0=auto)",
                            value=0.0,
                            minimum=0.0,
                        )
                        show_wilor_meshes = gr.Checkbox(
                            value=True, label="Show WiLoR MANO hand meshes"
                        )
                        show_wilor_joints = gr.Checkbox(
                            value=True, label="Show WiLoR hand joint skeletons"
                        )

            with gr.Row():
                frame_slider = gr.Slider(
                    0, 1, value=0, step=1, label="Frame", interactive=False, scale=8
                )
                playback_fps = gr.Slider(
                    1,
                    24,
                    value=DEFAULT_PLAYBACK_FPS,
                    step=1,
                    label="Playback FPS",
                    scale=3,
                )
                play_button = gr.Button("Play", scale=1, min_width=90)
            status = gr.Markdown("Verify the paths, then click **Load scene**.")
            viewer = PointCloudCanvas(
                label="Static scene + current dynamic frame",
                elem_id="pointcloud-viewer",
            )
            gr.Markdown("### Synchronized source video and SAM3 tracking")
            with gr.Row():
                source_video_player = gr.Video(
                    label="Source video",
                    interactive=False,
                    height=400,
                    include_audio=False,
                    autoplay=False,
                    format="mp4",
                    buttons=[],
                    elem_id="source-video-player",
                )
                sam3_tracking_player = gr.Video(
                    label="SAM3 tracking",
                    interactive=False,
                    height=400,
                    include_audio=False,
                    autoplay=False,
                    format="mp4",
                    buttons=[],
                    elem_id="sam3-tracking-player",
                )

        load_inputs = [
            sequence_root,
            experiment,
            max_points,
            color_mode,
            object_representation,
            show_pose_axes,
            axis_basis_mode,
            axis_length,
            show_foundationpose_axes,
            show_foundationpose_bbox,
            foundationpose_axis_length,
            show_camera_frustum,
            camera_frustum_length,
            show_wilor_meshes,
            show_wilor_joints,
            hide_arm_pointcloud,
            hide_static_scene,
        ]
        render_inputs = [
            frame_slider,
            state,
            max_points,
            color_mode,
            object_representation,
            show_pose_axes,
            axis_basis_mode,
            axis_length,
            show_foundationpose_axes,
            show_foundationpose_bbox,
            foundationpose_axis_length,
            show_camera_frustum,
            camera_frustum_length,
            show_wilor_meshes,
            show_wilor_joints,
            hide_arm_pointcloud,
            hide_static_scene,
        ]
        load_button.click(
            load_sequence_scene,
            inputs=load_inputs,
            outputs=[
                state,
                frame_slider,
                viewer,
                status,
                source_video_player,
                sam3_tracking_player,
            ],
            show_progress="full",
        )
        frame_slider.release(
            render_frame,
            inputs=render_inputs,
            outputs=[viewer, status],
            show_progress="minimal",
            concurrency_limit=1,
            concurrency_id="pointcloud-render",
        )
        max_points.change(
            load_sequence_scene,
            inputs=load_inputs,
            outputs=[
                state,
                frame_slider,
                viewer,
                status,
                source_video_player,
                sam3_tracking_player,
            ],
            show_progress="minimal",
            concurrency_limit=1,
            concurrency_id="pointcloud-render",
        )
        sequence_root.change(
            experiment_dropdown_update,
            inputs=[sequence_root],
            outputs=[experiment],
            show_progress="hidden",
        )
        experiment.change(
            load_sequence_scene,
            inputs=load_inputs,
            outputs=[
                state,
                frame_slider,
                viewer,
                status,
                source_video_player,
                sam3_tracking_player,
            ],
            show_progress="minimal",
            concurrency_limit=1,
            concurrency_id="pointcloud-render",
        )
        for render_control in (
            color_mode,
            object_representation,
            show_pose_axes,
            axis_basis_mode,
            axis_length,
            show_foundationpose_axes,
            show_foundationpose_bbox,
            foundationpose_axis_length,
            show_camera_frustum,
            camera_frustum_length,
            show_wilor_meshes,
            show_wilor_joints,
            hide_arm_pointcloud,
            hide_static_scene,
        ):
            render_control.change(
                render_frame,
                inputs=render_inputs,
                outputs=[viewer, status],
                show_progress="minimal",
                concurrency_limit=1,
                concurrency_id="pointcloud-render",
            )
        play_button.click(
            toggle_playback,
            inputs=[is_playing],
            outputs=[is_playing, play_button, playback_timer],
            show_progress="hidden",
        )
        playback_fps.release(
            update_playback_rate,
            inputs=[playback_fps],
            outputs=[playback_timer],
            show_progress="hidden",
        )
        playback_timer.tick(
            advance_frame,
            inputs=render_inputs,
            outputs=[
                frame_slider,
                viewer,
                status,
            ],
            show_progress="hidden",
            concurrency_limit=1,
            concurrency_id="pointcloud-render",
            trigger_mode="always_last",
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View static + per-frame dynamic PLY point clouds")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_demo().queue(default_concurrency_limit=2).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
        css=DEMO_CSS,
    )

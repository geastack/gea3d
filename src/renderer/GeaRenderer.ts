// SPDX-License-Identifier: GPL-3.0-only
// GeaRenderer — draws a three-style scene through gea3d's native engine
// (gea3d/src/native/gea3d_native.cpp, compiled into the firmware via the
// app's `gea.nativeSources`). Each frame writes what changed in the TS scene
// graph to its native mirror, then ONE drawScene call culls, transforms,
// lights, sorts and rasterizes natively.
//
// With a retained UI mounted (on an engine that supports the underlay hook)
// the frame is composited UNDER the UI nodes in the engine's own flush, so
// HUD text and buttons stay on top without flicker. Elsewhere the frame is
// presented straight to the display, or into an image surface
// (setImageTarget) that a <canvas> node draws.

import { Camera } from '../three/camera'
import { advanceNativeFrame } from '../three/cache'
import { Object3D, Scene } from '../three/core'
import {
  createSurface,
  drawScene,
  renderOptions,
  renderTarget,
  stat,
  STAT_CLIP_DROPPED,
  STAT_CULLED,
  STAT_DROPPED,
  STAT_MESHES_CULLED,
  STAT_MESHES_DRAWN,
  STAT_NEAR_REJECTED,
  STAT_SUBMITTED,
} from '../native/api'

export interface GeaRendererOptions {
  // Primitives per frame (default 4096).
  triangleCapacity?: number
  // Per-pixel depth test for opaque geometry (exact intersections). Off:
  // nearest-first coverage, each pixel written once.
  depthBuffer?: boolean
}

export class GeaRenderer {
  width: number
  height: number
  autoClear: boolean
  // Render stats for the last frame
  trianglesDrawn: number
  trianglesDropped: number
  statLine: string

  private statFrames: number
  private imageTarget: number

  constructor(options?: GeaRendererOptions) {
    this.width = 0
    this.height = 0
    this.autoClear = true
    this.trianglesDrawn = 0
    this.trianglesDropped = 0
    this.statLine = ''
    this.statFrames = 0
    this.imageTarget = -1

    let capacity = 4096
    let depthBuffer = false
    if (options !== undefined) {
      const requested = options.triangleCapacity
      if (requested !== undefined) capacity = requested
      const depth = options.depthBuffer
      if (depth !== undefined) depthBuffer = depth
    }
    renderOptions(depthBuffer ? 1 : 0, capacity)
  }

  setSize(width: number, height: number): void {
    this.width = Math.floor(width)
    this.height = Math.floor(height)
  }

  // Renders into a native image surface instead of the display and returns
  // its image id: draw it with ctx.drawImage(imageFromId(id), x, y) on a
  // <canvas> node. The fallback for engines without the UI underlay.
  setImageTarget(width: number, height: number): number {
    if (this.imageTarget < 0) this.imageTarget = createSurface(width, height)
    renderTarget(this.imageTarget)
    this.setSize(width, height)
    return this.imageTarget
  }

  render(scene: Scene, camera: Camera): void {
    if (this.width <= 0 || this.height <= 0) {
      // Use window.* rather than Display.width/height: both are host
      // property→method bridges, but the emitter lowers window.innerWidth
      // reliably, whereas Display.width intermittently emits as a member ref
      // (uncalled method) on larger programs. App code always calls setSize
      // first, so this is only a safety net.
      this.setSize(window.innerWidth, window.innerHeight)
    }

    advanceNativeFrame()
    scene.syncNative(-1)
    scene.syncEnvironment(this.autoClear)
    syncCameraGraph(scene, camera)

    const drawn = drawScene(scene.nativeNode.id, camera.nativeNode.id, this.width, this.height)
    this.trianglesDrawn = drawn < 0 ? 0 : drawn
    this.trianglesDropped = stat(STAT_DROPPED)
    // The stat line refreshes every 120 frames rather than per frame: it is
    // a string build, and nothing needs it fresher.
    this.statFrames++
    if (this.statFrames >= 120) {
      this.statLine =
        'submitted=' + stat(STAT_SUBMITTED) +
        ' near=' + stat(STAT_NEAR_REJECTED) +
        ' culled=' + stat(STAT_CULLED) +
        ' clip=' + stat(STAT_CLIP_DROPPED) +
        ' drawn=' + this.trianglesDrawn +
        ' meshes=' + stat(STAT_MESHES_DRAWN) +
        ' offscreen=' + stat(STAT_MESHES_CULLED)
      this.statFrames = 0
    }
  }
}

// A camera outside the scene graph (the common three.js setup) syncs its
// own graph; one inside it was synced with the scene.
function syncCameraGraph(scene: Scene, camera: Camera): void {
  const root: Object3D = camera.root()
  if (root.id !== scene.id) root.syncNative(-1)
}

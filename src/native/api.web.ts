// SPDX-License-Identifier: GPL-3.0-only
/// <reference lib="dom" />
// Browser implementation of gea3d's native engine for the DOM web target,
// which has no native code. Same exports as api.ts; an app selects it in its
// vite.web.config.ts (see gea3d/README.md):
//
//   resolve: { alias: [{ find: /^\.\.\/native\/api$/, replacement: '../native/api.web' }] }
//
// Frames are drawn with Canvas 2D into a <canvas> behind the app's DOM, so
// the retained UI stays on top. It is a preview renderer: faces are flat
// (vertex colours averaged), sorted back to front, and fog, lights, blending,
// lines, points, sprites and instances follow the native engine.

import {
  BACKGROUND_GRADIENT,
  BACKGROUND_NONE,
  FOG_EXP2,
  FOG_LINEAR,
  LIGHT_AMBIENT,
  LIGHT_DIRECTIONAL,
  LIGHT_HEMISPHERE,
  LIGHT_POINT,
  MATERIAL_ADDITIVE,
  MATERIAL_FOG,
  MATERIAL_SIZE_ATTENUATION,
  MATERIAL_TRANSPARENT,
  MATERIAL_VERTEX_COLORS,
  MATERIAL_VISIBLE,
  MATERIAL_WIREFRAME,
  NODE_CAMERA,
  NODE_GROUP,
  NODE_LIGHT,
  NODE_LINE,
  NODE_LINE_SEGMENTS,
  NODE_MESH,
  NODE_POINTS,
  NODE_SPRITE,
} from './constants'

export * from './constants'

type Matrix = Float32Array

interface WebBuffer {
  f32: Float32Array
  u32: Uint32Array
}

interface WebGeometry {
  position: number
  index: number
  color: number
  normal: number
}

interface WebMaterial {
  shade: number
  color: number[]
  opacity: number
  side: number
  flags: number
  size: number
}

interface WebNode {
  kind: number
  parent: number
  children: number[]
  position: number[]
  quaternion: number[]
  scale: number[]
  local: Matrix
  useMatrix: boolean
  world: Matrix
  visible: boolean
  frustumCulled: boolean
  renderOrder: number
  geometry: number
  material: number
  // An InstancedMesh draws its instances only (none at a count of 0).
  instanced: boolean
  instanceMatrices: number
  instanceColors: number
  instanceCount: number
  lightKind: number
  lightColor: number[]
  intensity: number
  groundColor: number[]
  distance: number
  decay: number
  target: number[]
  projection: Matrix
  near: number
  far: number
  background: number
  backgroundTop: number
  backgroundBottom: number
  fog: number
  fogColor: number
  fogNear: number
  fogFar: number
  fogDensity: number
}

// A primitive in screen space, painted back to front.
interface Shape {
  depth: number
  xs: number[]
  ys: number[]
  color: string
  alpha: number
  additive: boolean
  // The object's renderOrder: sorted by it first, then back to front.
  layer: number
  line: boolean
  // Stroke width in pixels: a line's width, or the hairline that hides face seams.
  width: number
}

interface Lights {
  ambient: number[]
  directions: number[][]
  directionalColors: number[][]
  points: number[][]
  pointColors: number[][]
  pointRanges: number[][]
  hemisphereUps: number[][]
  skies: number[][]
  grounds: number[][]
}

const buffers: (WebBuffer | null)[] = []
const geometries: (WebGeometry | null)[] = []
const materials: (WebMaterial | null)[] = []
const nodes: (WebNode | null)[] = []
const stats = [0, 0, 0, 0, 0, 0, 0, 0, 0]
const rayHit = [0, 0, 0, 0, -1, -1]
let canvas: HTMLCanvasElement | null = null

function allocate<T>(pool: (T | null)[], value: T): number {
  const free = pool.indexOf(null)
  if (free >= 0) {
    pool[free] = value
    return free
  }
  pool.push(value)
  return pool.length - 1
}

function identity(): Matrix {
  const m = new Float32Array(16)
  m[0] = m[5] = m[10] = m[15] = 1
  return m
}

// ---- buffers, geometry, materials

export function createBuffer(): number {
  return allocate(buffers, { f32: new Float32Array(0), u32: new Uint32Array(0) })
}

export function deleteBuffer(buffer: number): void {
  if (buffers[buffer]) buffers[buffer] = null
}

export function bufferDataF32(buffer: number, data: Float32Array): void {
  const slot = buffers[buffer]
  if (slot) slot.f32 = new Float32Array(data)
}

export function bufferDataU32(buffer: number, data: Uint32Array): void {
  const slot = buffers[buffer]
  if (slot) slot.u32 = new Uint32Array(data)
}

export function createGeometry(): number {
  return allocate(geometries, { position: -1, index: -1, color: -1, normal: -1 })
}

export function deleteGeometry(geometry: number): void {
  if (geometries[geometry]) geometries[geometry] = null
}

export function geometryBuffers(geometry: number, position: number, index: number, color: number, normal: number): void {
  const slot = geometries[geometry]
  if (slot) Object.assign(slot, { position, index, color, normal })
}

export function createMaterial(): number {
  return allocate(materials, { shade: 1, color: [1, 1, 1], opacity: 1, side: 0, flags: MATERIAL_VISIBLE, size: 1 })
}

export function deleteMaterial(material: number): void {
  if (materials[material]) materials[material] = null
}

export function materialSet(
  material: number,
  shade: number,
  r: number,
  g: number,
  b: number,
  opacity: number,
  side: number,
  flags: number,
  size: number,
): void {
  const slot = materials[material]
  if (slot) Object.assign(slot, { shade, color: [r, g, b], opacity, side, flags, size })
}

// ---- nodes

export function createNode(kind: number): number {
  return allocate(nodes, {
    kind,
    parent: -1,
    children: [],
    position: [0, 0, 0],
    quaternion: [0, 0, 0, 1],
    scale: [1, 1, 1],
    local: identity(),
    useMatrix: false,
    world: identity(),
    visible: true,
    frustumCulled: true,
    renderOrder: 0,
    geometry: -1,
    material: -1,
    instanced: false,
    instanceMatrices: -1,
    instanceColors: -1,
    instanceCount: 0,
    lightKind: 0,
    lightColor: [1, 1, 1],
    intensity: 1,
    groundColor: [1, 1, 1],
    distance: 0,
    decay: 2,
    target: [0, 0, 0],
    projection: identity(),
    near: 0.1,
    far: 2000,
    background: BACKGROUND_NONE,
    backgroundTop: 0,
    backgroundBottom: 0,
    fog: 0,
    fogColor: 0,
    fogNear: 1,
    fogFar: 1000,
    fogDensity: 0,
  })
}

function detach(id: number): void {
  const node = nodes[id]
  if (!node || node.parent < 0) return
  const parent = nodes[node.parent]
  if (parent) parent.children = parent.children.filter((child) => child !== id)
  node.parent = -1
}

export function deleteNode(node: number): void {
  const slot = nodes[node]
  if (!slot) return
  detach(node)
  for (const child of slot.children) {
    const childNode = nodes[child]
    if (childNode) childNode.parent = -1
  }
  nodes[node] = null
}

export function nodeParent(node: number, parent: number): void {
  const slot = nodes[node]
  if (!slot || slot.parent === parent) return
  detach(node)
  const parentNode = parent >= 0 ? nodes[parent] : null
  if (!parentNode) return
  slot.parent = parent
  parentNode.children.push(node)
}

export function nodeTransform(
  node: number,
  px: number, py: number, pz: number,
  qx: number, qy: number, qz: number, qw: number,
  sx: number, sy: number, sz: number,
): void {
  const slot = nodes[node]
  if (!slot) return
  slot.position = [px, py, pz]
  slot.quaternion = [qx, qy, qz, qw]
  slot.scale = [sx, sy, sz]
  slot.useMatrix = false
}

export function nodeMatrix(node: number, elements: Float32Array): void {
  const slot = nodes[node]
  if (!slot) return
  slot.local.set(elements.subarray(0, 16))
  slot.useMatrix = true
}

export function nodeFlags(node: number, visible: number, frustumCulled: number, renderOrder: number): void {
  const slot = nodes[node]
  if (!slot) return
  slot.visible = visible !== 0
  slot.frustumCulled = frustumCulled !== 0
  slot.renderOrder = renderOrder
}

export function nodeGeometry(node: number, geometry: number, material: number): void {
  const slot = nodes[node]
  if (slot) Object.assign(slot, { geometry, material })
}

export function nodeInstances(node: number, matrices: number, colors: number, count: number): void {
  const slot = nodes[node]
  if (slot) Object.assign(slot, { instanced: true, instanceMatrices: matrices, instanceColors: colors, instanceCount: count })
}

export function nodeLight(
  node: number,
  kind: number,
  r: number, g: number, b: number,
  intensity: number,
  groundR: number, groundG: number, groundB: number,
  distance: number,
  decay: number,
): void {
  const slot = nodes[node]
  if (slot) Object.assign(slot, { lightKind: kind, lightColor: [r, g, b], intensity, groundColor: [groundR, groundG, groundB], distance, decay })
}

export function nodeLightTarget(node: number, x: number, y: number, z: number): void {
  const slot = nodes[node]
  if (slot) slot.target = [x, y, z]
}

export function cameraProjection(node: number, elements: Float32Array, near: number, far: number): void {
  const slot = nodes[node]
  if (!slot) return
  slot.projection.set(elements.subarray(0, 16))
  slot.near = near
  slot.far = far
}

export function sceneEnvironment(
  scene: number,
  background: number,
  top: number,
  bottom: number,
  fog: number,
  fogColor: number,
  fogNear: number,
  fogFar: number,
  fogDensity: number,
): void {
  const slot = nodes[scene]
  if (slot) Object.assign(slot, { background, backgroundTop: top, backgroundBottom: bottom, fog, fogColor, fogNear, fogFar, fogDensity })
}

// ---- frame

export function renderOptions(depthBuffer: number, capacity: number): void {
  // Painter's order only in the browser.
}

export function renderTarget(surface: number): void {
  // Always the page canvas in the browser.
}

export function createSurface(width: number, height: number): number {
  return -1
}

export function stat(which: number): number {
  return stats[which] ?? 0
}

export function drawScene(scene: number, camera: number, width: number, height: number): number {
  const sceneNode = nodes[scene]
  const cameraNode = nodes[camera]
  if (!sceneNode || !cameraNode) return 0

  stats.fill(0)
  updateWorld(scene, null)
  worldOf(camera)
  const view = invertAffine(cameraNode.world)
  const frame = new FrameBuilder(sceneNode, cameraNode, view, width, height)
  frame.collect(scene)
  frame.paint(ensureCanvas(width, height))
  stats[7] = frame.shapes.length
  return frame.shapes.length
}

// ---- raycast

export function raycast(
  node: number,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  near: number,
  far: number,
): number {
  const slot = nodes[node]
  const geometry = slot ? geometries[slot.geometry] : null
  const positions = geometry ? buffers[geometry.position] : null
  const material = slot ? materials[slot.material] : null
  if (!slot || slot.kind !== NODE_MESH || !geometry || !positions || !material) return 0

  const length = Math.hypot(dx, dy, dz)
  if (length === 0) return 0
  const direction = [dx / length, dy / length, dz / length]
  const world = worldOf(node)
  const indexBuffer = buffers[geometry.index]
  const indices = indexBuffer && indexBuffer.u32.length > 0 ? indexBuffer.u32 : null
  const p = positions.f32
  const triangles = Math.floor((indices ? indices.length : p.length / 3) / 3)
  const instances = slot.instanced ? buffers[slot.instanceMatrices] : null
  const instanceCount = slot.instanced ? (instances ? Math.min(slot.instanceCount, Math.floor(instances.f32.length / 16)) : 0) : 1
  let best = far
  let hit = false
  for (let instance = 0; instance < instanceCount; instance++) {
    const m = instances ? multiply(world, instances.f32.subarray(instance * 16, instance * 16 + 16)) : world
    const inverse = invertAffine(m)
    const o = transformPoint(inverse, ox, oy, oz)
    const d = transformDirection(inverse, direction[0], direction[1], direction[2])
    for (let t = 0; t < triangles; t++) {
      const a = (indices ? indices[t * 3] : t * 3) * 3
      const b = (indices ? indices[t * 3 + 1] : t * 3 + 1) * 3
      const c = (indices ? indices[t * 3 + 2] : t * 3 + 2) * 3
      const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]]
      const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]]
      const h = cross(d, e2)
      const det = dot(e1, h)
      if (Math.abs(det) < 1e-9) continue
      if ((material.side === 0 && det < 0) || (material.side === 1 && det > 0)) continue
      const s = [o[0] - p[a], o[1] - p[a + 1], o[2] - p[a + 2]]
      const u = dot(s, h) / det
      if (u < 0 || u > 1) continue
      const q = cross(s, e1)
      const v = dot(d, q) / det
      if (v < 0 || u + v > 1) continue
      const distance = dot(e2, q) / det
      if (distance < near || distance >= best) continue
      best = distance
      hit = true
      rayHit.splice(0, 6, distance, ox + direction[0] * distance, oy + direction[1] * distance, oz + direction[2] * distance, t, instances ? instance : -1)
    }
  }
  return hit ? 1 : 0
}

export function rayResult(which: number): number {
  return rayHit[which] ?? 0
}

// ---- scene graph math

function compose(node: WebNode): Matrix {
  const [x, y, z, w] = node.quaternion
  const [sx, sy, sz] = node.scale
  const m = new Float32Array(16)
  m[0] = (1 - 2 * (y * y + z * z)) * sx
  m[1] = 2 * (x * y + w * z) * sx
  m[2] = 2 * (x * z - w * y) * sx
  m[4] = 2 * (x * y - w * z) * sy
  m[5] = (1 - 2 * (x * x + z * z)) * sy
  m[6] = 2 * (y * z + w * x) * sy
  m[8] = 2 * (x * z + w * y) * sz
  m[9] = 2 * (y * z - w * x) * sz
  m[10] = (1 - 2 * (x * x + y * y)) * sz
  m[12] = node.position[0]
  m[13] = node.position[1]
  m[14] = node.position[2]
  m[15] = 1
  return m
}

function localOf(node: WebNode): Matrix {
  return node.useMatrix ? node.local : compose(node)
}

function updateWorld(id: number, parentWorld: Matrix | null): void {
  const node = nodes[id]
  if (!node) return
  node.world = parentWorld ? multiply(parentWorld, localOf(node)) : localOf(node)
  for (const child of node.children) updateWorld(child, node.world)
}

function worldOf(id: number): Matrix {
  const node = nodes[id]
  if (!node) return identity()
  node.world = node.parent >= 0 ? multiply(worldOf(node.parent), localOf(node)) : localOf(node)
  return node.world
}

function multiply(a: Float32Array, b: Float32Array): Matrix {
  const out = new Float32Array(16)
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] = a[row] * b[col * 4] + a[4 + row] * b[col * 4 + 1] + a[8 + row] * b[col * 4 + 2] + a[12 + row] * b[col * 4 + 3]
    }
  }
  return out
}

function invertAffine(m: Float32Array): Matrix {
  const [a00, a10, a20] = [m[0], m[1], m[2]]
  const [a01, a11, a21] = [m[4], m[5], m[6]]
  const [a02, a12, a22] = [m[8], m[9], m[10]]
  const c00 = a11 * a22 - a12 * a21
  const c01 = a12 * a20 - a10 * a22
  const c02 = a10 * a21 - a11 * a20
  const det = a00 * c00 + a01 * c01 + a02 * c02
  const out = identity()
  if (det === 0) return out
  const inv = 1 / det
  out[0] = c00 * inv
  out[1] = c01 * inv
  out[2] = c02 * inv
  out[4] = (a02 * a21 - a01 * a22) * inv
  out[5] = (a00 * a22 - a02 * a20) * inv
  out[6] = (a01 * a20 - a00 * a21) * inv
  out[8] = (a01 * a12 - a02 * a11) * inv
  out[9] = (a02 * a10 - a00 * a12) * inv
  out[10] = (a00 * a11 - a01 * a10) * inv
  const [tx, ty, tz] = [m[12], m[13], m[14]]
  out[12] = -(out[0] * tx + out[4] * ty + out[8] * tz)
  out[13] = -(out[1] * tx + out[5] * ty + out[9] * tz)
  out[14] = -(out[2] * tx + out[6] * ty + out[10] * tz)
  return out
}

function transformPoint(m: Float32Array, x: number, y: number, z: number): number[] {
  return [m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]]
}

function transformDirection(m: Float32Array, x: number, y: number, z: number): number[] {
  return [m[0] * x + m[4] * y + m[8] * z, m[1] * x + m[5] * y + m[9] * z, m[2] * x + m[6] * y + m[10] * z]
}

function cross(a: number[], b: number[]): number[] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

function dot(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

function normalize(v: number[]): number[] {
  const length = Math.hypot(v[0], v[1], v[2])
  return length > 0 ? [v[0] / length, v[1] / length, v[2] / length] : v
}

function hexChannels(hex: number): number[] {
  return [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255]
}

function cssColor(rgb: number[]): string {
  const channel = (value: number) => Math.max(0, Math.min(255, Math.floor(value * 255)))
  return `rgb(${channel(rgb[0])},${channel(rgb[1])},${channel(rgb[2])})`
}

// ---- frame building

class FrameBuilder {
  readonly shapes: Shape[] = []
  private readonly lights: Lights = {
    ambient: [0, 0, 0],
    directions: [],
    directionalColors: [],
    points: [],
    pointColors: [],
    pointRanges: [],
    hemisphereUps: [],
    skies: [],
    grounds: [],
  }
  private readonly planes: number[][]
  private readonly near: number
  private readonly orthographic: boolean
  // renderOrder of the node being drawn.
  private layer = 0

  constructor(
    private readonly scene: WebNode,
    private readonly camera: WebNode,
    private readonly view: Matrix,
    private readonly width: number,
    private readonly height: number,
  ) {
    this.near = Math.max(camera.near, 0.0001)
    const p = camera.projection
    this.orthographic = p[15] === 1 && p[11] === 0
    const rows = [0, 1, 2, 3].map((r) => [p[r], p[4 + r], p[8 + r], p[12 + r]])
    this.planes = []
    for (let i = 0; i < 3; i++) {
      for (const sign of [1, -1]) {
        const plane = rows[3].map((value, k) => value + sign * rows[i][k])
        const length = Math.hypot(plane[0], plane[1], plane[2]) || 1
        this.planes.push(plane.map((value) => value / length))
      }
    }
  }

  collect(root: number): void {
    const drawables: WebNode[] = []
    const visit = (id: number) => {
      const node = nodes[id]
      if (!node || !node.visible) return
      if (node.kind === NODE_LIGHT) this.addLight(node)
      else if (node.kind !== NODE_GROUP && node.kind !== NODE_CAMERA) drawables.push(node)
      for (const child of node.children) visit(child)
    }
    visit(root)
    for (const node of drawables) this.draw(node)
  }

  paint(target: HTMLCanvasElement): void {
    const ctx = target.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, this.width, this.height)
    const scene = this.scene
    if (scene.background !== BACKGROUND_NONE) {
      const top = cssColor(hexChannels(scene.backgroundTop))
      if (scene.background === BACKGROUND_GRADIENT) {
        const gradient = ctx.createLinearGradient(0, 0, 0, this.height)
        gradient.addColorStop(0, top)
        gradient.addColorStop(1, cssColor(hexChannels(scene.backgroundBottom)))
        ctx.fillStyle = gradient
      } else {
        ctx.fillStyle = top
      }
      ctx.fillRect(0, 0, this.width, this.height)
    }

    this.sortBackToFront()
    ctx.lineJoin = 'round'
    for (const shape of this.shapes) {
      ctx.globalAlpha = shape.alpha
      ctx.globalCompositeOperation = shape.additive ? 'lighter' : 'source-over'
      ctx.beginPath()
      ctx.moveTo(shape.xs[0], shape.ys[0])
      for (let i = 1; i < shape.xs.length; i++) ctx.lineTo(shape.xs[i], shape.ys[i])
      ctx.strokeStyle = shape.color
      ctx.lineWidth = shape.width
      if (shape.line) {
        ctx.stroke()
        continue
      }
      ctx.closePath()
      ctx.fillStyle = shape.color
      ctx.fill()
      // Hairline seams between antialiased neighbours disappear under a
      // stroke of the face's own colour.
      if (shape.alpha >= 1) ctx.stroke()
    }
    ctx.globalAlpha = 1
    ctx.globalCompositeOperation = 'source-over'
  }

  // Painter's order in the native engine's 1024 depth buckets, quantized
  // from the farthest primitive, so both engines resolve ties alike.
  private sortBackToFront(): void {
    let lo = Infinity
    let hi = -Infinity
    for (const shape of this.shapes) {
      lo = Math.min(lo, shape.depth)
      hi = Math.max(hi, shape.depth)
    }
    const scale = hi - lo > 1e-6 ? 1023 / (hi - lo) : 0
    const bucket = (shape: Shape) => Math.min(1023, Math.floor((hi - shape.depth) * scale))
    this.shapes.sort((a, b) => a.layer - b.layer || bucket(a) - bucket(b))
  }

  private addLight(node: WebNode): void {
    const lights = this.lights
    const color = node.lightColor.map((value) => value * node.intensity)
    const w = node.world
    if (node.lightKind === LIGHT_AMBIENT) {
      for (let k = 0; k < 3; k++) lights.ambient[k] += color[k]
    } else if (node.lightKind === LIGHT_DIRECTIONAL) {
      lights.directions.push(normalize(transformDirection(this.view, w[12] - node.target[0], w[13] - node.target[1], w[14] - node.target[2])))
      lights.directionalColors.push(color)
    } else if (node.lightKind === LIGHT_POINT) {
      lights.points.push(transformPoint(this.view, w[12], w[13], w[14]))
      lights.pointColors.push(color)
      lights.pointRanges.push([node.distance, node.decay])
    } else if (node.lightKind === LIGHT_HEMISPHERE) {
      lights.hemisphereUps.push(normalize(transformDirection(this.view, w[12], w[13], w[14])))
      lights.skies.push(color)
      lights.grounds.push(node.groundColor.map((value) => value * node.intensity))
    }
  }

  private light(n: number[], p: number[]): number[] {
    const l = this.lights
    const out = [...l.ambient]
    l.directions.forEach((d, i) => {
      const amount = Math.max(0, dot(n, d))
      for (let k = 0; k < 3; k++) out[k] += amount * l.directionalColors[i][k]
    })
    l.hemisphereUps.forEach((up, i) => {
      const t = dot(n, up) * 0.5 + 0.5
      for (let k = 0; k < 3; k++) out[k] += l.grounds[i][k] + (l.skies[i][k] - l.grounds[i][k]) * t
    })
    l.points.forEach((position, i) => {
      const toLight = [position[0] - p[0], position[1] - p[1], position[2] - p[2]]
      const distance = Math.hypot(toLight[0], toLight[1], toLight[2])
      if (distance === 0) return
      const amount = Math.max(0, dot(n, toLight) / distance)
      const [range, decay] = l.pointRanges[i]
      let falloff = decay > 0 ? 1 / Math.max(Math.pow(distance, decay), 0.01) : 1
      if (range > 0) falloff *= Math.pow(Math.max(0, Math.min(1, 1 - Math.pow(distance / range, 4))), 2)
      for (let k = 0; k < 3; k++) out[k] += amount * falloff * l.pointColors[i][k]
    })
    return out
  }

  private fogged(rgb: number[], distance: number, material: WebMaterial): number[] {
    const scene = this.scene
    if (!(material.flags & MATERIAL_FOG) || scene.fog === 0) return rgb
    let visibility = 1
    if (scene.fog === FOG_LINEAR && scene.fogFar > scene.fogNear) visibility = (scene.fogFar - distance) / (scene.fogFar - scene.fogNear)
    else if (scene.fog === FOG_EXP2) visibility = Math.exp(-Math.pow(scene.fogDensity * distance, 2))
    visibility = Math.max(0, Math.min(1, visibility))
    const fog = hexChannels(scene.fogColor)
    return rgb.map((value, k) => fog[k] + (value - fog[k]) * visibility)
  }

  private project(v: number[]): number[] {
    const p = this.camera.projection
    const x = p[0] * v[0] + p[4] * v[1] + p[8] * v[2] + p[12]
    const y = p[1] * v[0] + p[5] * v[1] + p[9] * v[2] + p[13]
    const w = p[3] * v[0] + p[7] * v[1] + p[11] * v[2] + p[15] || 1e-6
    return [(x / w + 1) * this.width * 0.5, (1 - y / w) * this.height * 0.5]
  }

  private visible(center: number[], radius: number): boolean {
    return this.planes.every((plane) => plane[0] * center[0] + plane[1] * center[1] + plane[2] * center[2] + plane[3] >= -radius)
  }

  private draw(node: WebNode): void {
    const material = materials[node.material]
    if (!material || !(material.flags & MATERIAL_VISIBLE)) return

    this.layer = node.renderOrder
    const modelView = multiply(this.view, node.world)
    if (node.kind === NODE_SPRITE) {
      this.drawSprite(modelView, material)
      return
    }
    const geometry = geometries[node.geometry]
    const positions = geometry ? buffers[geometry.position] : null
    if (!geometry || !positions) return

    const instances = node.instanced ? buffers[node.instanceMatrices] : null
    const tints = instances ? buffers[node.instanceColors] : null
    const count = node.instanced ? (instances ? Math.min(node.instanceCount, Math.floor(instances.f32.length / 16)) : 0) : 1
    for (let i = 0; i < count; i++) {
      const mv = instances ? multiply(modelView, instances.f32.subarray(i * 16, i * 16 + 16)) : modelView
      const tint = tints && tints.f32.length >= (i + 1) * 3 ? Array.from(tints.f32.subarray(i * 3, i * 3 + 3)) : [1, 1, 1]
      if (node.frustumCulled && !this.boundsVisible(positions.f32, mv)) {
        stats[6]++
        continue
      }
      stats[5]++
      this.drawGeometry(node.kind, geometry, positions.f32, material, mv, tint)
    }
  }

  private boundsVisible(p: Float32Array, mv: Matrix): boolean {
    let lo = [Infinity, Infinity, Infinity]
    let hi = [-Infinity, -Infinity, -Infinity]
    for (let i = 0; i < p.length; i += 3) {
      lo = [Math.min(lo[0], p[i]), Math.min(lo[1], p[i + 1]), Math.min(lo[2], p[i + 2])]
      hi = [Math.max(hi[0], p[i]), Math.max(hi[1], p[i + 1]), Math.max(hi[2], p[i + 2])]
    }
    const center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]
    const radius = Math.hypot(hi[0] - center[0], hi[1] - center[1], hi[2] - center[2])
    const scale = Math.sqrt(Math.max(mv[0] ** 2 + mv[1] ** 2 + mv[2] ** 2, mv[4] ** 2 + mv[5] ** 2 + mv[6] ** 2, mv[8] ** 2 + mv[9] ** 2 + mv[10] ** 2))
    return this.visible(transformPoint(mv, center[0], center[1], center[2]), radius * scale)
  }

  private drawGeometry(kind: number, geometry: WebGeometry, p: Float32Array, material: WebMaterial, mv: Matrix, tint: number[]): void {
    const vertexCount = Math.floor(p.length / 3)
    const view: number[][] = []
    for (let i = 0; i < vertexCount; i++) view.push(transformPoint(mv, p[i * 3], p[i * 3 + 1], p[i * 3 + 2]))
    const colorBuffer = material.flags & MATERIAL_VERTEX_COLORS ? buffers[geometry.color] : null
    const colors = colorBuffer && colorBuffer.f32.length >= vertexCount * 3 ? colorBuffer.f32 : null
    const base = (i: number) => material.color.map((value, k) => value * tint[k] * (colors ? colors[i * 3 + k] : 1))
    const indexBuffer = buffers[geometry.index]
    const indices = indexBuffer && indexBuffer.u32.length > 0 ? indexBuffer.u32 : null
    const at = (i: number) => (indices ? indices[i] : i)
    const count = indices ? indices.length : vertexCount

    if (kind === NODE_POINTS) {
      for (let i = 0; i < vertexCount; i++) this.drawPoint(view[i], base(i), material)
      return
    }
    if (kind === NODE_LINE || kind === NODE_LINE_SEGMENTS) {
      const step = kind === NODE_LINE ? 1 : 2
      for (let i = 0; i + 1 < count; i += step) this.drawLine(view[at(i)], view[at(i + 1)], base(at(i)), material)
      return
    }
    stats[0] += Math.floor(count / 3)
    for (let t = 0; t + 2 < count; t += 3) {
      const ids = [at(t), at(t + 1), at(t + 2)]
      const corners = ids.map((id) => view[id])
      if (material.flags & MATERIAL_WIREFRAME) {
        for (let k = 0; k < 3; k++) this.drawLine(corners[k], corners[(k + 1) % 3], base(ids[k]), material)
        continue
      }
      this.drawTriangle(corners, ids.map(base), material)
    }
  }

  private drawTriangle(corners: number[][], colors: number[][], material: WebMaterial): void {
    const nearZ = -this.near
    if (corners.every((c) => c[2] > nearZ)) {
      stats[1]++
      return
    }
    let n = cross(
      [corners[1][0] - corners[0][0], corners[1][1] - corners[0][1], corners[1][2] - corners[0][2]],
      [corners[2][0] - corners[0][0], corners[2][1] - corners[0][1], corners[2][2] - corners[0][2]],
    )
    const facing = this.orthographic ? -n[2] : dot(n, corners[0])
    const back = facing >= 0
    if ((material.side === 0 && back) || (material.side === 1 && !back)) {
      stats[2]++
      return
    }
    if (back) n = n.map((value) => -value)
    n = normalize(n)

    const centroid = [0, 1, 2].map((k) => (corners[0][k] + corners[1][k] + corners[2][k]) / 3)
    const average = [0, 1, 2].map((k) => (colors[0][k] + colors[1][k] + colors[2][k]) / 3)
    let rgb = average
    if (material.shade === 1) rgb = average.map((value, k) => value * this.light(n, centroid)[k])
    else if (material.shade === 2) rgb = [n[0] * 0.5 + 0.5, n[1] * 0.5 + 0.5, n[2] * 0.5 + 0.5]
    rgb = this.fogged(rgb, -centroid[2], material)

    const polygon = this.clip(corners)
    if (polygon.length < 3) {
      stats[3]++
      return
    }
    const screen = polygon.map((c) => this.project(c))
    this.shapes.push({
      depth: Math.max(...polygon.map((c) => -c[2])),
      xs: screen.map((s) => s[0]),
      ys: screen.map((s) => s[1]),
      color: cssColor(rgb),
      alpha: this.alpha(material),
      additive: (material.flags & MATERIAL_ADDITIVE) !== 0,
      layer: this.layer,
      line: false,
      width: 1,
    })
  }

  // Near-plane clip in view space.
  private clip(corners: number[][]): number[][] {
    const nearZ = -this.near
    const out: number[][] = []
    for (let e = 0; e < corners.length; e++) {
      const a = corners[e]
      const b = corners[(e + 1) % corners.length]
      const aIn = a[2] <= nearZ
      const bIn = b[2] <= nearZ
      if (aIn) out.push(a)
      if (aIn !== bIn) {
        const t = (nearZ - a[2]) / (b[2] - a[2])
        out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, nearZ])
      }
    }
    return out
  }

  private drawLine(a: number[], b: number[], rgb: number[], material: WebMaterial): void {
    const segment = this.clip([a, b, b]).slice(0, 2)
    if (segment.length < 2) return
    const screen = segment.map((c) => this.project(c))
    this.shapes.push({
      depth: Math.max(-segment[0][2], -segment[1][2]),
      xs: screen.map((s) => s[0]),
      ys: screen.map((s) => s[1]),
      color: cssColor(this.fogged(rgb, -(a[2] + b[2]) / 2, material)),
      alpha: this.alpha(material),
      additive: (material.flags & MATERIAL_ADDITIVE) !== 0,
      layer: this.layer,
      line: true,
      width: Math.min(3, Math.max(1, Math.round(material.size))),
    })
  }

  private drawPoint(p: number[], rgb: number[], material: WebMaterial): void {
    if (p[2] > -this.near) return
    const attenuate = material.flags & MATERIAL_SIZE_ATTENUATION && !this.orthographic
    const size = Math.max(1, attenuate ? (material.size * this.height * 0.5) / -p[2] : material.size)
    const [x, y] = this.project(p)
    const half = size / 2
    this.shapes.push({
      depth: -p[2],
      xs: [x - half, x + half, x + half, x - half],
      ys: [y - half, y - half, y + half, y + half],
      color: cssColor(this.fogged(rgb, -p[2], material)),
      alpha: this.alpha(material),
      additive: (material.flags & MATERIAL_ADDITIVE) !== 0,
      layer: this.layer,
      line: false,
      width: 1,
    })
  }

  private drawSprite(mv: Matrix, material: WebMaterial): void {
    const center = transformPoint(mv, 0, 0, 0)
    const halfX = Math.hypot(mv[0], mv[1], mv[2]) / 2
    const halfY = Math.hypot(mv[4], mv[5], mv[6]) / 2
    if (center[2] > -this.near || !this.visible(center, Math.max(halfX, halfY) * 2)) return
    const corners = [
      [center[0] - halfX, center[1] + halfY, center[2]],
      [center[0] + halfX, center[1] + halfY, center[2]],
      [center[0] + halfX, center[1] - halfY, center[2]],
      [center[0] - halfX, center[1] - halfY, center[2]],
    ].map((c) => this.project(c))
    this.shapes.push({
      depth: -center[2],
      xs: corners.map((c) => c[0]),
      ys: corners.map((c) => c[1]),
      color: cssColor(this.fogged(material.color, -center[2], material)),
      alpha: this.alpha(material),
      additive: (material.flags & MATERIAL_ADDITIVE) !== 0,
      layer: this.layer,
      line: false,
      width: 1,
    })
  }

  private alpha(material: WebMaterial): number {
    return material.flags & MATERIAL_TRANSPARENT ? Math.max(0, Math.min(1, material.opacity)) : 1
  }
}

// The frame canvas: fixed behind the app's DOM, one CSS pixel per pixel.
function ensureCanvas(width: number, height: number): HTMLCanvasElement {
  if (!canvas) {
    canvas = document.createElement('canvas')
    canvas.style.position = 'fixed'
    canvas.style.left = '0'
    canvas.style.top = '0'
    canvas.style.zIndex = '-1'
    canvas.style.pointerEvents = 'none'
    document.body.prepend(canvas)
    document.body.style.background = 'transparent'
  }
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
  }
  return canvas
}

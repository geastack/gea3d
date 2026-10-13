// SPDX-License-Identifier: GPL-3.0-only
// Typed port of the three.js scene-graph core (API-compatible subset).
//
// The scene graph is mirrored natively (gea3d_native.cpp, float32): every
// Object3D owns a native node and syncNative() writes only what changed since
// the last frame (a transform, a flag, a material), so an unchanged object
// costs a few comparisons per frame. World matrices, culling and drawing are
// native; the TS matrices below serve app queries such as getWorldPosition.
//
// geatsc-driven design rules (embedded mode, verified against the emitted
// C++ — see gea3d/README.md "compiler findings"):
// - NO nullable class-typed fields (`T | null` boxes to gea_cpp_value):
//   sentinels are used instead (parent === this, radius < 0, count === 0).
// - NO instanceof / downcasts (instanceof lowers to constant false):
//   classification is double-dispatch through virtual methods.
// - NO interface-typed fields, NO dynamic property bags, NO Symbol protocols.
// - NO statements before super() in derived constructors (miscompiles).

import { ValueCache } from './cache'
import { Fog } from './fog'
import { Color, Euler, Matrix4, Quaternion, Vector3 } from './math'
import { Material } from './materials'
import {
  bufferDataF32,
  bufferDataU32,
  createBuffer,
  createGeometry,
  createNode,
  deleteBuffer,
  deleteGeometry,
  deleteNode,
  geometryBuffers,
  nodeFlags,
  nodeGeometry,
  nodeInstances,
  nodeLight,
  nodeLightTarget,
  nodeMatrix,
  nodeParent,
  nodeTransform,
  raycast,
  rayResult,
  sceneEnvironment,
  BACKGROUND_FLAT,
  BACKGROUND_GRADIENT,
  BACKGROUND_NONE,
  LIGHT_AMBIENT,
  LIGHT_DIRECTIONAL,
  LIGHT_HEMISPHERE,
  LIGHT_POINT,
  NODE_GROUP,
  NODE_LIGHT,
  NODE_LINE,
  NODE_LINE_SEGMENTS,
  NODE_MESH,
  NODE_POINTS,
  NODE_SPRITE,
  RAY_DISTANCE,
  RAY_FACE,
  RAY_INSTANCE,
  RAY_POINT_X,
  RAY_POINT_Y,
  RAY_POINT_Z,
} from '../native/api'

let _object3DId = 0

export class Sphere {
  readonly center: Vector3
  radius: number

  constructor() {
    this.center = new Vector3()
    this.radius = -1 // < 0 = not computed yet
  }
}

export class BufferAttribute {
  readonly array: Float32Array
  readonly itemSize: number
  readonly count: number
  // Bumped by `needsUpdate = true`; the renderer re-uploads when it moves.
  version: number
  // Native buffer holding `array` as of `uploadedVersion` (-1 = none).
  nativeBuffer: number
  private uploadedVersion: number

  constructor(array: Float32Array, itemSize: number) {
    this.array = array
    this.itemSize = itemSize
    this.count = itemSize > 0 ? Math.floor(array.length / itemSize) : 0
    this.version = 0
    this.nativeBuffer = -1
    this.uploadedVersion = -1
  }

  get needsUpdate(): boolean {
    return false
  }

  set needsUpdate(value: boolean) {
    if (value) this.version++
  }

  getX(index: number): number {
    return this.array[index * this.itemSize]
  }

  getY(index: number): number {
    return this.array[index * this.itemSize + 1]
  }

  getZ(index: number): number {
    return this.array[index * this.itemSize + 2]
  }

  setXYZ(index: number, x: number, y: number, z: number): BufferAttribute {
    const i = index * this.itemSize
    this.array[i] = x
    this.array[i + 1] = y
    this.array[i + 2] = z
    return this
  }

  // Uploads the array when it changed; returns its native buffer (-1 when
  // the attribute is empty).
  upload(): number {
    if (this.count === 0) return -1

    if (this.nativeBuffer < 0) {
      this.nativeBuffer = createBuffer()
      this.uploadedVersion = -1
    }
    if (this.uploadedVersion !== this.version) {
      bufferDataF32(this.nativeBuffer, this.array)
      this.uploadedVersion = this.version
    }
    return this.nativeBuffer
  }

  releaseNative(): void {
    if (this.nativeBuffer < 0) return

    deleteBuffer(this.nativeBuffer)
    this.nativeBuffer = -1
    // Geometries holding this attribute see a change.
    this.version++
  }
}

function float32ArrayFromNumbers(values: number[]): Float32Array {
  const array = new Float32Array(values.length)
  for (let i = 0; i < values.length; i++) array[i] = values[i]
  return array
}

export class Float32BufferAttribute extends BufferAttribute {
  constructor(values: number[], itemSize: number) {
    // No statements before super() — geatsc miscompiles them.
    super(float32ArrayFromNumbers(values), itemSize)
  }
}

// InstancedMesh per-instance data (matrices: itemSize 16, colours: 3).
export class InstancedBufferAttribute extends BufferAttribute {}

// Fixed-field attribute container. Attributes are never null: "absent" is an
// empty attribute (count === 0).
export class GeometryAttributes {
  position: BufferAttribute
  normal: BufferAttribute
  color: BufferAttribute
  uv: BufferAttribute

  constructor() {
    this.position = new BufferAttribute(new Float32Array(0), 3)
    this.normal = new BufferAttribute(new Float32Array(0), 3)
    this.color = new BufferAttribute(new Float32Array(0), 3)
    this.uv = new BufferAttribute(new Float32Array(0), 2)
  }
}

export class BufferGeometry {
  readonly attributes: GeometryAttributes
  readonly boundingSphere: Sphere
  // Empty = non-indexed (sequential triangles over position).
  private indexArray: Uint32Array
  private indexVersion: number
  private uploadedIndexVersion: number
  // Native ids (-1 = not uploaded / absent). Plain numeric sentinels —
  // nullable or object-typed fields box in embedded geatsc.
  nativeGeometry: number
  nativePositionBuffer: number
  nativeIndexBuffer: number
  nativeColorBuffer: number
  private nativeNormalBuffer: number
  // Bumped when an attribute is replaced; with the attribute and index
  // versions it tells whether anything changed since the last sync.
  private attributeSets: number
  private syncedState: number

  constructor() {
    this.attributes = new GeometryAttributes()
    this.boundingSphere = new Sphere()
    this.indexArray = new Uint32Array(0)
    this.indexVersion = 0
    this.uploadedIndexVersion = -1
    this.nativeGeometry = -1
    this.nativePositionBuffer = -1
    this.nativeIndexBuffer = -1
    this.nativeColorBuffer = -1
    this.nativeNormalBuffer = -1
    this.attributeSets = 0
    this.syncedState = -1
  }

  setAttribute(name: string, attribute: BufferAttribute): BufferGeometry {
    if (name === 'position') this.attributes.position = attribute
    else if (name === 'normal') this.attributes.normal = attribute
    else if (name === 'color') this.attributes.color = attribute
    else if (name === 'uv') this.attributes.uv = attribute
    this.boundingSphere.radius = -1
    this.attributeSets++
    return this
  }

  getAttribute(name: string): BufferAttribute {
    if (name === 'position') return this.attributes.position
    if (name === 'normal') return this.attributes.normal
    if (name === 'color') return this.attributes.color
    return this.attributes.uv
  }

  setIndex(indices: number[]): BufferGeometry {
    const arr = new Uint32Array(indices.length)
    for (let i = 0; i < indices.length; i++) arr[i] = indices[i]
    this.indexArray = arr
    this.indexVersion++
    return this
  }

  // Returns the index array, lazily building sequential indices for
  // non-indexed geometry (cached).
  ensureIndexArray(): Uint32Array {
    if (this.indexArray.length > 0) return this.indexArray
    const count = this.attributes.position.count
    if (count === 0) return this.indexArray
    const arr = new Uint32Array(count)
    for (let i = 0; i < count; i++) arr[i] = i
    this.indexArray = arr
    this.indexVersion++
    return arr
  }

  triangleCount(): number {
    if (this.indexArray.length > 0) return Math.floor(this.indexArray.length / 3)
    return Math.floor(this.attributes.position.count / 3)
  }

  computeBoundingSphere(): void {
    const sphere = this.boundingSphere
    const pos = this.attributes.position
    if (pos.count === 0) {
      sphere.center.set(0, 0, 0)
      sphere.radius = 0
      return
    }
    const arr = pos.array
    const n = pos.count
    let minX = arr[0]
    let minY = arr[1]
    let minZ = arr[2]
    let maxX = minX
    let maxY = minY
    let maxZ = minZ
    for (let i = 1; i < n; i++) {
      const x = arr[i * 3]
      const y = arr[i * 3 + 1]
      const z = arr[i * 3 + 2]
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
      if (z < minZ) minZ = z
      if (z > maxZ) maxZ = z
    }
    const cx = (minX + maxX) * 0.5
    const cy = (minY + maxY) * 0.5
    const cz = (minZ + maxZ) * 0.5
    sphere.center.set(cx, cy, cz)
    let maxDistSq = 0
    for (let i = 0; i < n; i++) {
      const dx = arr[i * 3] - cx
      const dy = arr[i * 3 + 1] - cy
      const dz = arr[i * 3 + 2] - cz
      const distSq = dx * dx + dy * dy + dz * dz
      if (distSq > maxDistSq) maxDistSq = distSq
    }
    sphere.radius = Math.sqrt(maxDistSq)
  }

  // Uploads changed attributes and returns the native geometry.
  syncNative(): number {
    const attributes = this.attributes
    const state = attributes.position.version + attributes.color.version + attributes.normal.version + this.indexVersion + this.attributeSets
    if (this.nativeGeometry >= 0 && state === this.syncedState) return this.nativeGeometry

    if (this.nativeGeometry < 0) this.nativeGeometry = createGeometry()
    this.syncedState = state

    const position = attributes.position.upload()
    const color = attributes.color.upload()
    const normal = attributes.normal.upload()
    const index = this.uploadIndex()
    if (
      position !== this.nativePositionBuffer ||
      index !== this.nativeIndexBuffer ||
      color !== this.nativeColorBuffer ||
      normal !== this.nativeNormalBuffer
    ) {
      geometryBuffers(this.nativeGeometry, position, index, color, normal)
      this.nativePositionBuffer = position
      this.nativeIndexBuffer = index
      this.nativeColorBuffer = color
      this.nativeNormalBuffer = normal
    }
    return this.nativeGeometry
  }

  // Frees the native copies; the next render uploads again.
  dispose(): void {
    const attributes = this.attributes
    attributes.position.releaseNative()
    attributes.normal.releaseNative()
    attributes.color.releaseNative()
    if (this.nativeIndexBuffer >= 0) deleteBuffer(this.nativeIndexBuffer)
    if (this.nativeGeometry >= 0) deleteGeometry(this.nativeGeometry)
    this.nativeGeometry = -1
    this.nativePositionBuffer = -1
    this.nativeIndexBuffer = -1
    this.nativeColorBuffer = -1
    this.nativeNormalBuffer = -1
    this.uploadedIndexVersion = -1
    this.syncedState = -1
  }

  private uploadIndex(): number {
    if (this.indexArray.length === 0) return -1

    let buffer = this.nativeIndexBuffer
    if (buffer < 0) {
      buffer = createBuffer()
      this.uploadedIndexVersion = -1
    }
    if (this.uploadedIndexVersion !== this.indexVersion) {
      bufferDataU32(buffer, this.indexArray)
      this.uploadedIndexVersion = this.indexVersion
    }
    return buffer
  }
}

// NOTE: no module-scope or lazy scratch singletons (miscompile in embedded
// geatsc — see math.ts). Cold-path helpers allocate small temporaries.

// Typed collection target for a scene walk (kept for apps that walk scenes
// themselves). Mesh and Light register via collectSelf double dispatch.
export class SceneCollector {
  readonly meshes: Mesh[]
  meshCount: number
  readonly lights: Light[]
  lightCount: number

  constructor() {
    this.meshes = []
    this.meshCount = 0
    this.lights = []
    this.lightCount = 0
  }

  reset(): void {
    this.meshCount = 0
    this.lightCount = 0
  }

  addMesh(mesh: Mesh): void {
    if (this.meshCount < this.meshes.length) this.meshes[this.meshCount] = mesh
    else this.meshes.push(mesh)
    this.meshCount++
  }

  addLight(light: Light): void {
    if (this.lightCount < this.lights.length) this.lights[this.lightCount] = light
    else this.lights.push(light)
    this.lightCount++
  }
}

// A ray hit, three.js Raycaster style.
export class Intersection {
  distance: number
  readonly point: Vector3
  object: Object3D
  faceIndex: number
  instanceId: number

  constructor(object: Object3D) {
    this.distance = 0
    this.point = new Vector3()
    this.object = object
    this.faceIndex = -1
    this.instanceId = -1
  }
}

// Which view of an Object3D's rotation is current; the other is recomputed
// on its next read. `version` bumps on every rotation write.
class RotationState {
  eulerStale: boolean
  quaternionStale: boolean
  version: number

  constructor() {
    this.eulerStale = false
    this.quaternionStale = false
    this.version = 0
  }
}

// What an Object3D last wrote to its native node. Explicit matrices are
// sent every frame: the native node keeps them only when they changed.
class NativeNode {
  id: number
  parent: number
  // 0 nothing sent, 1 position/quaternion/scale, 2 explicit matrix
  mode: number
  px: number
  py: number
  pz: number
  sx: number
  sy: number
  sz: number
  rotationVersion: number
  visible: boolean
  frustumCulled: boolean
  renderOrder: number

  constructor() {
    this.id = -1
    this.parent = -2
    this.mode = 0
    this.px = 0
    this.py = 0
    this.pz = 0
    this.sx = 0
    this.sy = 0
    this.sz = 0
    this.rotationVersion = -1
    this.visible = true
    this.frustumCulled = true
    this.renderOrder = Number.NaN
  }

  reset(): void {
    this.id = -1
    this.parent = -2
    this.mode = 0
    this.renderOrder = Number.NaN
  }

  sendMatrix(elements: Float32Array): void {
    nodeMatrix(this.id, elements)
    this.mode = 2
  }
}

export class Object3D {
  readonly id: number
  name: string
  readonly position: Vector3
  readonly rotation: Euler
  readonly quaternion: Quaternion
  readonly scale: Vector3
  readonly up: Vector3
  readonly matrix: Matrix4
  readonly matrixWorld: Matrix4
  matrixAutoUpdate: boolean
  matrixWorldNeedsUpdate: boolean
  visible: boolean
  frustumCulled: boolean
  renderOrder: number
  // Sentinel: parent points at self when detached (nullable class fields
  // box). The flag is the truth — object identity comparison (`x === this`)
  // does not survive the gea_cpp_strict_equals lowering on wrapped pointers.
  parent: Object3D
  private parentAttached: boolean
  readonly children: Object3D[]
  isCamera: boolean
  // Native node kind (NODE_* in native/api.ts), set by subclasses.
  nodeKind: number
  readonly nativeNode: NativeNode
  private readonly rotationState: RotationState

  constructor() {
    _object3DId++
    this.id = _object3DId
    this.name = ''
    this.position = new Vector3()
    this.rotation = new Euler()
    this.quaternion = new Quaternion()
    this.scale = new Vector3(1, 1, 1)
    this.up = new Vector3(0, 1, 0)
    this.matrix = new Matrix4()
    this.matrixWorld = new Matrix4()
    this.matrixAutoUpdate = true
    this.matrixWorldNeedsUpdate = false
    this.visible = true
    this.frustumCulled = true
    this.renderOrder = 0
    this.parent = this
    this.parentAttached = false
    this.children = []
    this.isCamera = false
    this.nodeKind = NODE_GROUP
    this.nativeNode = new NativeNode()
    this.rotationState = new RotationState()

    // Euler and quaternion stay in sync lazily: a write marks the other view
    // stale, and a read of a stale view recomputes it once.
    const rotation = this.rotation
    const quaternion = this.quaternion
    const state = this.rotationState
    rotation._onChange(() => {
      state.quaternionStale = true
      state.eulerStale = false
      state.version++
    })
    quaternion._onChange(() => {
      state.eulerStale = true
      state.quaternionStale = false
      state.version++
    })
    rotation._onRead(() => {
      if (!state.eulerStale) return
      state.eulerStale = false
      rotation.setFromQuaternion(quaternion, '', false)
    })
    quaternion._onRead(() => {
      if (!state.quaternionStale) return
      state.quaternionStale = false
      quaternion.setFromEuler(rotation, false)
    })
  }

  hasParent(): boolean {
    return this.parentAttached
  }

  // The top of this object's graph (itself when detached).
  root(): Object3D {
    return this.parentAttached ? this.parent.root() : this
  }

  // Internal: writes the parent link from the child's side. `this.parent =
  // <param>` stores a real pointer; `object.parent = this` inside add() nulls
  // out (storing the enclosing `this` into another object's field miscompiles
  // in embedded geatsc — passing `this` as a call argument works).
  attachToParent(parent: Object3D): void {
    this.parent = parent
    this.parentAttached = true
  }

  add(object: Object3D): Object3D {
    if (object.parentAttached) object.parent.remove(object)
    // Through self(): geatsc emits `std::move` of the const receiver for a
    // bare `this` argument, which ESP-IDF's -Werror=redundant-move rejects.
    object.attachToParent(this.self())
    this.children.push(object)
    return this
  }

  private self(): Object3D {
    return this
  }

  remove(object: Object3D): Object3D {
    // Identity search by id — Array.indexOf on class instances relies on the
    // same strict-equals lowering that hasParent avoids.
    const children = this.children
    for (let i = 0; i < children.length; i++) {
      if (children[i].id === object.id) {
        object.parent = object
        object.parentAttached = false
        object.releaseNative()
        children.splice(i, 1)
        return this
      }
    }
    return this
  }

  clear(): Object3D {
    for (let i = 0; i < this.children.length; i++) {
      const child = this.children[i]
      child.parent = child
      child.parentAttached = false
      child.releaseNative()
    }
    this.children.length = 0
    return this
  }

  traverse(callback: (object: Object3D) => void): void {
    callback(this)
    const children = this.children
    for (let i = 0; i < children.length; i++) children[i].traverse(callback)
  }

  // Double-dispatch hook: subclasses that render register themselves.
  collectSelf(collector: SceneCollector): void {
    // base objects contribute nothing
  }

  collect(collector: SceneCollector): void {
    if (!this.visible) return
    this.collectSelf(collector)
    const children = this.children
    for (let i = 0; i < children.length; i++) children[i].collect(collector)
  }

  rotateX(angle: number): Object3D {
    return this.rotateOnAxisComponents(1, 0, 0, angle)
  }

  rotateY(angle: number): Object3D {
    return this.rotateOnAxisComponents(0, 1, 0, angle)
  }

  rotateZ(angle: number): Object3D {
    return this.rotateOnAxisComponents(0, 0, 1, angle)
  }

  // Rotates about a normalized axis in object space.
  rotateOnAxis(axis: Vector3, angle: number): Object3D {
    return this.rotateOnAxisComponents(axis.x, axis.y, axis.z, angle)
  }

  translateOnAxis(axis: Vector3, distance: number): Object3D {
    return this.translateOnAxisComponents(axis.x, axis.y, axis.z, distance)
  }

  translateX(distance: number): Object3D {
    return this.translateOnAxisComponents(1, 0, 0, distance)
  }

  translateY(distance: number): Object3D {
    return this.translateOnAxisComponents(0, 1, 0, distance)
  }

  translateZ(distance: number): Object3D {
    return this.translateOnAxisComponents(0, 0, 1, distance)
  }

  getWorldPosition(target: Vector3): Vector3 {
    this.updateWorldMatrix(true, false)
    return target.setFromMatrixPosition(this.matrixWorld)
  }

  // Orients the object toward the target point. Cameras look down their -Z;
  // other objects face +Z toward the target (three.js semantics). Unlike
  // three.js this takes numbers only — a `Vector3 | number` union parameter
  // boxes in geatsc; use lookAtTarget(v) for the vector form. No parent-
  // rotation compensation (fine for scene-root cameras).
  lookAt(x: number, y: number = 0, z: number = 0): void {
    let eyeX = this.position.x
    let eyeY = this.position.y
    let eyeZ = this.position.z
    if (this.parentAttached) {
      this.updateWorldMatrix(true, false)
      const we = this.matrixWorld.elements
      eyeX = we[12]
      eyeY = we[13]
      eyeZ = we[14]
    }
    const up = this.up
    if (this.isCamera) this.quaternion.setFromLookAt(eyeX, eyeY, eyeZ, x, y, z, up.x, up.y, up.z)
    else this.quaternion.setFromLookAt(x, y, z, eyeX, eyeY, eyeZ, up.x, up.y, up.z)
  }

  lookAtTarget(target: Vector3): void {
    this.lookAt(target.x, target.y, target.z)
  }

  updateMatrix(): void {
    this.matrix.compose(this.position, this.quaternion, this.scale)
    this.matrixWorldNeedsUpdate = true
  }

  updateMatrixWorld(force: boolean = false): void {
    if (this.matrixAutoUpdate) this.updateMatrix()
    let f = force
    if (this.matrixWorldNeedsUpdate || f) {
      if (this.hasParent()) this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix)
      else this.matrixWorld.copy(this.matrix)
      this.matrixWorldNeedsUpdate = false
      f = true
    }
    const children = this.children
    for (let i = 0; i < children.length; i++) children[i].updateMatrixWorld(f)
  }

  updateWorldMatrix(updateParents: boolean, updateChildren: boolean): void {
    if (updateParents && this.hasParent()) this.parent.updateWorldMatrix(true, false)
    if (this.matrixAutoUpdate) this.updateMatrix()
    if (this.hasParent()) this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix)
    else this.matrixWorld.copy(this.matrix)
    if (updateChildren) {
      const children = this.children
      for (let i = 0; i < children.length; i++) children[i].updateWorldMatrix(false, true)
    }
  }

  // ---- native mirror

  // Writes this subtree's changes to its native nodes. `parent` is the
  // native parent node (-1 for a root).
  syncNative(parent: number): void {
    const node = this.nativeNode
    if (node.id < 0) node.id = createNode(this.nodeKind)
    if (node.parent !== parent) {
      nodeParent(node.id, parent)
      node.parent = parent
    }
    if (node.visible !== this.visible || node.frustumCulled !== this.frustumCulled || node.renderOrder !== this.renderOrder) {
      nodeFlags(node.id, this.visible ? 1 : 0, this.frustumCulled ? 1 : 0, this.renderOrder)
      node.visible = this.visible
      node.frustumCulled = this.frustumCulled
      node.renderOrder = this.renderOrder
    }
    if (!this.visible) return

    this.syncTransform(node)
    this.syncSelf()
    const children = this.children
    for (let i = 0; i < children.length; i++) children[i].syncNative(node.id)
  }

  // Double-dispatch hook: subclasses write their own native state.
  syncSelf(): void {
    // base objects carry only a transform
  }

  // Frees this subtree's native nodes; a later sync recreates them.
  releaseNative(): void {
    const node = this.nativeNode
    if (node.id >= 0) deleteNode(node.id)
    node.reset()
    this.releaseSelf()
    const children = this.children
    for (let i = 0; i < children.length; i++) children[i].releaseNative()
  }

  // Double-dispatch hook: subclasses forget per-node native bindings.
  releaseSelf(): void {
    // nothing bound
  }

  // Double-dispatch hook for Raycaster: objects that can be hit append
  // their nearest hit along the world-space ray.
  raycastSelf(origin: Vector3, direction: Vector3, near: number, far: number, intersects: Intersection[]): void {
    // base objects are never hit
  }

  private syncTransform(node: NativeNode): void {
    if (!this.matrixAutoUpdate) {
      node.sendMatrix(this.matrix.elements)
      return
    }

    const p = this.position
    const s = this.scale
    const version = this.rotationState.version
    if (
      node.mode === 1 &&
      node.rotationVersion === version &&
      node.px === p.x && node.py === p.y && node.pz === p.z &&
      node.sx === s.x && node.sy === s.y && node.sz === s.z
    ) {
      return
    }
    const q = this.quaternion
    nodeTransform(node.id, p.x, p.y, p.z, q.x, q.y, q.z, q.w, s.x, s.y, s.z)
    node.mode = 1
    node.rotationVersion = version
    node.px = p.x
    node.py = p.y
    node.pz = p.z
    node.sx = s.x
    node.sy = s.y
    node.sz = s.z
  }

  private rotateOnAxisComponents(x: number, y: number, z: number, angle: number): Object3D {
    const half = angle / 2
    const s = Math.sin(half)
    this.quaternion.multiplyComponents(x * s, y * s, z * s, Math.cos(half))
    return this
  }

  // Moves along an object-space axis: the axis rotated by the quaternion.
  private translateOnAxisComponents(x: number, y: number, z: number, distance: number): Object3D {
    const q = this.quaternion
    const qx = q.x
    const qy = q.y
    const qz = q.z
    const qw = q.w
    const tx = 2 * (qy * z - qz * y)
    const ty = 2 * (qz * x - qx * z)
    const tz = 2 * (qx * y - qy * x)
    const p = this.position
    p.x += (x + qw * tx + qy * tz - qz * ty) * distance
    p.y += (y + qw * ty + qz * tx - qx * tz) * distance
    p.z += (z + qw * tz + qx * ty - qy * tx) * distance
    return this
  }
}

export class Group extends Object3D {}

// An object drawn from a geometry and a material (Mesh, Line, Points).
export class Drawable extends Object3D {
  geometry: BufferGeometry
  material: Material
  private boundGeometry: number
  private boundMaterial: number

  constructor(geometry: BufferGeometry, material: Material) {
    super()
    this.geometry = geometry
    this.material = material
    this.boundGeometry = -2
    this.boundMaterial = -2
  }

  syncSelf(): void {
    const geometry = this.geometry.syncNative()
    const material = this.material.syncNative()
    if (geometry === this.boundGeometry && material === this.boundMaterial) return

    nodeGeometry(this.nativeNode.id, geometry, material)
    this.boundGeometry = geometry
    this.boundMaterial = material
  }

  releaseSelf(): void {
    this.boundGeometry = -2
    this.boundMaterial = -2
  }
}

export class Mesh extends Drawable {
  constructor(geometry: BufferGeometry, material: Material) {
    super(geometry, material)
    this.nodeKind = NODE_MESH
  }

  collectSelf(collector: SceneCollector): void {
    collector.addMesh(this)
  }

  raycastSelf(origin: Vector3, direction: Vector3, near: number, far: number, intersects: Intersection[]): void {
    const node = this.nativeNode.id
    if (node < 0) return
    if (raycast(node, origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, near, far) === 0) return

    const hit = new Intersection(this)
    hit.distance = rayResult(RAY_DISTANCE)
    hit.point.set(rayResult(RAY_POINT_X), rayResult(RAY_POINT_Y), rayResult(RAY_POINT_Z))
    hit.faceIndex = rayResult(RAY_FACE)
    hit.instanceId = rayResult(RAY_INSTANCE)
    intersects.push(hit)
  }
}

// Many copies of one geometry, each placed by its own matrix (and tinted by
// an optional per-instance colour), drawn in one native call.
export class InstancedMesh extends Mesh {
  readonly instanceMatrix: InstancedBufferAttribute
  // Empty until setColorAt is first called.
  instanceColor: InstancedBufferAttribute
  count: number
  private readonly sentInstances: ValueCache

  constructor(geometry: BufferGeometry, material: Material, count: number) {
    super(geometry, material)
    this.count = Math.max(0, Math.floor(count))
    this.instanceMatrix = new InstancedBufferAttribute(new Float32Array(this.count * 16), 16)
    this.instanceColor = new InstancedBufferAttribute(new Float32Array(0), 3)
    this.sentInstances = new ValueCache(3)
    for (let i = 0; i < this.count; i++) {
      const offset = i * 16
      this.instanceMatrix.array[offset] = 1
      this.instanceMatrix.array[offset + 5] = 1
      this.instanceMatrix.array[offset + 10] = 1
      this.instanceMatrix.array[offset + 15] = 1
    }
  }

  setMatrixAt(index: number, matrix: Matrix4): void {
    const array = this.instanceMatrix.array
    const e = matrix.elements
    const offset = index * 16
    for (let i = 0; i < 16; i++) array[offset + i] = e[i]
  }

  getMatrixAt(index: number, matrix: Matrix4): void {
    const array = this.instanceMatrix.array
    const e = matrix.elements
    const offset = index * 16
    for (let i = 0; i < 16; i++) e[i] = array[offset + i]
  }

  setColorAt(index: number, color: Color): void {
    if (this.instanceColor.count === 0) {
      const colors = new Float32Array(this.instanceMatrix.count * 3)
      for (let i = 0; i < colors.length; i++) colors[i] = 1
      this.instanceColor = new InstancedBufferAttribute(colors, 3)
    }
    const offset = index * 3
    this.instanceColor.array[offset] = color.r
    this.instanceColor.array[offset + 1] = color.g
    this.instanceColor.array[offset + 2] = color.b
  }

  syncSelf(): void {
    super.syncSelf()
    const matrices = this.instanceMatrix.upload()
    const colors = this.instanceColor.upload()
    const count = Math.min(this.count, this.instanceMatrix.count)
    const sent = this.sentInstances
    sent.begin()
    sent.put(0, matrices)
    sent.put(1, colors)
    sent.put(2, count)
    if (sent.dirty) nodeInstances(this.nativeNode.id, matrices, colors, count)
  }

  releaseSelf(): void {
    super.releaseSelf()
    this.sentInstances.invalidate()
  }
}

// A polyline through consecutive vertices.
export class Line extends Drawable {
  constructor(geometry: BufferGeometry, material: Material) {
    super(geometry, material)
    this.nodeKind = NODE_LINE
  }
}

// Independent segments between vertex pairs.
export class LineSegments extends Line {
  constructor(geometry: BufferGeometry, material: Material) {
    super(geometry, material)
    this.nodeKind = NODE_LINE_SEGMENTS
  }
}

// A square per vertex, `material.size` pixels (or world units with
// sizeAttenuation) across.
export class Points extends Drawable {
  constructor(geometry: BufferGeometry, material: Material) {
    super(geometry, material)
    this.nodeKind = NODE_POINTS
  }
}

// A camera-facing rectangle of scale.x by scale.y world units (its geometry
// stays empty).
export class Sprite extends Drawable {
  constructor(material: Material) {
    super(new BufferGeometry(), material)
    this.nodeKind = NODE_SPRITE
  }
}

export const LIGHT_KIND_AMBIENT = LIGHT_AMBIENT
export const LIGHT_KIND_DIRECTIONAL = LIGHT_DIRECTIONAL
export const LIGHT_KIND_HEMISPHERE = LIGHT_HEMISPHERE
export const LIGHT_KIND_POINT = LIGHT_POINT

// A Light's fields as last written to the native engine.
class SentLight {
  valid: boolean
  kind: number
  r: number
  g: number
  b: number
  intensity: number
  groundR: number
  groundG: number
  groundB: number
  distance: number
  decay: number
  tx: number
  ty: number
  tz: number

  constructor() {
    this.valid = false
    this.kind = 0
    this.r = 0
    this.g = 0
    this.b = 0
    this.intensity = 0
    this.groundR = 0
    this.groundG = 0
    this.groundB = 0
    this.distance = 0
    this.decay = 0
    this.tx = 0
    this.ty = 0
    this.tz = 0
  }

  matchesLight(light: Light): boolean {
    const c = light.color
    const g = light.groundColor
    return (
      this.valid && this.kind === light.lightKind &&
      this.r === c.r && this.g === c.g && this.b === c.b && this.intensity === light.intensity &&
      this.groundR === g.r && this.groundG === g.g && this.groundB === g.b &&
      this.distance === light.distance && this.decay === light.decay
    )
  }

  takeLight(light: Light): void {
    const c = light.color
    const g = light.groundColor
    this.kind = light.lightKind
    this.r = c.r
    this.g = c.g
    this.b = c.b
    this.intensity = light.intensity
    this.groundR = g.r
    this.groundG = g.g
    this.groundB = g.b
    this.distance = light.distance
    this.decay = light.decay
  }
}

export class Light extends Object3D {
  readonly color: Color
  intensity: number
  lightKind: number
  // DirectionalLight aim point (unused sentinel instance for other kinds).
  readonly target: Object3D
  // HemisphereLight ground color (white for other kinds).
  readonly groundColor: Color
  // PointLight range (0 = unlimited) and falloff exponent.
  distance: number
  decay: number
  private readonly sent: SentLight

  constructor(color: number = 0xffffff, intensity: number = 1) {
    super()
    this.color = new Color(color)
    this.intensity = intensity
    this.lightKind = LIGHT_KIND_AMBIENT
    this.target = new Object3D()
    this.groundColor = new Color(0xffffff)
    this.distance = 0
    this.decay = 2
    this.nodeKind = NODE_LIGHT
    this.sent = new SentLight()
  }

  collectSelf(collector: SceneCollector): void {
    collector.addLight(this)
  }

  syncSelf(): void {
    const node = this.nativeNode.id
    const c = this.color
    const g = this.groundColor
    const sent = this.sent
    if (!sent.matchesLight(this)) {
      nodeLight(node, this.lightKind, c.r, c.g, c.b, this.intensity, g.r, g.g, g.b, this.distance, this.decay)
      sent.takeLight(this)
    }

    const t = this.target.position
    if (!sent.valid || sent.tx !== t.x || sent.ty !== t.y || sent.tz !== t.z) {
      nodeLightTarget(node, t.x, t.y, t.z)
      sent.tx = t.x
      sent.ty = t.y
      sent.tz = t.z
    }
    sent.valid = true
  }

  releaseSelf(): void {
    this.sent.valid = false
  }
}

export class AmbientLight extends Light {}

// Shines from `position` toward `target` (three.js semantics). The default
// position (0, 1, 0) with target at the origin lights from straight above.
export class DirectionalLight extends Light {
  constructor(color: number = 0xffffff, intensity: number = 1) {
    super(color, intensity)
    this.lightKind = LIGHT_KIND_DIRECTIONAL
    this.position.set(0, 1, 0)
  }
}

// Sky colour from above (along `position`, default up) blending to the
// ground colour from below.
export class HemisphereLight extends Light {
  constructor(skyColor: number = 0xffffff, groundColor: number = 0xffffff, intensity: number = 1) {
    super(skyColor, intensity)
    this.lightKind = LIGHT_KIND_HEMISPHERE
    this.groundColor.setHex(groundColor)
    this.position.set(0, 1, 0)
  }
}

// Shines in every direction from `position`, fading with distance.
export class PointLight extends Light {
  constructor(color: number = 0xffffff, intensity: number = 1, distance: number = 0, decay: number = 2) {
    super(color, intensity)
    this.lightKind = LIGHT_KIND_POINT
    this.distance = distance
    this.decay = decay
  }
}

export class Scene extends Object3D {
  // Hex colour (like `new Color(hex).getHex()`); -1 = don't clear.
  backgroundColor: number
  // With a hex colour here the background is a vertical gradient from
  // backgroundColor (top) to this (bottom); -1 = flat.
  backgroundBottom: number
  // Distance fog toward fog.color; a Fog with far <= near is off.
  fog: Fog
  private readonly sentEnvironment: ValueCache

  constructor() {
    super()
    this.backgroundColor = 0x000000
    this.backgroundBottom = -1
    this.fog = new Fog(0x000000, 0, 0)
    this.sentEnvironment = new ValueCache(9)
  }

  // Writes the background and fog once the scene has its node (after
  // syncNative); `clear` = false keeps the pixels under the 3D view.
  syncEnvironment(clear: boolean): void {
    let background = BACKGROUND_NONE
    if (clear && this.backgroundColor >= 0) background = this.backgroundBottom >= 0 ? BACKGROUND_GRADIENT : BACKGROUND_FLAT
    const fog = this.fog
    const fogMode = fog.mode()
    const fogColor = fog.color.getHex()
    const sent = this.sentEnvironment
    sent.begin()
    sent.put(0, background)
    sent.put(1, this.backgroundColor)
    sent.put(2, this.backgroundBottom)
    sent.put(3, fogMode)
    sent.put(4, fogColor)
    sent.put(5, fog.near)
    sent.put(6, fog.far)
    sent.put(7, fog.density)
    if (!sent.dirty) return

    sceneEnvironment(this.nativeNode.id, background, this.backgroundColor, this.backgroundBottom, fogMode, fogColor, fog.near, fog.far, fog.density)
  }

  releaseSelf(): void {
    this.sentEnvironment.invalidate()
  }
}

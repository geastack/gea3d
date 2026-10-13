// SPDX-License-Identifier: GPL-3.0-only
// TS face of gea3d's native engine (gea3d_native.{h,cpp}). The `gea3dNative`
// global is bound to gea::host::gea3d::* by the geatsc-plugin-gea host shims;
// every native call in the library goes through the functions below. The DOM
// web target, which has no native code, swaps this module for api.web.ts.

declare const gea3dNative: {
  createBuffer(): number
  deleteBuffer(buffer: number): void
  bufferDataF32(buffer: number, data: Float32Array): void
  bufferDataU32(buffer: number, data: Uint32Array): void
  createGeometry(): number
  deleteGeometry(geometry: number): void
  geometryBuffers(geometry: number, position: number, index: number, color: number, normal: number): void
  createMaterial(): number
  deleteMaterial(material: number): void
  materialSet(material: number, shade: number, r: number, g: number, b: number, opacity: number, side: number, flags: number, size: number): void
  createNode(kind: number): number
  deleteNode(node: number): void
  nodeParent(node: number, parent: number): void
  nodeTransform(node: number, px: number, py: number, pz: number, qx: number, qy: number, qz: number, qw: number, sx: number, sy: number, sz: number): void
  nodeMatrix(node: number, elements: Float32Array): void
  nodeFlags(node: number, visible: number, frustumCulled: number, renderOrder: number): void
  nodeGeometry(node: number, geometry: number, material: number): void
  nodeInstances(node: number, matrices: number, colors: number, count: number): void
  nodeLight(node: number, kind: number, r: number, g: number, b: number, intensity: number, groundR: number, groundG: number, groundB: number, distance: number, decay: number): void
  nodeLightTarget(node: number, x: number, y: number, z: number): void
  cameraProjection(node: number, elements: Float32Array, near: number, far: number): void
  sceneEnvironment(scene: number, background: number, top: number, bottom: number, fog: number, fogColor: number, fogNear: number, fogFar: number, fogDensity: number): void
  renderOptions(depthBuffer: number, capacity: number): void
  renderTarget(surface: number): void
  createSurface(width: number, height: number): number
  drawScene(scene: number, camera: number, width: number, height: number): number
  stat(which: number): number
  raycast(node: number, ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, near: number, far: number): number
  rayResult(which: number): number
}

export * from './constants'

export function createBuffer(): number {
  return gea3dNative.createBuffer()
}

export function deleteBuffer(buffer: number): void {
  gea3dNative.deleteBuffer(buffer)
}

export function bufferDataF32(buffer: number, data: Float32Array): void {
  gea3dNative.bufferDataF32(buffer, data)
}

export function bufferDataU32(buffer: number, data: Uint32Array): void {
  gea3dNative.bufferDataU32(buffer, data)
}

export function createGeometry(): number {
  return gea3dNative.createGeometry()
}

export function deleteGeometry(geometry: number): void {
  gea3dNative.deleteGeometry(geometry)
}

export function geometryBuffers(geometry: number, position: number, index: number, color: number, normal: number): void {
  gea3dNative.geometryBuffers(geometry, position, index, color, normal)
}

export function createMaterial(): number {
  return gea3dNative.createMaterial()
}

export function deleteMaterial(material: number): void {
  gea3dNative.deleteMaterial(material)
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
  gea3dNative.materialSet(material, shade, r, g, b, opacity, side, flags, size)
}

export function createNode(kind: number): number {
  return gea3dNative.createNode(kind)
}

export function deleteNode(node: number): void {
  gea3dNative.deleteNode(node)
}

export function nodeParent(node: number, parent: number): void {
  gea3dNative.nodeParent(node, parent)
}

export function nodeTransform(
  node: number,
  px: number, py: number, pz: number,
  qx: number, qy: number, qz: number, qw: number,
  sx: number, sy: number, sz: number,
): void {
  gea3dNative.nodeTransform(node, px, py, pz, qx, qy, qz, qw, sx, sy, sz)
}

export function nodeMatrix(node: number, elements: Float32Array): void {
  gea3dNative.nodeMatrix(node, elements)
}

export function nodeFlags(node: number, visible: number, frustumCulled: number, renderOrder: number): void {
  gea3dNative.nodeFlags(node, visible, frustumCulled, renderOrder)
}

export function nodeGeometry(node: number, geometry: number, material: number): void {
  gea3dNative.nodeGeometry(node, geometry, material)
}

export function nodeInstances(node: number, matrices: number, colors: number, count: number): void {
  gea3dNative.nodeInstances(node, matrices, colors, count)
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
  gea3dNative.nodeLight(node, kind, r, g, b, intensity, groundR, groundG, groundB, distance, decay)
}

export function nodeLightTarget(node: number, x: number, y: number, z: number): void {
  gea3dNative.nodeLightTarget(node, x, y, z)
}

export function cameraProjection(node: number, elements: Float32Array, near: number, far: number): void {
  gea3dNative.cameraProjection(node, elements, near, far)
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
  gea3dNative.sceneEnvironment(scene, background, top, bottom, fog, fogColor, fogNear, fogFar, fogDensity)
}

export function renderOptions(depthBuffer: number, capacity: number): void {
  gea3dNative.renderOptions(depthBuffer, capacity)
}

export function renderTarget(surface: number): void {
  gea3dNative.renderTarget(surface)
}

export function createSurface(width: number, height: number): number {
  return gea3dNative.createSurface(width, height)
}

export function drawScene(scene: number, camera: number, width: number, height: number): number {
  return gea3dNative.drawScene(scene, camera, width, height)
}

export function stat(which: number): number {
  return gea3dNative.stat(which)
}

export function raycast(
  node: number,
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  near: number,
  far: number,
): number {
  return gea3dNative.raycast(node, ox, oy, oz, dx, dy, dz, near, far)
}

export function rayResult(which: number): number {
  return gea3dNative.rayResult(which)
}

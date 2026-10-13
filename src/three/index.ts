// SPDX-License-Identifier: GPL-3.0-only
// The 'three'-compatible export surface. App code written against this module
// should read like idiomatic three.js.

export { clamp, DEG2RAD, Color, Euler, Matrix4, Quaternion, Vector2, Vector3, Vector4 } from './math'
export {
  BufferAttribute,
  BufferGeometry,
  Drawable,
  Float32BufferAttribute,
  GeometryAttributes,
  Group,
  InstancedBufferAttribute,
  Intersection,
  Object3D,
  Scene,
  SceneCollector,
  Sphere,
} from './core'
export { Camera, OrthographicCamera, PerspectiveCamera } from './camera'
export {
  AdditiveBlending,
  BackSide,
  DoubleSide,
  FrontSide,
  LineBasicMaterial,
  Material,
  MeshBasicMaterial,
  MeshLambertMaterial,
  MeshNormalMaterial,
  MeshPhongMaterial,
  MeshStandardMaterial,
  NoBlending,
  NormalBlending,
  PointsMaterial,
  SpriteMaterial,
} from './materials'
export type {
  LineBasicMaterialParameters,
  MaterialParameters,
  MeshMaterialParameters,
  PointsMaterialParameters,
  SpriteMaterialParameters,
} from './materials'
export { Fog, FogExp2 } from './fog'
export { AmbientLight, DirectionalLight, HemisphereLight, Light, PointLight } from './lights'
export { InstancedMesh, Line, LineSegments, Mesh, Points, Sprite } from './mesh'
export { Ray, Raycaster } from './raycaster'
export {
  BoxGeometry,
  ConeGeometry,
  CylinderGeometry,
  PlaneGeometry,
  SphereGeometry,
  TorusGeometry,
  TorusKnotGeometry,
} from './geometries'

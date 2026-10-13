// SPDX-License-Identifier: GPL-3.0-only
// Material subset. Lambert/Phong/Standard shade as lambert: per vertex when
// the geometry has normals (Gouraud), per face with flatShading or without
// normals. Vertex colours interpolate across each face. Parameter objects
// take hex numbers for color (not Color/string unions).

import { currentNativeFrame } from './cache'
import { Color } from './math'
import {
  createMaterial,
  deleteMaterial,
  materialSet,
  MATERIAL_ADDITIVE,
  MATERIAL_DEPTH_TEST,
  MATERIAL_DEPTH_WRITE,
  MATERIAL_FLAT_SHADING,
  MATERIAL_FOG,
  MATERIAL_SIZE_ATTENUATION,
  MATERIAL_TRANSPARENT,
  MATERIAL_VERTEX_COLORS,
  MATERIAL_VISIBLE,
  MATERIAL_WIREFRAME,
} from '../native/api'

export const FrontSide = 0
export const BackSide = 1
export const DoubleSide = 2

// Blending modes (three.js values). Additive adds the colour (times opacity)
// to what is behind it, drawn after the opaque pass like transparent faces.
export const NoBlending = 0
export const NormalBlending = 1
export const AdditiveBlending = 2

export const SHADE_UNLIT = 0
export const SHADE_LAMBERT = 1
export const SHADE_NORMAL = 2

export interface MaterialParameters {
  color?: number
  vertexColors?: boolean
  side?: number
  transparent?: boolean
  opacity?: number
  blending?: number
  fog?: boolean
  flatShading?: boolean
  wireframe?: boolean
  depthTest?: boolean
  depthWrite?: boolean
  // PointsMaterial: size in pixels, or world units with sizeAttenuation.
  size?: number
  sizeAttenuation?: boolean
  // LineBasicMaterial: pixels across (1 to 3).
  linewidth?: number
}

export type MeshMaterialParameters = MaterialParameters
export type LineBasicMaterialParameters = MaterialParameters
export type PointsMaterialParameters = MaterialParameters
export type SpriteMaterialParameters = MaterialParameters

// The fields of a Material as last written to the native engine.
class SentMaterial {
  valid: boolean
  r: number
  g: number
  b: number
  opacity: number
  side: number
  size: number
  blending: number
  vertexColors: boolean
  transparent: boolean
  visible: boolean
  fog: boolean
  flatShading: boolean
  wireframe: boolean
  depthTest: boolean
  depthWrite: boolean
  sizeAttenuation: boolean

  constructor() {
    this.valid = false
    this.r = 0
    this.g = 0
    this.b = 0
    this.opacity = 0
    this.side = 0
    this.size = 0
    this.blending = 0
    this.vertexColors = false
    this.transparent = false
    this.visible = false
    this.fog = false
    this.flatShading = false
    this.wireframe = false
    this.depthTest = false
    this.depthWrite = false
    this.sizeAttenuation = false
  }

  matches(m: Material, size: number): boolean {
    const c = m.color
    return (
      this.valid &&
      this.r === c.r && this.g === c.g && this.b === c.b &&
      this.opacity === m.opacity && this.side === m.side && this.size === size && this.blending === m.blending &&
      this.vertexColors === m.vertexColors && this.transparent === m.transparent && this.visible === m.visible &&
      this.fog === m.fog && this.flatShading === m.flatShading && this.wireframe === m.wireframe &&
      this.depthTest === m.depthTest && this.depthWrite === m.depthWrite && this.sizeAttenuation === m.sizeAttenuation
    )
  }

  take(m: Material, size: number): void {
    const c = m.color
    this.valid = true
    this.r = c.r
    this.g = c.g
    this.b = c.b
    this.opacity = m.opacity
    this.side = m.side
    this.size = size
    this.blending = m.blending
    this.vertexColors = m.vertexColors
    this.transparent = m.transparent
    this.visible = m.visible
    this.fog = m.fog
    this.flatShading = m.flatShading
    this.wireframe = m.wireframe
    this.depthTest = m.depthTest
    this.depthWrite = m.depthWrite
    this.sizeAttenuation = m.sizeAttenuation
  }
}

export class Material {
  readonly color: Color
  vertexColors: boolean
  side: number
  transparent: boolean
  opacity: number
  blending: number
  visible: boolean
  fog: boolean
  flatShading: boolean
  wireframe: boolean
  depthTest: boolean
  depthWrite: boolean
  size: number
  sizeAttenuation: boolean
  linewidth: number
  // Shading selector used by the renderer: 0 unlit, 1 lambert, 2 normal.
  readonly shadeMode: number
  // Native material id (-1 = not created yet).
  nativeMaterial: number
  private readonly sent: SentMaterial
  private syncedFrame: number

  constructor(shadeMode: number, params?: MaterialParameters) {
    this.color = new Color()
    this.vertexColors = false
    this.side = FrontSide
    this.transparent = false
    this.opacity = 1
    this.blending = NormalBlending
    this.visible = true
    this.fog = true
    this.flatShading = false
    this.wireframe = false
    this.depthTest = true
    this.depthWrite = true
    this.size = 1
    this.sizeAttenuation = true
    this.linewidth = 1
    this.shadeMode = shadeMode
    this.nativeMaterial = -1
    this.sent = new SentMaterial()
    this.syncedFrame = -1
    if (params !== undefined) this.setValues(params)
  }

  setValues(params: MaterialParameters): void {
    const c = params.color
    if (c !== undefined) this.color.setHex(c)
    const vc = params.vertexColors
    if (vc !== undefined) this.vertexColors = vc
    const side = params.side
    if (side !== undefined) this.side = side
    const transparent = params.transparent
    if (transparent !== undefined) this.transparent = transparent
    const opacity = params.opacity
    if (opacity !== undefined) this.opacity = opacity
    const blending = params.blending
    if (blending !== undefined) this.blending = blending
    const fog = params.fog
    if (fog !== undefined) this.fog = fog
    const flatShading = params.flatShading
    if (flatShading !== undefined) this.flatShading = flatShading
    const wireframe = params.wireframe
    if (wireframe !== undefined) this.wireframe = wireframe
    const depthTest = params.depthTest
    if (depthTest !== undefined) this.depthTest = depthTest
    const depthWrite = params.depthWrite
    if (depthWrite !== undefined) this.depthWrite = depthWrite
    const size = params.size
    if (size !== undefined) this.size = size
    const sizeAttenuation = params.sizeAttenuation
    if (sizeAttenuation !== undefined) this.sizeAttenuation = sizeAttenuation
    const linewidth = params.linewidth
    if (linewidth !== undefined) this.linewidth = linewidth
  }

  // Writes the material if it changed (at most once per frame); returns its
  // native id.
  syncNative(): number {
    if (this.nativeMaterial < 0) {
      this.nativeMaterial = createMaterial()
      this.sent.valid = false
    }
    const frame = currentNativeFrame()
    if (this.syncedFrame === frame) return this.nativeMaterial

    this.syncedFrame = frame
    const size = this.nativeSize()
    if (this.sent.matches(this, size)) return this.nativeMaterial

    const c = this.color
    materialSet(this.nativeMaterial, this.shadeMode, c.r, c.g, c.b, this.opacity, this.side, this.nativeFlags(), size)
    this.sent.take(this, size)
    return this.nativeMaterial
  }

  // The native size slot: point size, or a line material's width.
  protected nativeSize(): number {
    return this.size
  }

  // Frees the native material; the next render recreates it.
  dispose(): void {
    if (this.nativeMaterial >= 0) deleteMaterial(this.nativeMaterial)
    this.nativeMaterial = -1
    this.syncedFrame = -1
  }

  private nativeFlags(): number {
    let flags = 0
    if (this.vertexColors) flags |= MATERIAL_VERTEX_COLORS
    if (this.transparent) flags |= MATERIAL_TRANSPARENT
    if (this.blending === AdditiveBlending) flags |= MATERIAL_ADDITIVE
    if (this.fog) flags |= MATERIAL_FOG
    if (this.flatShading) flags |= MATERIAL_FLAT_SHADING
    if (this.wireframe) flags |= MATERIAL_WIREFRAME
    if (this.depthTest) flags |= MATERIAL_DEPTH_TEST
    if (this.depthWrite) flags |= MATERIAL_DEPTH_WRITE
    if (this.sizeAttenuation) flags |= MATERIAL_SIZE_ATTENUATION
    if (this.visible) flags |= MATERIAL_VISIBLE
    return flags
  }
}

export class MeshBasicMaterial extends Material {
  constructor(params?: MaterialParameters) {
    super(SHADE_UNLIT, params)
  }
}

export class MeshLambertMaterial extends Material {
  constructor(params?: MaterialParameters) {
    super(SHADE_LAMBERT, params)
  }
}

export class MeshPhongMaterial extends Material {
  constructor(params?: MaterialParameters) {
    super(SHADE_LAMBERT, params)
  }
}

export class MeshStandardMaterial extends Material {
  constructor(params?: MaterialParameters) {
    super(SHADE_LAMBERT, params)
  }
}

export class MeshNormalMaterial extends Material {
  constructor(params?: MaterialParameters) {
    super(SHADE_NORMAL, params)
  }
}

// Unlit colour for Line and LineSegments, `linewidth` pixels across (1 to 3).
export class LineBasicMaterial extends Material {
  constructor(params?: MaterialParameters) {
    super(SHADE_UNLIT, params)
  }

  protected nativeSize(): number {
    return this.linewidth
  }
}

// Unlit colour for Points; `size` sets the square's extent.
export class PointsMaterial extends Material {
  constructor(params?: MaterialParameters) {
    super(SHADE_UNLIT, params)
  }
}

// Unlit colour for Sprite.
export class SpriteMaterial extends Material {
  constructor(params?: MaterialParameters) {
    super(SHADE_UNLIT, params)
  }
}

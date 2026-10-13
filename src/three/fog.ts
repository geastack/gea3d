// SPDX-License-Identifier: GPL-3.0-only
// Distance fog: geometry blends toward `color` with distance from the camera.

import { Color } from './math'

export const FOG_MODE_NONE = 0
export const FOG_MODE_LINEAR = 1
export const FOG_MODE_EXP2 = 2

// Linear fog: none at `near`, full at `far` (off when far <= near).
export class Fog {
  readonly color: Color
  near: number
  far: number
  // FogExp2 density (0 for linear fog).
  density: number
  isFogExp2: boolean

  constructor(color: number, near: number = 1, far: number = 1000) {
    this.color = new Color(color)
    this.near = near
    this.far = far
    this.density = 0
    this.isFogExp2 = false
  }

  mode(): number {
    if (this.isFogExp2) return this.density > 0 ? FOG_MODE_EXP2 : FOG_MODE_NONE
    return this.far > this.near ? FOG_MODE_LINEAR : FOG_MODE_NONE
  }
}

// Exponential squared fog: 1 - exp(-(density * distance)^2).
export class FogExp2 extends Fog {
  constructor(color: number, density: number = 0.00025) {
    super(color, 0, 0)
    this.density = density
    this.isFogExp2 = true
  }
}

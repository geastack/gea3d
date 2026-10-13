// SPDX-License-Identifier: GPL-3.0-only
import { Object3D } from './core'
import { DEG2RAD, Matrix4 } from './math'
import { cameraProjection, NODE_CAMERA } from '../native/api'

export class Camera extends Object3D {
  readonly matrixWorldInverse: Matrix4
  readonly projectionMatrix: Matrix4
  readonly projectionMatrixInverse: Matrix4
  // Clip planes live on the base so the renderer never needs to downcast.
  near: number
  far: number
  isOrthographicCamera: boolean
  // Bumped by updateProjectionMatrix; the native camera follows it.
  projectionVersion: number
  private sentProjectionVersion: number

  constructor() {
    super()
    this.matrixWorldInverse = new Matrix4()
    this.projectionMatrix = new Matrix4()
    this.projectionMatrixInverse = new Matrix4()
    this.near = 0.1
    this.far = 2000
    this.isOrthographicCamera = false
    this.isCamera = true
    this.nodeKind = NODE_CAMERA
    this.projectionVersion = 0
    this.sentProjectionVersion = -1
  }

  updateMatrixWorld(force: boolean = false): void {
    super.updateMatrixWorld(force)
    this.matrixWorldInverse.copy(this.matrixWorld).invert()
  }

  // Recomputes projectionMatrix from the camera's parameters.
  updateProjectionMatrix(): void {
    this.projectionChanged()
  }

  protected projectionChanged(): void {
    this.projectionMatrixInverse.copy(this.projectionMatrix).invert()
    this.projectionVersion++
  }

  syncSelf(): void {
    if (this.sentProjectionVersion === this.projectionVersion) return

    cameraProjection(this.nativeNode.id, this.projectionMatrix.elements, this.near, this.far)
    this.sentProjectionVersion = this.projectionVersion
  }

  releaseSelf(): void {
    this.sentProjectionVersion = -1
  }
}

export class PerspectiveCamera extends Camera {
  fov: number
  aspect: number
  zoom: number

  constructor(fov: number = 50, aspect: number = 1, near: number = 0.1, far: number = 2000) {
    super()
    this.fov = fov
    this.aspect = aspect
    this.near = near
    this.far = far
    this.zoom = 1
    this.updateProjectionMatrix()
  }

  updateProjectionMatrix(): void {
    const near = this.near
    const top = (near * Math.tan(DEG2RAD * 0.5 * this.fov)) / this.zoom
    const height = 2 * top
    const width = this.aspect * height
    const left = -0.5 * width
    this.projectionMatrix.makePerspective(left, left + width, top, top - height, near, this.far)
    this.projectionChanged()
  }
}

export class OrthographicCamera extends Camera {
  left: number
  right: number
  top: number
  bottom: number
  zoom: number

  constructor(
    left: number = -1,
    right: number = 1,
    top: number = 1,
    bottom: number = -1,
    near: number = 0.1,
    far: number = 2000,
  ) {
    super()
    this.left = left
    this.right = right
    this.top = top
    this.bottom = bottom
    this.near = near
    this.far = far
    this.zoom = 1
    this.isOrthographicCamera = true
    this.updateProjectionMatrix()
  }

  updateProjectionMatrix(): void {
    const dx = (this.right - this.left) / (2 * this.zoom)
    const dy = (this.top - this.bottom) / (2 * this.zoom)
    const cx = (this.right + this.left) / 2
    const cy = (this.top + this.bottom) / 2
    this.projectionMatrix.makeOrthographic(cx - dx, cx + dx, cy + dy, cy - dy, this.near, this.far)
    this.projectionChanged()
  }
}

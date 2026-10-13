// SPDX-License-Identifier: GPL-3.0-only
// Picking against uploaded mesh geometry, three.js Raycaster style. The ray
// tests run natively against each mesh's triangles in world space; points,
// lines and sprites are not hit.

import { Camera } from './camera'
import { Intersection, Object3D } from './core'
import { Matrix4, Vector2, Vector3 } from './math'

export class Ray {
  readonly origin: Vector3
  // Normalized.
  readonly direction: Vector3

  constructor() {
    this.origin = new Vector3()
    this.direction = new Vector3(0, 0, -1)
  }

  set(origin: Vector3, direction: Vector3): Ray {
    this.origin.copy(origin)
    this.direction.copy(direction).normalize()
    return this
  }
}

export class Raycaster {
  readonly ray: Ray
  near: number
  far: number
  private readonly unproject: Matrix4

  constructor(near: number = 0, far: number = Infinity) {
    this.ray = new Ray()
    this.near = near
    this.far = far
    this.unproject = new Matrix4()
  }

  set(origin: Vector3, direction: Vector3): void {
    this.ray.set(origin, direction)
  }

  // Ray from the camera through `coords` in normalized device coordinates
  // (x and y in -1..1, y up).
  setFromCamera(coords: Vector2, camera: Camera): void {
    camera.updateMatrixWorld(false)
    const origin = this.ray.origin
    const direction = this.ray.direction
    this.unproject.multiplyMatrices(camera.matrixWorld, camera.projectionMatrixInverse)
    if (camera.isOrthographicCamera) {
      origin.set(coords.x, coords.y, -1).applyMatrix4(this.unproject)
      direction.set(0, 0, -1).transformDirection(camera.matrixWorld)
      return
    }
    origin.setFromMatrixPosition(camera.matrixWorld)
    direction.set(coords.x, coords.y, 0.5).applyMatrix4(this.unproject).sub(origin).normalize()
  }

  // Hits on `object` (and its descendants when recursive), nearest first.
  intersectObject(object: Object3D, recursive: boolean = true): Intersection[] {
    const intersects: Intersection[] = []
    object.root().syncNative(-1)
    this.collect(object, recursive, intersects)
    sortByDistance(intersects)
    return intersects
  }

  intersectObjects(objects: Object3D[], recursive: boolean = true): Intersection[] {
    const intersects: Intersection[] = []
    for (let i = 0; i < objects.length; i++) {
      objects[i].root().syncNative(-1)
      this.collect(objects[i], recursive, intersects)
    }
    sortByDistance(intersects)
    return intersects
  }

  private collect(object: Object3D, recursive: boolean, intersects: Intersection[]): void {
    if (!object.visible) return

    object.raycastSelf(this.ray.origin, this.ray.direction, this.near, this.far, intersects)
    if (!recursive) return

    const children = object.children
    for (let i = 0; i < children.length; i++) this.collect(children[i], true, intersects)
  }
}

// Insertion sort: hit lists are short.
function sortByDistance(intersects: Intersection[]): void {
  for (let i = 1; i < intersects.length; i++) {
    const hit = intersects[i]
    let j = i - 1
    while (j >= 0 && intersects[j].distance > hit.distance) {
      intersects[j + 1] = intersects[j]
      j--
    }
    intersects[j + 1] = hit
  }
}

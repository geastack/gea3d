// SPDX-License-Identifier: GPL-3.0-only
// Last values written to a native object: put() records each, and dirty
// says whether any differed from the previous write (NaN = never written).
export class ValueCache {
  private readonly values: number[]
  private changed: boolean

  constructor(size: number) {
    this.values = []
    for (let i = 0; i < size; i++) this.values.push(Number.NaN)
    this.changed = false
  }

  begin(): void {
    this.changed = false
  }

  put(index: number, value: number): void {
    if (this.values[index] === value) return
    this.values[index] = value
    this.changed = true
  }

  get dirty(): boolean {
    return this.changed
  }

  invalidate(): void {
    for (let i = 0; i < this.values.length; i++) this.values[i] = Number.NaN
  }
}

// Counts renderer frames, so objects shared by several meshes (materials)
// are written to the native engine at most once per frame.
let nativeFrameNumber = 0

export function advanceNativeFrame(): void {
  nativeFrameNumber++
}

export function currentNativeFrame(): number {
  return nativeFrameNumber
}

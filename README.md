# gea3d

`@geastack/gea3d` provides a typed Three.js-style scene graph and a native
software 3D renderer for Gea devices. The native engine keeps a scene mirror,
updates changed transforms and materials, and performs culling, lighting,
clipping and rasterization in one scene draw per frame.

It supports perspective and orthographic cameras, meshes and instancing,
lines, points, sprites, fog, blending, raycasting, and optional depth buffering.
Lambert, Phong and Standard materials use Lambert lighting rather than full
Three.js material parity. This package does not implement WebGL; upstream Three.js
runs natively through WebGL on ANGLE with
[`@geastack/native-webgl-angle`](https://github.com/geastack/native-webgl-angle).

## Install

Install this package in your Gea app or npm workspace:

```sh
npm install @geastack/gea3d @geastack/core @geastack/cli
```

`@geastack/cli` provides the `gea` command and brings the board targets with
it.

Import the renderer and scene classes through the public package exports:

```ts
import { GeaRenderer } from '@geastack/gea3d'
import { Scene, PerspectiveCamera } from '@geastack/gea3d/three'

const scene = new Scene()
const camera = new PerspectiveCamera(52, 1, 1, 1000)
const renderer = new GeaRenderer({ triangleCapacity: 4096, depthBuffer: true })
renderer.setSize(480, 480)
renderer.render(scene, camera)
```

For native firmware, declare the renderer's C++ source in your app's
`package.json`. Gea resolves the installed package and its adjacent headers:

```json
{
  "gea": {
    "nativeSources": ["@geastack/gea3d/native/gea3d_native.cpp"]
  }
}
```

Register your board once with `npx gea setup`, which names its alias, then run
the app through the Gea CLI from your project root:

```sh
npx gea run --board <alias> --app <your-app-id>
```

The Lantern, Aviator, Neon and Orbit apps in
[geastack/demos](https://github.com/geastack/demos) demonstrate scene setup,
responsive panel sizes, input, animation and composition with retained UI.
They depend on the published `@geastack/gea3d` and name its native source as
above; no vendored renderer is required.

## Package contents

- `src/three`: typed scene graph, cameras, materials, math and raycasting.
- `src/renderer`: the native scene renderer and image-target presentation.
- `src/native`: C++ engine, rasterizer, host bridge and browser preview bridge.
- `src/raster`: the TypeScript software raster pipeline and presenter.

Native rendering can draw directly to the display, beneath retained UI through
Gea's underlay hook, or into an image target. The browser preview bridge in
`src/native/api.web.ts` uses Canvas 2D and requires the bundler to select it
instead of `src/native/api.ts`; it is not a WebGL renderer.

## Development

```sh
npm ci
npm run check
```

The demos use the published package, so a change here reaches them through a
release.

## License

GPL-3.0-only. See [LICENSE](LICENSE). Distribution of firmware containing this
renderer is subject to that license. Three.js attribution is retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// gea3d's scanline rasterizer. It draws one frame's primitives into any
// rectangular region of a pixel buffer (the whole framebuffer, or one row
// band of a streaming flush), so the same code serves every present path.
//
// Occlusion, two ways with identical results:
// - coverage (drawing straight into slow memory): opaque primitives are drawn
//   nearest-first and a per-row coverage bitmask lets each pixel be written
//   exactly once; pixels no primitive covers get the background.
// - painter's (drawing into a fast band buffer, where overdraw is cheap and
//   bookkeeping is not): the background is laid first, then opaque
//   primitives farthest-first.
// With a depth buffer (opt-in, or in coverage mode when the frame has
// blended primitives) opaque spans also record 16-bit depth, and blended
// primitives are depth-tested against it. Painter's mode without an opt-in
// depth buffer draws blended primitives in the same back-to-front order as
// opaque ones instead, so they cost no depth bookkeeping.
//
// Edges are stepped in 16.16 fixed point from a position every triangle
// sharing the edge computes identically (its first row centre), so shared
// edges tile without cracks or double hits under the pixel-centre rule.
//
// The hot paths use no 64-bit division, no popcount and no libm calls: they
// run on Cortex-M7 parts where those are software routines.

#include "pixel.h"

#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define GEA3D_INLINE inline __attribute__((always_inline))
#else
#define GEA3D_INLINE inline
#endif

namespace gea::host::gea3d::raster {

using Pixel = gea::framework::graphics::pixel::native_t;
namespace pixel = gea::framework::graphics::pixel;

enum PrimType : std::uint8_t {
	kTriangle = 0,
	kLine = 1,
};

enum PrimFlag : std::uint8_t {
	kSmooth = 1,  // per-vertex colours interpolated across the primitive
	kBlend = 2,   // alpha-blended (drawn back-to-front after the opaque pass)
	kAdd = 4,     // with kBlend: added to the pixel (scaled by alpha), not mixed
};

// Screen coordinates are 28.4 fixed point; depth is near/w in (0, 1], larger
// is nearer. Colours are 0..255 per vertex.
struct Prim {
	std::int32_t x[3];
	std::int32_t y[3];
	float depth[3];
	std::uint8_t r[3];
	std::uint8_t g[3];
	std::uint8_t b[3];
	std::uint8_t type;
	std::uint8_t flags;
	std::uint8_t alpha;
	std::uint8_t width;  // lines: pixels across (1..3)
	Pixel color;
	std::int16_t rowY0;
	std::int16_t rowY1;
	// Painter's key: view distance of the primitive's farthest vertex.
	float key;
};

enum BackgroundMode : std::uint8_t {
	kBackgroundNone = 0,
	kBackgroundFlat = 1,
	kBackgroundGradient = 2,
};

struct Background {
	std::uint8_t mode = kBackgroundNone;
	std::uint8_t top[3] = {0, 0, 0};
	std::uint8_t bottom[3] = {0, 0, 0};
};

// One frame's sorted primitives, ready to rasterize any number of times.
struct Frame {
	// Primitive records: the fixed pool (faster memory, when set) until a
	// frame outgrows it, then primStore on the heap.
	Prim *prims = nullptr;
	int primCapacity = 0;
	Prim *fixedPrims = nullptr;
	int fixedCapacity = 0;
	std::vector<Prim> primStore;
	// Dense per-primitive sort inputs (emission order), so sorting never
	// walks the primitive records.
	std::vector<float> sortKeys;
	std::vector<std::uint8_t> blended;
	// Per primitive, its object's renderOrder; the sort orders by it first
	// when the frame holds more than one (minLayer < maxLayer).
	std::vector<std::int32_t> layers;
	int minLayer = 0;
	int maxLayer = 0;
	std::vector<std::uint32_t> emittedRows;
	std::vector<std::int32_t> order;  // opaque nearest-first, then blended farthest-first
	// Per sorted position: the primitive's row range packed as
	// (rowY0 & 0xFFFF) | (rowY1 << 16), so a band finds the primitives it
	// touches without reading their records. In fast memory when it fits.
	const std::uint32_t *rows = nullptr;
	std::vector<std::uint32_t> rowStore;
	int count = 0;
	int opaqueCount = 0;
	int blendCount = 0;
	int width = 0;
	int height = 0;
	bool depthBuffer = false;  // opaque spans depth-test (true z-buffer)
	Background background;
};

// Scratch reused across regions and frames. `fixed` (optional) is faster
// memory the coverage bookkeeping uses when a region fits in it. The block
// remembers its shape and which rows the last region touched, so the next
// region of the same shape clears only those.
struct Scratch {
	std::vector<std::uint32_t> coverage;
	std::vector<std::uint16_t> depth;
	std::uint32_t *fixed = nullptr;
	std::size_t fixedWords = 0;
	const std::uint32_t *block = nullptr;
	int blockWidth = 0;
	int blockHeight = 0;
	int dirtyFirst = 0;
	int dirtyLast = -1;
};

namespace detail {

GEA3D_INLINE int minInt(int a, int b) { return a < b ? a : b; }
GEA3D_INLINE int maxInt(int a, int b) { return a > b ? a : b; }
// Branch-free on FPv5 (VMINNM/VMAXNM): a float compare's transfer to the
// integer flags stalls the Cortex-M7 pipeline.
GEA3D_INLINE float clampFloat(float v, float lo, float hi) { return __builtin_fminf(__builtin_fmaxf(v, lo), hi); }
GEA3D_INLINE int clampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
GEA3D_INLINE int ctz(std::uint32_t value) { return __builtin_ctz(value); }

// Round to nearest, ties away from zero (VRINTA on FPv5, no libm call).
GEA3D_INLINE std::int32_t roundToInt(float value) { return static_cast<std::int32_t>(__builtin_roundf(value)); }

constexpr float kFixedOne = 65536.0f;
constexpr float kMaxSlope = 16000.0f;  // px per row; steeper edges span under one row

// An edge's x at successive pixel-row centres, 16.16 px. Both triangles that
// share the edge start from its own first row centre, so they agree exactly.
struct Edge {
	std::int32_t x = 0;
	std::int32_t step = 0;

	GEA3D_INLINE void init(std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1, int row)
	{
		const float slope = clampFloat(static_cast<float>(x1 - x0) / static_cast<float>(y1 - y0), -kMaxSlope, kMaxSlope);
		const int top = (y0 - 8 + 15) >> 4;
		const float xTop = (static_cast<float>(x0) + static_cast<float>(top * 16 + 8 - y0) * slope) * (1.0f / 16.0f);
		step = roundToInt(slope * kFixedOne);
		x = roundToInt(xTop * kFixedOne) + static_cast<std::int32_t>(static_cast<std::int64_t>(row - top) * step);
	}

	GEA3D_INLINE void advance() { x += step; }

	// First pixel whose centre is at or right of the edge.
	GEA3D_INLINE int pixel() const { return (x + 32767) >> 16; }
};

// value(px, py) = base + px * dx + py * dy over one triangle.
struct Plane {
	float base = 0.0f;
	float dx = 0.0f;
	float dy = 0.0f;

	GEA3D_INLINE void init(const float *sx, const float *sy, float invArea, float a0, float a1, float a2)
	{
		const float e1 = a1 - a0;
		const float e2 = a2 - a0;
		dx = (e1 * (sy[2] - sy[0]) - e2 * (sy[1] - sy[0])) * invArea;
		dy = (e2 * (sx[1] - sx[0]) - e1 * (sx[2] - sx[0])) * invArea;
		base = a0 - sx[0] * dx - sy[0] * dy;
	}

	GEA3D_INLINE float at(float px, float py) const { return base + px * dx + py * dy; }
};

constexpr float kColorScale = 65536.0f;  // colours step in 16.16
constexpr float kDepthScale = 65535.0f * 4096.0f;  // depth steps in 16.12

// 8-bit channels to a native pixel, truncating like the rest of the canvas.
GEA3D_INLINE Pixel packColor(int r, int g, int b)
{
	const std::uint32_t rrggbbaa = (static_cast<std::uint32_t>(r) << 24) | (static_cast<std::uint32_t>(g) << 16) | (static_cast<std::uint32_t>(b) << 8) | 0xFFu;
	return pixel::nativeFromRrggbbaa(rrggbbaa);
}

GEA3D_INLINE Pixel packRgb(std::int32_t r, std::int32_t g, std::int32_t b)
{
#if GEA_PIXEL_FORMAT_IS_8888 || GEA_PIXEL_FORMAT_IS_GRAY
	return packColor(r >> 16, g >> 16, b >> 16);
#else
	const std::uint16_t rgb565 = static_cast<std::uint16_t>(((r >> 8) & 0xF800) | ((g >> 13) & 0x07E0) | (b >> 19));
	return pixel::fromRgb565(rgb565);
#endif
}

// A constant-colour run: paired 32-bit stores on 16-bit framebuffers.
GEA3D_INLINE void fillPixels(Pixel *dst, int count, Pixel color)
{
#if GEA_PIXEL_FORMAT_IS_8888 || GEA_PIXEL_FORMAT_IS_GRAY || GEA_PIXEL_PIE_FILL16
	pixel::fillNative(dst, count, color);
#else
	if (count <= 0) return;
	if ((reinterpret_cast<std::uintptr_t>(dst) & 2u) != 0) {
		*dst++ = color;
		count--;
	}
	const std::uint32_t pair = static_cast<std::uint32_t>(color) | (static_cast<std::uint32_t>(color) << 16);
	std::uint32_t *wide = reinterpret_cast<std::uint32_t *>(dst);
	const int pairs = count >> 1;
	int i = 0;
	// Eight words per step: the compiler emits paired (STRD) stores.
	for (; i + 8 <= pairs; i += 8) {
		wide[i] = pair;
		wide[i + 1] = pair;
		wide[i + 2] = pair;
		wide[i + 3] = pair;
		wide[i + 4] = pair;
		wide[i + 5] = pair;
		wide[i + 6] = pair;
		wide[i + 7] = pair;
	}
	for (; i < pairs; i++) wide[i] = pair;
	if ((count & 1) != 0) dst[count - 1] = color;
#endif
}

// `fg` scaled by alpha (0..255) added to `bg`, each channel saturating.
GEA3D_INLINE Pixel addPixel(Pixel fg, Pixel bg, int alpha)
{
	const int scale = alpha + (alpha >> 7);
#if GEA_PIXEL_FORMAT_IS_8888 || GEA_PIXEL_FORMAT_IS_GRAY
	int fr = 0, fgreen = 0, fb = 0, fa = 0, br = 0, bgreen = 0, bb = 0, ba = 0;
	pixel::unpackNative8(fg, &fr, &fgreen, &fb, &fa);
	pixel::unpackNative8(bg, &br, &bgreen, &bb, &ba);
	return packColor(minInt(255, br + ((fr * scale) >> 8)), minInt(255, bgreen + ((fgreen * scale) >> 8)), minInt(255, bb + ((fb * scale) >> 8)));
#else
	const std::uint32_t f = pixel::toRgb565(fg);
	const std::uint32_t b = pixel::toRgb565(bg);
	const int r = minInt(31, static_cast<int>(b >> 11) + ((static_cast<int>(f >> 11) * scale) >> 8));
	const int g = minInt(63, static_cast<int>((b >> 5) & 63) + ((static_cast<int>((f >> 5) & 63) * scale) >> 8));
	const int bl = minInt(31, static_cast<int>(b & 31) + ((static_cast<int>(f & 31) * scale) >> 8));
	return pixel::fromRgb565(static_cast<std::uint16_t>((r << 11) | (g << 5) | bl));
#endif
}

// A blended primitive's colour over `bg`: added or mixed by alpha.
GEA3D_INLINE Pixel blendPixel(Pixel fg, Pixel bg, int alpha, bool add)
{
	return add ? addPixel(fg, bg, alpha) : pixel::blendNative(fg, bg, alpha);
}

// The bits of a 16.16 colour channel that reach the pixel (packRgb): a run
// whose channels keep these bits is one flat colour.
#if GEA_PIXEL_FORMAT_IS_8888 || GEA_PIXEL_FORMAT_IS_GRAY
constexpr int kShiftR = 16, kShiftG = 16, kShiftB = 16;
#else
constexpr int kShiftR = 19, kShiftG = 18, kShiftB = 19;
#endif

// Pixels a channel at `value` stepping `step` keeps its visible bits: the
// first step that changes them, or `limit` when it never does before that.
GEA3D_INLINE int steadyFor(std::int32_t value, std::int32_t step, int shift, int limit)
{
	if (step > 0) {
		const std::int32_t next = ((value >> shift) + 1) << shift;
		return static_cast<int>(static_cast<std::uint32_t>(next - value + step - 1) / static_cast<std::uint32_t>(step));
	}
	if (step < 0) {
		const std::int32_t floor = (value >> shift) << shift;
		return static_cast<int>(static_cast<std::uint32_t>(value - floor) / static_cast<std::uint32_t>(-step)) + 1;
	}
	return limit;
}

GEA3D_INLINE int levelChanges(std::int32_t start, std::int32_t end, int shift)
{
	const int change = (end >> shift) - (start >> shift);
	return change < 0 ? -change : change;
}

// One packed colour per pixel; on 16-bit framebuffers pairs of pixels share
// a 32-bit store (little-endian: the first pixel is the low half).
GEA3D_INLINE void shadeEach(Pixel *dst, int count, std::int32_t r, std::int32_t g, std::int32_t b, std::int32_t dr, std::int32_t dg, std::int32_t db)
{
#if GEA_PIXEL_FORMAT_IS_8888 || GEA_PIXEL_FORMAT_IS_GRAY || GEA_PIXEL_PIE_FILL16
	for (int i = 0; i < count; i++) {
		dst[i] = packRgb(r, g, b);
		r += dr;
		g += dg;
		b += db;
	}
#else
	if (count > 0 && (reinterpret_cast<std::uintptr_t>(dst) & 2u) != 0) {
		*dst++ = packRgb(r, g, b);
		r += dr;
		g += dg;
		b += db;
		count--;
	}
	std::uint32_t *wide = reinterpret_cast<std::uint32_t *>(dst);
	for (; count >= 2; count -= 2) {
		const std::uint32_t first = packRgb(r, g, b);
		const std::uint32_t second = packRgb(r + dr, g + dg, b + db);
		*wide++ = first | (second << 16);
		r += 2 * dr;
		g += 2 * dg;
		b += 2 * db;
	}
	if (count > 0) *reinterpret_cast<Pixel *>(wide) = packRgb(r, g, b);
#endif
}

// A Gouraud span whose channels stay in 0..0xFFFFFF (16.16). Spans that
// change colour rarely are filled as flat runs between the changes; the
// pixels are exactly those of the per-pixel walk.
GEA3D_INLINE void shadePixels(Pixel *dst, int count, std::int32_t r, std::int32_t g, std::int32_t b, std::int32_t dr, std::int32_t dg, std::int32_t db)
{
	const int last = count - 1;
	const int changes = levelChanges(r, r + dr * last, kShiftR) + levelChanges(g, g + dg * last, kShiftG) + levelChanges(b, b + db * last, kShiftB);
	if (count < 4 * (changes + 1)) {
		shadeEach(dst, count, r, g, b, dr, dg, db);
		return;
	}

	int nextR = steadyFor(r, dr, kShiftR, count);
	int nextG = steadyFor(g, dg, kShiftG, count);
	int nextB = steadyFor(b, db, kShiftB, count);
	int i = 0;
	while (i < count) {
		const int end = minInt(minInt(nextR, nextG), minInt(nextB, count));
		const std::int32_t ri = r + dr * i, gi = g + dg * i, bi = b + db * i;
		fillPixels(dst + i, end - i, packRgb(ri, gi, bi));
		i = end;
		if (i == count) return;

		if (nextR == i) nextR = i + steadyFor(r + dr * i, dr, kShiftR, count);
		if (nextG == i) nextG = i + steadyFor(g + dg * i, dg, kShiftG, count);
		if (nextB == i) nextB = i + steadyFor(b + db * i, db, kShiftB, count);
	}
}

}  // namespace detail

// The primitives touching one band, in sorted order (indices into
// Frame::prims), the first `opaque` of them opaque.
struct BandList {
	const std::int32_t *items = nullptr;
	int count = 0;
	int opaque = 0;
};

enum class Occlusion : std::uint8_t {
	kCoverage,  // each pixel written once: for slow target memory
	kPainter,   // background first, then back to front: for fast buffers
};

class Rasterizer {
public:
	Rasterizer(const Frame &frame, Scratch &scratch, Pixel *base, int stride, int x0, int y0, int x1, int y1, Occlusion occlusion = Occlusion::kCoverage)
	    : frame_(frame), scratch_(scratch), base_(base), stride_(stride), x0_(x0), y0_(y0), x1_(x1), y1_(y1), painter_(occlusion == Occlusion::kPainter)
	{
		width_ = x1 - x0 + 1;
		height_ = y1 - y0 + 1;
		if (width_ <= 0 || height_ <= 0) return;

		if (!painter_) bindCoverage();
		useDepth_ = frame.depthBuffer || (frame.blendCount > 0 && !painter_);
		if (useDepth_) {
			scratch.depth.assign(static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_), 0u);
			depth_ = scratch.depth.data();
		}
	}

	// Coverage mode: the whole frame, nearest-first.
	void run()
	{
		if (width_ <= 0 || height_ <= 0) return;

		const int opaque = frame_.opaqueCount;
		const int total = opaque + frame_.blendCount;
		drawSorted(0, opaque, false);
		fillBackground();
		drawSorted(opaque, total, true);
		scratch_.dirtyFirst = dirtyFirst_;
		scratch_.dirtyLast = dirtyLast_;
	}

	// Painter's mode: the background, then the band's primitives. Opaque
	// farthest-first is the exact reverse of the nearest-first order; a
	// depth-tested frame keeps nearest-first so equal depths resolve as in
	// coverage mode.
	void runPainter(const BandList &band)
	{
		if (width_ <= 0 || height_ <= 0) return;

		fillBackground();
		if (frame_.depthBuffer) {
			for (int k = 0; k < band.opaque; k++) draw(frame_.prims[band.items[k]], false);
			for (int k = band.opaque; k < band.count; k++) draw(frame_.prims[band.items[k]], true);
			return;
		}

		// Back to front: the opaque run reversed, merged with the blended run
		// (already farthest-first) by layer, then depth; opaque wins ties.
		const std::int32_t *layers = frame_.layers.data();
		int opaque = band.opaque - 1;
		int blended = band.opaque;
		while (opaque >= 0 || blended < band.count) {
			bool takeBlended = opaque < 0;
			if (!takeBlended && blended < band.count) {
				const std::int32_t o = band.items[opaque];
				const std::int32_t b = band.items[blended];
				takeBlended = layers[b] < layers[o] || (layers[b] == layers[o] && frame_.prims[b].key > frame_.prims[o].key);
			}
			if (takeBlended) {
				draw(frame_.prims[band.items[blended++]], true);
				continue;
			}
			draw(frame_.prims[band.items[opaque--]], false);
		}
	}

private:
	struct SpanState {
		const Prim *prim;
		detail::Plane r, g, b, z;
		std::int32_t dr = 0, dg = 0, db = 0;  // colour steps per pixel, 16.16
		bool shortSteps = false;  // no span's colour ramp overflows 32 bits
		bool smooth;
		bool blend;
		bool depthTest;
		bool depthWrite;
	};

	const Frame &frame_;
	Scratch &scratch_;
	Pixel *base_;
	int stride_;
	int x0_, y0_, x1_, y1_;
	int width_ = 0;
	int height_ = 0;
	int words_ = 0;
	int summaryWords_ = 0;
	// Per row: coverage bits, a summary bit per full coverage word, the
	// count of coverage words that still have open pixels, and whether any
	// opaque primitive touched the row.
	std::uint32_t *coverage_ = nullptr;
	std::uint32_t *summary_ = nullptr;
	std::int32_t *openWords_ = nullptr;
	std::int32_t *touched_ = nullptr;
	int dirtyFirst_ = 0;
	int dirtyLast_ = -1;
	std::uint16_t *depth_ = nullptr;
	bool useDepth_ = false;
	const bool painter_;

	void bindCoverage()
	{
		words_ = (width_ + 31) >> 5;
		summaryWords_ = (words_ + 31) >> 5;
		const std::size_t rows = static_cast<std::size_t>(height_);
		const std::size_t coverageWords = static_cast<std::size_t>(words_) * rows;
		const std::size_t summaryWords = static_cast<std::size_t>(summaryWords_) * rows;
		const std::size_t total = coverageWords + summaryWords + 2 * rows;
		std::uint32_t *block = scratch_.fixed;
		if (total > scratch_.fixedWords) {
			if (scratch_.coverage.size() < total) scratch_.coverage.resize(total);
			block = scratch_.coverage.data();
		}
		coverage_ = block;
		summary_ = block + coverageWords;
		openWords_ = reinterpret_cast<std::int32_t *>(summary_ + summaryWords);
		touched_ = openWords_ + rows;

		const bool sameShape = scratch_.block == block && scratch_.blockWidth == width_ && scratch_.blockHeight == height_;
		const int first = sameShape ? scratch_.dirtyFirst : 0;
		const int last = sameShape ? scratch_.dirtyLast : height_ - 1;
		for (int row = first; row <= last; row++) resetRow(row);
		scratch_.block = block;
		scratch_.blockWidth = width_;
		scratch_.blockHeight = height_;
	}

	// Region-relative row back to "nothing covered". Bits past the right
	// edge start covered, so a word full of real pixels reads 0xFFFFFFFF.
	void resetRow(int row)
	{
		std::uint32_t *bits = coverage_ + static_cast<std::size_t>(row) * static_cast<std::size_t>(words_);
		std::memset(bits, 0, static_cast<std::size_t>(words_) * sizeof(std::uint32_t));
		if ((width_ & 31) != 0) bits[words_ - 1] = ~((1u << (width_ & 31)) - 1u);
		std::memset(summary_ + static_cast<std::size_t>(row) * static_cast<std::size_t>(summaryWords_), 0, static_cast<std::size_t>(summaryWords_) * sizeof(std::uint32_t));
		openWords_[row] = words_;
		touched_[row] = 0;
	}

	GEA3D_INLINE void touch(int local)
	{
		if (touched_[local]) return;

		touched_[local] = 1;
		if (dirtyLast_ < dirtyFirst_) {
			dirtyFirst_ = local;
			dirtyLast_ = local;
			return;
		}
		dirtyFirst_ = detail::minInt(dirtyFirst_, local);
		dirtyLast_ = detail::maxInt(dirtyLast_, local);
	}

	// Sorted positions [first, last) that touch the region's rows.
	void drawSorted(int first, int last, bool blendPass)
	{
		for (int i = first; i < last; i++) {
			if (!overlapsRegion(frame_.rows[i])) continue;
			draw(frame_.prims[static_cast<std::size_t>(frame_.order[static_cast<std::size_t>(i)])], blendPass);
		}
	}

	GEA3D_INLINE bool overlapsRegion(std::uint32_t rows) const
	{
		const int first = static_cast<std::int16_t>(rows & 0xFFFFu);
		const int last = static_cast<std::int16_t>(rows >> 16);
		return last >= y0_ && first <= y1_;
	}

	void draw(const Prim &prim, bool blendPass)
	{
		if (prim.type != kLine) {
			drawTriangle(prim, blendPass);
			return;
		}
		if (painter_ && !blendPass && !useDepth_) {
			drawLinePainter(prim);
			return;
		}
		drawLine(prim, blendPass);
	}

	void drawTriangle(const Prim &prim, bool blendPass)
	{
		int i0 = 0, i1 = 1, i2 = 2;
		if (prim.y[i0] > prim.y[i1]) swapIndex(i0, i1);
		if (prim.y[i0] > prim.y[i2]) swapIndex(i0, i2);
		if (prim.y[i1] > prim.y[i2]) swapIndex(i1, i2);
		const std::int32_t X0 = prim.x[i0], Y0 = prim.y[i0];
		const std::int32_t X1 = prim.x[i1], Y1 = prim.y[i1];
		const std::int32_t X2 = prim.x[i2], Y2 = prim.y[i2];

		const std::int64_t cross = static_cast<std::int64_t>(X1 - X0) * (Y2 - Y0) - static_cast<std::int64_t>(Y1 - Y0) * (X2 - X0);
		if (cross == 0) return;

		const int yStart = (Y0 - 8 + 15) >> 4;
		const int ySplit = (Y1 - 8 + 15) >> 4;
		const int yEnd = (Y2 - 8 + 15) >> 4;
		const int rowFirst = detail::maxInt(yStart, y0_);
		const int rowLast = detail::minInt(yEnd - 1, y1_);
		if (rowFirst > rowLast) return;

		const SpanState span = spanState(prim, blendPass, i0, i1, i2);
		const Walk walk{X0, Y0, X1, Y1, X2, Y2, rowFirst, rowLast, ySplit, cross > 0};

		// Painter's order without blending or depth: spans go straight to the band.
		if (painter_ && !span.blend && !useDepth_) {
			if (span.smooth) {
				walkRows(walk, [this, &span](int row, int xa, int xb) { shadeRow(span, row, xa, xb - xa); });
				return;
			}
			const Pixel color = prim.color;
			Pixel *const base = base_;
			const int stride = stride_;
			walkRows(walk, [base, stride, color](int row, int xa, int xb) { detail::fillPixels(base + static_cast<std::ptrdiff_t>(row) * stride + xa, xb - xa, color); });
			return;
		}
		walkRows(walk, [this, &span](int row, int xa, int xb) { emitRow(span, row, xa, xb); });
	}

	// A triangle's corners sorted by y (28.4), its rows clipped to the region,
	// and whether the long edge (top to bottom corner) is on the left.
	struct Walk {
		std::int32_t x0, y0, x1, y1, x2, y2;
		int rowFirst, rowLast, ySplit;
		bool longLeft;
	};

	// Calls emit(row, xa, xb) for the pixels [xa, xb) of each row, clipped to
	// the region and non-empty.
	template <typename Emit>
	GEA3D_INLINE void walkRows(const Walk &w, Emit emit)
	{
		detail::Edge longEdge;
		longEdge.init(w.x0, w.y0, w.x2, w.y2, w.rowFirst);
		detail::Edge shortEdge;
		int row = w.rowFirst;
		if (row < w.ySplit) {
			shortEdge.init(w.x0, w.y0, w.x1, w.y1, row);
			row = walkSection(longEdge, shortEdge, w.longLeft, row, detail::minInt(w.ySplit - 1, w.rowLast), emit);
		}
		if (row > w.rowLast || w.y2 == w.y1) return;

		shortEdge.init(w.x1, w.y1, w.x2, w.y2, row);
		walkSection(longEdge, shortEdge, w.longLeft, row, w.rowLast, emit);
	}

	template <typename Emit>
	GEA3D_INLINE int walkSection(detail::Edge &longEdge, detail::Edge &shortEdge, bool longLeft, int row, int last, Emit &emit)
	{
		// Local copies stay in registers: pixel stores cannot alias them.
		detail::Edge left = longLeft ? longEdge : shortEdge;
		detail::Edge right = longLeft ? shortEdge : longEdge;
		const int first = x0_;
		const int limit = x1_ + 1;
		for (; row <= last; row++) {
			const int xa = detail::maxInt(left.pixel(), first);
			const int xb = detail::minInt(right.pixel(), limit);
			if (xa < xb) emit(row, xa, xb);
			left.advance();
			right.advance();
		}
		longEdge = longLeft ? left : right;
		return row;
	}

	static GEA3D_INLINE void swapIndex(int &a, int &b)
	{
		const int t = a;
		a = b;
		b = t;
	}

	SpanState spanState(const Prim &prim, bool blendPass, int i0, int i1, int i2) const
	{
		SpanState span;
		span.prim = &prim;
		span.smooth = (prim.flags & kSmooth) != 0;
		span.blend = blendPass;
		span.depthTest = useDepth_ && (blendPass || frame_.depthBuffer);
		span.depthWrite = useDepth_ && !blendPass;
		if (!span.smooth && !useDepth_) return span;

		const float sx[3] = {prim.x[i0] * (1.0f / 16.0f), prim.x[i1] * (1.0f / 16.0f), prim.x[i2] * (1.0f / 16.0f)};
		const float sy[3] = {prim.y[i0] * (1.0f / 16.0f), prim.y[i1] * (1.0f / 16.0f), prim.y[i2] * (1.0f / 16.0f)};
		const float area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]);
		if (area == 0.0f) {
			span.smooth = false;
			return span;
		}

		const float invArea = 1.0f / area;
		if (span.smooth) {
			span.r.init(sx, sy, invArea, prim.r[i0], prim.r[i1], prim.r[i2]);
			span.g.init(sx, sy, invArea, prim.g[i0], prim.g[i1], prim.g[i2]);
			span.b.init(sx, sy, invArea, prim.b[i0], prim.b[i1], prim.b[i2]);
			span.dr = static_cast<std::int32_t>(span.r.dx * detail::kColorScale);
			span.dg = static_cast<std::int32_t>(span.g.dx * detail::kColorScale);
			span.db = static_cast<std::int32_t>(span.b.dx * detail::kColorScale);
			const std::int64_t steepest = detail::maxInt(detail::maxInt(absStep(span.dr), absStep(span.dg)), absStep(span.db));
			span.shortSteps = steepest * width_ < 0x7F000000;
		}
		if (useDepth_) span.z.init(sx, sy, invArea, prim.depth[i0], prim.depth[i1], prim.depth[i2]);
		return span;
	}

	// Pixels [xa, xb) of `row`, already clipped to the region.
	GEA3D_INLINE void emitRow(const SpanState &span, int row, int xa, int xb)
	{
		if (span.blend) {
			fillRun(span, row, xa, xb - xa);
			return;
		}
		if (painter_) {
			flushRun(span, row, base_ + static_cast<std::ptrdiff_t>(row) * stride_ + x0_, xa - x0_, xb - x0_, !span.smooth && !useDepth_);
			return;
		}

		const int local = row - y0_;
		std::uint32_t *bits = coverage_ + static_cast<std::size_t>(local) * static_cast<std::size_t>(words_);
		std::uint32_t *summary = summary_ + static_cast<std::size_t>(local) * static_cast<std::size_t>(summaryWords_);
		const int begin = xa - x0_;
		const int end = xb - x0_;
		if (frame_.depthBuffer) {
			// Z-buffer: every pixel is depth-tested; coverage only tracks what
			// the background must not paint.
			fillRun(span, row, xa, xb - xa);
			touch(local);
			markCovered(bits, summary, local, begin, end);
			return;
		}

		const bool flat = !span.smooth && !useDepth_;
		Pixel *line = base_ + static_cast<std::ptrdiff_t>(row) * stride_ + x0_;
		if (!touched_[local]) {
			// Nothing owns this row yet: the whole span is open.
			flushRun(span, row, line, begin, end, flat);
			touch(local);
			markCovered(bits, summary, local, begin, end);
			return;
		}
		if (openWords_[local] == 0) return;

		coverSpan(span, row, line, bits, summary, local, begin, end, flat);
	}

	// Coverage walk: only the open pixels of [begin, end) are painted. The
	// row summary skips straight to the words that still have open pixels.
	void coverSpan(const SpanState &span, int row, Pixel *line, std::uint32_t *bits, std::uint32_t *summary, int local, int begin, int end, bool flat)
	{
		const int firstWord = begin >> 5;
		const int lastWord = (end - 1) >> 5;
		if (spanOpen(bits, begin, end, firstWord, lastWord)) {
			flushRun(span, row, line, begin, end, flat);
			markCovered(bits, summary, local, begin, end);
			return;
		}

		int runStart = 0, runEnd = -1;
		for (int group = firstWord >> 5; group <= lastWord >> 5; group++) {
			std::uint32_t candidates = ~summary[group] & wordRange(group, firstWord, lastWord);
			while (candidates != 0) {
				const int word = (group << 5) + detail::ctz(candidates);
				candidates &= candidates - 1;

				const int from = detail::maxInt(begin, word << 5);
				const int to = detail::minInt(end, (word + 1) << 5);
				const int count = to - from;
				const std::uint32_t mask = count >= 32 ? 0xFFFFFFFFu : (((1u << count) - 1u) << (from & 31));
				const std::uint32_t covered = bits[word];
				std::uint32_t open = ~covered & mask;
				if (open == 0) continue;

				const std::uint32_t now = covered | mask;
				bits[word] = now;
				if (now == 0xFFFFFFFFu) {
					summary[group] |= 1u << (word & 31);
					openWords_[local]--;
				}
				// Each run of open bits extends the pending run or starts a new one.
				while (open != 0) {
					const int bit = detail::ctz(open);
					const std::uint32_t shifted = open >> bit;
					const int length = shifted == (0xFFFFFFFFu >> bit) ? 32 - bit : detail::ctz(~shifted);
					const int runX = (word << 5) + bit;
					if (runX != runEnd) {
						if (runEnd > runStart) flushRun(span, row, line, runStart, runEnd, flat);
						runStart = runX;
					}
					runEnd = runX + length;
					open = length >= 32 ? 0u : open & ~(((1u << length) - 1u) << bit);
				}
			}
		}
		if (runEnd > runStart) flushRun(span, row, line, runStart, runEnd, flat);
	}

	// Whether no pixel of [begin, end) is covered yet.
	static GEA3D_INLINE bool spanOpen(const std::uint32_t *bits, int begin, int end, int firstWord, int lastWord)
	{
		for (int word = firstWord; word <= lastWord; word++) {
			const int from = detail::maxInt(begin, word << 5);
			const int count = detail::minInt(end, (word + 1) << 5) - from;
			const std::uint32_t mask = count >= 32 ? 0xFFFFFFFFu : (((1u << count) - 1u) << (from & 31));
			if ((bits[word] & mask) != 0) return false;
		}
		return true;
	}

	// Bits of summary word `group` for coverage words [first, last].
	static GEA3D_INLINE std::uint32_t wordRange(int group, int first, int last)
	{
		const int lo = detail::maxInt(first - (group << 5), 0);
		const int hi = detail::minInt(last - (group << 5), 31);
		const std::uint32_t upTo = hi >= 31 ? 0xFFFFFFFFu : (1u << (hi + 1)) - 1u;
		return upTo & ~((1u << lo) - 1u);
	}

	// Paints region-relative pixels [start, end) of a row.
	GEA3D_INLINE void flushRun(const SpanState &span, int row, Pixel *line, int start, int end, bool flat)
	{
		if (flat) {
			detail::fillPixels(line + start, end - start, span.prim->color);
			return;
		}
		fillRun(span, row, start + x0_, end - start);
	}

	// Marks region-relative pixels [x, end) covered.
	GEA3D_INLINE void markCovered(std::uint32_t *bits, std::uint32_t *summary, int local, int x, int end)
	{
		while (x < end) {
			const int word = x >> 5;
			const int last = detail::minInt((word + 1) << 5, end);
			const int count = last - x;
			const std::uint32_t before = bits[word];
			const std::uint32_t now = before | (count >= 32 ? 0xFFFFFFFFu : (((1u << count) - 1u) << (x & 31)));
			bits[word] = now;
			if (now == 0xFFFFFFFFu && before != 0xFFFFFFFFu) {
				summary[word >> 5] |= 1u << (word & 31);
				openWords_[local]--;
			}
			x = last;
		}
	}

	// A Gouraud span without blending or depth.
	GEA3D_INLINE void shadeRow(const SpanState &span, int row, int x, int count)
	{
		const float px = static_cast<float>(x) + 0.5f;
		const float py = static_cast<float>(row) + 0.5f;
		const std::int32_t r = colorStart(span.r, px, py);
		const std::int32_t g = colorStart(span.g, px, py);
		const std::int32_t b = colorStart(span.b, px, py);
		const int last = count - 1;
		const bool inside = span.shortSteps ? channelsInRange(r, g, b, span, last) : inRange(r, span.dr, last) && inRange(g, span.dg, last) && inRange(b, span.db, last);
		if (!inside) {
			fillRun(span, row, x, count);
			return;
		}

		detail::shadePixels(base_ + static_cast<std::ptrdiff_t>(row) * stride_ + x, count, r, g, b, span.dr, span.dg, span.db);
	}

	void fillRun(const SpanState &span, int row, int x, int count)
	{
		Pixel *dst = base_ + static_cast<std::ptrdiff_t>(row) * stride_ + x;
		std::uint16_t *zrow = useDepth_ ? depth_ + static_cast<std::size_t>(row - y0_) * static_cast<std::size_t>(width_) + (x - x0_) : nullptr;
		const float px = static_cast<float>(x) + 0.5f;
		const float py = static_cast<float>(row) + 0.5f;

		std::int32_t z = 0, dz = 0;
		if (useDepth_) {
			z = static_cast<std::int32_t>(detail::clampFloat(span.z.at(px, py), 0.0f, 1.0f) * detail::kDepthScale);
			dz = static_cast<std::int32_t>(span.z.dx * detail::kDepthScale);
		}

		const bool plain = !span.blend && !span.depthTest;
		if (plain && !span.smooth) {
			detail::fillPixels(dst, count, span.prim->color);
			if (span.depthWrite) writeDepth(zrow, count, z, dz);
			return;
		}

		std::int32_t r = 0, g = 0, b = 0, dr = 0, dg = 0, db = 0;
		if (span.smooth) {
			r = colorStart(span.r, px, py);
			g = colorStart(span.g, px, py);
			b = colorStart(span.b, px, py);
			dr = span.dr;
			dg = span.dg;
			db = span.db;
		}

		// Gouraud without blending or depth test: the common lit span. The
		// ramp is linear, so in-range ends mean every pixel is in range.
		const int last = count - 1;
		if (plain && inRange(r, dr, last) && inRange(g, dg, last) && inRange(b, db, last)) {
			detail::shadePixels(dst, count, r, g, b, dr, dg, db);
			if (span.depthWrite) writeDepth(zrow, count, z, dz);
			return;
		}

		const Pixel flat = span.prim->color;
		const int alpha = span.prim->alpha;
		const bool add = (span.prim->flags & kAdd) != 0;
		for (int i = 0; i < count; i++) {
			bool visible = true;
			if (span.depthTest) {
				const std::uint16_t depth = static_cast<std::uint16_t>(z >> 12);
				visible = depth > zrow[i];
				if (visible && span.depthWrite) zrow[i] = depth;
			} else if (span.depthWrite) {
				zrow[i] = static_cast<std::uint16_t>(z >> 12);
			}
			if (visible) {
				const Pixel color = span.smooth ? detail::packRgb(clampColor(r), clampColor(g), clampColor(b)) : flat;
				dst[i] = span.blend ? detail::blendPixel(color, dst[i], alpha, add) : color;
			}
			z += dz;
			r += dr;
			g += dg;
			b += db;
		}
	}

	static GEA3D_INLINE int absStep(std::int32_t step) { return step < 0 ? -step : step; }

	// inRange for all three channels, when their ramps fit 32 bits: a value
	// outside 0..0xFFFFFF at either end sets a bit of the top byte.
	static GEA3D_INLINE bool channelsInRange(std::int32_t r, std::int32_t g, std::int32_t b, const SpanState &span, int last)
	{
		const std::uint32_t ends = static_cast<std::uint32_t>(r | g | b | (r + span.dr * last) | (g + span.dg * last) | (b + span.db * last));
		return (ends & 0xFF000000u) == 0;
	}

	static GEA3D_INLINE bool inRange(std::int32_t start, std::int32_t step, int steps)
	{
		const std::int64_t end = static_cast<std::int64_t>(start) + static_cast<std::int64_t>(step) * steps;
		return start >= 0 && start <= 0xFFFFFF && end >= 0 && end <= 0xFFFFFF;
	}

	static GEA3D_INLINE std::int32_t colorStart(const detail::Plane &plane, float px, float py)
	{
		return static_cast<std::int32_t>(plane.at(px, py) * detail::kColorScale);
	}

	static GEA3D_INLINE std::int32_t clampColor(std::int32_t value)
	{
		if (value < 0) return 0;
		if (value > 0xFFFFFF) return 0xFFFFFF;
		return value;
	}

	static void writeDepth(std::uint16_t *zrow, int count, std::int32_t z, std::int32_t dz)
	{
		for (int i = 0; i < count; i++) {
			zrow[i] = static_cast<std::uint16_t>(z >> 12);
			z += dz;
		}
	}

	// A line, `width` pixels across its minor axis. Opaque lines own their
	// pixels in the coverage mask like triangles do.
	void drawLine(const Prim &prim, bool blendPass)
	{
		const bool depthTest = useDepth_ && (blendPass || frame_.depthBuffer);
		walkLine(prim, [this, &prim, blendPass, depthTest](int x, int y, Pixel color, float t) { plotLine(prim, x, y, color, t, blendPass, depthTest); });
	}

	// Painter's order without blending or depth: pixels go straight to the
	// band; a flat line fills its runs along each row or column at once.
	void drawLinePainter(const Prim &prim)
	{
		Pixel *const base = base_;
		const int stride = stride_;
		if ((prim.flags & kSmooth) != 0) {
			walkLine(prim, [base, stride](int x, int y, Pixel color, float) { base[static_cast<std::ptrdiff_t>(y) * stride + x] = color; });
			return;
		}

		LineSpan w;
		if (!lineSpan(prim, w)) return;

		const Pixel color = prim.color;
		if (!w.xMajor) {
			// One run across per row.
			for (int m = w.first; m <= w.last; m++) {
				const int n = w.minor >> 16;
				const int n0 = detail::maxInt(n + w.lo, w.minorLo);
				const int n1 = detail::minInt(n + w.hi, w.minorHi);
				if (n0 <= n1) detail::fillPixels(base + static_cast<std::ptrdiff_t>(m) * stride + n0, n1 - n0 + 1, color);
				w.minor += w.minorStep;
			}
			return;
		}

		// Runs along a row, as long as the row stays the same.
		int m = w.first;
		while (m <= w.last) {
			const int n = w.minor >> 16;
			int run = w.last - m + 1;
			if (w.minorStep > 0) run = detail::minInt(run, static_cast<int>((static_cast<std::uint32_t>(((n + 1) << 16) - w.minor) + static_cast<std::uint32_t>(w.minorStep) - 1) / static_cast<std::uint32_t>(w.minorStep)));
			if (w.minorStep < 0) run = detail::minInt(run, static_cast<int>(static_cast<std::uint32_t>(w.minor - (n << 16)) / static_cast<std::uint32_t>(-w.minorStep)) + 1);
			const int n0 = detail::maxInt(n + w.lo, w.minorLo);
			const int n1 = detail::minInt(n + w.hi, w.minorHi);
			for (int k = n0; k <= n1; k++) detail::fillPixels(base + static_cast<std::ptrdiff_t>(k) * stride + m, run, color);
			m += run;
			w.minor += w.minorStep * run;
		}
	}

	// A line's walk over the region: its major axis (walked toward increasing
	// coordinates from end `from`), the stretch [first, last] of it that can
	// land in the region, and the minor coordinate there in 16.16 (+ 0.5, so
	// `minor >> 16` is the rounded pixel).
	struct LineSpan {
		bool xMajor;
		int from;
		int lo, hi;            // pixels across, around the minor coordinate
		int minorLo, minorHi;  // the region along the minor axis
		int first, last;
		std::int32_t minor, minorStep;
		float t, inv;          // along the line at `first`, and per pixel
	};

	GEA3D_INLINE bool lineSpan(const Prim &prim, LineSpan &w) const
	{
		const float xa = prim.x[0] * (1.0f / 16.0f) - 0.5f;
		const float ya = prim.y[0] * (1.0f / 16.0f) - 0.5f;
		const float xb = prim.x[1] * (1.0f / 16.0f) - 0.5f;
		const float yb = prim.y[1] * (1.0f / 16.0f) - 0.5f;
		const float dx = xb - xa;
		const float dy = yb - ya;
		w.xMajor = (dx < 0.0f ? -dx : dx) > (dy < 0.0f ? -dy : dy);

		w.from = 0;
		float ma = w.xMajor ? xa : ya, mb = w.xMajor ? xb : yb;
		float na = w.xMajor ? ya : xa, nb = w.xMajor ? yb : xb;
		if (ma > mb) {
			swapFloat(ma, mb);
			swapFloat(na, nb);
			w.from = 1;
		}
		const float span = mb - ma;
		w.inv = span > 0.0f ? 1.0f / span : 0.0f;
		const float slope = (nb - na) * w.inv;
		const int across = detail::maxInt(prim.width, 1);
		w.lo = -((across - 1) >> 1);
		w.hi = w.lo + across - 1;
		w.minorLo = w.xMajor ? y0_ : x0_;
		w.minorHi = w.xMajor ? y1_ : x1_;

		w.first = detail::maxInt(detail::roundToInt(ma), w.xMajor ? x0_ : y0_);
		w.last = detail::minInt(detail::roundToInt(mb), w.xMajor ? x1_ : y1_);
		// Where the minor coordinate can round into the region, with slack.
		if (slope != 0.0f) {
			const float perMinor = 1.0f / slope;
			float m0 = ma + (static_cast<float>(w.minorLo - 1 - w.hi) - na) * perMinor;
			float m1 = ma + (static_cast<float>(w.minorHi + 1 - w.lo) - na) * perMinor;
			if (slope < 0.0f) swapFloat(m0, m1);
			w.first = detail::maxInt(w.first, static_cast<int>(detail::clampFloat(m0, -32768.0f, 32767.0f)) - 1);
			w.last = detail::minInt(w.last, static_cast<int>(detail::clampFloat(m1, -32768.0f, 32767.0f)) + 1);
		} else if (na < static_cast<float>(w.minorLo - 1 - w.hi) || na > static_cast<float>(w.minorHi + 1 - w.lo)) {
			return false;
		}
		if (w.first > w.last) return false;

		const float lead = static_cast<float>(w.first) - ma;
		w.minor = detail::roundToInt((na + lead * slope) * detail::kFixedOne) + 32768;
		w.minorStep = detail::roundToInt(slope * detail::kFixedOne);
		w.t = lead * w.inv;
		return true;
	}

	// Calls visit(x, y, color, t) for a line's pixels in the region: one per
	// column (x-major) or row (y-major) along its major axis, `width` across
	// the other. `t` runs 0..1 from the first end to the second.
	template <typename Visit>
	GEA3D_INLINE void walkLine(const Prim &prim, Visit visit)
	{
		LineSpan w;
		if (!lineSpan(prim, w)) return;

		const int to = 1 - w.from;
		const bool smooth = (prim.flags & kSmooth) != 0;
		std::int32_t r = 0, g = 0, b = 0, dr = 0, dg = 0, db = 0;
		if (smooth) {
			const float start = detail::clampFloat(w.t, 0.0f, 1.0f);
			lineChannel(prim.r[w.from], prim.r[to], start, w.inv, &r, &dr);
			lineChannel(prim.g[w.from], prim.g[to], start, w.inv, &g, &dg);
			lineChannel(prim.b[w.from], prim.b[to], start, w.inv, &b, &db);
		}

		for (int m = w.first; m <= w.last; m++) {
			const Pixel color = smooth ? detail::packRgb(clampColor(r), clampColor(g), clampColor(b)) : prim.color;
			const int n = w.minor >> 16;
			const float along = detail::clampFloat(w.from == 0 ? w.t : 1.0f - w.t, 0.0f, 1.0f);
			const int n0 = detail::maxInt(n + w.lo, w.minorLo);
			const int n1 = detail::minInt(n + w.hi, w.minorHi);
			for (int k = n0; k <= n1; k++) visit(w.xMajor ? m : k, w.xMajor ? k : m, color, along);
			w.minor += w.minorStep;
			w.t += w.inv;
			r += dr;
			g += dg;
			b += db;
		}
	}

	static GEA3D_INLINE void swapFloat(float &a, float &b)
	{
		const float t = a;
		a = b;
		b = t;
	}

	// A colour channel along a line in 16.16: its value at `t` and its step
	// per pixel (`inv` = 1 / the line's length along its major axis).
	static GEA3D_INLINE void lineChannel(int from, int to, float t, float inv, std::int32_t *value, std::int32_t *step)
	{
		const float delta = static_cast<float>(to - from);
		*value = static_cast<std::int32_t>((static_cast<float>(from) + delta * t) * detail::kColorScale);
		*step = static_cast<std::int32_t>(delta * inv * detail::kColorScale);
	}

	// One line pixel at step parameter `t`.
	GEA3D_INLINE void plotLine(const Prim &prim, int x, int y, Pixel color, float t, bool blendPass, bool depthTest)
	{
		if (x < x0_ || x > x1_ || y < y0_ || y > y1_) return;

		const int local = y - y0_;
		const int col = x - x0_;
		std::uint32_t *bits = painter_ ? nullptr : coverage_ + static_cast<std::size_t>(local) * static_cast<std::size_t>(words_);
		if (bits && !blendPass && !frame_.depthBuffer && (bits[col >> 5] & (1u << (col & 31)))) return;

		if (useDepth_) {
			std::uint16_t &stored = depth_[static_cast<std::size_t>(local) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(col)];
			const float d = prim.depth[0] + (prim.depth[1] - prim.depth[0]) * t;
			const std::uint16_t depth = static_cast<std::uint16_t>(detail::clampFloat(d, 0.0f, 1.0f) * 65535.0f);
			if (depthTest && depth <= stored) return;
			if (!blendPass) stored = depth;
		}

		Pixel &dst = base_[static_cast<std::ptrdiff_t>(y) * stride_ + x];
		dst = blendPass ? detail::blendPixel(color, dst, prim.alpha, (prim.flags & kAdd) != 0) : color;
		if (blendPass || !bits) return;

		touch(local);
		markCovered(bits, summary_ + static_cast<std::size_t>(local) * static_cast<std::size_t>(summaryWords_), local, col, col + 1);
	}

	void fillBackground()
	{
		const Background &bg = frame_.background;
		if (bg.mode == kBackgroundNone) return;

		const float span = frame_.height > 1 ? 1.0f / static_cast<float>(frame_.height - 1) : 0.0f;
		const Pixel flat = detail::packColor(bg.top[0], bg.top[1], bg.top[2]);
		for (int row = y0_; row <= y1_; row++) {
			Pixel color = flat;
			if (bg.mode == kBackgroundGradient) {
				const float t = detail::clampFloat(static_cast<float>(row) * span, 0.0f, 1.0f);
				color = detail::packColor(lerp(bg.top[0], bg.bottom[0], t), lerp(bg.top[1], bg.bottom[1], t), lerp(bg.top[2], bg.bottom[2], t));
			}
			fillUncovered(row, color);
		}
	}

	static GEA3D_INLINE int lerp(int a, int b, float t) { return static_cast<int>(static_cast<float>(a) + static_cast<float>(b - a) * t + 0.5f); }

	void fillUncovered(int row, Pixel color)
	{
		const int local = row - y0_;
		Pixel *line = base_ + static_cast<std::ptrdiff_t>(row) * stride_ + x0_;
		if (painter_ || !touched_[local]) {
			detail::fillPixels(line, width_, color);
			return;
		}
		if (openWords_[local] == 0) return;

		const std::uint32_t *bits = coverage_ + static_cast<std::size_t>(local) * static_cast<std::size_t>(words_);
		int x = 0;
		while (x < width_) {
			const int word = x >> 5;
			const std::uint32_t covered = bits[word];
			const int last = detail::minInt((word + 1) << 5, width_);
			if (covered == 0xFFFFFFFFu) {
				x = last;
				continue;
			}
			if (covered == 0) {
				int runEnd = last;
				while (runEnd < width_ && bits[runEnd >> 5] == 0) runEnd = detail::minInt(runEnd + 32, width_);
				detail::fillPixels(line + x, runEnd - x, color);
				x = runEnd;
				continue;
			}
			// Runs of open pixels within the word, found by bit scans.
			std::uint32_t open = ~covered;
			while (open != 0) {
				const int bit = detail::ctz(open);
				const std::uint32_t shifted = open >> bit;
				const int length = shifted == (0xFFFFFFFFu >> bit) ? 32 - bit : detail::ctz(~shifted);
				const int start = (word << 5) + bit;
				detail::fillPixels(line + start, detail::minInt(start + length, last) - start, color);
				open = length >= 32 ? 0u : open & ~(((1u << length) - 1u) << bit);
			}
			x = last;
		}
	}
};

}  // namespace gea::host::gea3d::raster

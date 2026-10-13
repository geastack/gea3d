// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Single-precision matrix helpers for the native scene graph. Matrices are
// column-major float[16] in three.js element order (m[column * 4 + row]).

#include <cmath>
#include <cstring>

namespace gea::host::gea3d::math {

inline void identity(float *m)
{
	std::memset(m, 0, sizeof(float) * 16);
	m[0] = 1.0f;
	m[5] = 1.0f;
	m[10] = 1.0f;
	m[15] = 1.0f;
}

// out = a * b. `out` may alias neither input.
inline void multiply(float *out, const float *a, const float *b)
{
	for (int col = 0; col < 4; col++) {
		const float b0 = b[col * 4];
		const float b1 = b[col * 4 + 1];
		const float b2 = b[col * 4 + 2];
		const float b3 = b[col * 4 + 3];
		for (int row = 0; row < 4; row++) {
			out[col * 4 + row] = a[row] * b0 + a[4 + row] * b1 + a[8 + row] * b2 + a[12 + row] * b3;
		}
	}
}

// Affine multiply: both inputs have a (0, 0, 0, 1) bottom row.
inline void multiplyAffine(float *out, const float *a, const float *b)
{
	for (int col = 0; col < 4; col++) {
		const float b0 = b[col * 4];
		const float b1 = b[col * 4 + 1];
		const float b2 = b[col * 4 + 2];
		for (int row = 0; row < 3; row++) {
			out[col * 4 + row] = a[row] * b0 + a[4 + row] * b1 + a[8 + row] * b2;
		}
		out[col * 4 + 3] = 0.0f;
	}
	out[12] += a[12];
	out[13] += a[13];
	out[14] += a[14];
	out[15] = 1.0f;
}

// Matrix4.compose(position, quaternion, scale).
inline void compose(float *m, const float *p, const float *q, const float *s)
{
	const float x = q[0], y = q[1], z = q[2], w = q[3];
	const float x2 = x + x, y2 = y + y, z2 = z + z;
	const float xx = x * x2, xy = x * y2, xz = x * z2;
	const float yy = y * y2, yz = y * z2, zz = z * z2;
	const float wx = w * x2, wy = w * y2, wz = w * z2;
	m[0] = (1.0f - (yy + zz)) * s[0];
	m[1] = (xy + wz) * s[0];
	m[2] = (xz - wy) * s[0];
	m[3] = 0.0f;
	m[4] = (xy - wz) * s[1];
	m[5] = (1.0f - (xx + zz)) * s[1];
	m[6] = (yz + wx) * s[1];
	m[7] = 0.0f;
	m[8] = (xz + wy) * s[2];
	m[9] = (yz - wx) * s[2];
	m[10] = (1.0f - (xx + yy)) * s[2];
	m[11] = 0.0f;
	m[12] = p[0];
	m[13] = p[1];
	m[14] = p[2];
	m[15] = 1.0f;
}

// Inverse of an affine matrix through its 3x3 cofactors. Returns false (and
// writes identity) when the matrix is singular.
inline bool invertAffine(float *out, const float *m)
{
	const float a00 = m[0], a01 = m[4], a02 = m[8];
	const float a10 = m[1], a11 = m[5], a12 = m[9];
	const float a20 = m[2], a21 = m[6], a22 = m[10];
	const float c00 = a11 * a22 - a12 * a21;
	const float c01 = a12 * a20 - a10 * a22;
	const float c02 = a10 * a21 - a11 * a20;
	const float det = a00 * c00 + a01 * c01 + a02 * c02;
	if (det == 0.0f) {
		identity(out);
		return false;
	}

	const float inv = 1.0f / det;
	out[0] = c00 * inv;
	out[1] = c01 * inv;
	out[2] = c02 * inv;
	out[3] = 0.0f;
	out[4] = (a02 * a21 - a01 * a22) * inv;
	out[5] = (a00 * a22 - a02 * a20) * inv;
	out[6] = (a01 * a20 - a00 * a21) * inv;
	out[7] = 0.0f;
	out[8] = (a01 * a12 - a02 * a11) * inv;
	out[9] = (a02 * a10 - a00 * a12) * inv;
	out[10] = (a00 * a11 - a01 * a10) * inv;
	out[11] = 0.0f;
	const float tx = m[12], ty = m[13], tz = m[14];
	out[12] = -(out[0] * tx + out[4] * ty + out[8] * tz);
	out[13] = -(out[1] * tx + out[5] * ty + out[9] * tz);
	out[14] = -(out[2] * tx + out[6] * ty + out[10] * tz);
	out[15] = 1.0f;
	return true;
}

inline void transformPoint(const float *m, float x, float y, float z, float *out)
{
	out[0] = m[0] * x + m[4] * y + m[8] * z + m[12];
	out[1] = m[1] * x + m[5] * y + m[9] * z + m[13];
	out[2] = m[2] * x + m[6] * y + m[10] * z + m[14];
}

inline void transformDirection(const float *m, float x, float y, float z, float *out)
{
	out[0] = m[0] * x + m[4] * y + m[8] * z;
	out[1] = m[1] * x + m[5] * y + m[9] * z;
	out[2] = m[2] * x + m[6] * y + m[10] * z;
}

inline float maxScaleOnAxis(const float *m)
{
	const float sx = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
	const float sy = m[4] * m[4] + m[5] * m[5] + m[6] * m[6];
	const float sz = m[8] * m[8] + m[9] * m[9] + m[10] * m[10];
	float s = sx > sy ? sx : sy;
	if (sz > s) s = sz;
	return std::sqrt(s);
}

inline bool normalize3(float *v)
{
	const float lenSq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	if (lenSq <= 0.0f) return false;
	const float inv = 1.0f / std::sqrt(lenSq);
	v[0] *= inv;
	v[1] *= inv;
	v[2] *= inv;
	return true;
}

// Six clip-space planes (a, b, c, d; inside when a*x + b*y + c*z + d >= 0) of
// a combined projection * view matrix, normalized so d measures distance.
inline void frustumPlanes(const float *m, float planes[6][4])
{
	const float rows[4][4] = {
		{m[0], m[4], m[8], m[12]},
		{m[1], m[5], m[9], m[13]},
		{m[2], m[6], m[10], m[14]},
		{m[3], m[7], m[11], m[15]},
	};
	for (int i = 0; i < 3; i++) {
		for (int k = 0; k < 4; k++) {
			planes[i * 2][k] = rows[3][k] + rows[i][k];
			planes[i * 2 + 1][k] = rows[3][k] - rows[i][k];
		}
	}
	for (int i = 0; i < 6; i++) {
		float *plane = planes[i];
		const float len = std::sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
		if (len <= 0.0f) continue;
		const float inv = 1.0f / len;
		for (int k = 0; k < 4; k++) plane[k] *= inv;
	}
}

inline bool sphereInFrustum(const float planes[6][4], const float *center, float radius)
{
	for (int i = 0; i < 6; i++) {
		const float *p = planes[i];
		if (p[0] * center[0] + p[1] * center[1] + p[2] * center[2] + p[3] < -radius) return false;
	}
	return true;
}

}  // namespace gea::host::gea3d::math

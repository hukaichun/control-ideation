/* Fly a horizontal circle, using only the state: the reference point on the circle is
 * chosen by the current position angle, so there is no clock and nothing to fall behind.
 * Altitude is held with the same linear feedback as hold.c. */
#include "xpol_abi.h"

#define LAM 1.5f
#define RADIUS 3.0f
#define OMEGA 0.41887902f   /* 2*pi/15 rad/s */
#define ALT (-5.0f)

void xpol_step(void *state, const xpol_math_t *m, const float *x, float *u)
{
	(void)state;
	const float k0 = LAM * LAM * LAM * LAM, k1 = 4 * LAM * LAM * LAM, k2 = 6 * LAM * LAM, k3 = 4 * LAM;
	const float w2 = OMEGA * OMEGA, w4 = w2 * w2;

	const float th = m->atan2f(x[1], x[0]);
	const float c = m->cosf(th), s = m->sinf(th);

	/* reference state on the circle at angle th: p, v, a, j, snap */
	const float p[2] = {RADIUS * c, RADIUS * s};
	const float v[2] = {-RADIUS * OMEGA * s, RADIUS * OMEGA * c};
	const float a[2] = {-w2 * p[0], -w2 * p[1]};
	const float j[2] = {-w2 * v[0], -w2 * v[1]};
	const float snap[2] = {w4 * p[0], w4 * p[1]};

	for (int i = 0; i < 2; i++) {
		u[i] = snap[i] - (k0 * (x[i] - p[i]) + k1 * (x[3 + i] - v[i]) + k2 * (x[6 + i] - a[i]) + k3 * (x[9 + i] - j[i]));
	}

	u[2] = -(k0 * (x[2] - ALT) + k1 * x[5] + k2 * x[8] + k3 * x[11]);
}

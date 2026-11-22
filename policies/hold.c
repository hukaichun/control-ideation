/* Hold a fixed point with a linear state feedback on the flat state.
 * Each axis is a 4th-order integrator chain; poles at (s + LAM)^4. */
#include "xpol_abi.h"

#define LAM 1.5f

static const float TARGET[3] = {2.0f, 0.0f, -5.0f};   /* NED, m */

void xpol_step(void *state, const xpol_math_t *m, const float *x, float *u)
{
	(void)state;
	(void)m;
	const float k0 = LAM * LAM * LAM * LAM, k1 = 4 * LAM * LAM * LAM, k2 = 6 * LAM * LAM, k3 = 4 * LAM;

	for (int i = 0; i < 3; i++) {
		u[i] = -(k0 * (x[i] - TARGET[i]) + k1 * x[3 + i] + k2 * x[6 + i] + k3 * x[9 + i]);
	}
}

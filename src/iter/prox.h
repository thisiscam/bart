/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __PROX_H
#define __PROX_H

#include "misc/cppwrap.h"

struct operator_p_s;
struct linop_s;


extern const struct operator_p_s* prox_normaleq_create(const struct linop_s* op, const _Complex float* y);
extern const struct operator_p_s* prox_leastsquares_create(unsigned int N, const long dims[__VLA(N)], float lambda, const _Complex float* y);
extern const struct operator_p_s* prox_l2ball_create(unsigned int N, const long dims[__VLA(N)], float eps, const _Complex float* center);
extern const struct operator_p_s* prox_zero_create(unsigned int N, const long dims[__VLA(N)]);

extern const struct operator_p_s* prox_lineq_create(const struct linop_s* op, const _Complex float* y);

#include "misc/cppwrap.h"
#endif

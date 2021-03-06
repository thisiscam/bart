/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __OPS_H
#define __OPS_H

#include "misc/cppwrap.h"


typedef void (*operator_fun_t)(const void* _data, unsigned int N, void* args[__VLA(N)]);
typedef void (*operator_p_fun_t)(const void* _data, float mu, _Complex float* _dst, const _Complex float* _src);
typedef void (*operator_del_t)(const void* op);



struct operator_s;
struct operator_p_s;


// create functions

extern const struct operator_s* operator_create(unsigned int ON, const long out_dims[__VLA(ON)],
		unsigned int IN, const long in_dims[__VLA(IN)],
		void* data, operator_fun_t apply, operator_del_t del);

extern const struct operator_s* operator_create2(unsigned int ON, const long out_dims[__VLA(ON)], const long out_strs[__VLA(ON)],
		unsigned int IN, const long in_dims[__VLA(IN)], const long in_strs[__VLA(IN)],
		void* data, operator_fun_t apply, operator_del_t del);

extern const struct operator_p_s* operator_p_create(unsigned int ON, const long out_dims[__VLA(ON)],
			unsigned int IN, const long in_dims[__VLA(IN)], void* data,
			operator_p_fun_t apply, operator_del_t del);

extern const struct operator_p_s* operator_p_create2(unsigned int ON, const long out_dims[__VLA(ON)], const long out_strs[__VLA(ON)],
		unsigned int IN, const long in_dims[__VLA(IN)], const long in_strs[__VLA(IN)],
		void* data, operator_p_fun_t apply, operator_del_t del);


extern const struct operator_s* operator_identity_create(unsigned int N, const long dims[__VLA(N)]);

extern const struct operator_s* operator_chain(const struct operator_s* a, const struct operator_s* b);
extern const struct operator_s* operator_chainN(unsigned int N, const struct operator_s* ops[__VLA(N)]);

//extern const struct operator_s* operator_mul(const struct operator_s* a, const struct operator_s* b);
//extern const struct operator_s* operator_sum(const struct operator_s* a, const struct operator_s* b);
extern const struct operator_s* operator_stack(unsigned int D, unsigned int E, const struct operator_s* a, const struct operator_s* b);

// del functions
extern void operator_free(const struct operator_s* x);
extern void operator_p_free(const struct operator_p_s* x);

extern const struct operator_s* operator_ref(const struct operator_s* x);
extern const struct operator_p_s* operator_p_ref(const struct operator_p_s* x);

// apply functions
extern void operator_apply(const struct operator_s* op, unsigned int IN, const long idims[__VLA(IN)], _Complex float* dst, const long ON, const long odims[__VLA(ON)], const _Complex float* src);
extern void operator_apply2(const struct operator_s* op, unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], _Complex float* dst, const long ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)], const _Complex float* src);
extern void operator_p_apply(const struct operator_p_s* op, float mu, unsigned int IN, const long idims[__VLA(IN)], _Complex float* dst, const long ON, const long odims[__VLA(ON)], const _Complex float* src);
extern void operator_p_apply2(const struct operator_p_s* op, float mu, unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], _Complex float* dst, const long ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)], const _Complex float* src);


extern void operator_apply_unchecked(const struct operator_s* op, _Complex float* dst, const _Complex float* src);
extern void operator_p_apply_unchecked(const struct operator_p_s* op, float mu,  _Complex float* dst, const _Complex float* src);


// get functions
struct iovec_s;
const struct iovec_s* operator_domain(const struct operator_s* op); 
const struct iovec_s* operator_codomain(const struct operator_s* op); 

const struct iovec_s* operator_p_domain(const struct operator_p_s* op); 
const struct iovec_s* operator_p_codomain(const struct operator_p_s* op); 

extern void* operator_get_data(const struct operator_s* op);
extern void* operator_p_get_data(const struct operator_p_s* x);








// iter helper functions
extern void operator_iter(void* _o, float* _dst, const float* _src);
extern void operator_p_iter(void* _o, float lambda, float* _dst, const float* _src);

#include "misc/cppwrap.h"

#endif

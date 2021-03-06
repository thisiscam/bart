/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2015 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 *
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops.h"
#include "num/iovec.h"

#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/misc.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"

#include "iter/iter.h"
#include "iter/iter2.h"

#include "noncart/nufft.h"

#include "sense/recon.h"
#include "sense/model.h"
#include "sense/optcom.h"

#include "wavelet2/wavelet.h"
#include "wavelet3/wavthresh.h"

#include "lowrank/lrthresh.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"



static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-l1/-l2] [-r lambda] [-t <trajectory>] <kspace> <sensitivities> <output>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Parallel-imaging compressed-sensing reconstruction.\n"
		"\n"
		"-l1/-l2\t\ttoggle l1-wavelet or l2 regularization.\n"
		"-r lambda\tregularization parameter\n"
		"-c\t\treal-value constraint\n"
		"-s step\t\titeration stepsize\n"
		"-i maxiter\tnumber of iterations\n"
		"-t trajectory\tk-space trajectory\n"
#ifdef BERKELEY_SVN
		"-n \t\tdisable random wavelet cycle spinning\n"
		"-g \t\tuse GPU\n"
		"-p pat\t\tpattern or weights\n"
#endif
	);
}


static
const struct linop_s* sense_nc_init(const long max_dims[DIMS], const long map_dims[DIMS], const complex float* maps, const long ksp_dims[DIMS], const long traj_dims[DIMS], const complex float* traj, struct nufft_conf_s conf, _Bool use_gpu)
{
	long coilim_dims[DIMS];
	long img_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);

	const struct linop_s* fft_op = nufft_create(DIMS, ksp_dims, coilim_dims, traj_dims, traj, NULL, conf, use_gpu);
	const struct linop_s* maps_op = maps2_create(coilim_dims, map_dims, img_dims, maps, use_gpu);

	const struct linop_s* lop = linop_chain(maps_op, fft_op);

	linop_free(maps_op);
	linop_free(fft_op);

	return lop;
}


int main_pics(int argc, char* argv[])
{
	// Initialize default parameters

	struct sense_conf conf = sense_defaults;

	bool admm = false;
	bool ist = false;
	bool use_gpu = false;
	bool l1wav = false;
	bool tvreg = false;
	bool lowrank = false;
	bool randshift = true;
	unsigned int maxiter = 30;
	float step = 0.95;
	float lambda = 0.;
	unsigned int rflags = 7;

	// Start time count

	double start_time = timestamp();

	// Read input options
	struct nufft_conf_s nuconf = nufft_conf_defaults;
	nuconf.toeplitz = false;

	float restrict_fov = -1.;
	const char* pat_file = NULL;
	const char* traj_file = NULL;
	const char* image_truth_file = NULL;
	bool im_truth = false;
	bool scale_im = false;
	bool eigen = false;

	bool hogwild = false;
	bool fast = false;
	float admm_rho = iter_admm_defaults.rho;

	if (0 == strcmp(basename(argv[0]), "sense"))
		debug_printf(DP_WARN, "The \'sense\' command is deprecated. Use \'pics\' instead.\n");

	int c;
	while (-1 != (c = getopt(argc, argv, "Fq:l:r:s:i:u:o:O:f:t:cT:Imngehp:Sd:R:H"))) {
		switch(c) {

		case 'I':
			ist = true;
			break;

		case 'e':
			eigen = true;
			break;

		case 'H':
			hogwild = true;
			break;

		case 'F':
			fast = true;
			break;

		case 'T':
			im_truth = true;
			image_truth_file = strdup(optarg);
			assert(NULL != image_truth_file);
			break;

		case 'd':
			debug_level = atoi(optarg);
			break;

		case 'r':
			lambda = atof(optarg);
			break;

		case 'R':
			rflags = atoi(optarg);
			break;

		case 'O':
			conf.rwiter = atoi(optarg);
			break;

		case 'o':
			conf.gamma = atof(optarg);
			break;

		case 's':
			step = atof(optarg);
			break;

		case 'i':
			maxiter = atoi(optarg);
			break;

		case 'u':
			admm_rho = atof(optarg);
			break;

		case 'l':
			if (0 == strcmp("1", optarg)) {

				l1wav = true;

			} else
			if (0 == strcmp("2", optarg)) {

				l1wav = false;

			} else
			if (0 == strcmp("v", optarg)) {

				l1wav = false;
				tvreg = true;
				admm = true;

			} else
			if (0 == strcmp("r", optarg)) {

				lowrank = true;
				l1wav = false;

			} else {

				usage(argv[0], stderr);
				exit(1);
			}
			break;

		case 'q':
			conf.cclambda = atof(optarg);
			break;

		case 'c':
			conf.rvc = true;
			break;

		case 'f':
			restrict_fov = atof(optarg);
			break;

		case 'm':
			admm = true;
			break;

		case 'g':
			use_gpu = true;
			break;

		case 'p':
			pat_file = strdup(optarg);
			break;

		case 't':
			traj_file = strdup(optarg);
			break;

		case 'S':
			scale_im = true;
			break;

		case 'n':
			randshift = false;
			break;

		case 'h':
			usage(argv[0], stdout);
			help();
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 3) {

		usage(argv[0], stderr);
		exit(1);
	}

	long max_dims[DIMS];
	long map_dims[DIMS];
	long pat_dims[DIMS];
	long img_dims[DIMS];
	long coilim_dims[DIMS];
	long ksp_dims[DIMS];
	long traj_dims[DIMS];


	// load kspace and maps and get dimensions

	complex float* kspace = load_cfl(argv[optind + 0], DIMS, ksp_dims);
	complex float* maps = load_cfl(argv[optind + 1], DIMS, map_dims);


	complex float* traj = NULL;

	if (NULL != traj_file)
		traj = load_cfl(traj_file, DIMS, traj_dims);


	md_copy_dims(DIMS, max_dims, ksp_dims);
	md_copy_dims(5, max_dims, map_dims);

	md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);
	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);

	if (!md_check_compat(DIMS, ~(MD_BIT(MAPS_DIM)|FFT_FLAGS), img_dims, map_dims))
		error("Dimensions of image and sensitivities do not match!\n");

	assert(1 == ksp_dims[MAPS_DIM]);


	(use_gpu ? num_init_gpu : num_init)();

	// print options

	if (use_gpu)
		debug_printf(DP_INFO, "GPU reconstruction\n");

	if (map_dims[MAPS_DIM] > 1) 
		debug_printf(DP_INFO, "%ld maps.\nESPIRiT reconstruction.\n", map_dims[MAPS_DIM]);

	if (l1wav)
		debug_printf(DP_INFO, "l1-wavelet regularization\n");

	if (hogwild)
		debug_printf(DP_INFO, "Hogwild stepsize\n");

	if (ist)
		debug_printf(DP_INFO, "Use IST\n");

	if (im_truth)
		debug_printf(DP_INFO, "Compare to truth\n");



	// initialize sampling pattern

	complex float* pattern = NULL;

	if (NULL != pat_file) {

		pattern = load_cfl(pat_file, DIMS, pat_dims);

		assert(md_check_compat(DIMS, COIL_FLAG, ksp_dims, pat_dims));

	} else {

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
		pattern = md_alloc(DIMS, pat_dims, CFL_SIZE);
		estimate_pattern(DIMS, ksp_dims, COIL_DIM, pattern, kspace);
	}


	if ((NULL != traj_file) && (NULL == pat_file)) {

		md_free(pattern);
		pattern = NULL;
		nuconf.toeplitz = true;

	} else {

		// print some statistics

		long T = md_calc_size(DIMS, pat_dims);
		long samples = (long)pow(md_znorm(DIMS, pat_dims, pattern), 2.);

		debug_printf(DP_INFO, "Size: %ld Samples: %ld Acc: %.2f\n", T, samples, (float)T / (float)samples);
	}

	if (NULL == traj_file) {

		fftmod(DIMS, ksp_dims, FFT_FLAGS, kspace, kspace);
		fftmod(DIMS, map_dims, FFT_FLAGS, maps, maps);
	}

	// apply fov mask to sensitivities

	if (-1. != restrict_fov) {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;

		apply_mask(DIMS, map_dims, maps, restrict_dims);
	}


	// initialize forward_op

	const struct linop_s* forward_op = NULL;

	if (NULL == traj_file)
		forward_op = sense_init(max_dims, FFT_FLAGS|COIL_FLAG|MAPS_FLAG, maps, use_gpu);
	else
		forward_op = sense_nc_init(max_dims, map_dims, maps, ksp_dims, traj_dims, traj, nuconf, use_gpu);

	// apply scaling

	float scaling = 0.;

	if (NULL == traj_file) {

		scaling = estimate_scaling(ksp_dims, NULL, kspace);

	} else {

		complex float* adj = md_alloc(DIMS, img_dims, CFL_SIZE);

		linop_adjoint(forward_op, DIMS, img_dims, adj, DIMS, ksp_dims, kspace);
		scaling = estimate_scaling_norm(1., md_calc_size(DIMS, img_dims), adj, false);

		md_free(adj);
	}

	if (scaling != 0.)
		md_zsmul(DIMS, ksp_dims, kspace, kspace, 1. / scaling);

	if (eigen) {

		double maxeigen = estimate_maxeigenval(forward_op->normal);

		debug_printf(DP_INFO, "Maximum eigenvalue: %.2e\n", maxeigen);

		step /= maxeigen;
	}


	// initialize thresh_op
	const struct operator_p_s* thresh_op = NULL;

	if (l1wav) {

		long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
		minsize[0] = MIN(img_dims[0], 16);
		minsize[1] = MIN(img_dims[1], 16);
		minsize[2] = MIN(img_dims[2], 16);

		if (7 == rflags) {

			thresh_op = prox_wavethresh_create(DIMS, img_dims, FFT_FLAGS, minsize, lambda, randshift, use_gpu);

		} else {

			unsigned int wflags = 0;
			for (unsigned int i = 0; i < DIMS; i++) {

				if ((1 < img_dims[i]) && MD_IS_SET(rflags, i)) {

					wflags = MD_SET(wflags, i);
					minsize[i] = MIN(img_dims[i], 16);
				}
			}

			thresh_op = prox_wavelet3_thresh_create(DIMS, img_dims, wflags, minsize, lambda, randshift);
		}
	}

	if (lowrank) {

		long blkdims[1][DIMS];

		// add a very basic lowrank penalty
		int levels = llr_blkdims(blkdims, MD_BIT(TIME_DIM), img_dims, img_dims[TIME_DIM]);

		assert(1 == levels);
		img_dims[LEVEL_DIM] = levels;

		for(int l = 0; l < levels; l++)
			blkdims[l][MAPS_DIM] = img_dims[MAPS_DIM];

		unsigned int mflags = 6;
		int remove_mean = 0;

		thresh_op = lrthresh_create(img_dims, randshift, mflags, (const long (*)[DIMS])blkdims, lambda, false, remove_mean, use_gpu);
	}



	complex float* image = create_cfl(argv[optind + 2], DIMS, img_dims);
	md_clear(DIMS, img_dims, image, CFL_SIZE);


	long img_truth_dims[DIMS];
	complex float* image_truth = NULL;

	if (im_truth) {

		image_truth = load_cfl(image_truth_file, DIMS, img_truth_dims);
		//md_zsmul(DIMS, img_dims, image_truth, image_truth, 1. / scaling);
	}


	// initialize algorithm

	italgo_fun2_t italgo = iter2_call_iter;
	struct iter_call_s iter2_data;

	void* iconf = &iter2_data;

	struct iter_conjgrad_conf cgconf;
	struct iter_fista_conf fsconf;
	struct iter_ist_conf isconf;
	struct iter_admm_conf mmconf;

	if (!(l1wav || tvreg || lowrank)) {

		cgconf = iter_conjgrad_defaults;
		cgconf.maxiter = maxiter;
		cgconf.l2lambda = lambda;

		iter2_data.fun = iter_conjgrad;
		iter2_data._conf = &cgconf;

	} else if (ist) {

		isconf = iter_ist_defaults;
		isconf.maxiter = maxiter;
		isconf.step = step;
		isconf.hogwild = hogwild;

		iter2_data.fun = iter_ist;
		iter2_data._conf = &isconf;

	} else if (admm) {

		debug_printf(DP_INFO, "ADMM\n");

		mmconf = iter_admm_defaults;
		mmconf.maxiter = maxiter;
		mmconf.rho = admm_rho;
		mmconf.hogwild = hogwild;
		mmconf.fast = fast;
//		mmconf.dynamic_rho = true;
		mmconf.ABSTOL = 0.;
		mmconf.RELTOL = 0.;

		italgo = iter2_admm;
		iconf = &mmconf;

	} else {

		fsconf = iter_fista_defaults;
		fsconf.maxiter = maxiter;
		fsconf.step = step;
		fsconf.hogwild = hogwild;

		iter2_data.fun = iter_fista;
		iter2_data._conf = &fsconf;
	}

	// TV operator

	const struct linop_s* tv_op = tvreg ? grad_init(DIMS, img_dims, rflags)
					: linop_identity_create(DIMS, img_dims);

	if (tvreg) {

		thresh_op = prox_thresh_create(DIMS + 1,
					linop_codomain(tv_op)->dims,
					lambda, MD_BIT(DIMS), use_gpu);
	}



	if (use_gpu) 
#ifdef USE_CUDA
		sense_recon2_gpu(&conf, max_dims, image, forward_op, pat_dims, pattern,
				italgo, iconf, 1, MAKE_ARRAY(thresh_op), admm ? MAKE_ARRAY(tv_op) : NULL,
				ksp_dims, kspace, image_truth);
#else
		assert(0);
#endif
	else
		sense_recon2(&conf, max_dims, image, forward_op, pat_dims, pattern,
				italgo, iconf, 1, MAKE_ARRAY(thresh_op), admm ? MAKE_ARRAY(tv_op) : NULL,
				ksp_dims, kspace, image_truth);

	if (scale_im)
		md_zsmul(DIMS, img_dims, image, image, scaling);

	// clean up

	if (NULL != pat_file)
		unmap_cfl(DIMS, pat_dims, pattern);
	else
		md_free(pattern);


	unmap_cfl(DIMS, map_dims, maps);
	unmap_cfl(DIMS, ksp_dims, kspace);
	unmap_cfl(DIMS, img_dims, image);

	if (NULL != traj)
		unmap_cfl(DIMS, traj_dims, traj);

	if (im_truth) {

		free((void*)image_truth_file);
		unmap_cfl(DIMS, img_dims, image_truth);
	}


	double end_time = timestamp();

	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);
	exit(0);
}



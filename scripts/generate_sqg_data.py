#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sdecast.sqg.solver import SQG, rfft2, irfft2
import os
from tqdm import tqdm
from argparse import ArgumentParser
import uuid
import multiprocessing

def parse_args():
    parser = ArgumentParser(
        description=__doc__, formatter_class=__import__("argparse").RawDescriptionHelpFormatter)
    parser.add_argument('--N', type=int, default=64, help='Grid size')
    parser.add_argument('--hrs', type=float, default=3.0,
                        help='Interval between frames in hours (may be fractional, e.g. 0.01)')
    parser.add_argument('--n_traj', type=int, default=1,
                        help='Number of trajectories')
    parser.add_argument('--n_times', type=int, default=100,
                        help='Number of time steps')
    parser.add_argument('--data_path', type=str, default='data/sqg',
                        help='Directory to write trajectories into')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.data_path, exist_ok=True)
    print(
        f"Generating data with N={args.N}, n_traj={args.n_traj}, n_times={args.n_times}, data_path={args.data_path}")

    versions = args.n_traj

    for version in tqdm(range(versions)):
        '''
        “dt" is directly related to “N" through the Courant-Friedrichs-Lewy condition — “dt" should become smaller if you increase “N",
        otherwise the numerical scheme becomes unstable.

        “diff_efold” is the numerical dissipation which ensures energy does not accumulate erroneously at the largest wavenumbers (smallest scales).

        As long the model does not crash, it will be ok to proceed.
        '''
        if args.N == 1024:
            N = 1024
            dt = 40
            diff_efold = 900.
        elif args.N == 512:
            N = 512
            dt = 90
            diff_efold = 1800.
        elif args.N == 256:
            N = 256
            dt = 90
            diff_efold = 86400./16.
        elif args.N == 192:
            N = 192
            dt = 300
            diff_efold = 86400./8.
        elif args.N == 128:
            N = 128
            dt = 600
            diff_efold = 86400./3.
        elif args.N == 96:
            N = 96
            dt = 900
            diff_efold = 86400./3.
        elif args.N == 64:
            N = 64
            dt = 1200
            diff_efold = 86400.
        elif args.N == 32:
            N = 32
            dt = 2400
            diff_efold = 86400. * 3
        elif args.N == 16:
            N = 16
            dt = 4800
            diff_efold = 86400. * 9

        norder = 8
        dealias = True

        dek = 0
        nsq = 1.e-4
        f = 1.e-4
        g = 9.8
        theta0 = 300
        H = 10.e3
        r = dek*nsq/f
        U = 30
        Lr = np.sqrt(nsq)*H/f
        L = 20.*Lr
        tdiab = 10.*86400
        symmetric = True
        scalefact = f*theta0/g

        pv = np.random.normal(0, 100., size=(2, N, N)).astype(float)

        nexp = 20
        x = np.arange(0, 2.*np.pi, 2.*np.pi/N)
        y = np.arange(0., 2.*np.pi, 2.*np.pi/N)
        x, y = np.meshgrid(x, y)
        x = x.astype(float)
        y = y.astype(float)
        pv[1] = pv[1]+2000.*(np.sin(x/2)**(2*nexp)*np.sin(y)**nexp)
        for k in range(2):
            pv[k] = pv[k] - pv[k].mean()

        threads = int(os.getenv('OMP_NUM_THREADS', '1'))

        print(f'Available CPU threads: {multiprocessing.cpu_count()}')
        print(f'Using {threads} threads')
        print('Using %d threads' % threads)

        precision = 'single'

        hrs = args.hrs
        outputinterval = hrs * 3600.
        timesteps_per_frame = max(1, int(np.ceil(outputinterval / dt)))
        dt = outputinterval / timesteps_per_frame

        model = SQG(pv, nsq=nsq, f=f, U=U, H=H, r=r, tdiab=tdiab, dt=dt,
                    diff_order=norder, diff_efold=diff_efold,
                    dealias=dealias, symmetric=symmetric, threads=threads,
                    precision=precision, tstart=0)

        n_days_min = 300
        tmin = n_days_min * 86400.
        n_days = args.n_times*hrs/24 + n_days_min
        print(f"Simulating {n_days} days")
        tmax = n_days * 86400.
        nsteps = int(tmax/outputinterval)
        assert (
            tmax-tmin)/outputinterval == args.n_times, f"nsteps {(tmax-tmin)/outputinterval} != n_times {args.n_times}"

        print(
            f"Running SQG turbulence simulation, generating {(tmax-tmin)/outputinterval} frames, each {outputinterval/3600.} hours apart.")

        model.timesteps = timesteps_per_frame

        random_id = str(uuid.uuid4())[:4]
        savedata = f'{args.data_path}/sqg_N{N}_{hrs}hrly_steps_{args.n_times}_{version}_{random_id}'

        if savedata is not None:
            from netCDF4 import Dataset
            nc = Dataset(f'{savedata}.nc', mode='w',
                         format='NETCDF4_CLASSIC')
            nc.r = model.r
            nc.f = model.f
            nc.U = model.U
            nc.L = model.L
            nc.H = model.H
            nc.g = g
            nc.theta0 = theta0
            nc.nsq = model.nsq
            nc.tdiab = model.tdiab
            nc.dt = model.dt
            nc.diff_efold = model.diff_efold
            nc.diff_order = model.diff_order
            nc.symmetric = int(model.symmetric)
            nc.dealias = int(model.dealias)
            x = nc.createDimension('x', N)
            y = nc.createDimension('y', N)
            z = nc.createDimension('z', 2)
            t = nc.createDimension('t', None)
            pvvar =\
                nc.createVariable(
                    'pv', float, ('t', 'z', 'y', 'x'), zlib=True)
            pvvar.units = 'K'
            xvar = nc.createVariable('x', float, ('x',))
            xvar.units = 'meters'
            yvar = nc.createVariable('y', float, ('y',))
            yvar.units = 'meters'
            zvar = nc.createVariable('z', float, ('z',))
            zvar.units = 'meters'
            tvar = nc.createVariable('t', float, ('t',))
            tvar.units = 'seconds'
            xvar[:] = np.arange(0, model.L, model.L/N)
            yvar[:] = np.arange(0, model.L, model.L/N)
            zvar[0] = 0
            zvar[1] = model.H

        nout = 0

        levplot = 1
        t = 0.0
        initial_t = t

        if hasattr(tmax, 'size'):
            tmax_value = float(tmax.size)
        else:
            tmax_value = float(tmax)

        total_iterations = int(
            (tmax_value - t) / (model.dt * model.timesteps)) + 1

        progress_bar = tqdm(total=total_iterations,
                            desc="Simulation progress", unit="steps")

        steps_completed = 0
        while t < tmax:
            model.advance()
            t = model.t
            pv = irfft2(model.pvspec)
            hr = t/3600.

            steps_completed += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"Hours": f"{hr:.2f}"})

            if savedata is not None and t >= tmin:
                pvvar[nout, :, :, :] = pv
                tvar[nout] = t
                nc.sync()
                if t >= tmax:
                    nc.close()
                nout = nout + 1

        progress_bar.close()

        nc = Dataset(f'{savedata}.nc', 'r')
        X = np.array(nc['pv'][:])
        nc.close()
        np.save(f'{savedata}.npy', X)

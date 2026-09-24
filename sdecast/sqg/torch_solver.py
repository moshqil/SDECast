import numpy as np
import os
from tqdm import tqdm
from argparse import ArgumentParser
import uuid
import multiprocessing
import torch
import torch.nn as nn

# model parameters.
'''
“dt" is directly related to “N" through the Courant-Friedrichs-Lewy condition — “dt" should become smaller if you increase “N",
otherwise the numerical scheme becomes unstable.

“diff_efold” is the numerical dissipation which ensures energy does not accumulate erroneously at the largest wavenumbers (smallest scales).

As long the model does not crash, it will be ok to proceed.
'''

class torchSQG:
    def __init__(
        self,
        pv,
        f=1.0e-4,
        nsq=1.0e-4,
        L=20.0e6,
        H=10.0e3,
        U=30.0,
        r=0.0,
        tdiab=10.0 * 86400,
        diff_order=8,
        diff_efold=None,
        symmetric=True,
        dt=None,
        dealias=True,
        precision="single",
        tstart=0,
        device='cpu'
    ):
        # initialize SQG model.
        if pv.shape[1] != 2:
            raise ValueError("1st dim of pv should be 2")
        N = pv.shape[-1]  # number of grid points in each direction
        # N should be even
        if N % 2:
            raise ValueError("N must be even (powers of 2 are fastest)")
        if dt is None:  # time step must be specified
            raise ValueError("must specify time step")
        if diff_efold is None:  # efolding time scale for diffusion must be specified
            raise ValueError("must specify efolding time scale for diffusion")
        self.N = N
        if precision == "single":
            # ffts in single precision (faster)
            dtype = torch.float32
        elif precision == "double":
            # ffts in double precision
            dtype = torch.float64
        else:
            msg = "precision must be 'single' or 'double'"
            raise ValueError(msg)
        # force arrays to be float32 for precision='single' (ffts are twice as fast)
        # Brunt-Vaisalla (buoyancy) freq squared
        self.nsq = nsq # torch.tensor(nsq, dtype=dtype, device=device)
        self.f = f# torch.tensor(f, dtype=dtype, device=device)  # coriolis
        self.H = H #torch.tensor(H, dtype=dtype, device=device)  # height of upper boundary
        self.U = U #torch.tensor(U, dtype=dtype, device=device)  # basic state velocity at z = H
        self.L = L #torch.tensor(L, dtype=dtype, device=device)  # size of square domain.
        self.dt = dt# torch.tensor(dt, dtype=dtype, device=device)  # time step (seconds)
        self.dealias = dealias  # if True, dealiasing applied using 2/3 rule.
        if r < 1.0e-10:
            self.ekman = False
        else:
            self.ekman = True
        self.r = r #torch.tensor(r, dtype=dtype, device=device)  # Ekman damping (at z=0)
        self.tdiab = tdiab #torch.tensor(tdiab, dtype=dtype, device=device)  # thermal relaxation damping.
        self.t = tstart #torch.tensor(tstart, dtype=dtype, device=device)  # initialize time counter
        # setup basic state pv (for thermal relaxation)
        self.symmetric = symmetric  # symmetric jet, or jet with U=0 at sfc.
        y = torch.arange(0, L, L / N, dtype=dtype, device=device)
        pvbar = torch.zeros((2, N), dtype=dtype, device=device)
        pi = np.pi #torch.tensor(np.pi, dtype=dtype)
        l = 2.0 * pi / L
        mu = l * np.sqrt(nsq) * H / f
        if symmetric:
            # symmetric version, no difference between upper and lower
            # boundary.
            # l = 2.*pi/L and mu = l*N*H/f
            # u = -0.5*U*np.sin(l*y)*np.sinh(mu*(z-0.5*H)/H)*np.sin(l*y)/np.sinh(0.5*mu)
            # theta = (f*theta0/g)*(0.5*U*mu/(l*H))*np.cosh(mu*(z-0.5*H)/H)*
            # np.cos(l*y)/np.sinh(0.5*mu)
            # + theta0 + (theta0*nsq*z/g)
            pvbar[:] = (
                -(mu * 0.5 * U / (l * H))
                * np.cosh(0.5 * mu)
                * torch.cos(l * y)
                / np.sinh(0.5 * mu)
            )
        else:
            # asymmetric version, equilibrium state has no flow at surface and
            # temp gradient slightly weaker at sfc.
            # u = U*np.sin(l*y)*np.sinh(mu*z/H)*np.sin(l*y)/np.sinh(mu)
            # theta = (f*theta0/g)*(U*mu/(l*H))*np.cosh(mu*z/H)*
            # np.cos(l*y)/np.sinh(mu)
            # + theta0 + (theta0*nsq*z/g)
            pvbar[:] = -(mu * U / (l * H)) * torch.cos(l * y) / np.sinh(mu)
            pvbar[1, :] = pvbar[0, :] * np.cosh(mu)
        #pvbar.shape = (2, N, 1)
        pvbar = pvbar.unsqueeze(-1).expand(-1, -1, N)
        #pvbar = pvbar * torch.ones((2, N, N), dtype=dtype, device=device)
        self.pvbar = pvbar
        self.pvspec_eq = torch.fft.rfft2(pvbar)  # state to relax to with timescale tdiab
        self.pvspec = torch.fft.rfft2(pv)  # initial pv field (spectral)
        # spectral stuff
        k = (N * torch.fft.fftfreq(N, device=device))[0: (N // 2) + 1]
        l = N * torch.fft.fftfreq(N, device=device)
        k, l = torch.meshgrid(k, l, indexing="xy")
        k = k.to(dtype)
        l = l.to(dtype)
        # dimensionalize wavenumbers.
        k = 2.0 * pi * k / self.L
        l = 2.0 * pi * l / self.L
        k = k.unsqueeze(0)
        l = l.unsqueeze(0)
        ksqlsq = k ** 2 + l ** 2
        self.k = k
        self.l = l
        self.ksqlsq = ksqlsq
        self.ik = (1.0j * k).to(torch.complex64)
        self.il = (1.0j * l).to(torch.complex64)
        if dealias:  # arrays needed for dealiasing nonlinear Jacobian
            k_pad = ((3 * N // 2) * torch.fft.fftfreq(3 * N // 2, device=device)
                     )[0: (3 * N // 4) + 1]
            l_pad = (3 * N // 2) * torch.fft.fftfreq(3 * N // 2, device=device)
            k_pad, l_pad = torch.meshgrid(k_pad, l_pad, indexing="xy")
            k_pad = k_pad.to(dtype)
            l_pad = l_pad.to(dtype)
            k_pad = 2.0 * pi * k_pad / self.L
            l_pad = 2.0 * pi * l_pad / self.L
            k_pad=k_pad.unsqueeze(0)
            l_pad=l_pad.unsqueeze(0)
            self.ik_pad = (1.0j * k_pad).to(torch.complex64)
            self.il_pad = (1.0j * l_pad).to(torch.complex64)
        mu = torch.sqrt(ksqlsq) * np.sqrt(self.nsq) * self.H / self.f
        #mu = torch.clip()
        mu=mu.clip(torch.finfo(mu.dtype).eps)  # clip to avoid NaN

        self.Hovermu = self.H / mu
        mu = mu.to(torch.float64)  # cast to avoid overflow in sinh
        self.tanhmu = torch.tanh(mu).to(dtype)  # cast back to original type
        self.sinhmu = torch.sinh(mu).to(dtype)
        self.diff_order = diff_order # torch.tensor(diff_order, dtype=dtype)  # hyperdiffusion order
        self.diff_efold = diff_efold # torch.tensor(diff_efold, dtype=dtype)  # hyperdiff time scale
        ktot = torch.sqrt(ksqlsq)
        ktotcutoff = pi * N / self.L # torch.tensor(pi * N / self.L, dtype=dtype)
        # integrating factor for hyperdiffusion
        # with efolding time scale for diffusion of shortest wave (N/2)
        self.hyperdiff = torch.exp(
            (-self.dt / self.diff_efold) *
            (ktot / ktotcutoff) ** self.diff_order
        )
        # number of timesteps to advance each call to 'advance' method.
        self.timesteps = 1
        self.hyperdiff_coef = (ktot / ktotcutoff) ** self.diff_order / self.diff_efold

    def invert(self, pvspec=None):
        if pvspec is None:
            pvspec = self.pvspec
        # invert boundary pv to get streamfunction
        # psispec = torch.empty((2, self.N, self.N // 2 + 1), dtype=pvspec.dtype, device = pvspec.device)
        psispec = torch.empty_like(pvspec)
        psispec[:, 0] = self.Hovermu * (
            (pvspec[:, 1] / self.sinhmu) - 
            (pvspec[:, 0] / self.tanhmu)
        )
        psispec[:, 1] = self.Hovermu * (
            (pvspec[:, 1] / self.tanhmu) - 
            (pvspec[:, 0] / self.sinhmu)
        )
        return psispec

    def invert_inverse(self, psispec=None):
        if psispec is None:
            psispec = self.invert(self.pvspec)
        # given streamfunction, return PV
        # pvspec = torch.empty((2, self.N, self.N // 2 + 1), dtype=psispec.dtype, device = psispec.device)
        pvspec = torch.empty_like(psispec)

        alpha = self.Hovermu
        th = self.tanhmu
        sh = self.sinhmu
        tmp1 = 1.0 / sh ** 2 - 1.0 / th ** 2
        tmp1[:, 0, 0] = 1.0
        pvspec[:, 0] = (
            (psispec[:, 0] / th) - 
            (psispec[:, 1] / sh)
            ) / (alpha * tmp1)
        pvspec[:, 1] = (
            (psispec[:, 0] / sh) - 
            (psispec[:, 1] / th)
            ) / (alpha * tmp1)
        pvspec[:, :, 0, 0] = 0.0  # area mean PV not determined by streamfunction
        return pvspec

    def advance(self, pv=None):
        # given total pv on grid, advance forward
        # number of timesteps given by 'timesteps' instance var.
        # if pv not specified, use pvspec instance variable.
        if pv is not None:
            self.pvspec = torch.fft.rfft2(pv)
        for n in range(self.timesteps):
            self.timestep()
        return torch.fft.irfft2(self.pvspec)

    def specpad(self, specarr):
        # pad spectral arrays with zeros to get
        # interpolation to 3/2 larger grid using inverse fft.
        # take care of normalization factor for inverse transform.
        B = specarr.shape[0]
        specarr_pad = torch.zeros(
            (B, 2, 3 * self.N // 2, 3 * self.N // 4 + 1), 
            dtype=specarr.dtype, device=specarr.device)
        specarr_pad[:, :, 0: self.N // 2, 0: self.N // 2] = (
            2.25 * specarr[:, :, 0: self.N // 2, 0: self.N // 2])
        specarr_pad[:, :, -self.N // 2:, 0: self.N // 2] = (
            2.25 * specarr[:, :, -self.N // 2:, 0: self.N // 2])
        # include negative Nyquist frequency.
        specarr_pad[:, :, 0: self.N // 2, self.N // 2] = \
            torch.conj(2.25 * specarr[:, :, 0: self.N // 2, -1])
        specarr_pad[:, :, -self.N // 2:, self.N // 2] = \
            torch.conj(2.25 * specarr[:, :, -self.N // 2:, -1])
        return specarr_pad

    def spectrunc(self, specarr):
        # truncate spectral array using 2/3 rule.
        B = specarr.shape[0]
        specarr_trunc = torch.zeros((B, 2, self.N, self.N // 2 + 1),
            dtype=specarr.dtype, device=specarr.device)
        specarr_trunc[:, :, 0: self.N // 2, 0: self.N // 2] = \
            specarr[:, :, 0: self.N // 2, 0: self.N // 2]
        specarr_trunc[:, :, -self.N // 2:, 0: self.N // 2] = \
            specarr[:, :, -self.N // 2:, 0: self.N // 2]
        return specarr_trunc

    def xyderiv(self, specarr):
        if not self.dealias:
            xderiv = torch.fft.irfft2(self.ik * specarr)
            yderiv = torch.fft.irfft2(self.il * specarr)
        else:  # pad spectral coeffs with zeros for dealiased jacobian
            specarr_pad = self.specpad(specarr)
            xderiv = torch.fft.irfft2(self.ik_pad * specarr_pad)
            yderiv = torch.fft.irfft2(self.il_pad * specarr_pad)
        return xderiv, yderiv

    def gettend(self, pvspec=None):
        # compute tendencies of pv on z=0,H
        # invert pv to get streamfunction
        if pvspec is None:
            pvspec = self.pvspec
        psispec = self.invert(pvspec)
        # nonlinear jacobian and thermal relaxation
        psix, psiy = self.xyderiv(psispec)
        pvx, pvy = self.xyderiv(pvspec)
        jacobian = psix * pvy - psiy * pvx
        jacobianspec = torch.fft.rfft2(jacobian)
        if self.dealias:  # 2/3 rule: truncate spectral coefficients of jacobian
            jacobianspec = self.spectrunc(jacobianspec)
        dpvspecdt = (1.0 / self.tdiab) * \
            (self.pvspec_eq - pvspec) - jacobianspec
        # Ekman damping at boundaries.
        if self.ekman:
            dpvspecdt[:, 0] += self.r * self.ksqlsq * psispec[:, 0]
            # for asymmetric jet (U=0 at sfc), no Ekman layer at lid
            if self.symmetric:
                dpvspecdt[:, 1] -= self.r * self.ksqlsq * psispec[:, 1]
        # save wind field
        self.u = -psiy
        self.v = psix
        return dpvspecdt

    def timestep(self):
        # update pv using 4th order runge-kutta time step with
        # implicit "integrating factor" treatment of hyperdiffusion.
        self.rkstep = 0
        k1 = self.dt * self.gettend(self.pvspec)
        self.rkstep = 1
        k2 = self.dt * self.gettend(self.pvspec + 0.5 * k1)
        self.rkstep = 2
        k3 = self.dt * self.gettend(self.pvspec + 0.5 * k2)
        self.rkstep = 3
        k4 = self.dt * self.gettend(self.pvspec + k3)
        pvspecnew = self.pvspec + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        self.pvspec = self.hyperdiff * pvspecnew
        self.t += self.dt  # increment time
    
    def drift(self, pv):
        # compute drift term.
        pvspec = torch.fft.rfft2(pv)
        dpvspecdt = self.gettend(pvspec)
        return torch.fft.irfft2(dpvspecdt)

class SQGPrior(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std
        self.model = None  # lazy init

    def _init_model(self, pv):
        device = pv.device

        N = 64
        dt = 1200
        diff_efold = 86400.

        norder = 8  # order of hyperdiffusion
        dealias = True  # dealiased with 2/3 rule?

        # Ekman damping coefficient r=dek*N**2/f, dek = ekman depth = sqrt(2.*Av/f))
        # Av (turb viscosity) = 2.5 gives dek = sqrt(5/f) = 223
        # for ocean Av is 1-5, land 5-50 (Lin and Pierrehumbert, 1988)
        # corresponding to ekman depth of 141-316 m over ocean.
        # spindown time of a barotropic vortex is tau = H/(f*dek), 10 days for
        # H=10km, f=0.0001, dek=100m.
        dek = 0  # applied only at surface if symmetric=False
        nsq = 1.e-4
        f = 1.e-4
        g = 9.8
        theta0 = 300
        H = 10.e3  # lid height
        r = dek*nsq/f
        U = 30  # jet speed
        Lr = np.sqrt(nsq)*H/f  # Rossby radius
        L = 20.*Lr
        # thermal relaxation time scale
        tdiab = 10.*86400  # in seconds
        # (if False, asymmetric equilibrium jet with zero wind at sfc)
        symmetric = True
        # parameter used to scale PV to temperature units.
        scalefact = f*theta0/g
        precision = "single"  # single or double precision for FFTs (single is faster)

        # initialize qg model instance
        # pv = torch.from_numpy(pv).to(torch.float32)
        self.model = torchSQG(pv, nsq=nsq, f=f, U=U, H=H, r=r, tdiab=tdiab, dt=dt,
                    diff_order=norder, diff_efold=diff_efold,
                    dealias=dealias, symmetric=symmetric,
                    precision=precision, tstart=0, device=device)
        
    
    def forward(self, z, t):
        if self.model is None:
            self._init_model(z)
        drift = self.model.drift(z * self.std) / self.std
        drift = drift * 3 * 3600 # 20 min internal steps times 3 h model steps
        return drift
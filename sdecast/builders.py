from sdecast.matching import MatchingSDE_SQG
from sdecast.prior import PriorSDE, PriorObservation
from sdecast.posterior import PosteriorAffine

def sde_matching_SQG(nx, channels, sample_length, hidden_channels=32, vol_type="const", s_max=1, max_sigma=0.1, s_min=0.01,
                     interpolant_type="flexible", sigma_type="diag", random_dt=False, expressive_time_emb=True,
                     prior_obs_std=0.1, noise_dim=32, channel_mult_emb=1, context_size=2,
                     regularization_factor=None, cond_channels=0, spectra_path=None, drift_spectra_path=None,
                     divergence_target=None, h_hours=3.0, use_true_drift=False):
    p_observe = PriorObservation(noise_std=prior_obs_std)

    p_sde = PriorSDE(nx, channels, hidden_channels, vol_type=vol_type, max_sigma=max_sigma, cond_channels=cond_channels,
                     use_true_drift=use_true_drift, h_hours=h_hours)
    q_affine = PosteriorAffine(nx, channels, hidden_channels, s_max=s_max, s_min=s_min,
                               interpolant_type=interpolant_type, sigma_type=sigma_type, cond_dt=random_dt, sample_length=sample_length,
                               noise_dim=noise_dim, channel_mult_emb=channel_mult_emb, context_size=context_size,
                               expressive_time_emb=expressive_time_emb, cond_channels=cond_channels)

    sde_matching = MatchingSDE_SQG(p_sde, q_affine, p_observe, sample_length=sample_length, nx=nx,
                                   regularization_factor=regularization_factor,
                                   spectra_path=spectra_path, drift_spectra_path=drift_spectra_path,
                                   divergence_target=divergence_target, h_hours=h_hours)
    return sde_matching

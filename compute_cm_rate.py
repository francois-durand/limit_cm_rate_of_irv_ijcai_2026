import numpy as np
from svvamp import GeneratorProfileIc, RuleIRV


def irv_cm_rate(n_v, n_c, n_samples):
    """Compute the CM rate of IRV.

    Adapted from https://github.com/francois-durand/irv-cm-aamas-2025/blob/main/compute_cm_rate.py.

    Parameters
    ----------
    n_v: int
        Number of voters.
    n_c: int
        Number of candidates.
    n_samples: int
        Number of samples.

    Returns
    -------
    float
        CM rate of IRV.

    Examples
    --------
        >>> n_v = 51
        >>> n_c = 4
        >>> theta = .2
        >>> n_samples = 100
        >>> np.random.seed(42)
        >>> irv_cm_rate(n_v, n_c, theta, n_samples)
        0.08
    """
    n_profiles_cm = 0
    profile_generator = GeneratorProfileIc(n_v=n_v, n_c=n_c)
    for _ in range(n_samples):
        # The option 'precheck_heuristic=False' makes the code almost twice faster here, with the same results.
        if RuleIRV(cm_option='exact', precheck_heuristic=False)(profile_generator()).is_cm_:
            n_profiles_cm += 1
    return n_profiles_cm / n_samples

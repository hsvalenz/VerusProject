import numpy as np 

def polar_to_cartesian(r: float, theta: float, phi: float) -> np.ndarray:
    """Convert polar coordinates to cartesian coordinates.

    Parameters
    ----------
    r : float
        radial distance
    theta : float
        azimuthal angle
    phi : float
        polar angle

    Returns
    -------
    np.ndarray
        cartesian coordinates
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def rel_dist(pos1:np.ndarray, pos2: np.ndarray) -> float:
    """The relative distance between two positions.

    Parameters
    ----------
    pos1 : np.ndarray
        the first position
    pos2 : np.ndarray
        the second position

    Returns
    -------
    float
        the euclidean distance between the two positions
    """
    rel_d = np.linalg.norm(pos1 - pos2)
    return rel_d

def delta_v(control: np.ndarray, m: float = 12.0, step_size: float = 1.0):
    """The sum of thrust used divided by the deputy's mass during a step in the simulation

    Parameters
    ----------
    control : np.ndarray
        the control vector of the deputy's thrust outputs
    m : float, optional
        deputy mass, by default 12.0
    step_size : float, optional
        the amount of time between simulation steps

    Returns
    -------
    float
        the deputy's delta_v
    """
    dv = np.sum(np.abs(control)) / m * step_size
    return dv

def delta_v(v: np.ndarray, prev_v: np.ndarray) -> np.ndarray:
    """The change in velocity

    Parameters
    ----------
    v : np.ndarray
        the current velocity
    prev_v : np.ndarray
        the previous velocity

    Returns
    -------
    float
        the change in velocity
    """
    v_norm = np.linalg.norm(v)
    prev_v_norm = np.linalg.norm(prev_v)
    return v_norm - prev_v_norm

import jax
import jax.numpy as jnp
import openmdao.api as om

from gtoc13.odes import solar_sail_ode, modeq_ode
from gtoc13.constants import MU_ALTAIRA, R0


class SolarSailODEComp(om.JaxExplicitComponent):
    """
    OpenMDAO component that computes solar sail ODE derivatives.

    Options:
        num_nodes: Number of nodes in each trajectory
 
    Inputs:
        r: Position vectors, shape (num_nodes, 3) in km
        v: Velocity vectors, shape (num_nodes, 3) in km/s
        u_n: Sail normal unit vectors, shape (num_nodes, 3)

    Outputs:
        drdt: Position derivatives, shape (num_nodes, 3) in km/s
        dvdt: Velocity derivatives, shape (num_nodes, 3) in km/s^2
        cos_alpha: Cosine of cone angle, shape (num_nodes,)
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes in each trajectory')

    def setup(self):
        num_nodes = self.options['num_nodes']

        # Inputs - shape is (num_nodes, N, 3) where num_nodes is first
        self.add_input('r', shape=(num_nodes, 3), units='km',
                       desc='Position vectors')
        self.add_input('v', shape=(num_nodes, 3), units='km/s',
                       desc='Velocity vectors')
        self.add_input('u_n', shape=(num_nodes, 3), units='unitless',
                    desc='Sail normal unit vectors')

        # Outputs - shape is (num_nodes, N, 3)
        self.add_output('drdt', shape=(num_nodes, 3), units='km/s',
                        desc='Position derivatives')
        self.add_output('dvdt', shape=(num_nodes, 3), units='km/s**2',
                        desc='Velocity derivatives')
        self.add_output('a_grav', shape=(num_nodes, 3), units='km/s**2',
                        desc='Acceleration due to gravity')
        self.add_output('a_sail', shape=(num_nodes, 3), units='km/s**2',
                        desc='Acceleration due to solar sail')
        self.add_output('cos_alpha', shape=(num_nodes,), units='unitless',
                        desc='Cosine of cone angle')
        self.add_output('u_n_norm', shape=(num_nodes,), units='unitless')

        # Create vectorized version of solar_sail_ode
        self._solar_sail_ode_vec = jax.vmap(solar_sail_ode,
                                            in_axes=(0, 0, 0,
                                                     None, None))
        
    def compute_primal(self, r, v, u_n):
        """
        Compute ODE derivatives for solar sail trajectories.

        Parameters
        ----------
        r : jnp.ndarray
            Position vectors, shape (num_nodes, 3)
        v : jnp.ndarray
            Velocity vectors, shape (num_nodes, 3)
        u_n : jnp.ndarray
            Sail normal unit vectors, shape (num_nodes, 3)

        Returns
        -------
        drdt : jnp.ndarray
            Position derivatives, shape (num_nodes, 3)
        dvdt : jnp.ndarray
            Velocity derivatives, shape (num_nodes, 3)
        a_grav : jnp.ndarray
            Acceleration vector due to gravity (num_nodes, 3)
        a_sail : jnp.ndarray
            Acceleration vector due to solar sail (num_nodes, 3)
        cos_alpha : jnp.ndarray
            Cosine of cone angle, shape (num_nodes,)
        u_n_norm : jnp.ndarray
            The magnitude of the array normal. (num_nodes,)
        """
        drdt, dvdt, a_grav, a_sail, cos_alpha = self._solar_sail_ode_vec(
            r, v, u_n, MU_ALTAIRA, R0
        )

        u_n_norm = jnp.linalg.norm(u_n, axis=-1)

        return drdt, dvdt, a_grav, a_sail, cos_alpha, u_n_norm


class SolarSailRadialControlODEComp(om.JaxExplicitComponent):
    """
    OpenMDAO component that computes solar sail ODE derivatives.

    Options:
        num_nodes: Number of nodes in each trajectory
 
    Inputs:
        r: Position vectors, shape (num_nodes, 3) in km
        v: Velocity vectors, shape (num_nodes, 3) in km/s
        u_n: Sail normal unit vectors, shape (num_nodes, 3)

    Outputs:
        drdt: Position derivatives, shape (num_nodes, 3) in km/s
        dvdt: Velocity derivatives, shape (num_nodes, 3) in km/s^2
        cos_alpha: Cosine of cone angle, shape (num_nodes,)
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes in each trajectory')

    def setup(self):
        num_nodes = self.options['num_nodes']

        # Inputs - shape is (num_nodes, N, 3) where num_nodes is first
        self.add_input('r', shape=(num_nodes, 3), units='km',
                       desc='Position vectors')
        self.add_input('v', shape=(num_nodes, 3), units='km/s',
                       desc='Velocity vectors')

        # Outputs - shape is (num_nodes, N, 3)
        self.add_output('drdt', shape=(num_nodes, 3), units='km/s',
                        desc='Position derivatives')
        self.add_output('dvdt', shape=(num_nodes, 3), units='km/s**2',
                        desc='Velocity derivatives')
        self.add_output('a_grav', shape=(num_nodes, 3), units='km/s**2',
                        desc='Acceleration due to gravity')
        self.add_output('a_sail', shape=(num_nodes, 3), units='km/s**2',
                        desc='Acceleration due to solar sail')
        self.add_output('cos_alpha', shape=(num_nodes,), units='unitless',
                        desc='Cosine of cone angle')
        self.add_output('u_n', shape=(num_nodes, 3), units='unitless',
                    desc='Sail normal unit vectors')
        self.add_output('u_n_norm', shape=(num_nodes,), units='unitless')

        # Create vectorized version of solar_sail_ode
        self._solar_sail_ode_vec = jax.vmap(solar_sail_ode,
                                            in_axes=(0, 0, 0,
                                                     None, None))
        
    def compute_primal(self, r, v):
        """
        Compute ODE derivatives for solar sail trajectories.

        Parameters
        ----------
        r : jnp.ndarray
            Position vectors, shape (num_nodes, 3)
        v : jnp.ndarray
            Velocity vectors, shape (num_nodes, 3)

        Returns
        -------
        drdt : jnp.ndarray
            Position derivatives, shape (num_nodes, 3)
        dvdt : jnp.ndarray
            Velocity derivatives, shape (num_nodes, 3)
        a_grav : jnp.ndarray
            Acceleration vector due to gravity (num_nodes, 3)
        a_sail : jnp.ndarray
            Acceleration vector due to solar sail (num_nodes, 3)
        cos_alpha : jnp.ndarray
            Cosine of cone angle, shape (num_nodes,)
        u_n : jnp.ndarray
            Sail normal unit vectors, shape (num_nodes, 3)
        u_n_norm : jnp.ndarray
            The magnitude of the array normal. (num_nodes,)
        """
        r_mag = jnp.linalg.norm(r, axis=-1, keepdims=True)
        u_n = -r / r_mag

        drdt, dvdt, a_grav, a_sail, cos_alpha = self._solar_sail_ode_vec(
            r, v, u_n, MU_ALTAIRA, R0
        )

        u_n_norm = jnp.linalg.norm(u_n, axis=-1)

        return drdt, dvdt, a_grav, a_sail, cos_alpha, u_n, u_n_norm


class SolarSailVectorizedODEComp(om.JaxExplicitComponent):
    """
    OpenMDAO component that computes solar sail ODE derivatives.

    This component can propagate multiple trajectories simultaneously.

    Options:
        mu: Gravitational parameter (default: MU_ALTAIRA)
        r0: Reference distance for solar sail (default: R0)
        num_nodes: Number of nodes in each trajectory
        N: Number of trajectory instances to propagate simultaneously (default: 1)

    Inputs:
        r: Position vectors, shape (num_nodes, N, 3) in km
        v: Velocity vectors, shape (num_nodes, N, 3) in km/s
        dt_dtau: Time conversion factor, shape (num_nodes, N)
        u_n: Sail normal unit vectors, shape (num_nodes, N, 3)

    Outputs:
        drdt: Position derivatives, shape (num_nodes, N, 3) in km/s
        dvdt: Velocity derivatives, shape (num_nodes, N, 3) in km/s^2
        cos_alpha: Cosine of cone angle, shape (num_nodes, N)
    """

    def initialize(self):
        self.options.declare('mu', default=MU_ALTAIRA,
                             desc='Gravitational parameter')
        self.options.declare('r0', default=R0,
                             desc='Reference distance for solar sail in km')
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes in each trajectory')
        self.options.declare('N', default=1, types=int,
                             desc='Number of trajectory instances')

    def setup(self):
        num_nodes = self.options['num_nodes']
        N = self.options['N']

        # Inputs - shape is (num_nodes, N, 3) where num_nodes is first
        self.add_input('r', shape=(num_nodes, N, 3), units='km',
                       desc='Position vectors')
        self.add_input('v', shape=(num_nodes, N, 3), units='km/s',
                       desc='Velocity vectors')
        self.add_input('dt_dtau', shape=(num_nodes, N), units='s',
                       desc='Time conversion factor')
        self.add_input('u_n', shape=(num_nodes, N, 3), units='unitless',
                       desc='Sail normal unit vectors')

        # Outputs - shape is (num_nodes, N, 3)
        self.add_output('drdt', shape=(num_nodes, N, 3), units='km',
                        desc='Position derivatives')
        self.add_output('dvdt', shape=(num_nodes, N, 3), units='km/s',
                        desc='Velocity derivatives')
        self.add_output('a_grav', shape=(num_nodes, N, 3), units='km/s**2',
                        desc='Acceleration due to gravity')
        self.add_output('a_sail', shape=(num_nodes, N, 3), units='km/s**2',
                        desc='Acceleration due to solar sail')
        self.add_output('cos_alpha', shape=(num_nodes, N), units='unitless',
                        desc='Cosine of cone angle')
        self.add_output('u_n_norm', shape=(num_nodes, N), units='unitless')

        # Store options as instance attributes for use in compute_primal
        self._mu = self.options['mu']
        self._r0 = self.options['r0']

        # Create vectorized version of solar_sail_ode
        # Input shape: (num_nodes, N, 3) for r, v, u_n and (num_nodes, N) for dt_dtau
        # solar_sail_ode expects scalars/1D arrays: r(3,), v(3,), dt_dtau(), u_n(3,)
        # Strategy: vmap over axis 0 (num_nodes), then vmap over axis 0 again (N after first vmap removes num_nodes)
        self._solar_sail_ode_vec = jax.vmap(
            jax.vmap(solar_sail_ode, in_axes=(0, 0, 0, 0, None, None)),  # vmap over N (axis 0 after outer vmap)
            in_axes=(0, 0, 0, 0, None, None)  # vmap over num_nodes (axis 0)
        )

    def get_self_statics(self):
        """
        Declare static inputs to the compute_primal method.
        """
        return (self._mu, self._r0)

    def compute_primal(self, r, v, dt_dtau, u_n):
        """
        Compute ODE derivatives for solar sail trajectories.

        Parameters
        ----------
        r : jnp.ndarray
            Position vectors, shape (num_nodes, N, 3)
        v : jnp.ndarray
            Velocity vectors, shape (num_nodes, N, 3)
        dt_dtau : jnp.ndarray
            Time conversion factor, shape (num_nodes, N)
        u_n : jnp.ndarray
            Sail normal unit vectors, shape (num_nodes, N, 3)

        Returns
        -------
        drdt : jnp.ndarray
            Position derivatives, shape (num_nodes, N, 3)
        dvdt : jnp.ndarray
            Velocity derivatives, shape (num_nodes, N, 3)
        a_grav : jnp.ndarray
            Acceleration vector due to gravity (num_nodes, N, 3)
        a_sail : jnp.ndarray
            Acceleration vector due to solar sail (num_nodes, N, 3)
        cos_alpha : jnp.ndarray
            Cosine of cone angle, shape (num_nodes, N)
        u_n_norm : jnp.ndarray
            The magnitude of the array normal.
        """
        drdt, dvdt, a_grav, a_sail, cos_alpha = self._solar_sail_ode_vec(
            r, v, dt_dtau, u_n, self._mu, self._r0
        )

        u_n_norm = jnp.linalg.norm(u_n, axis=-1)

        return drdt, dvdt, a_grav, a_sail, cos_alpha, u_n_norm


class ModeqODEComp(om.JaxExplicitComponent):
    """
    OpenMDAO component that computes modified equinoctial element ODE derivatives.

    This component propagates trajectories using modified equinoctial elements (MEE),
    which avoid singularities for near-circular and near-equatorial orbits. The
    solar sail perturbations are included in the dynamics.

    Options
    -------
    num_nodes : int
        Number of nodes in the trajectory

    Inputs
    ------
    p : ndarray, shape (num_nodes,)
        Semi-latus rectum in km
    f : ndarray, shape (num_nodes,)
        Eccentricity vector x-component (dimensionless)
    g : ndarray, shape (num_nodes,)
        Eccentricity vector y-component (dimensionless)
    h : ndarray, shape (num_nodes,)
        Inclination vector x-component (dimensionless)
    k : ndarray, shape (num_nodes,)
        Inclination vector y-component (dimensionless)
    L : ndarray, shape (num_nodes,)
        True longitude in radians
    u_n : ndarray, shape (num_nodes, 3)
        Sail normal unit vectors (dimensionless)

    Outputs
    -------
    pdot : ndarray, shape (num_nodes,)
        Rate of change of semi-latus rectum (km/s)
    fdot : ndarray, shape (num_nodes,)
        Rate of change of f (1/s)
    gdot : ndarray, shape (num_nodes,)
        Rate of change of g (1/s)
    hdot : ndarray, shape (num_nodes,)
        Rate of change of h (1/s)
    kdot : ndarray, shape (num_nodes,)
        Rate of change of k (1/s)
    Ldot : ndarray, shape (num_nodes,)
        Rate of change of true longitude (rad/s)
    cos_alpha : ndarray, shape (num_nodes,)
        Cosine of sail cone angle (for constraint enforcement)
    r : ndarray, shape (num_nodes, 3)
        Cartesian position vectors in km
    v : ndarray, shape (num_nodes, 3)
        Cartesian velocity vectors in km/s
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes in the trajectory')

    def setup(self):
        num_nodes = self.options['num_nodes']

        # Inputs - Modified Equinoctial Elements
        self.add_input('p', shape=(num_nodes,), units='km',
                       desc='Semi-latus rectum')
        self.add_input('f', shape=(num_nodes,), units=None,
                       desc='Eccentricity vector x-component')
        self.add_input('g', shape=(num_nodes,), units=None,
                       desc='Eccentricity vector y-component')
        self.add_input('h', shape=(num_nodes,), units=None,
                       desc='Inclination vector x-component')
        self.add_input('k', shape=(num_nodes,), units=None,
                       desc='Inclination vector y-component')
        self.add_input('L', shape=(num_nodes,), units='rad',
                       desc='True longitude')
        self.add_input('u_n', shape=(num_nodes, 3), units='unitless',
                       desc='Sail normal unit vectors')

        # Outputs - MEE rates
        self.add_output('pdot', shape=(num_nodes,), units='km/s',
                        desc='Rate of change of semi-latus rectum')
        self.add_output('fdot', shape=(num_nodes,), units='1/s',
                        desc='Rate of change of f')
        self.add_output('gdot', shape=(num_nodes,), units='1/s',
                        desc='Rate of change of g')
        self.add_output('hdot', shape=(num_nodes,), units='1/s',
                        desc='Rate of change of h')
        self.add_output('kdot', shape=(num_nodes,), units='1/s',
                        desc='Rate of change of k')
        self.add_output('Ldot', shape=(num_nodes,), units='rad/s',
                        desc='Rate of change of true longitude')
        self.add_output('cos_alpha', shape=(num_nodes,), units='unitless',
                        desc='Cosine of sail cone angle')

        # Additional outputs - Cartesian states for visualization/analysis
        self.add_output('r', shape=(num_nodes, 3), units='km',
                        desc='Cartesian position vectors')
        self.add_output('v', shape=(num_nodes, 3), units='km/s',
                        desc='Cartesian velocity vectors')

        # Create vectorized version of modeq_ode
        # modeq_ode signature: (p, f, g, h, k, L, u_n, mu, r0) -> (pdot, fdot, gdot, hdot, kdot, Ldot, cos_alpha, r_vec, v_vec)
        # We need to vmap over the first axis (num_nodes) for p, f, g, h, k, L, u_n
        # mu and r0 are broadcast (scalar constants)
        self._modeq_ode_vec = jax.vmap(modeq_ode,
                                       in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))

    def compute_primal(self, p, f, g, h, k, L, u_n):
        """
        Compute modified equinoctial element ODE derivatives.

        Parameters
        ----------
        p : jnp.ndarray, shape (num_nodes,)
            Semi-latus rectum in km
        f : jnp.ndarray, shape (num_nodes,)
            Eccentricity vector x-component
        g : jnp.ndarray, shape (num_nodes,)
            Eccentricity vector y-component
        h : jnp.ndarray, shape (num_nodes,)
            Inclination vector x-component
        k : jnp.ndarray, shape (num_nodes,)
            Inclination vector y-component
        L : jnp.ndarray, shape (num_nodes,)
            True longitude in radians
        u_n : jnp.ndarray, shape (num_nodes, 3)
            Sail normal unit vectors

        Returns
        -------
        pdot : jnp.ndarray, shape (num_nodes,)
            Rate of change of semi-latus rectum
        fdot : jnp.ndarray, shape (num_nodes,)
            Rate of change of f
        gdot : jnp.ndarray, shape (num_nodes,)
            Rate of change of g
        hdot : jnp.ndarray, shape (num_nodes,)
            Rate of change of h
        kdot : jnp.ndarray, shape (num_nodes,)
            Rate of change of k
        Ldot : jnp.ndarray, shape (num_nodes,)
            Rate of change of true longitude
        cos_alpha : jnp.ndarray, shape (num_nodes,)
            Cosine of sail cone angle
        r : jnp.ndarray, shape (num_nodes, 3)
            Cartesian position vectors
        v : jnp.ndarray, shape (num_nodes, 3)
            Cartesian velocity vectors
        """
        pdot, fdot, gdot, hdot, kdot, Ldot, cos_alpha, r, v = self._modeq_ode_vec(
            p, f, g, h, k, L, u_n, MU_ALTAIRA, R0
        )

        return pdot, fdot, gdot, hdot, kdot, Ldot, cos_alpha, r, v


class ModeqRadialControlODEComp(om.JaxExplicitComponent):
    """
    OpenMDAO component that computes modified equinoctial element ODE derivatives.

    This component propagates trajectories using modified equinoctial elements (MEE),
    which avoid singularities for near-circular and near-equatorial orbits. The
    solar sail perturbations are included in the dynamics.

    This component assumes that the sail normal vector is computed internally.

    Options
    -------
    num_nodes : int
        Number of nodes in the trajectory

    Inputs
    ------
    p : ndarray, shape (num_nodes,)
        Semi-latus rectum in km
    f : ndarray, shape (num_nodes,)
        Eccentricity vector x-component (dimensionless)
    g : ndarray, shape (num_nodes,)
        Eccentricity vector y-component (dimensionless)
    h : ndarray, shape (num_nodes,)
        Inclination vector x-component (dimensionless)
    k : ndarray, shape (num_nodes,)
        Inclination vector y-component (dimensionless)
    L : ndarray, shape (num_nodes,)
        True longitude in radians

    Outputs
    -------
    pdot : ndarray, shape (num_nodes,)
        Rate of change of semi-latus rectum (km/s)
    fdot : ndarray, shape (num_nodes,)
        Rate of change of f (1/s)
    gdot : ndarray, shape (num_nodes,)
        Rate of change of g (1/s)
    hdot : ndarray, shape (num_nodes,)
        Rate of change of h (1/s)
    kdot : ndarray, shape (num_nodes,)
        Rate of change of k (1/s)
    Ldot : ndarray, shape (num_nodes,)
        Rate of change of true longitude (rad/s)
    cos_alpha : ndarray, shape (num_nodes,)
        Cosine of sail cone angle (for constraint enforcement)
    r : ndarray, shape (num_nodes, 3)
        Cartesian position vectors in km
    v : ndarray, shape (num_nodes, 3)
        Cartesian velocity vectors in km/s
    u_n : ndarray, shape (num_nodes, 3)
        Sail normal unit vectors (dimensionless)
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes in the trajectory')

    def setup(self):
        num_nodes = self.options['num_nodes']

        # Inputs - Modified Equinoctial Elements
        self.add_input('p', shape=(num_nodes,), units='km',
                       desc='Semi-latus rectum')
        self.add_input('f', shape=(num_nodes,), units=None,
                       desc='Eccentricity vector x-component')
        self.add_input('g', shape=(num_nodes,), units=None,
                       desc='Eccentricity vector y-component')
        self.add_input('h', shape=(num_nodes,), units=None,
                       desc='Inclination vector x-component')
        self.add_input('k', shape=(num_nodes,), units=None,
                       desc='Inclination vector y-component')
        self.add_input('L', shape=(num_nodes,), units='rad',
                       desc='True longitude')

        # Outputs - MEE rates
        self.add_output('pdot', shape=(num_nodes,), units='km/s',
                        desc='Rate of change of semi-latus rectum')
        self.add_output('fdot', shape=(num_nodes,), units='1/s',
                        desc='Rate of change of f')
        self.add_output('gdot', shape=(num_nodes,), units='1/s',
                        desc='Rate of change of g')
        self.add_output('hdot', shape=(num_nodes,), units='1/s',
                        desc='Rate of change of h')
        self.add_output('kdot', shape=(num_nodes,), units='1/s',
                        desc='Rate of change of k')
        self.add_output('Ldot', shape=(num_nodes,), units='rad/s',
                        desc='Rate of change of true longitude')
        self.add_output('cos_alpha', shape=(num_nodes,), units='unitless',
                        desc='Cosine of sail cone angle')

        # Additional outputs - Cartesian states for visualization/analysis
        self.add_output('r', shape=(num_nodes, 3), units='km',
                        desc='Cartesian position vectors')
        self.add_output('v', shape=(num_nodes, 3), units='km/s',
                        desc='Cartesian velocity vectors')
        
        self.add_output('u_n', shape=(num_nodes, 3), units='unitless',
                       desc='Sail normal unit vectors')

        # Create vectorized version of modeq_ode
        # modeq_ode signature: (p, f, g, h, k, L, u_n, mu, r0) -> (pdot, fdot, gdot, hdot, kdot, Ldot, cos_alpha, r_vec, v_vec)
        # We need to vmap over the first axis (num_nodes) for p, f, g, h, k, L, u_n
        # mu and r0 are broadcast (scalar constants)
        self._modeq_ode_vec = jax.vmap(modeq_ode,
                                       in_axes=(0, 0, 0, 0, 0, 0, 0, None, None))

    def compute_primal(self, p, f, g, h, k, L, u_n):
        """
        Compute modified equinoctial element ODE derivatives.

        Parameters
        ----------
        p : jnp.ndarray, shape (num_nodes,)
            Semi-latus rectum in km
        f : jnp.ndarray, shape (num_nodes,)
            Eccentricity vector x-component
        g : jnp.ndarray, shape (num_nodes,)
            Eccentricity vector y-component
        h : jnp.ndarray, shape (num_nodes,)
            Inclination vector x-component
        k : jnp.ndarray, shape (num_nodes,)
            Inclination vector y-component
        L : jnp.ndarray, shape (num_nodes,)
            True longitude in radians
        u_n : jnp.ndarray, shape (num_nodes, 3)
            Sail normal unit vectors

        Returns
        -------
        pdot : jnp.ndarray, shape (num_nodes,)
            Rate of change of semi-latus rectum
        fdot : jnp.ndarray, shape (num_nodes,)
            Rate of change of f
        gdot : jnp.ndarray, shape (num_nodes,)
            Rate of change of g
        hdot : jnp.ndarray, shape (num_nodes,)
            Rate of change of h
        kdot : jnp.ndarray, shape (num_nodes,)
            Rate of change of k
        Ldot : jnp.ndarray, shape (num_nodes,)
            Rate of change of true longitude
        cos_alpha : jnp.ndarray, shape (num_nodes,)
            Cosine of sail cone angle
        r : jnp.ndarray, shape (num_nodes, 3)
            Cartesian position vectors
        v : jnp.ndarray, shape (num_nodes, 3)
            Cartesian velocity vectors
        """
        pdot, fdot, gdot, hdot, kdot, Ldot, cos_alpha, r, v = self._modeq_ode_vec(
            p, f, g, h, k, L, u_n, MU_ALTAIRA, R0
        )

        return pdot, fdot, gdot, hdot, kdot, Ldot, cos_alpha, r, v

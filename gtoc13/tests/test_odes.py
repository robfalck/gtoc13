import unittest

from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController
import jax.numpy as jnp

from gtoc13 import bodies_data, ballistic_ode, solar_sail_ode, MU_ALTAIRA


class TestODES(unittest.TestCase):

    def test_ballistic_ode(self):

        term = ODETerm(ballistic_ode)
        solver = Dopri5()
        body_id = 3

        t0 = 0.0
        tf = bodies_data[body_id].get_period(units='TU')
        save_at = SaveAt(ts=jnp.linspace(t0, tf, 100))

        # Get initial state and flatten it to [x, y, z, vx, vy, vz]
        state0 = bodies_data[body_id].get_state(0.0, time_units='TU', distance_units='DU')
        y0 = jnp.concatenate([state0.r, state0.v])
        mu_canon = 1.0

        solution = diffeqsolve(term, solver, args=(mu_canon,),
                               t0=t0, t1=tf, dt0=0.01*tf, y0=y0,
                               saveat=save_at)

        print(f"Solution shape: {solution.ys.shape}")
        print(f"\nInitial state (DU, DU/TU):")
        print(f"  r0 = {solution.ys[0, :3]}")
        print(f"  v0 = {solution.ys[0, 3:]}")
        print(f"\nFinal state (DU, DU/TU):")
        print(f"  rf = {solution.ys[-1, :3]}")
        print(f"  vf = {solution.ys[-1, 3:]}")
        print(f"\nDifference after one period:")
        print(f"  Δr = {jnp.linalg.norm(solution.ys[-1, :3] - solution.ys[0, :3])} DU")
        print(f"  Δv = {jnp.linalg.norm(solution.ys[-1, 3:] - solution.ys[0, 3:])} DU/TU")

        # After one orbital period, state should return to initial conditions
        self.assertLess(jnp.linalg.norm(solution.ys[-1, :3] - solution.ys[0, :3]), 1e-4,
                        "Position should return to initial value after one period")
        self.assertLess(jnp.linalg.norm(solution.ys[-1, 3:] - solution.ys[0, 3:]), 1e-5,
                        "Velocity should return to initial value after one period")


if __name__ == '__main__':
    unittest.main()

# def f(t, y, args):
#     return -y

# term = ODETerm(f)
# y0 = jnp.array([2., 3.])

# # Save at 50 equally spaced timesteps from 0 to 1
# ts = jnp.linspace(0, 1, 50)
# saveat = SaveAt(ts=ts)

# solution = diffeqsolve(term, solver, t0=0, t1=1,
#                        dt0=0.1, y0=y0, saveat=saveat)

# print(f"Solution shape: {solution.ys.shape}")
# print(f"First few timesteps: {ts[:5]}")
# print(f"First few solution values:\n{solution.ys[:5]}")
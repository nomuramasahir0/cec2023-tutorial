"""
The code is adapted from https://github.com/CyberAgentAILab/cmaes/blob/main/tools/cmaes_visualizer.py.
Example:
    python3 xnes_viewer.py --function quadratic --frames 25 --interval 800
    python3 xnes_viewer.py --function rosenbrock --frames 25 --interval 800
    python3 xnes_viewer.py --function himmelblau --frames 25 --interval 800
    python3 xnes_viewer.py --function six-hump-camel --frames 25 --interval 800

"""
import argparse

import numpy as np
from scipy import stats

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import rcParams
import math

from typing import cast
from typing import Optional


_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


class XNES:
    """xNES stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::

           import numpy as np
           from cmaes import XNES

           def quadratic(x1, x2):
               return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

           optimizer = XNES(mean=np.zeros(2), sigma=1.3)

           for generation in range(50):
               solutions = []
               for _ in range(optimizer.population_size):
                   # Ask a parameter
                   x = optimizer.ask()
                   value = quadratic(x[0], x[1])
                   solutions.append((x, value))
                   print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")

               # Tell evaluation values.
               optimizer.tell(solutions)

    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter (optional).

        n_max_resampling:
            A maximum number of resampling parameters (default: 100).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.

        seed:
            A seed number (optional).

        population_size:
            A population size (optional).

    """

    # Paper: https://dl.acm.org/doi/10.1145/1830483.1830557

    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        bounds: Optional[np.ndarray] = None,
        n_max_resampling: int = 100,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
    ):
        assert sigma > 0, "sigma must be non-zero positive value"

        assert np.all(
            np.abs(mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

        n_dim = len(mean)
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))
        assert population_size > 0, "popsize must be non-zero positive value."

        w_hat = np.log(population_size / 2 + 1) - np.log(
            np.arange(1, population_size + 1)
        )
        w_hat[np.where(w_hat < 0)] = 0
        weights = w_hat / sum(w_hat) - (1.0 / population_size)

        self._n_dim = n_dim
        self._popsize = population_size

        # weights
        self._weights = weights

        # learning rate
        self._eta_mean = 1.0
        self._eta_sigma = (3 / 5) * (3 + math.log(n_dim)) / (n_dim * math.sqrt(n_dim))
        self._eta_B = self._eta_sigma

        # distribution parameter
        self._mean = mean.copy()
        self._sigma = sigma
        self._B = np.eye(n_dim)

        # bounds contains low and high of each parameter.
        assert bounds is None or _is_valid_bounds(bounds, mean), "invalid bounds"
        self._bounds = bounds
        self._n_max_resampling = n_max_resampling

        self._g = 0
        self._rng = np.random.RandomState(seed)

        # Termination criteria
        self._tolx = 1e-12 * sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * n_dim / population_size)
        self._funhist_values = np.empty(self._funhist_term * 2)

    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._n_dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)

    def set_bounds(self, bounds: Optional[np.ndarray]) -> None:
        """Update boundary constraints"""
        assert bounds is None or _is_valid_bounds(bounds, self._mean), "invalid bounds"
        self._bounds = bounds

    def ask(self) -> np.ndarray:
        """Sample a parameter"""
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_feasible(x):
                return x
        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        return x

    def _sample_solution(self) -> np.ndarray:
        z = self._rng.randn(self._n_dim)  # ~ N(0, I)
        x = self._mean + self._sigma * self._B.dot(z)  # ~ N(m, σ^2 B B^T)
        return x

    def _is_feasible(self, param: np.ndarray) -> bool:
        if self._bounds is None:
            return True
        return cast(
            bool,
            np.all(param >= self._bounds[:, 0]) and np.all(param <= self._bounds[:, 1]),
        )  # Cast bool_ to bool.

    def _repair_infeasible_params(self, param: np.ndarray) -> np.ndarray:
        if self._bounds is None:
            return param

        # clip with lower and upper bound.
        param = np.where(param < self._bounds[:, 0], self._bounds[:, 0], param)
        param = np.where(param > self._bounds[:, 1], self._bounds[:, 1], param)
        return param

    def tell(self, solutions: list[tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""

        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

        self._g += 1
        solutions.sort(key=lambda s: s[1])

        # Stores 'best' and 'worst' values of the
        # last 'self._funhist_term' generations.
        funhist_idx = 2 * (self.generation % self._funhist_term)
        self._funhist_values[funhist_idx] = solutions[0][1]
        self._funhist_values[funhist_idx + 1] = solutions[-1][1]

        z_k = np.array(
            [
                np.linalg.inv(self._sigma * self._B).dot(s[0] - self._mean)
                for s in solutions
            ]
        )

        # natural gradient estimation in local coordinate
        G_delta = np.sum(
            [self._weights[i] * z_k[i, :] for i in range(self.population_size)], axis=0
        )
        G_M = np.sum(
            [
                self._weights[i]
                * (np.outer(z_k[i, :], z_k[i, :]) - np.eye(self._n_dim))
                for i in range(self.population_size)
            ],
            axis=0,
        )
        G_sigma = G_M.trace() / self._n_dim
        G_B = G_M - G_sigma * np.eye(self._n_dim)

        # parameter update
        self._mean += self._eta_mean * self._sigma * np.dot(self._B, G_delta)
        self._sigma *= math.exp((self._eta_sigma / 2.0) * G_sigma)
        self._B = self._B.dot(_expm((self._eta_B / 2.0) * G_B))

    def should_stop(self) -> bool:
        A = self._B.dot(self._B.T)
        A = (A + A.T) / 2
        E2, V = np.linalg.eigh(A)
        E = np.sqrt(np.where(E2 < 0, _EPS, E2))
        diagA = np.diag(A)

        # Stop if the range of function values of the recent generation is below tolfun.
        if (
            self.generation > self._funhist_term
            and np.max(self._funhist_values) - np.min(self._funhist_values)
            < self._tolfun
        ):
            return True

        # Stop if detecting divergent behavior.
        if self._sigma * np.max(E) > self._tolxup:
            return True

        # No effect coordinates: stop if adding 0.2-standard deviations
        # in any single coordinate does not change m.
        if np.any(self._mean == self._mean + (0.2 * self._sigma * np.sqrt(diagA))):
            return True

        # No effect axis: stop if adding 0.1-standard deviation vector in
        # any principal axis direction of C does not change m. "pycma" check
        # axis one by one at each generation.
        i = self.generation % self.dim
        if np.all(self._mean == self._mean + (0.1 * self._sigma * E[i] * V[:, i])):
            return True

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        condition_cov = np.max(E) / np.min(E)
        if condition_cov > self._tolconditioncov:
            return True

        return False


def _is_valid_bounds(bounds: Optional[np.ndarray], mean: np.ndarray) -> bool:
    if bounds is None:
        return True
    if (mean.size, 2) != bounds.shape:
        return False
    if not np.all(bounds[:, 0] <= mean):
        return False
    if not np.all(mean <= bounds[:, 1]):
        return False
    return True


def _expm(mat: np.ndarray) -> np.ndarray:
    D, U = np.linalg.eigh(mat)
    expD = np.exp(D)
    return U @ np.diag(expD) @ U.T


parser = argparse.ArgumentParser()
parser.add_argument(
    "--function",
    choices=["quadratic", "himmelblau", "rosenbrock", "six-hump-camel"],
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
)
parser.add_argument(
    "--frames",
    type=int,
    default=100,
)
parser.add_argument(
    "--interval",
    type=int,
    default=20,
)
args = parser.parse_args()

rcParams["figure.figsize"] = 10, 5
fig, (ax1, ax2) = plt.subplots(1, 2)

color_dict = {
    "red": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "blue": ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    "yellow": ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
}
bw = LinearSegmentedColormap("BlueWhile", color_dict)


def himmelbleu(x1, x2):
    return (x1**2 + x2 - 11.0) ** 2 + (x1 + x2**2 - 7.0) ** 2


def himmelbleu_contour(x1, x2):
    return np.log(himmelbleu(x1, x2) + 1)


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


def quadratic_contour(x1, x2):
    return np.log(quadratic(x1, x2) + 1)


def rosenbrock(x1, x2):
    return 100 * (x2 - x1**2) ** 2 + (x1 - 1) ** 2


def rosenbrock_contour(x1, x2):
    return np.log(rosenbrock(x1, x2) + 1)


def six_hump_camel(x1, x2):
    return (
        (4 - 2.1 * (x1**2) + (x1**4) / 3) * (x1**2)
        + x1 * x2
        + (-4 + 4 * x2**2) * (x2**2)
    )


def six_hump_camel_contour(x1, x2):
    return np.log(six_hump_camel(x1, x2) + 1.0316)


function_name = ""
if args.function == "quadratic":
    function_name = "Quadratic function"
    objective = quadratic
    contour_function = quadratic_contour
    global_minimums = [
        (3.0, -2.0),
    ]
    # input domain
    x1_lower_bound, x1_upper_bound = -4, 4
    x2_lower_bound, x2_upper_bound = -4, 4
elif args.function == "himmelblau":
    function_name = "Himmelblau function"
    objective = himmelbleu
    contour_function = himmelbleu_contour
    global_minimums = [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ]
    # input domain
    x1_lower_bound, x1_upper_bound = -4, 4
    x2_lower_bound, x2_upper_bound = -4, 4
elif args.function == "rosenbrock":
    # https://www.sfu.ca/~ssurjano/rosen.html
    function_name = "Rosenbrock function"
    objective = rosenbrock
    contour_function = rosenbrock_contour
    global_minimums = [
        (1, 1),
    ]
    # input domain
    # x1_lower_bound, x1_upper_bound = -5, 10
    # x2_lower_bound, x2_upper_bound = -5, 10
    x1_lower_bound, x1_upper_bound = -1, 2
    x2_lower_bound, x2_upper_bound = -1, 2
elif args.function == "six-hump-camel":
    # https://www.sfu.ca/~ssurjano/camel6.html
    function_name = "Six-hump camel function"
    objective = six_hump_camel
    contour_function = six_hump_camel_contour
    global_minimums = [
        (0.0898, -0.7126),
        (-0.0898, 0.7126),
    ]
    # input domain
    x1_lower_bound, x1_upper_bound = -3, 3
    x2_lower_bound, x2_upper_bound = -2, 2
else:
    raise ValueError("invalid function type")


seed = args.seed
bounds = np.array([[x1_lower_bound, x1_upper_bound], [x2_lower_bound, x2_upper_bound]])
sigma = (x1_upper_bound - x2_lower_bound) / 5
# optimizer = CMA(mean=np.zeros(2), sigma=sigma, bounds=bounds, seed=seed)
optimizer = XNES(mean=np.zeros(2), sigma=sigma, bounds=bounds, seed=seed)
solutions = []
trial_number = 0
rng = np.random.RandomState(seed)


def init():
    ax1.set_xlim(x1_lower_bound, x1_upper_bound)
    ax1.set_ylim(x2_lower_bound, x2_upper_bound)
    ax2.set_xlim(x1_lower_bound, x1_upper_bound)
    ax2.set_ylim(x2_lower_bound, x2_upper_bound)

    # Plot 4 local minimum value
    for m in global_minimums:
        ax1.plot(m[0], m[1], "y*", ms=10)
        ax2.plot(m[0], m[1], "y*", ms=10)

    # Plot contour of himmelbleu function
    x1 = np.arange(x1_lower_bound, x1_upper_bound, 0.01)
    x2 = np.arange(x2_lower_bound, x2_upper_bound, 0.01)
    x1, x2 = np.meshgrid(x1, x2)

    ax1.contour(x1, x2, contour_function(x1, x2), 30, cmap=bw)




def update(frame):
    global solutions, optimizer, trial_number
    if len(solutions) == optimizer.population_size:
        optimizer.tell(solutions)
        solutions = []

    n_sample = optimizer.population_size
    for i in range(n_sample):
        x = optimizer.ask()
        evaluation = objective(x[0], x[1])

        # Plot sample points
        ax1.plot(x[0], x[1], "o", c="r", label="2d", alpha=0.5)

        solution = (
            x,
            evaluation,
        )
        solutions.append(solution)
    trial_number += n_sample

    # Update title
    fig.suptitle(f"xNES {function_name} trial={trial_number}")

    # Plot multivariate gaussian distribution
    x, y = np.mgrid[
        x1_lower_bound:x1_upper_bound:0.01, x2_lower_bound:x2_upper_bound:0.01
    ]
    cov = optimizer._sigma**2 * optimizer._B.dot(optimizer._B.T)
    cov = (cov + cov.T) / 2.
    rv = stats.multivariate_normal(optimizer._mean, cov)
    pos = np.dstack((x, y))
    ax2.contourf(x, y, rv.pdf(pos))

    if frame % 50 == 0:
        print(f"Processing frame {frame}")


def main():
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=args.frames,
        init_func=init,
        blit=False,
        interval=args.interval,
    )
    ani.save(f"./tmp/{args.function}.mp4")


if __name__ == "__main__":
    main()

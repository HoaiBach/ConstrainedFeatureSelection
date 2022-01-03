import numpy as np
from Utility import Helpers
from Problem import FeatureSelection

"""
Implement the modified CSO: "A modified competitive swarm optimizer for large scale optimization problems"
Idea is to divide th pop into 3 parts, each competition contains 3 individuals: 1 winner and 2 losers
The losers learn from the winner.
"""


class MCSO:

    def __init__(self, problem: FeatureSelection, pop_size=201, max_evaluations=10000,
                 phi1=0.1, phi2=0.2, topology='ring', max_pos=1.0, min_pos=0.0):
        self.problem = problem
        # ensure popsize is multiple of 3
        if pop_size % 3 == 1:
            pop_size = pop_size + 2
        elif pop_size % 3 == 2:
            pop_size = pop_size + 1
        self.pop_size = pop_size
        self.max_evaluations = max_evaluations
        self.phi1 = phi1
        self.phi2 = phi2
        self.topology = topology  # note topology can be 'global' or 'ring'
        self.max_pos = max_pos
        self.min_pos = min_pos
        self.no_evaluations = 0
        # assert minimisation
        assert self.problem.minimized

    def evaluate_pop(self, pop):
        """
        Evaluate the pop of individuals, return the corresponding fitness
        :param pop:
        :return:
        """
        fit_errors = [self.problem.fitness(ind) for ind in pop]
        self.no_evaluations += len(pop)
        return fit_errors

    def evaluate_pop_loocv(self, pop):
        """
        Evaluate the pop of individuals, return the corresponding fitness
        :param pop:
        :return:
        """
        fit_errors = [self.problem.fitness_loocv(ind) for ind in pop]
        self.no_evaluations += len(pop)
        return fit_errors

    def evolve(self):
        # Init the population
        dim = self.problem.dim
        pop_positions = self.problem.init_pop(self.pop_size)
        pop_vels = np.zeros((self.pop_size, dim))
        pop_err = np.zeros(self.pop_size)
        pop_fit = np.zeros(self.pop_size)

        # Evaluate the initialised pop
        fit_errors = self.evaluate_pop_loocv(pop_positions)
        for idx, (fit, err) in enumerate(fit_errors):
            pop_err[idx] = err
            pop_fit[idx] = fit

        # Prepare the neighborhood topology
        if self.topology == 'global':
            neighbors = np.array([np.arange(self.pop_size)] * self.pop_size)
        elif self.topology == 'ring':
            neighbors = [[idx - 1, idx, idx + 1] for idx in np.arange(start=1, stop=self.pop_size - 1)]
            # for first ind
            first_ind = [[self.pop_size - 1, 0, 1]]
            first_ind.extend(neighbors)
            neighbors = first_ind
            # for last ind
            neighbors.append([self.pop_size - 2, self.pop_size - 1, 0])
            neighbors = np.array(neighbors)
        else:
            raise Exception('Topology %s is not implemented.' % self.topology)
        assert len(neighbors) == self.pop_size

        # Start the evolution
        best_idx = np.argsort(pop_fit)[0]
        best_sol = np.copy(pop_positions[best_idx])
        best_fit = pop_fit[best_idx]
        best_err = pop_err[best_idx]

        evolutionary_process = '***************************************\n'
        best_subset = self.problem.position_2_solution(best_sol)[0]
        fRate = len(best_subset) / self.problem.dim
        evolutionary_process += 'At %d: %.4f, %.4f, %.2f ~%s\n' % (self.no_evaluations, best_fit, best_err, fRate,
                                                                   ', '.join(['%d' % ele for ele in best_subset]))
        evolutionary_process += '\t\t 0 infeasible solutions\n'
        pop_diversity = Helpers.population_stat(pop_positions)
        evolutionary_process += '\t\t Pop diversity: %f\n' % pop_diversity
        evolutionary_process += '\t\t Pop fit: %s\n' % ', '.join(['%.4f' % fit for fit in pop_fit])

        while self.no_evaluations < self.max_evaluations:
            indices_pool = np.arange(self.pop_size)
            np.random.shuffle(indices_pool)

            next_pop = np.zeros((self.pop_size, self.problem.dim))
            next_vels = np.zeros((self.pop_size, self.problem.dim))
            next_pop_fit = np.array([None] * self.pop_size)
            next_pop_err = np.array([None] * self.pop_size)

            to_evaluate = np.array([], dtype=int)

            for idx1, idx2, idx3 in zip(indices_pool[0::3], indices_pool[1::3], indices_pool[2::3]):
                # contrasting 3 random individuals
                # fitness(winner) <= fitness(loser1) <= fitness(loser2)
                triple_indices = np.array([idx1, idx2, idx3])
                triple_fitness = np.array([pop_fit[idx1], pop_fit[idx2], pop_fit[idx3]])
                triple_indices_sorted = triple_indices[np.argsort(triple_fitness)]
                winner_idx = triple_indices_sorted[0]
                loser1_idx = triple_indices_sorted[1]
                loser2_idx = triple_indices_sorted[2]

                # add winner to next pop
                next_pop[winner_idx] = np.copy(pop_positions[winner_idx])
                next_vels[winner_idx] = np.copy(pop_vels[winner_idx])
                next_pop_fit[winner_idx] = pop_fit[winner_idx]
                next_pop_err[winner_idx] = pop_err[winner_idx]

                # update both losers using different phi values
                for loser_idx, phi in zip([loser1_idx, loser2_idx], [self.phi1, self.phi2]):
                    r1 = np.random.rand(dim)
                    r2 = np.random.rand(dim)
                    r3 = np.random.rand(dim)

                    x_ave = np.average(pop_positions[neighbors[loser_idx]], axis=0)
                    vel = r1 * pop_vels[loser_idx] + r2 * (pop_positions[winner_idx] - pop_positions[loser_idx]) \
                          + phi * r3 * (x_ave - pop_positions[loser_idx])
                    new_pos = pop_positions[loser_idx] + vel
                    new_pos[new_pos > self.max_pos] = self.max_pos
                    new_pos[new_pos < self.min_pos] = self.min_pos

                    # add new position of loser to the next pop
                    next_pop[loser_idx] = new_pos
                    next_vels[loser_idx] = vel
                    to_evaluate = np.append(to_evaluate, loser_idx)

            assert len(next_pop) == len(pop_positions)

            eval_fit_errs = self.evaluate_pop_loocv(next_pop[to_evaluate])
            for idx, (fit, err) in zip(to_evaluate, eval_fit_errs):
                next_pop_fit[idx] = fit
                next_pop_err[idx] = err

            pop_positions = next_pop
            pop_vels = next_vels
            pop_fit = next_pop_fit
            pop_err = next_pop_err

            # update the best ind if necessary
            for fit, err, sol in zip(pop_fit, pop_err, pop_positions):
                if self.problem.is_better(fit, best_fit):
                    best_fit = fit
                    best_sol = np.copy(sol)
                    best_err = err

            # Prepare the evolutionary information
            best_subset = self.problem.position_2_solution(best_sol)[0]
            fRate = len(best_subset) / self.problem.dim
            evolutionary_process += 'At %d: %.4f, %.4f, %.2f ~%s\n' % (self.no_evaluations, best_fit, best_err, fRate,
                                                                       ', '.join(['%d' % ele for ele in best_subset]))
            pop_diversity = Helpers.population_stat(pop_positions)
            evolutionary_process += '\t\t Pop diversity: %f\n' % pop_diversity
            evolutionary_process += '\t\t Pop fit: %s\n' % ', '.join(['%.4f' % fit for fit in pop_fit])

        return best_sol, evolutionary_process

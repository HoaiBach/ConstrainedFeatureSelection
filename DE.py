"""
Implement JADE: Adaptive differential evolution with optional external archive
"""
from Problem import MIRFFeatureSelection
import numpy as np
from Utility import Helpers


class DE:

    def __init__(self, problem: MIRFFeatureSelection, pop_size=100, max_evaluations=10000,
                 min_pos=0.0, max_pos=1.0, F=0.8, CR=0.7):
        self.problem = problem
        self.pop_size = pop_size
        self.max_evaluations = max_evaluations
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.F = F
        self.CR = CR
        self.no_evaluations = 0

    def evaluate_pop(self, pop):
        """
        Evaluate the pop of individuals, return the corresponding fitness
        :param pop:
        :return:
        """
        fit_errors = [self.problem.fitness(ind) for ind in pop]
        self.no_evaluations += len(pop)
        return fit_errors

    def evaluate(self, sol):
        """
        Evaluate an individual sol
        :param sol:
        :return:
        """
        fit_error = self.problem.fitness(sol)
        self.no_evaluations += 1
        return fit_error

    def evolve(self):
        # initialise the population
        dim = self.problem.dim
        pop_positions = self.problem.init_pop(self.pop_size)
        pop_err = np.zeros(self.pop_size)
        pop_fit = np.zeros(self.pop_size)

        # Evaluate the initialised pop
        fit_errors = self.evaluate_pop(pop_positions)
        for idx, (fit, err) in enumerate(fit_errors):
            pop_err[idx] = err
            pop_fit[idx] = fit

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
            for idx, ind in enumerate(pop_positions):
                while True:
                    idx_r1 = np.random.randint(self.pop_size)
                    if idx_r1 != idx:
                        break
                x_r1 = pop_positions[idx_r1]

                while True:
                    idx_r2 = np.random.randint(self.pop_size)
                    if idx_r2 != idx and idx_r2 != idx_r1:
                        break
                x_r2 = pop_positions[idx_r2]

                while True:
                    idx_r3 = np.random.randint(self.pop_size)
                    if idx_r3 != idx and idx_r3 != idx_r1 and idx_r3 != idx_r2:
                        break
                x_r3 = pop_positions[idx_r3]

                # mutation
                mutant = x_r1 + self.F*(x_r3-x_r2)
                mutant[mutant < self.min_pos] = self.min_pos
                mutant[mutant > self.max_pos] = self.max_pos

                # crossover
                cr_rnd = np.random.rand(dim)
                j_rnd = np.random.randint(dim)
                mask = cr_rnd < self.CR
                mask[j_rnd] = True
                trial = np.copy(ind)
                trial[mask] = mutant[mask]

                # selection
                trial_fit, trial_err = self.evaluate(trial)
                if not self.problem.is_better(pop_fit[idx], trial_fit):
                    pop_positions[idx] = np.copy(trial)
                    pop_fit[idx] = trial_fit
                    pop_err[idx] = trial_err
                    # check to update the best solution
                    if self.problem.is_better(pop_fit[idx], best_fit):
                        best_fit = pop_fit[idx]
                        best_err = pop_err[idx]
                        best_sol = np.copy(pop_positions[idx])

            # Prepare the evolutionary information
            best_subset = self.problem.position_2_solution(best_sol)[0]
            fRate = len(best_subset) / self.problem.dim
            evolutionary_process += 'At %d: %.4f, %.4f, %.2f ~%s\n' % (self.no_evaluations, best_fit, best_err, fRate,
                                                                       ', '.join(['%d' % ele for ele in best_subset]))
            pop_diversity = Helpers.population_stat(pop_positions)
            evolutionary_process += '\t\t Pop diversity: %f\n' % pop_diversity
            evolutionary_process += '\t\t Pop fit: %s\n' % ', '.join(['%.4f' % fit for fit in pop_fit])

        return best_sol, evolutionary_process


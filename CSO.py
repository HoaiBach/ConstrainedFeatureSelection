import numpy as np
from Utility import Helpers
from Problem import FeatureSelection
import WorldPara

class CSO:

    def __init__(self, problem: FeatureSelection, cond_constrain, pop_size=100, max_iterations=100,
                 phi=0.05, topology='ring', max_pos=1.0, min_pos=0.0, parallel='False'):
        self.problem = problem
        # ensure even pop_size
        if pop_size % 2 == 1:
            pop_size = pop_size + 1
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.phi = phi
        self.topology = topology  # note topology can be 'global' or 'ring'
        self.max_pos = max_pos
        self.min_pos = min_pos
        self.parallel = parallel
        self.cond_constrain = cond_constrain
        self.data_surrogate = np.empty((0, self.problem.dim+1), dtype=float)

    def evaluate_pop(self, pop):
        """
        Evaluate the pop of individuals, return the corresponding fitness
        :param pop:
        :return:
        """
        if self.parallel:
            fit_errors = self.problem.fitness_parallel(pop)
        else:
            fit_errors = [self.problem.fitness(ind) for ind in pop]
        return fit_errors

    def evaluate_pop_loocv(self, pop):
        """
        Evaluate the pop of individuals, return the corresponding fitness
        :param pop:
        :return:
        """
        if self.parallel:
            fit_errors = self.problem.fitness_loocv_parallel(pop)
        else:
            fit_errors = [self.problem.fitness_loocv(ind) for ind in pop]
        return fit_errors

    def check_feasible(self, fit, err):
        if WorldPara.CONSTRAIN_TYPE == 'err':
            return err <= self.cond_constrain
        elif WorldPara.CONSTRAIN_TYPE == 'fit':
            return fit <= self.cond_constrain

    def evolve(self):
        # Init the population
        dim = self.problem.dim
        pop_positions = self.problem.init_pop(self.pop_size)
        pop_vels = np.zeros((self.pop_size, dim))
        pop_err = np.zeros(self.pop_size)
        pop_fit = np.zeros(self.pop_size)

        fit_errors = self.evaluate_pop_loocv(pop_positions)
        for idx, (fit, err) in enumerate(fit_errors):
            pop_err[idx] = err
            pop_fit[idx] = fit

        if self.topology == 'global':
            neighbors = np.array([np.arange(self.pop_size)] * self.pop_size)
        elif self.topology == 'ring':
            neighbors = [[idx - 1, idx, idx + 1] for idx in np.arange(start=1, stop=self.pop_size - 1)]
            # for first ind
            first_ind = [[self.pop_size - 1, 0, 1]]
            first_ind.extend(neighbors)
            neighbors = first_ind
            # for last ind
            neighbors.append([self.pop_size-2, self.pop_size - 1, 0])
            neighbors = np.array(neighbors)
        else:
            raise Exception('Topology %s is not implemented.' % self.topology)
        assert len(neighbors) == self.pop_size

        # Start the evolution
        best_idx = np.argsort(pop_fit)[0]
        best_sol = np.copy(pop_positions[best_idx])
        best_fit = pop_fit[best_idx]
        best_err = pop_err[best_idx]

        no_iterations = 0
        stuck_for = 0
        best_updated = False

        evolutionary_process = '***************************************\n'
        best_subset = self.problem.position_2_solution(best_sol)[0]
        fRate = len(best_subset)/self.problem.dim
        evolutionary_process += 'At %d: %.4f, %.4f, %.2f ~%s\n' % (no_iterations, best_fit, best_err, fRate,
                                                                   ', '.join(['%d' % ele for ele in best_subset]))
        evolutionary_process += '\t\t 0 infeasible solutions\n'
        pop_diversity = Helpers.population_stat(pop_positions)
        evolutionary_process += '\t\t Pop diversity: %f\n' % pop_diversity
        evolutionary_process += '\t\t Pop fit: %s\n' % ', '.join(['%.4f' % fit for fit in pop_fit])
        evolutionary_process += '\t\t Stuck for: %d\n' % stuck_for
        evolutionary_process += '\t\t Constrained type: %s\n' % WorldPara.CONSTRAIN_TYPE
        evolutionary_process += '\t\t Constrained cond: %f\n' % self.cond_constrain

        while no_iterations < self.max_iterations:
            next_pop = np.zeros((self.pop_size, self.problem.dim))
            next_vels = np.zeros((self.pop_size, self.problem.dim))
            next_pop_fit = np.array([None]*self.pop_size)
            next_pop_err = np.array([None]*self.pop_size)

            to_evaluate = np.array([], dtype=int)
            no_infeasible = 0

            indices_pool = np.arange(self.pop_size)
            np.random.shuffle(indices_pool)

            feasible_indices = []
            if not (WorldPara.CONSTRAIN_MODE is None):
                feasible_indices = np.array([idx for idx in np.arange(self.pop_size)
                                             if self.check_feasible(pop_fit[idx], pop_err[idx])])

            for idx1, idx2 in zip(indices_pool[0::2], indices_pool[1::2]):

                # update the data to train surrogate
                if WorldPara.MUTATE_WINNER:
                    sol1, fit1 = pop_positions[idx1], pop_fit[idx1]
                    sol2, fit2 = pop_positions[idx2], pop_fit[idx2]
                    instance = self.problem.surrogate_prep_ins(sol1, fit1, sol2, fit2)
                    self.data_surrogate = np.append(self.data_surrogate, [instance], axis=0)

                if not self.check_feasible(pop_fit[idx1], pop_err[idx1]):
                    no_infeasible += 1
                if not self.check_feasible(pop_fit[idx2], pop_err[idx2]):
                    no_infeasible += 1

                if not (WorldPara.CONSTRAIN_MODE is None):
                    # contrasting 2 random individuals
                    if (not self.check_feasible(pop_fit[idx1], pop_err[idx1])) and \
                            (not self.check_feasible(pop_fit[idx2], pop_err[idx2])) and \
                            len(feasible_indices) >= 2:
                        # both have to learn from feasible solutions
                        fea_sols = np.random.choice(feasible_indices, size=2, replace=False)
                        for learn_idx, master_idx in zip([idx1, idx2], fea_sols):
                            r1 = np.random.rand(dim)
                            r2 = np.random.rand(dim)
                            x_ave = np.average(pop_positions[neighbors[learn_idx]], axis=0)
                            vel = r1 * pop_vels[learn_idx] + r2 * (pop_positions[master_idx] - pop_positions[learn_idx]
                                                                   ) + self.phi * (x_ave - pop_positions[learn_idx])
                            new_pos = pop_positions[learn_idx] + vel
                            new_pos[new_pos > self.max_pos] = self.max_pos
                            new_pos[new_pos < self.min_pos] = self.min_pos

                            # add new position of loser to the next pop
                            next_pop[learn_idx] = new_pos
                            next_vels[learn_idx] = vel
                            to_evaluate = np.append(to_evaluate, learn_idx)
                    else:
                        if self.check_feasible(pop_fit[idx1], pop_err[idx1]) and \
                                not (self.check_feasible(pop_fit[idx2], pop_err[idx2])):
                            winner_idx = idx1
                            loser_idx = idx2
                        elif self.check_feasible(pop_fit[idx2], pop_err[idx2]) and \
                                not (self.check_feasible(pop_fit[idx1], pop_err[idx1])):
                            winner_idx = idx2
                            loser_idx = idx1
                        elif self.problem.is_better(pop_fit[idx1], pop_fit[idx2]):
                            winner_idx = idx1
                            loser_idx = idx2
                        else:
                            winner_idx = idx2
                            loser_idx = idx1

                        # add winner to next pop
                        next_pop[winner_idx] = np.copy(pop_positions[winner_idx])
                        next_vels[winner_idx] = np.copy(pop_vels[winner_idx])
                        next_pop_fit[winner_idx] = pop_fit[winner_idx]
                        next_pop_err[winner_idx] = pop_fit[winner_idx]

                        # update loser
                        r1 = np.random.rand(dim)
                        r2 = np.random.rand(dim)
                        x_ave = np.average(pop_positions[neighbors[loser_idx]], axis=0)
                        vel = r1 * pop_vels[loser_idx] + r2 * (pop_positions[winner_idx] - pop_positions[loser_idx]) \
                              + self.phi * (x_ave - pop_positions[loser_idx])
                        new_pos = pop_positions[loser_idx] + vel
                        new_pos[new_pos > self.max_pos] = self.max_pos
                        new_pos[new_pos < self.min_pos] = self.min_pos

                        # add new position of loser to the next pop
                        next_pop[loser_idx] = new_pos
                        next_vels[loser_idx] = vel
                        to_evaluate = np.append(to_evaluate, loser_idx)
                else:
                    # contrasting 2 random individuals
                    if self.problem.is_better(pop_fit[idx1], pop_fit[idx2]):
                        winner_idx = idx1
                        loser_idx = idx2
                    else:
                        winner_idx = idx2
                        loser_idx = idx1

                    # add winner to next pop
                    next_pop[winner_idx] = np.copy(pop_positions[winner_idx])
                    next_vels[winner_idx] = np.copy(pop_vels[winner_idx])
                    next_pop_fit[winner_idx] = pop_fit[winner_idx]
                    next_pop_err[winner_idx] = pop_err[winner_idx]

                    # update loser
                    r1 = np.random.rand(dim)
                    r2 = np.random.rand(dim)
                    x_ave = np.average(pop_positions[neighbors[loser_idx]], axis=0)
                    vel = r1 * pop_vels[loser_idx] + r2 * (pop_positions[winner_idx] - pop_positions[loser_idx]) \
                          + self.phi * (x_ave - pop_positions[loser_idx])
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
                    best_updated = True

            if WorldPara.CONSTRAIN_MODE == 'hybrid' and no_iterations >= 3*self.max_iterations//4:
                WorldPara.CONSTRAIN_TYPE = 'fit'

            if no_iterations > self.max_iterations//2 and not (WorldPara.CONSTRAIN_MODE is None):
                if WorldPara.CONSTRAIN_TYPE == 'err':
                    self.cond_constrain = np.median(pop_err)
                elif WorldPara.CONSTRAIN_TYPE == 'fit':
                    self.cond_constrain = np.median(pop_fit)
                else:
                    raise Exception('Constrain %s is not implemented' % WorldPara.CONSTRAIN_TYPE)

            # Perform local search if necessary
            if WorldPara.MUTATE_WINNER and stuck_for >= 9:
                top_indices = np.argsort(pop_fit)[:int(0.1*self.pop_size)]

                self.problem.surrogate_build(self.data_surrogate)
                # firstly update the surrogate model

                top_mutants = np.empty((0, self.problem.dim))
                to_evaluate_mutants = np.array([], dtype=int)

                for top_idx in top_indices:
                    mutant, success = self.problem.local_search(pop_positions[top_idx])
                    if success:
                        top_mutants = np.append(top_mutants, [mutant], axis=0)
                        to_evaluate_mutants = np.append(to_evaluate_mutants, top_idx)

                top_mutants_fit_errs = self.evaluate_pop_loocv(top_mutants)
                # no_evaluations += len(winner_indices)
                for mutant_idx, (fit, err), mutant in zip(to_evaluate_mutants, top_mutants_fit_errs, top_mutants):
                    if self.problem.is_better(fit, pop_fit[mutant_idx]):
                        pop_positions[mutant_idx] = mutant
                        pop_fit[mutant_idx] = fit
                        pop_err[mutant_idx] = err
                        # update best if necessary
                        if self.problem.is_better(pop_fit[mutant_idx], best_fit):
                            best_sol = np.copy(pop_positions[mutant_idx])
                            best_fit = pop_fit[mutant_idx]
                            best_err = pop_err[mutant_idx]
                            best_updated = True

            # Prepare the evolutionary information
            best_subset = self.problem.position_2_solution(best_sol)[0]
            fRate = len(best_subset)/self.problem.dim
            evolutionary_process += 'At %d: %.4f, %.4f, %.2f ~%s\n' % (no_iterations+1, best_fit, best_err, fRate,
                                                                       ', '.join(['%d' % ele for ele in best_subset]))
            evolutionary_process += '\t\t %d infeasible solutions\n' % no_infeasible
            pop_diversity = Helpers.population_stat(pop_positions)
            evolutionary_process += '\t\t Pop diversity: %f\n' % pop_diversity
            evolutionary_process += '\t\t Pop fit: %s\n' % ', '.join(['%.4f' % fit for fit in pop_fit])

            if best_updated:
                stuck_for = 0
            else:
                stuck_for += 1
            evolutionary_process += '\t\t Stuck for: %d\n' % stuck_for
            evolutionary_process += '\t\t Constrained type: %s\n' % WorldPara.CONSTRAIN_TYPE
            evolutionary_process += '\t\t Constrained cond: %f\n' % self.cond_constrain

            # Reset the information for each iteration
            best_updated = False
            no_infeasible = 0

            no_iterations += 1

        return best_sol, evolutionary_process

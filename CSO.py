import numpy as np
from Utility import Helpers
from Problem import FeatureSelection

class CSO:

    def __init__(self, problem: FeatureSelection, pop_size=100, max_evaluations=10000,
                 phi=0.05, topology='global', max_pos=1.0, min_pos=0.0,
                 parallel='False'):
        self.problem = problem
        # ensure even pop_size
        if pop_size % 2 == 1:
            pop_size = pop_size + 1
        self.pop_size = pop_size
        self.max_evaluations = max_evaluations
        self.phi = phi
        # note topology can be 'global' or 'ring'
        self.topology = topology
        self.max_pos = max_pos
        self.min_pos = min_pos
        self.parallel = parallel

    def evaluate_pop(self, pop):
        """
        Evaluate the pop of individuals, return the corresponding fitness
        :param pop:
        :return:
        """
        if self.parallel:
            fits = self.problem.fitness_parallel(pop)
        else:
            fits = [self.problem.fitness(ind) for ind in pop]
        return fits

    def evolve(self):
        # Init the population
        dim = self.problem.dim
        pop_positions = self.problem.init_pop(self.pop_size)
        pop_vels = np.zeros((self.pop_size, dim))
        pop_fit = self.evaluate_pop(pop_positions)

        no_evaluations = len(pop_positions)
        evolutionary_process = ''

        if self.topology == 'global':
            neighbors = [np.arange(self.pop_size)] * self.pop_size
        elif self.topology == 'ring':
            neighbors = [[idx - 1, idx + 1] for idx in np.arange(start=1, stop=self.pop_size - 1)]
            first_ind = [[1, self.pop_size - 1]]
            first_ind.extend(neighbors)
            neighbors = first_ind
            neighbors.append([0, self.pop_size - 2])
            neighbors = np.array(neighbors)
        else:
            raise Exception('Topology %s is not implemented.' % self.topology)
        assert len(neighbors) == self.pop_size

        # Start the evolution
        best_sol = None
        best_fit = self.problem.worst_fitness()
        milestone = no_evaluations
        stuck_for = 0
        best_updated = False
        while no_evaluations < self.max_evaluations:
            next_pop = []
            next_pop_fitness = []
            next_vels = []
            to_evaluate = []
            no_infeasible = 0

            indices_pool = np.arange(self.pop_size)
            np.random.shuffle(indices_pool)

            import WorldPara
            feasible_indices = []
            if WorldPara.PENALISE_WORSE_THAN_FULL:
                feasible_indices = np.where(np.array(pop_fit) != float('inf'))[0]

            for idx1, idx2 in zip(indices_pool[0::2], indices_pool[1::2]):
                if pop_fit[idx1] == float('inf'):
                    no_infeasible += 1
                if pop_fit[idx2] == float('inf'):
                    no_infeasible += 1
                if WorldPara.PENALISE_WORSE_THAN_FULL:
                    # contrasting 2 random individuals
                    if pop_fit[idx1] == pop_fit[idx2] == float('inf') and len(feasible_indices) > 0:
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
                            next_pop.append(new_pos)
                            next_pop_fitness.append(None)
                            next_vels.append(vel)
                            to_evaluate.append(len(next_pop) - 1)
                    else:
                        if self.problem.is_better(pop_fit[idx1], pop_fit[idx2]):
                            winner_idx = idx1
                            loser_idx = idx2
                        else:
                            winner_idx = idx2
                            loser_idx = idx1

                        # add winner to next pop
                        next_pop.append(np.copy(pop_positions[winner_idx]))
                        next_pop_fitness.append(pop_fit[winner_idx])
                        next_vels.append(np.copy(pop_vels[winner_idx]))

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
                        next_pop.append(new_pos)
                        next_pop_fitness.append(None)
                        next_vels.append(vel)
                        to_evaluate.append(len(next_pop) - 1)
                else:
                    # contrasting 2 random individuals
                    if self.problem.is_better(pop_fit[idx1], pop_fit[idx2]):
                        winner_idx = idx1
                        loser_idx = idx2
                    else:
                        winner_idx = idx2
                        loser_idx = idx1

                    # add winner to next pop
                    next_pop.append(np.copy(pop_positions[winner_idx]))
                    next_pop_fitness.append(pop_fit[winner_idx])
                    next_vels.append(np.copy(pop_vels[winner_idx]))

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
                    next_pop.append(new_pos)
                    next_pop_fitness.append(None)
                    next_vels.append(vel)
                    to_evaluate.append(len(next_pop) - 1)

            assert len(next_pop) == len(pop_positions)
            next_pop = np.array(next_pop)

            eval_fits = self.evaluate_pop(next_pop[to_evaluate])
            for idx, fit in zip(to_evaluate, eval_fits):
                next_pop_fitness[idx] = fit
            no_evaluations += len(to_evaluate)

            pop_positions = np.array(next_pop)
            pop_fit = np.array(next_pop_fitness)
            pop_vels = np.array(next_vels)

            # update the best ind if necessary
            for fit, sol in zip(pop_fit, pop_positions):
                if self.problem.is_better(fit, best_fit):
                    best_fit = fit
                    best_sol = np.copy(sol)
                    best_updated = True

            if no_evaluations >= milestone:
                milestone += self.pop_size
                best_subset = self.problem.position_2_solution(best_sol)[0]
                f_weight = self.problem.f_weight
                fRate = len(best_subset) / len(best_sol)
                eRate = (best_fit - f_weight * fRate) / (1 - f_weight)
                evolutionary_process += 'At %d: %.4f, %.4f, %.2f ~%s\n' % (no_evaluations, best_fit, eRate, fRate,
                                                                           ', '.join(
                                                                               ['%d' % ele for ele in best_subset]))
                evolutionary_process += '\t\t %d infeasible solutions\n' % no_infeasible
                pop_diversity = Helpers.population_stat(pop_positions)
                evolutionary_process += '\t\t Pop diversity: %f\n' % pop_diversity
                evolutionary_process += '\t\t Pop fit: %s\n' % ', '.join(['%.4f' % fit for fit in pop_fit])

                if best_updated:
                    stuck_for = 0
                else:
                    stuck_for += 1
                best_updated = False
                evolutionary_process += '\t\t Stuck for: %d\n' % stuck_for

                if stuck_for >= 5 and WorldPara.ENHANCE_CONSTRAIN:
                    if WorldPara.ERR_CONSTRAIN:
                        # calculate pop_err first
                        sel_ratio = [len(self.problem.position_2_solution(ind)[0]) / self.problem.no_features
                                     for ind in pop_positions]
                        pop_error = np.array([])
                        for fit, ratio in zip(pop_fit, sel_ratio):
                            sel_err = (fit - self.problem.f_weight * ratio) / (1 - self.problem.f_weight)
                            pop_error = np.append(pop_error, sel_err)
                        new_constrain = np.median(pop_error)
                        invalid_indices = np.where(pop_error > new_constrain)
                    else:
                        pop_fit = np.array(pop_fit)
                        new_constrain = np.median(pop_fit)
                        invalid_indices = np.where(pop_fit > new_constrain)

                    self.problem.constrain_cond = new_constrain
                    pop_fit[invalid_indices] = float('inf')

                evolutionary_process += '\t\t Constrained error: %f\n' % self.problem.constrain_cond
                no_infeasible = 0

        return best_sol, evolutionary_process

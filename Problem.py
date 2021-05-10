import numpy as np
import abc
from multiprocessing import get_context
from sklearn.model_selection import StratifiedKFold as SKF, KFold
from Utility import Helpers
import WorldPara
from skfeature.function.similarity_based.reliefF import reliefF


class Problem(metaclass=abc.ABCMeta):

    def __init__(self, minimized):
        self.minimized = minimized
        self.no_instances = -1
        self.no_features = -1

    def worst_fitness(self):
        w_f = float('inf') if self.minimized else float('-inf')
        return w_f

    def is_better(self, first, second):
        if self.minimized:
            return first < second
        else:
            return first > second

    @abc.abstractmethod
    def fitness(self, sol):
        pass

    @abc.abstractmethod
    def fitness_parallel(self, sol_list):
        pass

    @abc.abstractmethod
    def position_2_solution(self, pos):
        pass


class FeatureSelection(Problem):

    def __init__(self, X, y, classifier, init_style, fratio_weight):
        Problem.__init__(self, minimized=True)
        self.X = X
        self.y = y
        self.no_instances, self.no_features = self.X.shape
        self.threshold = 0.6
        self.dim = self.no_features
        self.clf = classifier
        self.init_style = init_style
        self.f_weight = fratio_weight

        # stratified only applicable when enough instnaces for each class
        k = 10
        labels, counts = np.unique(self.y, return_counts=True)
        label_min = np.min(counts)
        if label_min < k:
            self.skf = KFold(n_splits=k, shuffle=True, random_state=1617)
            self.skf_valid = KFold(n_splits=k, shuffle=True, random_state=1990)
        else:
            self.skf = SKF(n_splits=k, shuffle=True, random_state=1617)
            self.skf_valid = SKF(n_splits=k, shuffle=True, random_state=1990)

        self.rf_scores = reliefF(self.X, self.y, k=1)
        self.rf_scores = self.rf_scores/np.sum(self.rf_scores)
        print(np.sum(self.rf_scores))

    def init_pop(self, pop_size):
        if self.init_style == 'Bing':
            fea_count = np.array([pop_size]*self.no_features, dtype=float)

            large_size = pop_size // 3
            small_size = pop_size - large_size
            pop = []
            for _ in range(small_size):
                no_sel = np.random.randint(1, self.no_features // 10 + 1)
                if WorldPara.INIT_STYLE == 'Random':
                    sel_fea = np.random.choice(range(0, self.no_features), size=no_sel, replace=False)
                elif WorldPara.INIT_STYLE == 'Diverse':
                    p = fea_count/np.sum(fea_count)
                    sel_fea = np.random.choice(range(0, self.no_features), size=no_sel, replace=False, p=p)
                    fea_count[sel_fea] -= 1.0
                elif WorldPara.INIT_STYLE == 'Relief':
                    sel_fea = np.random.choice(range(0, self.no_features), size=no_sel, replace=False, p=self.rf_scores)
                else:
                    raise Exception('%s init style is not implemented!!' % WorldPara.INIT_STYLE)

                ind = np.zeros(self.no_features, dtype=float)
                ind[sel_fea] = 1.0
                pop.append(ind)
            for _ in range(large_size):
                no_sel = np.random.randint(self.no_features // 2, self.no_features + 1)
                if WorldPara.INIT_STYLE == 'Random':
                    sel_fea = np.random.choice(range(0, self.no_features), size=no_sel, replace=False)
                elif WorldPara.INIT_STYLE == 'Diverse':
                    p = fea_count/np.sum(fea_count)
                    sel_fea = np.random.choice(range(0, self.no_features), size=no_sel, replace=False, p=p)
                    fea_count[sel_fea] -= 1.0
                elif WorldPara.INIT_STYLE == 'Relief':
                    sel_fea = np.random.choice(range(0, self.no_features), size=no_sel, replace=False, p=self.rf_scores)
                else:
                    raise Exception('%s init style is not implemented!!' % WorldPara.INIT_STYLE)
                ind = np.zeros(self.no_features, dtype=float)
                ind[sel_fea] = 1.0
                pop.append(ind)
            pop = np.array(pop)
            np.random.shuffle(pop)
        else:
            pop = np.random.rand((pop_size, self.no_features))

        return pop

    def position_2_solution(self, pos):
        assert len(pos) == self.dim

        selected_features = np.where(pos > self.threshold)[0]
        unselected_features = np.where(pos <= self.threshold)[0]

        return selected_features, unselected_features

    def fitness(self, sol):
        selected_features, unselected_features = self.position_2_solution(sol)
        if len(selected_features) == 0:
            fitness = self.worst_fitness()
            error = 1.0
        else:
            X_selected = self.X[:, selected_features]
            error = Helpers.kFoldCrossValidation(X_selected, self.y, self.clf, self.skf)

            sel_ratio = len(selected_features) / self.no_features
            fitness = (1.0 - self.f_weight) * error + self.f_weight * sel_ratio

        return fitness, error

    def fitness_loocv(self, sol):
        selected_features, unselected_features = self.position_2_solution(sol)
        if len(selected_features) == 0:
            fitness = self.worst_fitness()
            error = 1.0
        else:
            X_selected = self.X[:, selected_features]
            error = Helpers.LOOCV_1NN(X_selected, self.y)

            sel_ratio = len(selected_features) / self.no_features
            fitness = (1.0 - self.f_weight) * error + self.f_weight * sel_ratio

        return fitness, error

    def fitness_parallel(self, sol_list):
        with get_context('spawn').Pool(processes=4) as pool:
            fitness_list = pool.map(self.fitness, sol_list)
            pool.close()
            return fitness_list

    def fitness_loocv_parallel(self, sol_list):
        with get_context('spawn').Pool(processes=4) as pool:
            fitness_list = pool.map(self.fitness_loocv, sol_list)
            pool.close()
            return fitness_list

    def local_search(self, sol):
        selected_features, unselected_features = self.position_2_solution(sol)

        p0to1 = 1.0/len(sol)
        rf_unsel = self.rf_scores[unselected_features]
        p0to1_rf = p0to1*len(unselected_features)/np.sum(rf_unsel)*rf_unsel

        p1to0 = 10.0/len(sol)
        rf_inv_sel = 1.0-self.rf_scores[selected_features]
        p1to0_rf_inv = p1to0*len(selected_features)/np.sum(rf_inv_sel)*rf_inv_sel

        new_pos = np.copy(sol)
        for prob, fea in zip(p1to0_rf_inv, selected_features):
            if np.random.rand() < prob:
                # flip from selected to not selected
                new_pos[fea] = self.threshold - (sol[fea]-self.threshold)*self.threshold/(1.0-self.threshold)
        for prob, fea in zip(p0to1_rf, unselected_features):
            if np.random.rand() < prob:
                # flip from unselected to selected
                new_pos[fea] = self.threshold + (self.threshold-sol[fea])*(1.0-self.threshold)/self.threshold

        return new_pos



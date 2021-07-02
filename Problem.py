import numpy as np
import abc
from multiprocessing import get_context
from sklearn.model_selection import StratifiedKFold as SKF, KFold
from Utility import Helpers
import WorldPara
from skfeature.function.similarity_based.reliefF import reliefF
from sklearn.svm import LinearSVC as SVC


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

        self.scores = reliefF(self.X, self.y, k=1)
        self.scores = self.scores / np.sum(self.scores)

        # from Orange.data import Domain, Table
        # from Orange.preprocess.discretize import EntropyMDL
        # from Orange.preprocess import Discretize
        # from skfeature.utility.mutual_information import su_calculation
        # domain = Domain.from_numpy(X=X, Y=y)
        # table = Table.from_numpy(domain=domain, X=X, Y=y)
        # disc = Discretize()
        # disc.method = EntropyMDL(force=True)
        # table_dis = disc(table)
        # X_dis = table_dis.X
        # test_scores = []
        # for i in range(self.no_features):
        #     test_scores.append(su_calculation(X_dis[:, i], y))
        # test_scores = np.array(test_scores)
        # test_scores = test_scores/np.sum(test_scores)
        # self.scores = test_scores
        
        self.surrogate_clf = SVC(random_state=1617)

    def init_pop(self, pop_size):
        if self.init_style == 'Bing':
            large_size = pop_size // 3
            small_size = pop_size - large_size
            pop = []

            for _ in range(small_size):
                no_sel = np.random.randint(1, self.no_features // 10 + 1)
                sel_fea = np.random.choice(range(0, self.no_features), size=no_sel, replace=False)
                ind = np.zeros(self.no_features, dtype=float)
                ind[sel_fea] = 1.0
                pop.append(ind)

            for _ in range(large_size):
                no_sel = np.random.randint(self.no_features // 2, self.no_features + 1)
                sel_fea = np.random.choice(range(0, self.no_features), size=no_sel, replace=False)
                ind = np.zeros(self.no_features, dtype=float)
                ind[sel_fea] = 1.0
                pop.append(ind)

            pop = np.array(pop)
            np.random.shuffle(pop)
        else:
            pop = np.random.rand(pop_size, self.no_features)

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
            error = Helpers.LOOCV_NN(X_selected, self.y, k=WorldPara.NUM_NEIGHBORS)

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

        p0to1 = 1.0 / len(sol)
        rf_unsel = self.scores[unselected_features]
        p0to1_rf = p0to1 * len(unselected_features) / np.sum(rf_unsel) * rf_unsel

        p1to0 = 10.0 / len(sol)
        rf_inv_sel = 1.0 - self.scores[selected_features]
        p1to0_rf_inv = p1to0 * len(selected_features) / np.sum(rf_inv_sel) * rf_inv_sel

        no_iteration = 0
        while no_iteration < WorldPara.LOCAL_ITERATIONS:
            new_pos = np.copy(sol)
            for prob, fea in zip(p1to0_rf_inv, selected_features):
                if np.random.rand() < prob:
                    # flip from selected to not selected
                    new_pos[fea] = self.threshold - (sol[fea] - self.threshold) * self.threshold / (
                                1.0 - self.threshold)
            for prob, fea in zip(p0to1_rf, unselected_features):
                if np.random.rand() < prob:
                    # flip from unselected to selected
                    new_pos[fea] = self.threshold + (self.threshold - sol[fea]) * (
                                1.0 - self.threshold) / self.threshold

            if self.surrogate_check(new_pos, sol) == 1:
                return new_pos, True

            no_iteration += 1

        return sol, False

    def update_length(self, sol, max_length):
        """
        Update the sol with the given max_length
        :param sol:
        :param max_length:
        :return:
        """
        selected_features, unselected_features = self.position_2_solution(sol)

        rf_unsel = self.scores[unselected_features]
        prob_to_add = rf_unsel/np.sum(rf_unsel)

        rf_inv_sel = 1.0 - self.scores[selected_features]
        prob_to_remove = rf_inv_sel/np.sum(rf_inv_sel)

        no_iteration = 0
        best_ind = None

        new_length = np.random.randint(1, max_length)
        while no_iteration < WorldPara.LENGTH_ITERATIONS:
            # generating new positions
            new_pos = np.copy(sol)
            if new_length > len(selected_features):
                add_fea = np.random.choice(unselected_features, size=new_length-len(selected_features), p=prob_to_add)
                new_pos[add_fea] = 1.0
            else:
                rm_fea = np.random.choice(selected_features, size=len(selected_features)-new_length, p=prob_to_remove)
                new_pos[rm_fea] = 0.0

            if best_ind is None:
                best_ind = new_pos
            else:
                if self.surrogate_check(new_pos, best_ind) == 1:
                    best_ind = new_pos

            no_iteration += 1

        return best_ind

    def update_length_fix(self, length):
        """
        Update the sol with the given length, randomly
        :return: new solution
        """
        assert length > 0
        prob = self.scores / np.sum(self.scores)
        no_iteration = 0
        best_ind = None

        while no_iteration < WorldPara.LENGTH_ITERATIONS:
            # generating new positions
            sel_fea = np.random.choice(np.arange(self.no_features), size=length, replace=False, p=prob)
            new_pos = np.zeros(self.no_features, dtype=float)
            new_pos[sel_fea] = 1.0

            if best_ind is None:
                best_ind = new_pos
            else:
                if self.surrogate_check(new_pos, best_ind) == 1:
                    best_ind = new_pos
            no_iteration += 1

        return best_ind

    def surrogate_check(self, sol1, sol2):
        """
        Using the surrogate classifier to check which one is better sol1 or sol2
        :param sol1:
        :param sol2:
        :return: 1 if sol1 is better, 0 otherwise
        """
        sel1, _ = self.position_2_solution(sol1)
        fea1 = np.zeros(len(sol1), dtype=float)
        fea1[sel1] = 1.0

        sel2, _ = self.position_2_solution(sol2)
        fea2 = np.zeros(len(sol2), dtype=float)
        fea2[sel2] = 1.0

        data = fea1 - fea2
        label = self.surrogate_clf.predict([data])[0]
        return label

    def surrogate_build(self, data: list):
        """
        Using the data to train a surrogate classifier
        Each instance in the data is: sol1 - sol2
        The label is 1 if sol1 is better, the label is 0 otherwise
        :param data:
        :return:
        """
        data = np.array(data)
        X = data[:, :self.dim]
        y = np.ravel(data[:, self.dim:])
        self.surrogate_clf.fit(X=X, y=y)

    def surrogate_prep_ins(self, sol1, fit1, sol2, fit2):
        """
        Prepare instance to train the surrogate model
        the label of instance is 1 if fit1 is better than fit2, otherwise the label is 0
        :param sol1:
        :param sol2:
        :param fit1:
        :param fit2:
        :return:
        """
        sel1, _ = self.position_2_solution(sol1)
        fea1 = np.zeros(len(sol1), dtype=float)
        fea1[sel1] = 1.0

        sel2, _ = self.position_2_solution(sol2)
        fea2 = np.zeros(len(sol2), dtype=float)
        fea2[sel2] = 1.0

        instance = fea1 - fea2
        if self.is_better(fit1, fit2):
            label = 1.0
        else:
            label = 0.0
        instance = np.append(instance, label)
        return instance

import numpy as np
import abc
from multiprocessing import get_context
from sklearn.model_selection import StratifiedKFold as SKF, KFold
from sklearn.metrics import balanced_accuracy_score


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
        else:
            self.skf = SKF(n_splits=k, shuffle=True, random_state=1617)

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
            return self.worst_fitness(), None

        # error rate using K-fold
        error = 0.0
        X_selected = self.X[:, selected_features]
        for train_idx, test_idx in self.skf.split(X_selected, self.y):
            X_train, X_test = X_selected[train_idx, :], X_selected[test_idx, :]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            self.clf.fit(X_train, y_train)
            y_test_pred = self.clf.predict(X_test)
            error += 1.0 - balanced_accuracy_score(y_true=y_test, y_pred=y_test_pred)
        error = error / self.skf.n_splits

        sel_ratio = len(selected_features) / self.no_features
        fitness = (1.0 - self.f_weight) * error + self.f_weight * sel_ratio

        return fitness

    def fitness_parallel(self, sol_list):
        with get_context('spawn').Pool(processes=4) as pool:
            fitness_list = pool.map(self.fitness, sol_list)
            pool.close()
            return fitness_list


class ShiftedGriewank(Problem):

    def fitness_parallel(self, sol_list):
        pass

    def __init__(self, dim):
        Problem.__init__(self, minimized=True)
        self.dim = dim

    def fitness(self, sol):
        first = np.sum([ele ** 2 / 4000 for ele in sol])
        second = 1
        for idx, ele in enumerate(sol):
            second *= np.cos(ele / np.sqrt(idx + 1))
        return first - second + 1 - 180

    def position_2_solution(self, pos):
        pass

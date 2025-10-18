import numpy as np
import random
import time
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

OPS = ['+', '-', '*', '/', 'log', 'sin', 'cos']

class EvolutionaryOptimizer:
    """
    Evolutionary optimizer for feature synthesis and selection.
    Uses a simple Genetic Algorithm to minimize MSE on regression tasks.
    """

    def __init__(self,
                 maxtime=3600,
                 pop_size=30,
                 crossover_prob=0.7,
                 mutation_prob=0.3,
                 elitism=True,
                 patience=40,
                 random_state=42,
                 models=None,
                 min_genes=3,
                 max_genes=10,
                 complexity_penalty=0.001):
        """
        Evolutionary optimizer for feature synthesis.

        Each individual is a list of transformations (variable length).
        """
        self.maxtime = maxtime
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism = elitism
        self.patience = patience
        self.random_state = random_state
        self.min_genes = min_genes
        self.max_genes = max_genes
        self.complexity_penalty = complexity_penalty

        self.models = models or [
            LinearRegression(),
            Ridge(alpha=1.0),
            RandomForestRegressor(n_estimators=50, random_state=random_state)
        ]

        random.seed(random_state)
        np.random.seed(random_state)

        self.history = []
        self.best_ind = None
        self.best_fit = None

    # =====================================================
    # FUNCIONES INTERNAS
    # =====================================================

    def _apply(self, ind, X):
        """Aplica las transformaciones del individuo sobre X."""
        X_new = X.copy()
        for (i1, i2, op) in ind:
            try:
                if op == '+': X_new = np.c_[X_new, X[:, i1] + X[:, i2]]
                elif op == '-': X_new = np.c_[X_new, X[:, i1] - X[:, i2]]
                elif op == '*': X_new = np.c_[X_new, X[:, i1] * X[:, i2]]
                elif op == '/': X_new = np.c_[X_new, np.divide(X[:, i1], X[:, i2] + 1e-8)]
                elif op == 'log': X_new = np.c_[X_new, np.log(np.abs(X[:, i1]) + 1)]
                elif op == 'sin': X_new = np.c_[X_new, np.sin(X[:, i1])]
                elif op == 'cos': X_new = np.c_[X_new, np.cos(X[:, i1])]
            except Exception:
                continue
        return X_new

    def _fitness(self, ind, X, y):
        """Evalúa la calidad de un individuo (fitness = -MSE medio con penalización)."""
        X_new = self._apply(ind, X)
        n_samples = X_new.shape[0]

        # Adaptar nº de folds y modelos según tamaño del dataset
        if n_samples > 15000:
            n_splits = 2
            models = [LinearRegression(), Ridge(alpha=1.0)]
        elif n_samples > 5000:
            n_splits = 3
            models = [
                LinearRegression(),
                Ridge(alpha=1.0),
                RandomForestRegressor(n_estimators=30, random_state=self.random_state)
            ]
        else:
            n_splits = 5
            models = [
                LinearRegression(),
                Ridge(alpha=1.0),
                Lasso(alpha=0.001, max_iter=2000),
                RandomForestRegressor(n_estimators=50, random_state=self.random_state),
                GradientBoostingRegressor(random_state=self.random_state)
            ]

        # Submuestreo para datasets grandes
        if n_samples > 10000:
            idx = np.random.choice(n_samples, size=4000, replace=False)
            X_eval, y_eval = X_new[idx], y[idx]
        else:
            X_eval, y_eval = X_new, y

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        for model in models:
            try:
                cv = cross_val_score(model, X_eval, y_eval, cv=kf,
                                     scoring='neg_mean_squared_error')
                scores.append(np.mean(cv))
            except Exception:
                continue

        if not scores:
            return -np.inf

        mean_score = np.mean(scores)  # es negativo (por convención de sklearn)
        penalty = self.complexity_penalty * len(ind)
        return mean_score - penalty  # mayor es mejor (menos MSE penalizado)

    def _mutate(self, ind, n_features):
        """Mutación con posibilidad de añadir, eliminar o cambiar genes."""
        child = ind.copy()
        r = random.random()

        if r < 0.25 and len(child) > self.min_genes:
            # Eliminar un gen
            child.pop(random.randint(0, len(child) - 1))
        elif r < 0.5 and len(child) < self.max_genes:
            # Añadir un gen nuevo
            child.append((
                random.randint(0, n_features - 1),
                random.randint(0, n_features - 1),
                random.choice(OPS)
            ))
        else:
            # Cambiar un operador o índices
            pos = random.randint(0, len(child) - 1)
            child[pos] = (
                random.randint(0, n_features - 1),
                random.randint(0, n_features - 1),
                random.choice(OPS)
            )
        return child

    def _crossover(self, p1, p2):
        """Cruce simple adaptable a longitudes variables."""
        p1_len = len(p1)
        p2_len = len(p2)
        if p1_len < 2 or p2_len < 2:
            return p1.copy(), p2.copy()
        cut1 = random.randint(1, p1_len - 1)
        cut2 = random.randint(1, p2_len - 1)
        c1 = p1[:cut1] + p2[cut2:]
        c2 = p2[:cut2] + p1[cut1:]
        return c1[:self.max_genes], c2[:self.max_genes]

    # =====================================================
    # ENTRENAMIENTO PRINCIPAL
    # =====================================================

    def fit(self, X, y):
        start = time.time()
        n_features = X.shape[1]

        # Inicializar población
        population = [
            [(random.randint(0, n_features - 1),
              random.randint(0, n_features - 1),
              random.choice(OPS))
             for _ in range(random.randint(self.min_genes, self.max_genes))]
            for _ in range(self.pop_size)
        ]

        fitnesses = [self._fitness(ind, X, y) for ind in population]
        best = population[np.argmax(fitnesses)]
        best_fit = max(fitnesses)
        self.history = [(0, best_fit, np.mean(fitnesses))]

        no_improve = 0
        gen = 0

        while time.time() - start < self.maxtime:
            new_pop = []

            # Elitismo
            if self.elitism:
                elite = population[np.argmax(fitnesses)]
                new_pop.append(elite)

            # Generar nueva población
            while len(new_pop) < self.pop_size:
                if random.random() < self.crossover_prob:
                    p1, p2 = random.sample(population, 2)
                    c1, c2 = self._crossover(p1, p2)
                    new_pop.extend([c1, c2])
                else:
                    parent = random.choice(population)
                    child = parent.copy()
                    if random.random() < self.mutation_prob:
                        child = self._mutate(child, n_features)
                    new_pop.append(child)

            population = new_pop[:self.pop_size]
            fitnesses = [self._fitness(ind, X, y) for ind in population]

            gen_best = population[np.argmax(fitnesses)]
            gen_best_fit = max(fitnesses)

            # Actualizar mejor global
            if gen_best_fit > best_fit:
                best = gen_best
                best_fit = gen_best_fit
                no_improve = 0
            else:
                no_improve += 1

            gen += 1
            self.history.append((gen, best_fit, np.mean(fitnesses)))

            if no_improve >= self.patience:
                print(f"[STOP] Early stopping at gen {gen}")
                break

        self.best_ind = best
        self.best_fit = best_fit
        print(f"[DONE] Best fitness: {best_fit:.5f} | Time: {time.time() - start:.1f}s")
        return self

    def transform(self, X):
        """Aplica la mejor transformación encontrada al dataset."""
        if self.best_ind is None:
            raise ValueError("El modelo no ha sido entrenado aún (fit no llamado).")
        return self._apply(self.best_ind, X)

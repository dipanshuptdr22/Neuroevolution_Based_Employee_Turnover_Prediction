import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from random import randint, random
import math

# =============================================================================
# Configuration Parameters
# =============================================================================
MAX_FEATURES = 12
MIN_FEATURES = 8
MAX_LAYERS = 3
MIN_LAYERS = 1
MAX_NEURONS = 9
MIN_NEURONS = 5

GEN_COUNT = 3
POP_SIZE = 25
ELITE_COUNT = math.ceil(POP_SIZE * 0.1)
MUTATION_PROB = 0.01
TARGET_SCORE = 1.01
TOURNAMENT_SIZE = math.ceil(POP_SIZE * 0.2)

DATA_PATH = "/content/WA_Fn-UseC_-HR-Employee-Attrition.csv"
TARGET_COL = "Attrition"
OHE_COLUMNS = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]
BINARY_COLUMNS = ["Gender", "OverTime", "Over18"]
DROP_COLUMNS = ["EmployeeNumber", "EmployeeCount", "Over18", "DailyRate", "HourlyRate", "MonthlyRate", "StandardHours"]
SCALE_EXCLUDE = OHE_COLUMNS + BINARY_COLUMNS + ["EmployeeNumber", "Attrition", "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobSatisfaction", "StockOptionLevel", "WorkLifeBalance", "RelationshipSatisfaction"]

# =============================================================================
# Data Controller
# =============================================================================
class DataController:
    __instance = None   #this is implicitly static if defined in init then it is object attribute

    @staticmethod
    def shared():
        if DataController.__instance is None:
            DataController()
        return DataController.__instance

    def __init__(self):
        if DataController.__instance is not None:
            raise Exception("DataController is a singleton!")
        DataController.__instance = self

    def read(self, path):
        self.df = pd.read_csv(path)

    def preprocess(self):
        self.df.drop(columns=DROP_COLUMNS, inplace=True)
        for col in BINARY_COLUMNS:   #converts binaary columns to 0/i yes / no to 0/1
            if col in self.df:
                self.df[col] = self.df[col].astype('category').cat.codes
        self.df = pd.get_dummies(self.df, columns=OHE_COLUMNS) #transforms department column to hr and sales andthen assign the values for each row
        to_scale = self.df.columns.difference(SCALE_EXCLUDE)   #Give the remaining columns except the one in scale exclude
        scaler = StandardScaler()  # used from the scikit learn libaray to conveert the colmns to standard values having mean =0 and std deviation =1
        self.df[to_scale] = scaler.fit_transform(self.df[to_scale])  # fit calculates the mean and deviation of each columns in to_scale
                                                                     # transform uses those values to transform the data x_scaled=(x-mean)/std
    def split(self, target_col, test_fraction=0.2):
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col].astype('category').cat.codes
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_fraction, random_state=0)

    def get_train_X(self): return self.X_train
    def get_train_y(self): return self.y_train
    def get_feature_count(self): return len(self.X_train.columns)

# =============================================================================
# FeatureMask and ModelShape
# =============================================================================
class FeatureMask:
    def __init__(self, mask):
        self.mask = mask

    def mutate(self, rate):
        self.mask = [1 - bit if random() < rate else bit for bit in self.mask]

    def cross_with(self, partner):
        idx = randint(0, len(self.mask) - 1)
        return (
            self.mask[:idx+1] + partner.mask[idx+1:],
            partner.mask[:idx+1] + self.mask[idx+1:]
        )

    def raw(self): return self.mask

class ModelShape:
    def __init__(self, shape):
        self.shape = shape

    def mutate(self, rate):
        if random() < rate:
            for i in range(len(self.shape)):
                self.shape[i] = randint(MIN_NEURONS, MAX_NEURONS)

    def cross_with(self, partner):
        idx = randint(0, min(len(self.shape), len(partner.shape)) - 1)
        return (
            self.shape[:idx+1] + partner.shape[idx+1:],
            partner.shape[:idx+1] + self.shape[idx+1:]
        )

    def raw(self): return self.shape

# =============================================================================
# NeuroGA (Neural Network individual)
# =============================================================================
class NeuroGA(MLPClassifier):
    def __init__(self, feat_mask=None, shape=None):
        dc = DataController.shared()
        mask = feat_mask if feat_mask else [1] * dc.get_feature_count()   #values on which it is trained
        arch = shape if shape else [randint(MIN_NEURONS, MAX_NEURONS) for _ in range(randint(MIN_LAYERS, MAX_LAYERS))] # defined the architectur of the  model
        self.feat_mask = FeatureMask(mask) #both used to assign the model to the object which called this as abovve we onnly created it not assigned it to any model
        self.shape = ModelShape(arch)
        self.fitness = 0.0
        super().__init__(hidden_layer_sizes=tuple(arch), max_iter=3000, learning_rate_init=0.001, random_state=0)
        # super line is used to create the model using the super class of init mlp classifier
    @staticmethod
    def create(): return NeuroGA()

    def filter_X(self, X):   #used to drop the unnecessary columns in the mask not needed
        drop = [i for i, bit in enumerate(self.feat_mask.raw()) if bit == 0]
        return X.drop(X.columns[drop], axis=1)

    def evaluate_accuracy(self, X, y):
        filtered_X = self.filter_X(X)
        kf = KFold(n_splits=10, shuffle=True, random_state=0)  #we do this for each architecture on the training dataset
        scores = []
        for train_idx, test_idx in kf.split(filtered_X):
            self.fit(filtered_X.iloc[train_idx], y.iloc[train_idx])
            scores.append(self.score(filtered_X.iloc[test_idx], y.iloc[test_idx]))
        self.fitness = np.mean(scores)

    def cross_with(self, other):
        f1, f2 = self.feat_mask.cross_with(other.feat_mask)
        s1, s2 = self.shape.cross_with(other.shape)
        return NeuroGA(f1, s1), NeuroGA(f2, s2)

    def apply_mutation(self, rate):
        self.feat_mask.mutate(rate)
        self.shape.mutate(rate)

    def get_score(self): return self.fitness

    def __str__(self):
        return f"Features: {self.feat_mask.raw()}, Shape: {self.shape.raw()}, Accuracy: {self.fitness:.4f}"

# =============================================================================
# ModelPool and Evolver
# =============================================================================
class ModelPool:
    def __init__(self):
        self.members = []

    def add(self, member):
        self.members.append(member)
        self.members.sort(key=lambda m: m.get_score(), reverse=True)

    def size(self): return len(self.members)
    def top(self): return self.members[0]
    def all(self): return self.members

    def evaluate_fitness(self, X, y):   #this will store for  each neural netwwork its accuracy in its object.fitness attribute
        for m in self.members:
            m.evaluate_accuracy(X, y)

    @staticmethod
    def init(factory):
        pool = ModelPool()
        while pool.size() < POP_SIZE:
            pool.add(factory())
        return pool

class Evolver:
    @staticmethod
    def evolve(current):
        new_pool = ModelPool()
        elites = current.all()[:ELITE_COUNT]
        for elite in elites:
            new_pool.add(elite)
        while new_pool.size() < POP_SIZE:
            p1 = Evolver.select(current)
            p2 = Evolver.select(current)
            c1, c2 = p1.cross_with(p2)
            new_pool.add(c1)
            if new_pool.size() < POP_SIZE:
                new_pool.add(c2)
        for i in range(ELITE_COUNT, POP_SIZE):
            new_pool.all()[i].apply_mutation(MUTATION_PROB)
        return new_pool

    @staticmethod
    def select(pool):
        return max([pool.all()[randint(0, POP_SIZE - 1)] for _ in range(TOURNAMENT_SIZE)], key=lambda m: m.get_score())

# =============================================================================
# Execution Pipeline
# =============================================================================
def prepare_data():
    dc = DataController.shared()
    dc.read(DATA_PATH)
    dc.preprocess()
    dc.split(TARGET_COL)
    print("Training features:", dc.get_train_X().columns.tolist())
    return dc

def initialize_population():
    dc = DataController.shared()
    pool = ModelPool.init(NeuroGA.create)  #pool will have the pool of neural networks
    pool.evaluate_fitness(dc.get_train_X(), dc.get_train_y())    #evalueates the fitness of the neural networks in the pool
    print_generation(pool, 0)
    return pool

def print_generation(pool, gen):
    print(f"Generation #{gen} | Top Accuracy: {pool.top().get_score():.4f}")
    for i, member in enumerate(pool.all()):
        print(f"Model {i:02d}: {member}")

def main():
    prepare_data()
    dc = DataController.shared()
    population = initialize_population()
    generation = 1
    while generation < GEN_COUNT and population.top().get_score() < TARGET_SCORE:
        population = Evolver.evolve(population)
        population.evaluate_fitness(dc.get_train_X(), dc.get_train_y())
        print_generation(population, generation)
        generation += 1
    print("\nBest Evolved Neural Network:")
    print(population.top())

if __name__ == '__main__':
    main()

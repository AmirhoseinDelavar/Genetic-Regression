import random
from array import array

import numpy as np
from statistics import mean

# input size assumed 1000
POP_SIZE = 100
POP = []
xtrain = np
ytrain = np


def fitness(genes):
    global xtrain,ytrain
    loss = 0.0
    for (x, y) in zip(np.nditer(xtrain), np.nditer(ytrain)):
        yout = genes[0] * (x ** 2) + genes[1] * x + genes[2]
        loss += (y - yout) ** 2
    loss /= 1000.0
    return loss


def offspring(p0, p1):
    prob = random.random()

    # if prob is less than 0.45, insert first half of p0
    if prob < 0.45:
        genes = np.array([p0.genes[0], p1.genes[1], p1.genes[2]])
        newchild = Chromoson(genes=genes)

    # if prob is less than 0.90, insert first half of p1
    elif prob < 0.90:
        genes = np.array([p1.genes[0], p0.genes[1], p0.genes[2]])
        newchild = Chromoson(genes=genes)

    # otherwise insert random gene(mutate),
    else:
        newchild = Chromoson()
    return newchild


class Genetic:

    def __init__(self, x, y) -> None:
        global xtrain, ytrain
        super().__init__()
        global POP
        self.generation = 0
        # importing data
        xtrain = np.genfromtxt(x, delimiter=',')
        ytrain = np.genfromtxt(y, delimiter=',')
        # start random pop
        self.firstpop()
        # sort pop
        POP.sort(key=lambda ch: ch.fitness)
        # termination condition
        while self.condition():
            # log state in console
            avgloss = mean([x.fitness for x in POP])/POP_SIZE
            print("Generation: {}\tavgLoss: {}\tbestLoss: {}\tbestA: {}\tbestB: {}\tbestC: {}" \
                  .format(self.generation, avgloss, POP[0].fitness, POP[0].genes[0], POP[0].genes[1], POP[0].genes[2]))
            # keep 10% of top-fitness for new gen
            new_gen = []
            new_gen.extend(POP[:10])
            # make new offset for 90% of new gen
            for _ in range(90):
                p0 = random.choice(POP[:50])
                p1 = random.choice(POP[:50])
                ch = offspring(p0, p1)
                new_gen.append(ch)
            # change new gen to pop
            POP = new_gen
            # gen+ & sort new pop
            self.generation += 1
            POP.sort(key=lambda ch: ch.fitness)

    def firstpop(self):
        for _ in range(POP_SIZE):
            POP.append(Chromoson())

    def condition(self):
        if POP[0].fitness < 0.05 or self.generation > 150:
            avgloss = mean([x.fitness for x in POP])/POP_SIZE
            print("Generation: {}\tavgLoss: {}\tbestLoss: {}\tbestA: {}\tbestB: {}\tbestC: {}" \
                  .format(self.generation, avgloss, POP[0].fitness, POP[0].genes[0], POP[0].genes[1], POP[0].genes[2]))
            print("Finished!!")
            return False
        else:
            return True


class Chromoson:

    def __init__(self, **kwargs):
        if 'genes' not in kwargs.keys():
            # creating a new random genes
            self.genes = np.random.uniform(low=-50.00, high=50.01, size=3)
        else:
            self.genes = kwargs.get('genes')
        # calculate fitness for genes
        self.fitness = fitness(self.genes)


if __name__ == '__main__':
    Genetic("./x_train.csv", "./y_train.csv")

from random import randint, random
import numpy as np


n = 5
m = 2
L = np.array([8., 6., 7., 4., 5.])
S = np.sum(L)/m

class ant_colony_optimizer():
    def __init__(self, colony_number, L, m, Q=0.1, ro=0.5, alpha=0.5, betta=0.5):
        if alpha + betta != 1:
            raise ValueError("Сумма Альфа и Бетта должна быть равна 1")
        self.pheromons = {}
        self.colony_number = colony_number
        self.L = L
        self.m = m
        self.ro = ro
        self.alpha = alpha
        self.betta = betta
        self.Q = Q
        self.S = np.sum(L)/m
        self.last_solution = [[L]] + [[] for i in range(m - 1)]
    
    def objective(self, solution, output="MeanSquareDeviation"):
        m = len(solution) 
        benchmark = np.sum(np.sum(solution)) / m
        if output == "Benchmark":
            return benchmark
        elif output == "MeanSquareDeviation":
            return np.mean([(np.sum(solution[i]) - benchmark)**2 for i in range(len(solution))])
        elif output == "MeanAbsDeviation":
            return np.mean([np.abs(np.sum(solution[i]) - benchmark) for i in range(len(solution))])
        elif output == "MaxDeviation":
            l1 = [np.sum(solution[i]) for i in range(len(solution))]
            return np.max(l1) - np.min(l1)
    
    def get_pheromon(self, solution, S):
        return (np.mean([np.abs(np.sum(solution[i]) - S) for i in range(len(solution))])**2)

    def get_distance(self, solution, elem, subset, S):
        return 1/abs(S - np.mean(solution[subset] + [elem]))
    
    def receive_pheromon(self, pheromons, elem, subset):
        for i in list(pheromons.keys()):
            if i == (elem, subset):
                return 1 + pheromons[(elem, subset)]
        return 1
        
    def go_ant_go(self):
        subsets = [[] for i in range(self.m)]
        L1 = self.L[:]
        L1.sort()
        L1 = L1[::-1]
        
        #Пустим муравьишку
        while len(L1) > 0:
            old_solution = subsets
            r = random()
            candidate = L1[0]
            prob_set = []
            
            #Посчитаем вероятности для муравьишки
            for j in range(self.m):
                if j != 0:
                    prob_set.append(prob_set[j-1] + (self.receive_pheromon(
                        self.pheromons, candidate, j)**self.alpha) * (self.get_distance(
                        old_solution, candidate, j, self.S)**self.betta)/sum(
                        (self.receive_pheromon(
                            self.pheromons, candidate, i)**self.alpha) * (self.get_distance(
                            old_solution, candidate, i, self.S)**self.betta) for i in range(self.m)))
                else:
                    prob_set.append((self.receive_pheromon(
                        self.pheromons, candidate, j)**self.alpha) * (self.get_distance(
                        old_solution, candidate, j, self.S)**self.betta)/sum(
                        (self.receive_pheromon(
                            self.pheromons, candidate, i)**self.alpha) * (self.get_distance(
                            old_solution, candidate, i, self.S)**self.betta) for i in range(self.m)))
            
            #Двинемся дальше по дереву
            check = np.array(prob_set) >= r
            pos, = np.where(check == True)
            if len(pos) == 0:
                pos = [randint(0, self.m-1)]
            subsets[pos[0]].append(candidate)
            
            #Не забудем положить за муравьишкой ферамоны
            #ВАЖНО! Сделать выветривание необновленных феромонов
            delta = self.get_pheromon(subsets, self.S)
            if delta == 0:
                delta += 0.00000000000001
            if (candidate, pos[0]) in list(self.pheromons.keys()): 
                self.pheromons[candidate, pos[0]] += (1-self.ro)*self.pheromons[candidate, pos[0]] + self.Q/delta
            else:
                self.pheromons[candidate, pos[0]] = self.Q/delta
            
            L1 = L1[1:]
        
        if self.objective(self.last_solution) > self.objective(subsets):
            self.last_solution = subsets
    
    def go_colony_go(self):
        ant_i = 1
        while ant_i <= self.colony_number:
            self.go_ant_go()
            ant_i += 1
    
    def optimize_with_iterations(self, iterations):
        iteration = 1
        while iteration <= iterations:
            self.go_colony_go()
            print("Прошла итерация номер ", iteration, ", осталось ", iterations-iteration, " итераций.")
            iteration += 1
        return self.last_solution

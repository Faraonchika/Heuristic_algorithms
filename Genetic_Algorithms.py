import numpy as np
import random


n = 5
m = 2
L = np.array([8., 6., 7., 4., 5.])
S = np.sum(L)/m

#usual genetic algorithm
class Genetic_Algorithm():
    def __init__(self, data, m=20, population_size=100, generations_number=100, elite_size=0.05, 
                 mutation_probability = 0.1, max_mutations=15, init_solution=[]):
        #problem data
        self.data = data
        self.m = m
        self.ideal_sum = np.sum(data) / m
        
        #algorithm parameters
        self.population_size = int(population_size)
        self.generations_number = generations_number
        self.init_solution = init_solution
        self.elite_size = int(elite_size*population_size)
        self.mutation_probability = mutation_probability
        self.max_mutations = max_mutations
        self.history = []
        
        #generate initial population
        if len(self.init_solution) == 0:
            self.population = []
        else:
            self.population = [self.encode(data, init_solution)]
        for ind_num in range(population_size - len([init_solution])):
            ind = []
            for number in data:
                l = [0 for i in range(m)]
                r = random.randint(0, m-1)
                l[r] = 1
                ind.append(l)
            self.population.append(np.transpose(ind))
        self.population = np.array(self.population)
        
        #the best individual
        self.the_best = (10**100,)
        
    def first_position(self, L, number):
        i = -1
        pos = -1
        while pos == -1 and i <= len(L)-1:
            i += 1
            try:
                pos = L[i].index(number)
            except ValueError:
                pass
            except IndexError:
                return [i-1, pos]
        return [i, pos]
    
    #encode individual to a bit list
    def encode(self, data, solution):
        encoded = [[0 for i in range(len(data))] for j in range(len(solution))]
        for number in data:
            n_pos = data.index(number)
            sub_pos = self.first_position(solution, number)[0]
            encoded[sub_pos][n_pos] = 1
        return encoded
    
    
    #decode one chromosome(subset)
    def chromosome_decode(self, data, chromosome):
        new_l = []
        positions = np.where(np.array(chromosome) == 1)[0]
        for pos in positions:
            new_l.append(data[pos])
        return new_l
    
    
    #decode individual        
    def decode(self, data, final_solution):
        decoded = []
        for l in final_solution:
            decoded.append(self.chromosome_decode(data, l))
        return decoded
    
    
    #count chromosome error
    def chromosome_error(self, chromosome):
        return abs(sum(chromosome_decode(self.data, chromosome)) - self.ideal_sum)
    
    
    #count individual error
    def individual_error(self, individual):
        chromosome_errors = []
        for chromosome in individual:
            chromosome_errors.append(self.chromosome_error(chromosome))
        return sum(chromosome_errors)/self.m
    
    #count count max - min
    def max_min_error(self, individual):
        chromosome_errors = []
        for chromosome in individual:
            chromosome_errors.append(self.chromosome_error(chromosome))
        return max(chromosome_errors) - min(chromosome_errors) + 1
    
    
    def sort_chromosomes_by_error(self, individual):
        individual1 = list(individual)
        individual1.sort(key=lambda x: chromosome_error(x))
        return np.array(individual1)
    
    
    def sort_individuals_by_error(self):
        population1 = list(self.population)
        population1.sort(key=lambda x: individual_error(x))
        return np.array(population1)
    
    
    def to_set(self, individual):
        s = set()
        for chromosome in individual:
            s.add(tuple(chromosome))
        return s
    
    
    def to_np(self, individual):
        l = []
        for chromosome in list(individual):
            l.append(np.array(chromosome))
        return np.array(l)
    
    
    def fix_deformity(self, individual):
        #number is contained by exsactly 1 subset
        num_codes = individual.T
        for i in range(len(num_codes)):
            got_1 = False
            for j in range(len(num_codes[i])):
                if got_1:
                    num_codes[i][j] = 0
                if num_codes[i][j] == 1:
                    got_1 = True
        individual = num_codes.T

        #numbers that are not in subsets
        add_positions = []
        for i in range(len(num_codes)-1):
            if np.all(num_codes[i] == 0):
                add_positions.append(i)


        #add them to subsets with the lowest error
        individual = self.sort_chromosomes_by_error(individual)
        s = 0
        for pos in add_positions:
            individual[s][pos] = 1
            if s + 1 <= self.m - 1:
                s += 1
            else:
                s = 0
        
        #renew arrays
        individual = self.sort_chromosomes_by_error(individual)
        
        return individual
    
    
    def crossover(self, parent1, parent2):
        all_chrom = self.to_np(self.to_set(parent1) | self.to_set(parent2))

        #We give only the best chromosomes to children
        all_chrom = self.sort_chromosomes_by_error(all_chrom)
        child = []
        for i in range(m-1):
            child.append(all_chrom[0])
            all_chrom = all_chrom[1:]

        #add one random chromosome in order to keep diversity
        child.append(random.choice(all_chrom))
        child = np.array(child)
        
        return self.fix_deformity(child)
    
    
    def mutate(self, individual):
        # individuals can mutate with a certain probability
        mutation_count = random.randint(0, self.max_mutations - 1)
        
        new_individual = individual.T
        for i in range(mutation_count):
            if random.random() < self.mutation_probability:
                r = random.randint(0, len(new_individual)-1)
                new_set = random.randint(0, m-1)
                l = np.array([0 for i in range(m)])
                l[new_set] = 1
                new_individual[r] = l
        return new_individual.T
    
    
    def find_the_best(self):
        best = self.sort_individuals_by_error()[0]
        return (self.individual_error(best), best)
    
    
    def get_the_best(self):
        return decode(data, self.the_best[1])
    
    def make_GA_step(self):
        elite_inds = list(self.sort_individuals_by_error()[:self.elite_size])
        
        all_parents = [random.choices(self.population, weights=[self.max_min_error(ind) for ind in self.population]
                                      , k=2) 
               for i in range(self.population_size - self.elite_size)]
        
        #make children
        children = []
        for parents in all_parents:
            children.append(self.crossover(parents[0], parents[1]))
        
        #add elite
        new_population = elite_inds + children
        
        #mutation
        for i in range(len(new_population)):
            new_population[i] = self.mutate(new_population[i])
        
        #renew population
        self.population = np.array(new_population)
    
    
    def run_GA(self):
        for g in range(self.generations_number):
            the_best = self.find_the_best()
            if self.the_best[0] > the_best[0]:
                self.the_best = the_best
#             print("Generation", g+1, " The best ind:", self.the_best[0])
            
            self.history.append(self.the_best)
            
            self.make_GA_step()


#parallel genetic algorithm where few populations develope at the same time
class Parallel_Genetic_Algorithm(Genetic_Algorithm):
    def __init__(self, data, m=20, population_size=100, generations_number=100, elite_size=0.05, 
                 mutation_probability = 0.1, max_mutations=15, init_solution=[], 
                number_of_iterations=5, migration_part=0.2):
        
        Genetic_Algorithm.__init__(self, data, m=m, population_size=population_size, 
                                   generations_number=generations_number, elite_size=elite_size, 
                 mutation_probability = mutation_probability, max_mutations=max_mutations, init_solution=init_solution)
        
        #new params
        self.number_of_parallel_algorithms = 2
        self.number_of_iterations = number_of_iterations
        self.migration_part = migration_part
    
    def run_PGA(self):
        results = []
        for n in range(self.number_of_parallel_algorithms):
            GA = Genetic_Algorithm(data, m=self.m, population_size=self.population_size, 
                                   generations_number=self.generations_number, elite_size=self.elite_size, 
                                    mutation_probability=self.mutation_probability, 
                                   max_mutations=self.max_mutations, init_solution=self.init_solution)
            GA.run_GA()
            new_the_best = GA.find_the_best()
            if self.the_best[0] > new_the_best[0]:
                self.the_best = new_the_best
            results.append(GA.get_the_best())
        
        number_of_inds = int(self.migration_part*self.population_size)
        
        #migration
        shuffle = []
        for r in results:
            to_sh = random.sample(self.to_set(r), number_of_inds)
            for i in range(len(to_sh)):
                to_sh[i] = tuple(to_sh[i])
            shuffle.append((to_sh))
        
        results[0] = list(self.to_np(self.to_set(results[0]) - set(shuffle[0])))
        results[1] = list(self.to_np(self.to_set(results[1]) - set(shuffle[1])))
        
        for s in shuffle[1]:
             results[0].append(np.array(s))
        for s in shuffle[0]:
             results[1].append(np.array(s))
       
        for r in range(len(results)):
            for l in range(len(results[r])):
                results[r][l] = list(results[r][l])

        
        for i in range(self.number_of_iterations-1):
            for r in range(self.number_of_parallel_algorithms):
                GA = Genetic_Algorithm(data, m=self.m, population_size=self.population_size, 
                                   generations_number=self.generations_number, elite_size=self.elite_size, 
                                    mutation_probability=self.mutation_probability, 
                                   max_mutations=self.max_mutations, init_solution=results[r])
                GA.run_GA()
                new_the_best = GA.find_the_best()
                if self.the_best[0] > new_the_best[0]:
                    self.the_best = new_the_best
                results[r] = GA.get_the_best()


#micro genetic algorithm with super small colony 
class Micro_Genetic_Algorithm(Genetic_Algorithm):
    def __init__(self, data, m=20, population_size=100, generations_number=100, elite_size=0.05, 
                 mutation_probability = 0.1, max_mutations=15, init_solution=[], 
                number_of_iterations=5, migration_part=0.2):
        
        Genetic_Algorithm.__init__(self, data, m=m, population_size=population_size, 
                                   generations_number=generations_number, elite_size=elite_size, 
                 mutation_probability = mutation_probability, max_mutations=max_mutations, init_solution=init_solution)
        
        #new params
        self.number_of_iterations = number_of_iterations
    
    def run_MGA(self):
        GA = Genetic_Algorithm(data, m=self.m, population_size=self.population_size, 
                                   generations_number=self.generations_number, elite_size=self.elite_size, 
                                    mutation_probability=self.mutation_probability, 
                                   max_mutations=self.max_mutations, init_solution=self.init_solution)

        GA.run_GA()
        the_best = GA.find_the_best()

        best = GA.get_the_best()
        if the_best[0] < self.the_best[0]:
            self.the_best = the_best
        
        
        for i in range(self.number_of_iterations - 1):
            GA = Genetic_Algorithm(data, m=self.m, population_size=self.population_size, 
                                   generations_number=self.generations_number, elite_size=self.elite_size, 
                                    mutation_probability=self.mutation_probability, 
                                   max_mutations=self.max_mutations, init_solution=best)
            GA.run_GA()
            
            the_best = GA.find_the_best()
            best = GA.get_the_best()
            if the_best[0] < self.the_best[0]:
                self.the_best = the_best

import random

from deap import base, creator, tools
from typing import List, Tuple


class nsgaGA:
    def __init__(self, config: dict, parameter_ranges: dict):
        self.config = config
        self.parameter_ranges = parameter_ranges

        # Extract parameters from config
        self.POPULATION_SIZE = config["population_size"]
        self.NUM_GENERATIONS = config["num_generations"]
        self.CROSSOVER = config["crossover"]
        self.MUTATION = config["mutation"]
        self.CROSSOVER_PROBABILITY = config["crossover_probability"]
        self.MUTATION_PROBABILITY = config["mutation_probability"]

        # Initialize DEAP
        self._initialize_deap()


    def _initialize_deap(self):
        """Sets up DEAP framework components."""
        creator.create("Multi_Objective_Fitness", base.Fitness, weights=(-1.0, 1.0, 1.0, -1.0))
        creator.create("Individual", list, fitness=creator.Multi_Objective_Fitness)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", self.create_individual)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register genetic operators
        crossover_operator = getattr(tools, self.CROSSOVER["operator"])
        mutation_operator = getattr(tools, self.MUTATION["operator"])

        self.toolbox.register("mate", crossover_operator, **self.CROSSOVER["params"])
        self.toolbox.register("mutate", mutation_operator, **self.MUTATION["params"])
        self.toolbox.register("select", tools.selNSGA2)


    def create_individual(self) -> List[float]:
        """Creates an individual by generating random values for each parameter."""
        individual = []
        for range in self.parameter_ranges.values():
            min_val, max_val = range
            individual.append(random.uniform(min_val, max_val))
        return individual
    

    def run(self, evaluate_function) -> Tuple[List, List]:
        """Executes the genetic algorithm and returns the Pareto front and their fitness values.

        Args:
            evaluate_function (callable): Function to evaluate the fitness of an individual.

        Returns:
            Tuple[List, List]: Pareto front individuals and their fitness values.
        """
        self.toolbox.register("evaluate", evaluate_function)

        # Generate initial population
        population = self.toolbox.population(n=self.POPULATION_SIZE)

        for generation in range(self.NUM_GENERATIONS):
            print(f"\n--- Generation {generation + 1}/{self.NUM_GENERATIONS} ---")

            # Evaluate the individuals in the population
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Select the next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.CROSSOVER_PROBABILITY:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.MUTATION_PROBABILITY:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate individuals with an invalid fitness
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_individuals))
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit

            # Replace old population
            population[:] = offspring

            # Extract Pareto front for this generation
            front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
            print(f"Pareto front size: {len(front)}")
            for i, ind in enumerate(front):
                print()
                print(f"  Individual {i + 1}: {ind}")
                print(f"    Fitness: {ind.fitness.values}")

        # Final Pareto front
        front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        front_fitness = [ind.fitness.values for ind in front]

        print("\n--- Final Pareto Front ---")
        for i, ind in enumerate(front):
            print()
            print(f"  Individual {i + 1}: {ind}")
            print(f"    Fitness: {ind.fitness.values}")

        return front, front_fitness

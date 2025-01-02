import random

from deap import base, creator, tools
from typing import List, Tuple, Callable


class nsgaGA:
    def __init__(self, config: dict, parameter_ranges: dict):
        """
        Initializes the genetic algorithm with the given configuration and parameter ranges.

        Args:
            config (dict): Configuration dictionary with GA parameters (population size, generations, etc.).
            parameter_ranges (dict): Dictionary specifying the parameter ranges for the individuals.
        """
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
        """
        Sets up the DEAP framework components, including fitness function, individual creation,
        and genetic operators like crossover and mutation.

        Registers the following operators in the toolbox:
            - mate: Crossover operator
            - mutate: Mutation operator
            - select: NSGA2 selection operator
        """
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
        """
        Creates an individual by generating random values for each parameter in the parameter range.

        Returns:
            List[float]: A list of randomly generated values representing an individual in the population.
        """
        individual = []
        for range in self.parameter_ranges.values():
            min_val, max_val = range
            individual.append(random.uniform(min_val, max_val))
        return individual
     

    def run(self, evaluate_function: Callable[[List[float]], Tuple[float, float, float, float]]) -> Tuple[List, List]:
        """
        Executes the genetic algorithm and returns the Pareto front and their fitness values.

        Args:
            evaluate_function (Callable): Function to evaluate the fitness of an individual. It should return
                                          a tuple with four values (lap_time, max_speed, distance_covered, damage).

        Returns:
            Tuple[List, List]: A tuple containing two lists:
                - The first list is the Pareto front (a list of non-dominated individuals).
                - The second list contains the corresponding fitness values for the individuals in the Pareto front.
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

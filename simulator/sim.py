import random


def simulate(individual):
    """
    Simulate the performance of a car setup.
    
    Args:
        individual (list): The car setup to simulate.
    
    Returns:
        tuple: The fitness values (lap_time, max_speed, distance_covered, damage).
    """
    lap_time = random.uniform(120, 150) 
    max_speed = random.uniform(240, 290)
    distance_covered = random.uniform(5793, 57930)
    damage = random.uniform(0, 100)

    return lap_time, max_speed, distance_covered, damage
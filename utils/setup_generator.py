import random


def generate_random_setup(parameter_ranges: dict) -> dict:
    """
    Generates a random racing car setup with parameters defined in the given parameter ranges.

    Args:
        parameter_ranges (dict): A dictionary containing the parameter ranges.

    Returns:
        dict: A dictionary containing the random values for each car setup parameter, rounded to 4 decimals.
    """
    setup = {param: round(random.uniform(low, high), 4) for param, (low, high) in parameter_ranges.items()}
    
    return setup
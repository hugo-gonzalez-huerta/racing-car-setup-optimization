import json
import os

from typing import Dict


def load_config(file_path: str) -> Dict:
    """
    Loads the configuration settings from a JSON file.
    
    Args:
        file_path (str): Path to the configuration file.
    
    Returns:
        Dict: Configuration data as a dictionary.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If there is an error loading the JSON data.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found at: {file_path}")

    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from config file: {file_path}. Error: {e}")
    except PermissionError:
        raise PermissionError(f"Permission denied when trying to open the config file: {file_path}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the config file: {e}")

    return config


def load_param_ranges(config_file: str) -> Dict:
    """
    Loads the parameter ranges from a JSON configuration file.

    Args:
        config_file (str): Path to the JSON configuration file.

    Returns:
        dict: A dictionary containing the parameter ranges from the configuration file.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If there is an error loading the JSON data.
    """
    # Check if the file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Parameter ranges file not found at: {config_file}")

    try:
        with open(config_file, 'r') as file:
            parameter_ranges = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from parameter ranges file: {config_file}. Error: {e}")
    except PermissionError:
        raise PermissionError(f"Permission denied when trying to open the parameter ranges file: {config_file}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading the parameter ranges file: {e}")

    return parameter_ranges

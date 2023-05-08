import numpy as np


def generate_step_counts(total_steps, intervention_count, steepness=1):
    if intervention_count < 1:
        raise ValueError("Intervention count must be greater than 0.")

    # Create an array of logarithmically spaced numbers between 1 and the intervention_count
    logspace = np.logspace(0, np.log10(intervention_count), num=intervention_count)

    # Apply the steepness factor
    logspace = np.power(logspace, steepness)

    # Normalize the array such that the sum of its elements is 1
    normalized_logspace = logspace / np.sum(logspace)

    # Multiply the normalized array with the total number of steps and round it to integers
    step_counts = np.round(normalized_logspace * total_steps)

    # Adjust the last element to ensure the sum of step_counts is equal to total_steps
    step_counts[-1] = total_steps - np.sum(step_counts[:-1])

    return step_counts.astype(int)


def generate_epsilon_values(intervention_count, initial_epsilon, final_epsilon, steepness=1):
    if intervention_count < 1:
        raise ValueError("Intervention count must be greater than 0.")

    # Create an array of logarithmically spaced numbers between 1 and the intervention_count
    logspace = np.logspace(0, np.log10(intervention_count), num=intervention_count)

    # Apply the steepness factor
    logspace = np.power(logspace, steepness)

    # Normalize the array such that the last element is 1
    normalized_logspace = logspace / logspace[-1]

    # Reverse the array and scale it to the range between final_epsilon and initial_epsilon
    epsilon_values = (normalized_logspace[::-1] * (initial_epsilon - final_epsilon)) + final_epsilon

    # Calculate the adjustment factors for both the initial and final epsilon values
    init_epsilon_difference = initial_epsilon - epsilon_values[0]
    final_epsilon_difference = epsilon_values[-1] - final_epsilon

    # Adjust the epsilon values while maintaining the distribution
    epsilon_values[0] += init_epsilon_difference
    epsilon_values[-1] -= final_epsilon_difference
    epsilon_values[1:-1] += (init_epsilon_difference - final_epsilon_difference) / (intervention_count - 2)

    return epsilon_values


def create_step_filled_array(steps, count, total_steps):
    array = np.full(count - 1, steps)

    last_element = total_steps - steps * (count - 1)

    array = np.append(array, last_element)

    return array


def generate_epsilon_linear_array(start, end, count, steps, total_steps):
    step_filled_array = create_step_filled_array(steps, count, total_steps)
    epsilon_array = np.zeros(count)

    epsilon_array[0] = start
    epsilon_array[-1] = end

    total_epsilons = start - end
    for i in range(1, count - 1):
        epsilon_array[i] = start - total_epsilons * (step_filled_array[i - 1] / total_steps)
        start = epsilon_array[i]

    return epsilon_array

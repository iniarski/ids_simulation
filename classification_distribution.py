import random
import matplotlib.pyplot as plt
import numpy as np

import simulation

def plot_distribution(p : float, n_samples : int = 1000000):
    samples_xs = [random.random() for _ in range(n_samples)]
    values = list(map(lambda sample_x : simulation.get_model_output(p, sample_x), samples_xs))
    
    plt.hist(values, bins=100, density=True)

    plt.show()

def main():
    plot_distribution(0.95)

if __name__ == '__main__':
    main()
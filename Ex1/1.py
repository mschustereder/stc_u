import statistics
import numpy as np
import matplotlib.pyplot as plt

def draw_histogramm(lower_bound, upper_bound, num_of_intervals, samples):
    bound_range = upper_bound - lower_bound
    step = bound_range / num_of_intervals
    bins = np.arange(lower_bound, upper_bound + step, step)
    bin_counts = np.zeros(bins.shape)
    for sample in samples:
        for index, bin_upper_bound in enumerate(bins[1:]):
            if sample <= bin_upper_bound:
                bin_counts[index] += 1
                break

    relative_bin_counts = bin_counts / len(samples)
    
    plt.bar(bins, relative_bin_counts, width=step, align='edge', edgecolor='black', alpha=0.7)
    plt.xlim(lower_bound, upper_bound)
    plt.xlabel("Running times bins")
    plt.ylabel("Relative frequencies")
    plt.title("Running times of runners at the Graz Maration 2023")
    plt.show()


# The following sample summarizes the running times (minutes) of n = 25 runners at the Graz Marathon 2023.

sample = [ 128, 173, 184, 193, 201, 206, 209, 214, 219, 223, 229, 231, 234, 238, 242, 248, 253, 259, 265, 272, 283, 293, 299, 319, 353 ]

print(f"The sample: {sample}")

length = len(sample)

print(f"Sample length: {len(sample)}")

mean = sum(sample) / length

print(f"Mean: {mean}")

median = statistics.median(sample)

print(f"Median: {median}")

sample_array = np.array(sample)

mean_centered_array = sample_array - mean

standard_deviation = np.sqrt((1 / (length - 1)) * np.dot(mean_centered_array, mean_centered_array))

print(f"Standard deviation: {standard_deviation}")

draw_histogramm(100, 400, 6, sample_array)

draw_histogramm(100, 400, 12, sample_array)

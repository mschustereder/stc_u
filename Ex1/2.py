import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

sample_with_speed_limit = np.array([ 7,  9,  9,  9,  9,  9, 10, 11, 11, 12,
                            12, 12, 12, 12, 13, 13, 13, 14, 14, 15,
                            15, 15, 15, 15, 15, 16, 16, 16, 16, 16,
                            16, 17, 17, 17, 17, 17, 18, 19, 19, 19,
                            19, 20, 21, 21, 21, 22, 22, 22, 22, 22,
                            23, 23, 24, 24, 24, 25, 25, 25, 25, 25,
                            26, 26, 27, 29, 29, 32, 41, 41, 42 ])

print(len(sample_with_speed_limit))

sample_without_speed_limit = np.array([  8,  8,  9,  9,  9, 10, 11, 11, 12, 12,
                                13, 14, 14, 14, 14, 14, 14, 15, 15, 15,
                                15, 15, 15, 15, 16, 16, 16, 16, 17, 17,
                                17, 17, 17, 18, 18, 18, 18, 18, 18, 18,
                                18, 18, 19, 19, 19, 19, 19, 20, 20, 20,
                                20, 20, 20, 20, 21, 21, 21, 21, 21, 21,
                                21, 21, 22, 22, 22, 23, 23, 24, 24, 24,
                                24, 24, 25, 25, 25, 25, 26, 26, 26, 27,
                                27, 27, 27, 28, 28, 29, 29, 29, 29, 30,
                                30, 31, 31, 31, 31, 32, 32, 32, 34, 34,
                                35, 36, 36, 37, 37, 38, 39, 39, 40, 40,
                                42, 44, 47, 48, 49 ])

print(len(sample_without_speed_limit))

# the third and fourth central moments are called 'skewness' and ''kurtosis'

def get_mean(sample):
    n = len(sample)
    return sum(sample) / n

def get_likelihood_standard_deviation(sample):
    n = len(sample)
    mean = get_mean(sample)
    mean_centered_sample = sample - mean
    return np.sqrt((1 / n) * np.dot(mean_centered_sample, mean_centered_sample))

def get_skewness(sample):
    n = len(sample)
    mean = get_mean(sample)
    mean_centered_sample = sample - mean
    return ( (1 / n) * np.dot(np.multiply(mean_centered_sample, mean_centered_sample), mean_centered_sample) / (get_likelihood_standard_deviation(sample) ** 3) )

def get_kurtosis(sample):
    n = len(sample)
    mean = get_mean(sample)
    mean_centered_sample = sample - mean
    return ( (1 / n) * np.dot(np.multiply(np.multiply(mean_centered_sample, mean_centered_sample), mean_centered_sample), mean_centered_sample) / (get_likelihood_standard_deviation(sample) ** 4) )

skewness_with_limit = get_skewness(sample_with_speed_limit)
kurtosis_with_limit = get_kurtosis(sample_with_speed_limit)
skewness_without_limit = get_skewness(sample_without_speed_limit)
kurtosis_without_limit = get_kurtosis(sample_without_speed_limit)

print(f"skewness_with_limit: {skewness_with_limit}")
print(f"kurtosis_with_limit: {kurtosis_with_limit}")
print(f"skewness_without_limit: {skewness_without_limit}")
print(f"kurtosis_without_limit: {kurtosis_without_limit}")

print(f"skewness_with_limit using scipy: {stats.skew(sample_with_speed_limit)}")
print(f"kurtosis_with_limit using scipy: {stats.kurtosis(sample_with_speed_limit)}")
print(f"skewness_without_limit using scipy: {stats.skew(sample_without_speed_limit)}")
print(f"kurtosis_without_limit using scipy: {stats.kurtosis(sample_without_speed_limit)}")

plt.figure(figsize=(10, 6))
plt.boxplot([sample_with_speed_limit, sample_without_speed_limit], labels=[f"Sample with speed limit (n = {len(sample_with_speed_limit)})", f"Sample without speed limit (n = {len(sample_without_speed_limit)})"])
plt.title("Motorway accidents in Sweden with and without speed limit (boxplots).")
plt.ylabel("Number of accidents")
plt.show()

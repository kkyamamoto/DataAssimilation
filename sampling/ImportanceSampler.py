import numpy as np


def performResampling(unnormalizedWeights, originalSamples):
    nSamples = len(unnormalizedWeights)
    normalizedW = unnormalizedWeights / np.sum(unnormalizedWeights)
    # normalizedW=unnormalizedWeights
    histogram = np.full(nSamples, np.nan)
    occurrence = np.zeros(nSamples)
    cdf = np.full(nSamples, np.nan)

    sum = 0
    for i in range(nSamples):
        sum += normalizedW[i]
        cdf[i] = sum

    uniform = np.random.uniform(0, 1, nSamples)
    # uniform = np.random.uniform(0, np.sum(unnormalizedWeights), nSamples)

    for i in range(nSamples):
        n = 0
        while uniform[i] >= cdf[n]:
            n += 1

        histogram[i] = originalSamples[n]
        occurrence[n] += 1

    return histogram


def performResampling2D(unnormalizedWeights, originalSamples):
    nSamples = len(unnormalizedWeights)
    normalizedW = unnormalizedWeights / np.sum(unnormalizedWeights)
    # normalizedW=unnormalizedWeights
    histogram = np.full(nSamples, np.nan)
    occurrence = np.zeros(nSamples)
    cdf = np.full(nSamples, np.nan)

    sum = 0
    for i in range(nSamples):
        sum += normalizedW[i]
        cdf[i] = sum

    uniform = np.random.uniform(0, 1, nSamples)
    # uniform = np.random.uniform(0, np.sum(unnormalizedWeights), nSamples)

    for i in range(nSamples):
        n = 0
        while uniform[i] >= cdf[n]:
            n += 1

        histogram[i] = originalSamples[0][n]  # the originalSamples array
        # occurrence[n] += 1

    return histogram


def performResamplingND(dim, unnormalizedWeights, originalSamples):  # dim=dimension of the vector
    nSamples = len(unnormalizedWeights)
    normalizedW = unnormalizedWeights / np.sum(unnormalizedWeights)
    # normalizedW=unnormalizedWeights
    histogram = np.full((nSamples, dim), np.nan)
    occurrence = np.zeros(nSamples)
    cdf = np.full(nSamples, np.nan)

    sum = 0
    for i in range(nSamples):
        sum += normalizedW[i]
        cdf[i] = sum

    uniform = np.random.uniform(0, 1, nSamples)
    # uniform = np.random.uniform(0, np.sum(unnormalizedWeights), nSamples)

    for i in range(nSamples):
        n = 0
        while uniform[i] >= cdf[n]:
            n += 1

        histogram[i] = originalSamples[n]
        occurrence[n] += 1

    return histogram


# Equilvalent to the general version above with dim=3.
def performResampling3D(unnormalizedWeights, originalSamples):
    nSamples = len(unnormalizedWeights)
    normalizedW = unnormalizedWeights / np.sum(unnormalizedWeights)
    # normalizedW=unnormalizedWeights
    histogram = np.full((nSamples, 3), np.nan)
    occurrence = np.zeros(nSamples)
    cdf = np.full(nSamples, np.nan)

    sum = 0
    for i in range(nSamples):
        sum += normalizedW[i]
        cdf[i] = sum

    uniform = np.random.uniform(0, 1, nSamples)
    # uniform = np.random.uniform(0, np.sum(unnormalizedWeights), nSamples)

    for i in range(nSamples):
        n = 0
        while uniform[i] >= cdf[n]:
            n += 1

        histogram[i] = originalSamples[n]
        occurrence[n] += 1

    return histogram

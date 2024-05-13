import numpy as np
import misc_functions


def calculate_errors(sources, counter, mag_thresh):
    instrumental_magnitudes = np.array([s.inst_mags[counter] for s in sources])
    reference_magnitudes = np.array([s.ref_mag[0] for s in sources])
    jk_params = np.zeros((len(instrumental_magnitudes), 2))
    for i in range((len(instrumental_magnitudes))):
        x_sample = np.append(instrumental_magnitudes[:i], instrumental_magnitudes[i+1:])
        y_sample = np.append(reference_magnitudes[:i], reference_magnitudes[i+1:])
        jk_params[i] = misc_functions.ODR(x_sample, y_sample)
    mean_params = np.mean(jk_params, axis= 0)
    sig_params = np.std(jk_params, axis = 0)

    return [mean_params, sig_params]



import csv
import h5py
import numpy as np
import os


def lin_model(p, x):
    return p[0] * x + p[1]

def ODR(x_data, y_data):
    x_bar = np.mean(x_data)
    y_bar = np.mean(y_data)

    s_xx = 1/len(x_data) * np.sum((x_data - x_bar)**2)
    s_yy = 1/len(y_data) * np.sum((y_data - y_bar)**2)  
    s_xy = 1/len(x_data) * np.sum((x_data - x_bar) * (y_data - y_bar))

    b_0 = (s_yy - s_xx + np.sqrt((s_yy - s_xx)**2 + 4*s_xy**2))/(2*s_xy)
    b_1 = y_bar - b_0 * x_bar

    return [b_0, b_1]



    
def output_data(Nights, Sources, MAEs):
    output_name = input("Please enter name for output file with h5 extension. Ex: my_outputs.h5")
    path = os.path.join(os.getcwd(), output_name)
    with h5py.File(path, 'w') as hdf:
        for source in Sources:
                if not source.flagged:
                    group = hdf.create_group(f'Source_{source.source_id}')
                    total_obs = 0
                    maes = []
                    avg_mags = []
                    avg_errs = []
                    filter = []
                    for n, night in enumerate(Nights):
                        lb = int(total_obs)
                        ub = int(total_obs + len(night.aligned_images))
                        night_mags = source.calibrated_mags[lb:ub]
                        night_errs = source.errors[lb:ub]
                        group.create_dataset(f"Night_{n}_Mags", data = night_mags)
                        group.create_dataset(f"Night_{n}_Errs", data = night_errs)
                        group.create_dataset(f"Night_{n}_MJD_Times", data = night.mjd_times)
                        group.create_dataset(f"Night_{n}_UTC_Dates", data = night.date)

                        avg_mag = np.average(night_mags, weights = source.weights[lb:ub])
                        avg_err = np.mean(night_errs)

                        maes.append(np.mean(MAEs[lb:ub]))
                        avg_mags.append(avg_mag)
                        avg_errs.append(avg_err)
                        total_obs += len(night.aligned_images)
                        filter.append(night.obs_filter)
                    mag_data = group.create_dataset('Average_Mags', data = avg_mags)
                    group.create_dataset('Average_Errs', data = avg_errs)
                    position_string = source.position.to_string().split(' ')
                    parsed_position = [float(position_string[0]), float(position_string[1])]
                    mag_data.attrs['source_ra'] = parsed_position[0]
                    mag_data.attrs['source_dec'] = parsed_position[1]
                    mag_data.attrs['is_reference'] = source.is_reference
                    overall_avg = np.average(avg_mags, weights = avg_errs)
                    mag_data.attrs['Average_Mag'] = overall_avg
                    mag_data.attrs['filter'] = filter[0]
                    try:
                        mag_data.attrs['ref_mag'] = source.ref_mag
                        mag_data.attrs['ref_mag_err'] = source.ref_mag_err
                    except:
                        mag_data.attrs['ref_mag'] = '--'
                        mag_data.attrs['ref_mag_err'] = '--'

                    MAE = np.average(maes, weights = 1/(np.array(avg_errs)**2))
                    mag_data.attrs['MAE'] = MAE
                    
                    if len(Nights) > 1:
                        Chis = []
                        for i, m in enumerate(avg_mags):
                            chi_i = ((m - overall_avg)**2)/(avg_errs[i]**2)
                            Chis.append(chi_i)
                        dof = len(avg_mags) - 1
                        chi_dof = np.sum(Chis)/dof
                        source.add_chi(chi_dof)
                        mag_data.attrs['CHI_2'] = chi_dof
                    else:
                        Chis = []
                        for i, m in enumerate(source.calibrated_mags):
                            chi_i = ((m- overall_avg)**2) / (source.errors[i] **2)
                            Chis.append(chi_i)
                        dof = len(source.calibrated_mags) - 1
                        chi_dof = np.sum(Chis) / dof
                        source.add_chi(chi_dof)
                        mag_data.attrs["CHI_2"] = chi_dof
        
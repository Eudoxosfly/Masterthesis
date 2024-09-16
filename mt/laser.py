import numpy as np


def get_func_gen_settings(on_time, off_time, passes, focus_distance, legacy=False, verbose=False):
    spot_width, spot_length = get_spot_size(focus_distance)

    speed = np.round(spot_length / (on_time + off_time) / passes, 1)
    if legacy:
        speed = spot_length / (on_time + off_time) // passes
        spot_width = 2.8
    total_time_over_spot = spot_length / speed
    frequency = passes / total_time_over_spot
    duty_cycle = on_time / (on_time + off_time)
    total_time_on = total_time_over_spot * duty_cycle
    single_time_on = total_time_on / passes
    total_time_off = total_time_over_spot * (1 - duty_cycle)
    single_time_off = total_time_off / passes

    equivalent_energy = total_time_on * 30 / (spot_width * spot_length)

    settings = {"speed": speed, "frequency": frequency, "duty_cycle": duty_cycle,
                "total_time_over_spot": total_time_over_spot, "passes": passes, "set_on_time": on_time,
                "actual_on_time": single_time_on, "total_on_time": total_time_on, "set_off_time": off_time,
                "actual_off_time": single_time_off, "total_off_time": total_time_off,
                "equivalent_energy": equivalent_energy, }

    if verbose:
        legacy and print("WARNING: Legacy mode is on. Speed is rounded down to the nearest integer.")
        print("Settings:")
        print("Set on time: {:.0f} ms".format(on_time * 1e3))
        print("Set off time: {:.0f} ms".format(off_time * 1e3))
        print("Cycles: {}".format(passes))
        print("-" * 20)
        print("Speed: {} mm/s".format(speed))
        print("Frequency: {:.2f} Hz".format(frequency))
        print("Duty cycle: {:.0f} %".format(duty_cycle * 100))
        print("Total time over spot: {:.2f} s".format(total_time_over_spot))
        print("Equivalent energy: {:.2f} J/mm^2".format(equivalent_energy))
        print("-" * 20)
        print("Actual on time: {:.0f} ms".format(single_time_on * 1e3))
        print("Total on time: {:.0f} ms".format(total_time_on * 1e3))
        print("Actual off time: {:.0f} ms".format(single_time_off * 1e3))
        print("Total off time: {:.0f} ms".format(total_time_off * 1e3))
    return settings


def get_spot_size(distance):
    """Calculate the spot size (mm) in x and y direction for a given focus distance."""

    def get_size(coef, dist):
        return coef[0] * dist + coef[1]

    # coefficients from linear fitting
    x_coef = (0.0498, -0.294)
    y_coef = (0.0599, -0.285)

    x = np.round(get_size(x_coef, distance), 2)
    y = np.round(get_size(y_coef, distance), 2)

    return x, y

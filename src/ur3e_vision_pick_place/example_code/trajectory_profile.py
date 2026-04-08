import numpy as np

class TrajectoryProfile:
    def __init__(self):
        pass

    def trapezoid_time_scaled(self, L, vmax, amax, dt):
        t_acc = vmax / amax
        d_acc = 0.5 * amax * t_acc**2

        if 2 * d_acc > L:
            t_acc = np.sqrt(L / amax)
            t_flat = 0
            t_total = 2 * t_acc
        else:
            d_flat = L - 2 * d_acc
            t_flat = d_flat / vmax
            t_total = 2 * t_acc + t_flat

        t_list = []
        s_list = []

        t = 0
        while t <= t_total:
            if t < t_acc:
                s = 0.5 * amax * t**2
            elif t < t_acc + t_flat:
                s = d_acc + vmax * (t - t_acc)
            else:
                t_dec = t - (t_acc + t_flat)
                s = d_acc + (vmax * t_flat) + (vmax * t_dec - 0.5 * amax * t_dec**2)

            t_list.append(t)
            s_list.append(s)
            t += dt

        return np.array(t_list), np.array(s_list), t_total
    
    def trapezoid_multi(self, L_array, vmax, amax, dt):
        L_array = np.array(L_array)
        L_max = np.max(L_array)

        # Base trapezoid
        t_list, s_base, T = self.trapezoid_time_scaled(L_max, vmax, amax, dt)

        # Scale for each dimension
        s_scaled = []

        for L in L_array:
            if L > 1e-8:
                s_scaled.append(s_base * (L / L_max))
            else:
                s_scaled.append(np.zeros_like(s_base))

        return t_list, np.array(s_scaled), T
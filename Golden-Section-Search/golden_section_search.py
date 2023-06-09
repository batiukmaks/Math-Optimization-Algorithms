import numpy as np


class GoldenSectionSearch:
    def __init__(self, function):
        self.function = function

    def search(self, a_init, b_init, epsilon):
        c = (3 - np.sqrt(5)) / 2
        a, b, y, z, f_y, f_z = [], [], [], [], [], []
        y.append(a_init + c * (b_init - a_init))
        z.append(a_init + (1 - c) * (b_init - a_init))
        f_y.append(self.function(y[-1]))
        f_z.append(self.function(z[-1]))

        if f_y[-1] <= f_z[-1]:
            b.append(z[-1])
            a.append(a_init)
        else:
            a.append(y[-1])
            b.append(b_init)

        itr = 0
        while b[-1] - a[-1] > epsilon:
            itr += 1
            if f_y[-1] <= f_z[-1]:
                z.append(y[-1])
                f_z.append(f_y[-1])
                y.append(a[-1] + c * (b[-1] - a[-1]))
                f_y.append(self.function(y[-1]))
            else:
                y.append(z[-1])
                f_y.append(f_z[-1])
                z.append(a[-1] + (1 - c) * (b[-1] - a[-1]))
                f_z.append(self.function(z[-1]))

            if f_y[-1] <= f_z[-1]:
                a.append(a[-1])
                b.append(z[-1])
            else:
                a.append(y[-1])
                b.append(b[-1])

            print(
                f"{itr} | a = {a[-1]:.4f} | b = {b[-1]:.4f} | y = {y[-1]:.4f} | z = {z[-1]:.4f} | f(y) = {f_y[-1]:.4f} | f(z) = {f_z[-1]:.4f}"
            )

        if f_y[-1] <= f_z[-1]:
            x = y[-1]
            f_x = f_y[-1]
        else:
            x = z[-1]
            f_x = f_z[-1]

        return x, f_x

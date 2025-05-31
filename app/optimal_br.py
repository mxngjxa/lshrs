import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad as integrate
    

class OptimalBR:
    def false_positive(self, r, b):
        return integrate(lambda t: (1 - (1 - t**r)**b), 0, self.t0)[0]

    def false_negative(self, r, b):
        return integrate(lambda t: (1 - t**r)**b, self.t0, 1)[0]

    def br(self, n):
        self.t0 = 0.5

        best_fpr = float('inf')
        best_fnr = float('inf')
        best_b = None
        best_r = None

        # Iterate over possible values of r
        for r in range(1, n + 1):
            if n % r != 0:
                continue  # Skip if not evenly divisible

            # Calculate corresponding b
            b = n // r

            # Calculate false positive rate and false negative rate
            fpr = self.false_positive(r, b)
            fnr = self.false_negative(r, b)

            # Update best values if found
            if fpr < best_fpr or (fpr == best_fpr and fnr < best_fnr):
                best_fpr = fpr
                best_fnr = fnr
                best_b = b
                best_r = r

        return best_b, best_r

def main():
    testn = 256
    print(OptimalBR().br(testn))

if __name__ == "__main__":
    main()

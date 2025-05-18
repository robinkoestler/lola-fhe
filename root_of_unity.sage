## A fast primitive root of unity

def root(index, exponent=None, precision: int =53):
    """
    Calculate the primitive n-th root of unity. Using SageMath's zeta function for efficiency.
    """
    if exponent:
        exponent %= index
        return ComplexField(precision).zeta(index) ** exponent
    return ComplexField(precision).zeta(index)

def rootrange(root: ComplexNumber, rootindex: int, range: list):
    """
    Generates a list of values by raising the given root to the power of ((5**j) % rootindex) for each value in the given range.
    """
    return [root ** ((5**j) % rootindex) for j in range]

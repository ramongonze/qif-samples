"""Util functions."""

def float_equal(a,b):
    """Compares two float numbers wrt. to an epsilon."""
    eps = 10**(-6)
    return abs(a-b) <= eps

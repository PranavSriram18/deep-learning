
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]):
    if not a or not b or len(a[0]) != len(b):
        raise ValueError
    return [sum(a_ij * b_j for a_ij, b_j in zip(a_i, b)) for a_i in a]
def get_column(matrix: list[list[any]], column_num: int) -> list[any]:
    return [row[column_num] for row in matrix]

def dot(vector1, vector2):
    return sum(item1* item2 for item1, item2 in zip(vector1, vector2))

def matrixmul(matrix1:list[list[int|float]],
              matrix2:list[list[int|float]])-> list[list[int|float]]: 
    """ 
    Multiplies a mxn matrix and an nxk matrix, to produce mxk output
    """
    m = len(matrix1) 
    n = len(matrix1[0])
    n_2 = len(matrix2)
    k = len(matrix2[0])
    if (n!=n_2): return -1
    
   
    matrix3 = [dot(row, get_column(j)) for j in range(k) for row in matrix1]
    return matrix3
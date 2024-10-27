import random
import utils
from vector import Vector

class Matrix:

    def __init__(self, rows, columns):
        if rows < 0 or columns < 0:
            raise ValueError("rows and columns must be greater than 0")

        self.rows = rows
        self.columns = columns
        self.data = [[0] * columns for row in range(rows)]

    def randomise(self):
        for row_index in range(self.rows):
            self.data[row_index] = utils.make_random_list(self.columns)
        
        return self
    
    def set_data(self, data: list):
        self.rows = len(data)
        if self.rows == 0:
            self.columns = 0
        else:
            self.columns = len(data[0])

        self.data = data

    def set_column(self, column_index: int, data: list):
        if len(data) != self.rows:
            raise ValueError(f"The new column data (length: {len(data)}) must have as many elements as the number of rows in this matrix ({self.rows} rows)")
        
        for row_index in range(self.rows):
            self.data[row_index][column_index] = data[column_index]

    def __str__(self):
        result = f"\nMatrix dimensions: {self.rows} rows x {self.columns} columns\n"

        for row in self.data:
            for column in row:
                result += f"{column}, "
            result += "\n"
        result += "\n"

        return result

    def get_row(self, row_index):
        if row_index < 0:
            raise ValueError("Tried to get a negative row index from a matrix")
        if row_index >= self.rows:
            raise ValueError(f"Tried to get a non-existent row's data ({row_index} from a matrix with {self.rows} rows)")

        return self.data[row_index]

    def get_column(self, column_index):
        if column_index < 0:
            raise ValueError("Tried to get a negative column index from a matrix")
        if column_index >= self.columns:
            raise ValueError(f"Tried to get a non-existent column's data ({column_index} from a matrix with {self.columns} columns)")

        list = []
        for i in range(self.rows):
            list.append(self.data[i][column_index])

        return list

    def __add__(self, second):
        if self.rows != second.rows:
            raise ValueError(f"The rows in the first matrix ({self.rows}), must be equal to the rows in the second matrix ({second.rows})")
        if self.columns != second.columns:
            raise ValueError(f"The columns in the first matrix ({self.columns}), must be equal to the columns in the second matrix ({second.columns})")
        
        result_matrix = Matrix(0, 0)
        result_matrix.rows = self.rows
        result_matrix.columns = self.columns

        # Matrix addition is simply element-wise addition
        for row_index in range(self.rows):
            new_row = []
            for column_index in range(self.columns):
                new_row.append(self.data[row_index][column_index] + second.data[row_index][column_index])
            result_matrix.data.append(new_row)

        return result_matrix

    def multiply(self, second):
        # From the corresponding wikipedia page:
        # https://en.wikipedia.org/wiki/Matrix_multiplication
        # "The number of columns in the first matrix must be equal to the number
        # of rows in the second matrix"
        if self.columns != second.rows:
            raise ValueError(f"The number of columns in the first matrix ({self.columns}) is not equal to the number of rows in the second matrix ({second.rows})")

        if self.columns == 0:
            return self

        result_matrix = Matrix(0, 0)
        result_matrix.rows = self.rows
        result_matrix.columns = second.columns

        for i in range(result_matrix.rows):
            new_row = []

            for j in range(result_matrix.columns):
                # Each element in the new matrix is the dot product of two vectors:
                # row i in the first matrix
                # column j in the second matrix
                first_i_row = Vector(self.get_row(i))
                second_j_column = Vector(second.get_column(j))

                value_for_new_matrix = first_i_row.dot(second_j_column)
                new_row.append(value_for_new_matrix)

            result_matrix.data.append(new_row)

        return result_matrix

                
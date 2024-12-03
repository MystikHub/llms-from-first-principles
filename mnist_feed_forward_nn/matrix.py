"""Implements matrix storage and arithmetic operations"""

import utils
from PIL import Image


class Matrix:
    """Implements matrix storage and arithmetic operations"""

    def __init__(self, rows, columns):
        if rows < 0 or columns < 0:
            raise ValueError("rows and columns must be greater than 0")

        self.rows = rows
        self.columns = columns
        self.data = [[0] * columns for row in range(rows)]

    def randomise(self):
        """Randomises the data in each cell"""

        for row_index in range(self.rows):
            self.data[row_index] = utils.random_list(self.columns)

        return self

    def set_rows_columns(self, data: list):
        """Sets the matrix data (row-major order)"""

        self.rows = len(data)
        if self.rows == 0:
            self.columns = 0
        else:
            self.columns = len(data[0])

        self.data = data

    def set_column(self, column_index: int, data: list):
        """Sets the data of a column"""

        if len(data) != self.rows:
            raise ValueError(
                f"The new column data (length: {len(data)}) must have as many elements as the "
                f"number of rows in this matrix ({self.rows} rows)"
            )

        for row_index in range(self.rows):
            self.data[row_index][column_index] = data[column_index]

    def __str__(self):
        """Prints the matrix's dimensions followed by the data in each cell"""

        result = f"\nMatrix dimensions: {self.rows} rows x {self.columns} columns\n"

        for row in self.data:
            for column in row:
                result += f"{column}, "
            result += "\n"
        result += "\n"

        return result

    def get(self, row_index, column_index):
        """Returns the value at a given row and column"""

        if row_index < 0 or column_index < 0:
            raise ValueError("The row or column cannot be negative")
        if row_index >= self.rows:
            raise ValueError("Row index out of bounds")
        if column_index >= self.columns:
            raise ValueError("Column index out of bounds")

        return self.data[row_index][column_index]

    def get_row(self, row_index):
        """Returns the data at a specific row"""

        if row_index < 0:
            raise ValueError("Tried to get a negative row index from a matrix")
        if row_index >= self.rows:
            raise ValueError(
                f"Tried to get a non-existent row's data ({row_index} from a matrix with {self.rows} rows)"
            )

        return self.data[row_index]

    def get_column(self, column_index):
        """Returns the data at a specific column"""

        if column_index < 0:
            raise ValueError("Tried to get a negative column index from a matrix")
        if column_index >= self.columns:
            raise ValueError(
                f"Tried to get a non-existent column's data ({column_index} from a matrix with {self.columns} columns)"
            )

        column_list = []
        for i in range(self.rows):
            column_list.append(self.data[i][column_index])

        return column_list

    def __add__(self, second):
        """Returns the sum of two matrices as another matrix"""

        if self.rows != second.rows:
            raise ValueError(
                f"The rows in the first matrix ({self.rows}), "
                f"must be equal to the rows in the second matrix ({second.rows})"
            )
        if self.columns != second.columns:
            raise ValueError(
                f"The columns in the first matrix ({self.columns}), "
                f"must be equal to the columns in the second matrix ({second.columns})"
            )

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

    def __sub__(self, second):
        if self.rows != second.rows:
            raise ValueError(
                f"The rows in the first matrix ({self.rows}), "
                f"must be equal to the rows in the second matrix ({second.rows})"
            )
        if self.columns != second.columns:
            raise ValueError(
                f"The columns in the first matrix ({self.columns}), "
                f"must be equal to the columns in the second matrix ({second.columns})"
            )

        result_matrix = Matrix(0, 0)
        result_matrix.rows = self.rows
        result_matrix.columns = self.columns

        # Matrix subtraction is simply element-wise subtraction
        for row_index in range(self.rows):
            new_row = []
            for column_index in range(self.columns):
                new_row.append(self.data[row_index][column_index] - second.data[row_index][column_index])
            result_matrix.data.append(new_row)

        return result_matrix

    def multiply(self, second):
        """Multiplies this matrix with another and returns it as a new matrix"""

        if self.columns != second.rows:
            raise ValueError(
                f"The number of columns in the first matrix ({self.columns}) "
                f"is not equal to the number of rows in the second matrix ({second.rows})"
            )

        if self.columns == 0:
            return self

        result_matrix = Matrix(0, 0)
        result_matrix.rows = self.rows
        result_matrix.columns = second.columns

        for i in range(result_matrix.rows):
            new_row = []

            for j in range(result_matrix.columns):
                # Each element in the new matrix is the dot product of a column vector from this
                # matrix and a row vector from the second matrix

                value_for_new_matrix = 0
                for k in range(self.columns):
                    # element k from row i in this matrix
                    first_row_value = self.data[i][k]
                    # element k from column j in the other matrix
                    second_column_value = second.data[k][j]
                    value_for_new_matrix += first_row_value * second_column_value
                new_row.append(value_for_new_matrix)

            result_matrix.data.append(new_row)

        return result_matrix

    def elementwise_add(self, second):
        """Adds each value from the second matrix to this one"""

        if self.rows != second.rows or self.columns != second.columns:
            raise ValueError("The dimensions of the two matrices do not match")

        for i in range(self.rows):
            for j in range(self.columns):
                self.data[i][j] += second.data[i][j]

    def elementwise_multiply(self, second):
        """Elementwise multiplication, returned as a new matrix"""
        if self.rows != second.rows or self.columns != second.columns:
            raise ValueError("The dimensions of the matrices do not match")

        result_matrix = Matrix(self.rows, self.columns)
        for row_index in range(self.rows):
            for column_index in range(self.columns):
                result_matrix.data[row_index][column_index] = (
                    self.data[row_index][column_index] * second.data[row_index][column_index]
                )

        return result_matrix

    def multiply_scalar(self, multiplier):
        "Multiplies all the value in this matrix with the scalar value in a new matrix"
        result_matrix = Matrix(self.rows, self.columns)

        for i in range(self.rows):
            for j in range(self.columns):
                result_matrix.data[i][j] = self.data[i][j] * multiplier

        return result_matrix

    def divide_scalar(self, divisor):
        "Divides all the value in this matrix with the scalar value in a new matrix"
        result_matrix = Matrix(self.rows, self.columns)

        for i in range(self.rows):
            for j in range(self.columns):
                result_matrix.data[i][j] = self.data[i][j] / divisor

        return result_matrix

    def transpose(self):
        "Inverts the rows and columns"

        transpose_data = []
        for column_index in range(self.columns):
            transpose_data.append(self.get_column(column_index))

        # Swap the row and column properties
        tmp = self.rows
        self.rows = self.columns
        self.columns = tmp

        self.data = transpose_data

    def make_identity(self):
        "Sets the values in this matrix to 1 along the diagonal and 0 everywhere else"
        if self.rows != self.columns:
            raise ValueError(f"Cannot create an identity matrix with {self.rows} rows and {self.columns} columns")

        for row_index in range(self.rows):
            identity_row = [0] * self.columns
            identity_row[row_index] = 1
            self.data[row_index] = identity_row

    def visualise(self):
        """Renders the matrix values as a grayscale image using PIL"""
        if self.rows * self.columns < 1:
            raise ValueError("Cannot visualise a matrix with zero values")

        image = Image.new("L", (self.columns, self.rows), 255)
        # Data going into the image must be flattened (starting from the upper left corner)
        image_data = []
        for row in self.data:
            image_data += row

        # Matrix data can contain any negative or positive value. Normalise this from 0 -> 55
        minimum = min(image_data)
        maximum = max(image_data)
        if maximum == minimum:
            print(f"All values in this matrix are the same: {minimum}")
            return
        data_range = maximum - minimum
        normalised_data = [int((value - minimum) * (255 / data_range)) for value in image_data]

        image.putdata(normalised_data)
        image.show()

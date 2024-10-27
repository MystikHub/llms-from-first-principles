class Vector():

    def __init__(self, data: list):
        self.data = data
        self.n_elements = len(data)

    def __str__(self):
        result = f"\nVector dimensions: {self.n_elements} elements\n"

        for element in self.data:
            result += f"{element}, "
        result += "\n"

        return result
    
    def dot(self, other):
        if self.n_elements != other.n_elements:
            raise ValueError("Both vectors in a dot product operation must have the same number of elements")

        # The dot product of two vectors is the sum of multiplying each value from one vector with each value of the other vector
        dot_product = 0
        for element_index in range(self.n_elements):
            element_product = self.data[element_index] * other.data[element_index]
            dot_product += element_product
        
        return dot_product
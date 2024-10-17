class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = self.backward_impl
		self._prev = set(_children)
		self._op = _op
		
	def __repr__(self):
		return f"Value(data={self.data}, grad={self.grad})"

	def __add__(self, other):
		_children = (self, other)
		new_data = self.data + other.data
		new_op = "+"
		new_node = Value(new_data, _children, new_op)
		return new_node


	def __mul__(self, other):
		_children = (self, other)
		new_data = self.data * other.data
		new_op = "*"
		new_node = Value(new_data, _children, new_op)
		return new_node


	def relu(self):
		_children = (self,)
		new_data = 0 if self.data <= 0 else self.data
		op = 'relu'
		new_node = Value(new_data, _children, op)
		return new_node
	
	def backward(self):
		return self.backward_impl(1)
	
	def backward_impl(self, incoming_grad):
		self.grad += incoming_grad
		if self._op == '+':
			for child in self._prev:
				child._backward(self.grad)
		if self._op == '*':
			listprev = list(self._prev)
			child0 = listprev[0]
			child1 = listprev[1]
			newgrad0 = self.grad * child1.data
			newgrad1 = self.grad * child0.data
			child0._backward(newgrad0)
			child1._backward(newgrad1)
		if self._op == 'relu':
			if self.data > 0:
				list(self._prev)[0]._backward(self.grad)

		
        



	
		

	
		
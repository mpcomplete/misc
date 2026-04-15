import math

def maybeWrap(expr):
    return isinstance(expr, Expr) and expr or Constant(expr)

class Expr:
    # Foward-accumulation.
    # Returns a tuple (f(x), f'(x)) where f' is the partial derivative of f with respect to x.
    def evalAndDerive(self, x):
        return (0, 0)

    # Backward-accumulation (backpropagation).
    def backEval(self):
        #print(f"expr {self.val}")
        return self.val
    def backDerive(self, seed):
        pass
    def __add__(self, other):
        return Add(self, maybeWrap(other))
    def __radd__(self, other):
        return Add(maybeWrap(other), self)
    def __mul__(self, other):
        return Mul(self, maybeWrap(other))
    def __rmul__(self, other):
        return Mul(maybeWrap(other), self)

class Constant(Expr):
    def __init__(self, val):
        self.name = f"{val}"
        self.val = val
    def evalAndDerive(self, x):
        return (self.val, 0)

class Variable(Expr):
    def __init__(self, name, val):
        self.name = name
        self.val = val
        self.partial = 0
    def evalAndDerive(self, x):
        if isinstance(x, Variable) and self.name == x.name: return (self.val, 1)
        return (self.val, 0)
    def backDerive(self, seed):
        self.partial += seed
        print(f"variable {self.name}={self.val} of {seed} (now {self.partial})")

class Add(Expr):
    def __init__(self, a, b):
        self.name = "+"
        self.a = a
        self.b = b
    def evalAndDerive(self, x):
        (a, da) = self.a.evalAndDerive(x)
        (b, db) = self.b.evalAndDerive(x)
        return (a+b, da+db)
    def backEval(self):
        self.val = self.a.backEval() + self.b.backEval()
        #print(f"add {self.a.val} + {self.b.val} = {self.val}")
        return self.val
    def backDerive(self, seed):
        print(f"dadd: {self.a.name}={self.a.val} + {self.b.name}={self.b.val} with {seed}")
        self.a.backDerive(seed)
        self.b.backDerive(seed)

class Mul(Expr):
    def __init__(self, a, b):
        self.name = "*"
        self.a = a
        self.b = b
    def evalAndDerive(self, x):
        (a, da) = self.a.evalAndDerive(x)
        (b, db) = self.b.evalAndDerive(x)
        return (a*b, a*db + b*da)
    def backEval(self):
        self.val = self.a.backEval() * self.b.backEval()
        #print(f"mul {self.a.val} * {self.b.val} = {self.val}")
        return self.val
    def backDerive(self, seed):
        print(f"dmul: {self.a.name}={self.a.val} * {self.b.name}={self.b.val} with {seed}")
        self.a.backDerive(self.b.val * seed)
        self.b.backDerive(self.a.val * seed)

class Sin(Expr):
    def __init__(self, a):
        self.a = a
    def evalAndDerive(self, x):
        (a, da) = self.a.evalAndDerive(x)
        return (math.sin(a), da*math.cos(a))
    def backEval(self):
        self.val = math.sin(self.a.backEval())
        return self.val
    def backDerive(self, seed):
        self.a.backDerive(seed * math.cos(self.a.val))

def testFunc(fgen, compareFunc):
    for xval in [0, 1, 2, 3, 4, 5]:
        x = Variable('x', xval)
        f = fgen(x)
        (fx, dfx) = f.evalAndDerive(x)
        f.backEval()
        f.backDerive(1)
        print(f"f({xval}) = {fx} (backprop={f.val}); f'({xval}) = {dfx} (backprop={x.partial}); (compare to {compareFunc(xval)})")

# Function y = x+2
# derivative y' = 1
#testFunc(lambda x: Add(x, Constant(2)))

# Function y = 4x+2
# derivative y' = 1
#testFunc(lambda x: Add(x, Constant(2)))
# testFunc(lambda x: Add(Mul(x, Constant(4)), Constant(2)))

#w = Variable('w', 3)
#testFunc(lambda x: 4*w*x + 2, lambda x: (4*3*x + 2, 4*x))

# Function y = 7*x^2 + 2x + 3
# derivative y' = 14*x + 2
testFunc(lambda x: 7*x*x + 2*x + 3, lambda x: 14*x + 2)

#testFunc(lambda x: Sin(Mul(x, x)), lambda x: (math.sin(x*x), 2*x*math.cos(x*x)))
#testFunc(lambda x: Sin(x*x), lambda x: (math.sin(x*x), 2*x*math.cos(x*x)))

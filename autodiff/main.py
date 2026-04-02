class Expr:
    def evalAndDerive(self):
        return (0, 0)

class Constant(Expr):
    def __init__(self, val):
        self.val = val
    def evalAndDerive(self, x):
        return (self.val, 0)

class Variable(Expr):
    def __init__(self, name, val):
        self.name = name
        self.val = val
    def evalAndDerive(self, x):
        if isinstance(x, Variable) and self.name == x.name: return (self.val, 1)
        return (self.val, 0)

class Add(Expr):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def evalAndDerive(self, x):
        (a, da) = self.a.evalAndDerive(x)
        (b, db) = self.b.evalAndDerive(x)
        return (a+b, da+db)

class Mul(Expr):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def evalAndDerive(self, x):
        (a, da) = self.a.evalAndDerive(x)
        (b, db) = self.b.evalAndDerive(x)
        return (a*b, a*db + b*da)

def testFunc(fgen):
    for xval in [0, 1, 2, 3, 4, 5]:
        x = Variable('x', xval)
        f = fgen(x)
        (fx, dfx) = f.evalAndDerive(x)
        print(f"f({xval}) = {fx}; f'({xval}) = {dfx}")

# Function y = x+2
# derivative y' = 1
#testFunc(lambda x: Add(x, Constant(2)))

# Function y = 4x+2
# derivative y' = 1
#testFunc(lambda x: Add(x, Constant(2)))
testFunc(lambda x: Add(Mul(x, Constant(4)), Constant(2)))

# Function y = 7*x^2 + 2x + 3
# derivative y' = 14*x + 2
# eval at 1, 2, 3, 4

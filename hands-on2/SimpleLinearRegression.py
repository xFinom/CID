from operator import mul

class SimpleLinearRegression:
    def __init__(self, x: list, y: list) -> None:
        self.x = x
        self.y = y

        self.n = len(x)

        self.slope = None
        self.intercept = None

    def train(self) -> tuple:
        self.calculateSlope()
        self.calculateIntercept()

        return (self.intercept, self.slope)
    
    def predict (self, x: float) -> float:
        return x * self.slope + self.intercept
    
    def calculateSlope(self) -> None:
        numerator = self.n * sum(map(mul, self.x, self.y)) - sum(self.x) * sum(self.y)

        self.slope = numerator / self.calculateDenominator()

    def calculateIntercept(self) -> None:
        numerator = sum(map(mul, self.x, self.x)) * sum(self.y) - sum(self.x) * sum(map(mul, self.x, self.y))

        self.intercept = numerator / self.calculateDenominator()
    
    def calculateDenominator(self) -> float:
        return self.n * sum(map(mul, self.x, self.x)) - sum(self.x)**2
    
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.score= marks
    def avg(self):
        return sum(self.score)/len(self.score)
s1= Student("malik",[45, 50, 55])
print(s1.avg())         
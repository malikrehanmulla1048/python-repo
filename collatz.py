import sys
def collatz(num):
    if num%2==0:
        return num//2
    else:
        return 3*num+1
num1=int(input("enter a number:"))
print(num1,end=" ")
while True:
    num2=collatz(num1)
    print(num2,end=" ")
    if num2==1:
        sys.exit()
    else:
        num1=num2
        
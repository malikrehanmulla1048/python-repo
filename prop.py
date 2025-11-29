# write a python program for prime numbers
num=int(input("enter the number till where you want the prime numbers:"))
li=[]
if  num>1:
    for i in range(2,num+1):
        for j in (2,i/2):
            print("@")
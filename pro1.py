print("hello")
def prime_numbers(num):
    li=[]
    if  num>3:
        for i in range(2,num+1):
            for j in range(2,(i//2)+1):
                if i%j==0:
                    break
            else:
                li.append(i)
    elif num==2:
        li.append(2)
    elif num==3:
        li.append(2)
        li.append(3)
    else:
      print("no prime numbers")
    for k in li:
        print(k,end=" ")
#num=int(input("enter the number till where you want the prime numbers:"))
#prime_numbers(num)
import numpy  as np
def array_operations():
    array1=np.array([[1,2],[3,6]])
    print(array1)
    print("shape of the array:",array1.shape)
    A=array1.T
    print("tanspose of the array:\n",A)
    B=array1+A
    print("addition of two arrays:\n",B)
    C=array1@A
    print("multiplication of two arrays:\n",C)
def list_operations():
    nm=int(input("enter the number of elements in the list:"))
    l2=[]
    for i in range(nm):
        ele=int(input("enter the element:"))
        l2.append(ele)
    m=int(input("enter the element to be entered in the list:"))
    l2.append(m)
    r=int(input("enter the value to be removed from the list:"))
    if r in l2:
        l2.remove(r)
    else:
        print("element is  not present in the list")
    print("the sorted list is :\n",l2.sort())
    print("the element to be popped",l2.pop())
    print("the final list is :\n",l2)
    array_operations()
list_operations()
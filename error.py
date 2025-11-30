# def spam():
#     print(eggs) /"here it gave the output as error"/
#     eggs="spam local"
eggs="global spam"
# spam()
def spam():
    # print(eggs)
    eggs="spam local"
    print(eggs)
spam()
print(eggs)
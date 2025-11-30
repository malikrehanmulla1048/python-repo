# def spam():
#     print(eggs) /"here it gave the output as error"/?//name error//?
#     eggs="spam local"
# eggs="global spam"
# spam()
def spam():
    eggs="spam local"
    bacon()
    print(eggs)
def bacon():
    eggs="bacon local"
    shell="white"
    print(eggs)
spam()
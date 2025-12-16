import logging 
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')
def factorial(n):
    logging.debug("start of the factorial")
    total=1
    for i in range(1,n+1):
        total*=i
    logging.debug("end of the factorial of %s", n)
    return total
print(factorial(5))
logging.debug("end of the program")
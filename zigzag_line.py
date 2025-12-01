import sys, time
try:
    while True:
        for i in range(1,9):
            print("-"*(i*i))
            time.sleep(0.1)
        for i in range(7,0,-1):
            print("-"*(i*i))
            time.sleep(.1)
            # if i==1:
            #     sys.exit()
except KeyboardInterrupt:
    sys.exit()
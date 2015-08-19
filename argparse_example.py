import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-g',
                    dest='gp',action = 'store_true',
                    help='generate_prediction')

parser.add_argument('-e', 
                    dest='ec',action = 'store_true',
                    help='editorial_calendar')

parser.add_argument('-', 
                    dest='rt',action = 'store_true',
                    help='rising_term')

results = parser.parse_args()

if __name__=='__main__':
    if results.gp:
        print 'prediction generated'
    if results.ec:
        print 'editorial calendar generated'
    if results.rt:
        print 'rising terms generated'
        


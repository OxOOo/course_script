# -*- encoding: utf-8 -*-

# http://www.fateadm.com/index.html

import os, sys
sys.path.append(os.curdir)
from fateadm_api import FateadmApi

f = open('fateadm_auth', 'r')
PD_ID = f.readline().strip()
PD_KEY = f.readline().strip()
f.close()
if len(PD_ID) == 0 or len(PD_KEY) == 0:
    print 'you need to prepare a fateadm_auth file, see README.md for more details'
    sys.exit(1)

api = FateadmApi(None, None, PD_ID, PD_KEY)

def getBalance():
    return api.QueryBalcExtend()

def decode(filename):
    return api.PredictFromFileExtend('30500', filename).upper()

if __name__ == '__main__':
    print api.QueryBalcExtend()
    print api.PredictFromFileExtend('30500', 'temp.jpg')

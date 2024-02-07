from matplotlib import pyplot as plt
import numpy as np
import json
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file_name")
    args = parser.parse_args()
    with open(args.result_file_name, "r") as f:
        results = json.load(f)
    print(results[:][0])
    f1 = np.array(results)[:,0]
    f2 = np.array(results)[:,1]
    plt.plot(f1,f2,'.',color='r')
    plt.show()
    plt.savefig('a.png')
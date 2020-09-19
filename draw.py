import os
import matplotlib.pyplot as plt
import numpy as np

def draw(directory: str):
    cpts, npns, times = [], [], []
    for file in os.listdir(directory):
        if file.endswith(".out"):
            file_path = os.path.join(directory, file)
            print(file_path)
            params = file.split('-')
            n_nodes, cpt, npn, partition = int(params[0].replace("node", "")), int(params[1]), int(params[3]), params[5].split('.')[0]
            cpts.append(cpt)
            npns.append(npn)
            print("    ", n_nodes, cpt, npn, partition)
            with open(file_path) as f:
                time = int(f.readline().split(' ')[1])
                print("   ", time)
                times.append(time)
    
    # plot walltime
    x, y = cpts, times
    xy = zip(x, y)
    xy = sorted(xy)
    x_sorted = [i for i, _ in xy]
    y_sorted = [j for _, j in xy]
    plt.plot(x_sorted, y_sorted)
    plt.title("#nodes: " + str(n_nodes) + ", partitions: " + partition)
    plt.xlabel("cores per task")
    plt.ylabel("micro seconds")
    plt.show()


    # plot speedup
    with open(directory + "sequential.txt") as f:
        sequential_time = int(f.readline().split(' ')[1])
        print("   sequential time: ", sequential_time)
    speedup = [sequential_time / j for j in y_sorted]
    print(x_sorted)
    print(speedup)
    best_cpt = x_sorted[np.argmax(speedup)]
    plt.plot(x_sorted, speedup)
    plt.title("#nodes: " + str(n_nodes) + ", partitions: " + partition + ", best cpt: " + str(best_cpt))
    plt.xlabel("cores per task")
    plt.ylabel("speedup")
    plt.show()


if __name__ == "__main__":
    draw("./testing3/")

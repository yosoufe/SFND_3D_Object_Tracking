import subprocess
import re
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pickle

from multiprocessing import Pool
import os

with_cuda = False

detector_types = ["SHITOMASI", "HARRIS", "FAST",
                   "BRISK", "ORB", "AKAZE", "SIFT"]

matcher_types = ["MAT_BF", "MAT_FLANN"]
descriptor_types = ["BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"]
selector_types = ["SEL_NN", "SEL_KNN"]

if with_cuda:
    detector_types = detector_types + ["FAST_CUDA", "ORB_CUDA" ]
    matcher_types = matcher_types + ["MAT_BF_CUDA"]
    descriptor_types = descriptor_types + ["ORB_CUDA"]

all_lists = [detector_types, matcher_types, descriptor_types, selector_types]

excluded_pairs = [
    ["SHITOMASI", "AKAZE"],
    ["HARRIS", "AKAZE"],
    ["FAST", "AKAZE"],
    ["FAST_CUDA", "AKAZE"],
    ["BRISK", "AKAZE"],
    ["ORB", "AKAZE"],
    ["ORB_CUDA", "AKAZE"],
    ["SIFT", "AKAZE"],
    ["SIFT", "ORB"],
    ["SIFT", "ORB_CUDA"],
]

EXECUTABLE = "./3D_object_tracking"
WORKING_DIR = "build"

FLOATING_PATTERN_REGEX = "([0-9]*.[0-9]+)"


def run_command(detector_type="ORB",
                descriptor_type="BRISK",
                matcher_type="MAT_BF",
                selector_type="SEL_NN",
                minReflectiveness = 0.2,
                focus_on_proceding_vehicle=False,
                top_view=False,
                camera_view=False,
                start_index = 0,
                end_index = 17):
    """ 
    runs the command and returns the output line by line in a for loop
    """
    command = [EXECUTABLE,
               "--detector_type", detector_type,
               "--matcher_type", matcher_type,
               "--descriptor_type", descriptor_type,
               "--selector_type", selector_type,
               "-r", str(minReflectiveness),
               "-s", str(start_index),
               "-e", str(end_index)
               ]
    if focus_on_proceding_vehicle:
        command = command + ["-f"]

    if top_view:
        command = command + ["--top_view"]

    if camera_view:
        command = command + ["--camera_view"]
    
    print ("executed command: ", " ".join(command), command)

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=WORKING_DIR, universal_newlines=True)
    for line in iter(proc.stdout.readline, ""):
        yield line
    proc.stdout.close()
    return_code = proc.wait()

def plot_ttc_lidar(file):
    results = pickle.load( open( file, "rb" ) )
    results = results["res"]
    fig, ax = plt.subplots()
    fig.set_size_inches([10*2,8*2])
    legends = []
    for ttc, thre in results:
        legends.append("Threshold {}".format(thre))
        ax.plot(range(2, 2+len(ttc)),ttc)
    
    ax.set_ylim([-100,150])
    ax.legend(legends, loc='upper right')

    ax.set(xlabel='frame', ylabel='TTC (s)',
        title='TTC from LIDAR with different reflectiveness lower threshold')
    ax.grid()
    plt.show()
    fig.savefig("task_5.png")

def plot_ttc_camera(file):
    results = pickle.load( open( file, "rb" ) )
    results = results["res"]
    fig, ax = plt.subplots()
    fig.set_size_inches([10*2,8*2])
    legends = []
    for ttc, detector_t, desc_t in results:
        legends.append("Det, Mat: {}, {}".format(detector_t, desc_t))
        ax.plot(range(2, 2+len(ttc)),ttc)
    
    ax.set_ylim([-100,150])
    ax.legend(legends, loc='upper right')

    ax.set(xlabel='frame', ylabel='TTC (s)',
        title='TTC from CAMERA with different Detector and Descriptor')
    ax.grid()
    plt.show()
    fig.savefig("task_6.png")

def task_5_process(threshold):
    lidar_ttc_re = re.compile(".*TTC from Lidar: (.+).*")
    runner = run_command(minReflectiveness=threshold, end_index = 74)
    ttc = []
    for line in runner:
        match_ext = lidar_ttc_re.match(line)
        # print(line)
        if match_ext:
            # print(match_ext.group(1))
            ttc.append(float(match_ext.group(1)))
    return ttc, threshold

def task_5():
    print("Task 5: Run the executable with different reflective threshold")
    n = 10
    ttcs = []
    thresholds = [x/n for x in range (0,n)]
    # run different parameters on different processes in parallel.
    with Pool(min(os.cpu_count(),20)) as pool:
        res = pool.map(task_5_process, thresholds)
        print(res)
        res_to_save = {"res": res }
        file = "results/lidar_ttcs_vs_ths.p"
        pickle.dump( res_to_save, open( file, "wb" ) )
        plot_ttc_lidar(file)

def task_6_process(args):
    camera_ttc_re = re.compile(".*TTC from Camera: (.+).*") 
    detector_t = args[0]
    desc_t = args[1]
    runner = run_command(detector_type=detector_t,
                         descriptor_type=desc_t,
                         minReflectiveness=0.2,
                         start_index = 0,
                         end_index = 74)
    ttc = []
    for line in runner:
        match_ext = camera_ttc_re.match(line)
        if match_ext:
            ttc.append(float(match_ext.group(1)))
    return ttc, detector_t, desc_t

def task_6():
    print("Task 6: Run the executable with different detector descriptors")
    different_combinations = []
    for det in detector_types:
        for desc in descriptor_types:
            if [det, desc] in excluded_pairs:
                print([det, desc], "skipped")
                continue
            different_combinations.append([det, desc])
    with Pool(min(os.cpu_count(),20)) as pool:
        res = pool.map(task_6_process, different_combinations)
        print(res)
        res_to_save = {"res": res }
        file = "results/camera_ttcs_vs_types.p"
        pickle.dump( res_to_save, open( file, "wb" ) )
        plot_ttc_camera(file)


def tes_reg_exp():
    test_text = ["        TTC from Lidar: 8.813924       ", "        TTC from Lidar: -inf     "]
    camera_ttc_re = re.compile(".*TTC from Lidar: (.+).*")
    for te in test_text:
        match_ext = camera_ttc_re.match(te)
        if match_ext:
            print(match_ext.group(1), float(match_ext.group(1)))
    exit(0)


if __name__ == "__main__":
    # tes_reg_exp()
    # plot_ttc_lidar("results/lidar_ttcs_vs_ths.p")
    # exit(0)
    parser = argparse.ArgumentParser(description='A runner to run the executable inside' +
                                     'the build directory with different parameters')
    parser.add_argument('--tasks',type=int, choices=[5, 6], required=True, nargs='+',
                        help='choose the task number')

    args = parser.parse_args()

    for task_number in args.tasks:
        func_name = "task_{}".format(task_number)
        eval(func_name)()

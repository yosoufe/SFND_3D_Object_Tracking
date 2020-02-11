import subprocess
import re
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pickle

from multiprocessing import Pool
import os

with_cuda = True

detector_types = ["SHITOMASI", "HARRIS", "FAST",
                  "FAST_CUDA", "BRISK", "ORB", "ORB_CUDA", "AKAZE", "SIFT"]
matcher_types = ["MAT_BF", "MAT_FLANN", "MAT_BF_CUDA"]
descriptor_types = ["BRISK", "BRIEF", "ORB",
                    "ORB_CUDA", "FREAK", "AKAZE", "SIFT"]
selector_types = ["SEL_NN", "SEL_KNN"]
all_lists = [detector_types, matcher_types, descriptor_types, selector_types]

EXECUTABLE = "./3D_object_tracking"
WORKING_DIR = "build"

FLOATING_PATTERN_REGEX = "([0-9]*.[0-9]+)"

NUMBER_OF_EXPECTED_RESULTS = 17


def run_command(detector_type="ORB",
                descriptor_type="BRISK",
                matcher_type="MAT_BF",
                selector_type="SEL_NN",
                minReflectiveness = 0.2,
                focus_on_proceding_vehicle=False,
                top_view=False,
                camera_view=False):
    """ 
    runs the command and returns the output line by line in a for loop
    """
    command = [EXECUTABLE,
               "--detector_type", detector_type,
               "--matcher_type", matcher_type,
               "--descriptor_type", descriptor_type,
               "--selector_type", selector_type,
               "-r", str(minReflectiveness)
               ]
    if focus_on_proceding_vehicle:
        command = command + ["-f"]

    if top_view:
        command = command + ["--top_view"]

    if camera_view:
        command = command + ["--camera_view"]
    
    print ("executed command: ", command)

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=WORKING_DIR, universal_newlines=True)
    for line in iter(proc.stdout.readline, ""):
        yield line
    proc.stdout.close()
    return_code = proc.wait()


def save_plot_ttc_lidar(ttc: list, thresholds):
    # Data for plotting
    frame_number = range(2, len(ttc)+2)

    fig, ax = plt.subplots()
    ax.plot(frame_number, ttc)

    ax.set(xlabel='frame number', ylabel='TTC from Lidar (s)',
        title='TTC calculations for different thresholds')
    ax.grid()
    th = thresholds
    fig.savefig("t5t{}.png".format(th))
    plt.show()



def task_5():
    print("Task 5: Run the executable with different reflective threshold")
    # camera_ttc_re = re.compile(".*TTC from Lidar: {}.*".format(FLOATING_PATTERN_REGEX))
    camera_ttc_re = re.compile(".*TTC from Lidar: (.+).*")
    n = 10
    ttcs = []
    thresholds = []

    # pool = Pool(os.cpu_count())

    for threshold in [x/n for x in range (0,n)]:
        runner = run_command(minReflectiveness=threshold)
        ttc = []
        for line in runner:
            match_ext = camera_ttc_re.match(line)
            # print(line)
            if match_ext:
                # print(match_ext.group(1))
                ttc.append(float(match_ext.group(1)))
        assert (len(ttc) == NUMBER_OF_EXPECTED_RESULTS), "len(ttc): {} \nttc: {}".format(len(ttc), ttc)
        ttcs.append(ttc)
        thresholds.append(threshold)
        print(threshold, ttc)

    res_to_save = {"res": ttcs , "thresholds": thresholds}
    pickle.dump( res_to_save, open( "results/liadar_ttcs_vs_ths.p", "wb" ) )
        # save_plot_ttc_lidar(ttc, threshold)




def task_6():
    print("task 6")




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
    parser = argparse.ArgumentParser(description='A runner to run the executable inside' +
                                     'the build directory with different parameters')
    parser.add_argument('--tasks',type=int, choices=[5, 6], required=True, nargs='+',
                        help='choose the task number')

    args = parser.parse_args()

    for task_number in args.tasks:
        func_name = "task_{}".format(task_number)
        eval(func_name)()

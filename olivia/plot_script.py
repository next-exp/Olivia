import sys

from olivia.histogram_plot_functions import plot_histograms_from_file

if __name__ == "__main__":

    monfile  = sys.argv[1]
    outpath  = sys.argv[2]
    plot_histograms_from_file(monfile,out_path=outpath)

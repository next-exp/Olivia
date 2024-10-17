import sys

from olivia.histogram_functions import join_histograms_from_files
from olivia.histogram_plot_functions import plot_histograms_from_file

if __name__ == "__main__":

    monfile  = sys.argv[1]
    monfiles = sys.argv[2:]
    
    join_histograms_from_files([monfile]+monfiles, join_file=monfile)

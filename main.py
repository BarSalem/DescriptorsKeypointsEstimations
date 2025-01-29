from descriptors_comparison import compare_images as descriptor_experiment
from detectors_comparison import compare_images as detector_experiment


def main():
    detector_experiment()
    descriptor_experiment()

if __name__ == '__main__':
    main()
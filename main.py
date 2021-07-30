#Use 
'''
    Camera, Video: python main.py realtime
    Image_path: python main.py image --path images_test/image.jpg
'''

#import library 
import argparse
from webcam_realtime import realtime_webcam
from detect_image import detect_image
import tensorflow as tf

# for running realtime emotion detection
def run_realtime_emotion():
    realtime_webcam()

# to run emotion detection on image saved on disk
def run_detection_path(path):
    detect_image(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("func_name", type=str,
                        help="Select a function to run. <emo_realtime> or <emo_path>")
    parser.add_argument("--path", default=None, type=str,
                        help="Specify the complete path where the image is saved.")
    # parse the args
    args = parser.parse_args()

    #print('****ARGS: ' + str(args))

    if args.func_name == "realtime":
        run_realtime_emotion()
    elif args.func_name == "image":
        run_detection_path(args.path)
    else:
        print("Usage: python main.py <function name>")

if __name__ == '__main__':
    tf.debugging.set_log_device_placement(True)
    # Place tensors on the CPU
    with tf.device('/CPU:0'):
        main()
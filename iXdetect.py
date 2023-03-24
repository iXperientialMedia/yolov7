from detection_helpers import *
from tracking_helpers import *
from bridge_wrapper import *
from PIL import Image



def detect():
    detector = Detector(classes = [0]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
    detector.load_model('./yolov7x.pt',) # pass the path to the trained weight file
    # Initialise  class that binds detector and tracker in one class
    tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/market1501.pb", detector=detector)

    # output = None will not save the output video
    tracker.track_video("./inference/videos/wedding_1.mp4", output="./runs/detect/exp/wedding_1.avi", show_live = False, skip_frames = 0, count_objects = True, verbose=1)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    detect()
from detector import *

classFile = "openimagesv7.names"
videoPath = 0 # for webcam
threshold = 0.5

detector = Detector()
detector.readClasses(classFile)
detector.loadModel()
# detector.predictImage(imagePath, threshold)
detector.predictVideo(videoPath, threshold)

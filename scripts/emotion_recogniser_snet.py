#!/usr/bin/python
from cv_bridge import CvBridge
import subprocess
import base64
import rospy
import cv2
from recognisers.emotion import EmotionRecogniser
from ros_people_model.srv import Emotion
from ros_people_model.srv import EmotionResponse


def handle_request(req):
    image = bridge.imgmsg_to_cv2(req.image, "8UC3")
    # Convert image to some json format
    retval, buff = cv2.imencode('.jpg', image)
    retval, img_64 = base64.b64encode(buff)
    req = '{"image_type": "jpg", "image" : "' + str(img_64) + '"}'
    with open('image.json', 'wt') as f:
        f.write(str(req))
    p = subprocess.Popen(["snet", "mpe-client", "call_server", "0x38506005d6b25386aac998448ae5eb48f87f4277", "0","10","34.216.72.29:6205", "EmotionRecognition", "classify", "image.json"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    print(result)
    results = str(result)
    print(err)

    preprocessed_emotions = results[results.find('predictions'):results.find('bounding_boxes')].replace('predictions: ','').split("\\n")
    print(preprocessed_emotions)
    EMOTIONS = {
        "anger" : 0,
        "disgust" : 1,
        "fear": 2,
        "happy": 3,
        "sad": 4,
        "surprise": 5,
        "neutral": 6
    }
    emotions = []
    for emotion in preprocessed_emotions:
        p_emotion = emotion.replace('"','')
        if p_emotion != '':
            emotions.append(EMOTIONS[p_emotion])

    return EmotionResponse(emotions)


if __name__ == "__main__":

    try:
        rospy.init_node('emotion_recogniser_server_singnet')

        bridge = CvBridge()
        srv = rospy.Service('emotion_recogniser_snet', Emotion, handle_request)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass

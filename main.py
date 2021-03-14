import argparse


def train(debug: bool):
    from mask_detection_trainer import MaskDetectionTrainer
    print('Train Model')

    # Train args
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset. Having with_mask and without_mask dir")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str,
                    default="mask_detector.model",
                    help="path to output face mask detector model")

    args = vars(ap.parse_known_args()[0])

    detectorTrainer = MaskDetectionTrainer(
        args['dataset'], args['plot'], args['model'], debug)
    detectorTrainer.train_and_save()


def predict(debug: bool):
    print('Predict image')
    from mask_detector import MaskDetector

    ap = argparse.ArgumentParser()
    # Predict args
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-f", "--face", type=str,
                    default="face_detector",
                    help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
                    default="mask_detector.model",
                    help="Path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="Minimum probability to filter weak detections")

    args = vars(ap.parse_known_args()[0])

    detector = MaskDetector(
        args['image'], args['face'], args['model'], args['confidence'], debug)

    detector.predict()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    # Helper args
    ap.add_argument('--debug', nargs='?', const=True,
                    required=False, default=False)

    # Train | Predict check
    ap.add_argument('--train', nargs='?', const=train, dest='fun')
    ap.add_argument('--predict', nargs='?', const=predict, dest='fun')

    args = vars(ap.parse_known_args()[0])

    args['fun'](args['debug'])

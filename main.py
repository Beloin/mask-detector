import argparse


def train(args):
    None


def predict(args):
    None


ap = argparse.ArgumentParser()

# Train | Predict check
ap.add_argument('--train', const=train, dest='fun')
ap.add_argument('--predict', const=predict, dest='fun')

# Helper args
ap.add_argument('--debug', required=False, default=False, type=bool)
ap.add_argument('-h', '--help', required=False, type=bool)

# Train args
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

# Predict args
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to output face mask detector model")

args = vars(ap.parse_args())

if args['help']:
    ap.print_help()
else:
    # Roda a função pedida.
    args.fun(args)

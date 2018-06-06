import numpy as np

from keras.models import load_model

from util.preprocessor import unnormalize_car
from util.crop import show


def main():
    ball_0s = np.load('data/ball_0s.npy').astype(np.float16)
    car_0s = np.load('data/car_0s.npy').astype(np.float16)
    ball_1s = np.load('data/ball_1s.npy').astype(np.float16)
    car_1s = np.load('data/car_1s.npy').astype(np.float16)

    models = [
        'A-0.933.hdf5',
        'B-0.935.hdf5',
        'C-0.904.hdf5',
        'D-0.940.hdf5',
        'E-0.906.hdf5',
        'F-0.960.hdf5',
    ]

    preds = dict()

    for model in models:
        letter = model[0]

        model = load_model('model_data/reward_function/' + model)
        predictions_0 = (model.predict([ball_0s, car_0s]) >= 0.5).astype(np.uint8)
        predictions_1 = (model.predict([ball_1s, car_1s]) >= 0.5).astype(np.uint8)
        preds[letter] = (predictions_0, predictions_1)

    for m in preds.keys():
        print('Checking stats for model {}'.format(m))
        predictions_0, predictions_1 = preds[m]

        fp = predictions_0.sum()
        tn = predictions_0.size - fp

        tp = predictions_1.sum()
        fn = predictions_1.size - tp

        accuracy = (tp + tn) / (fp + tn + tp + fn)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        neg_precision = tn / (tn + fn)
        neg_recall = tn / (tn + fp)

        print('Accuracy: {}'.format(accuracy))

        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))

        print('Negative Precision: {}'.format(neg_precision))
        print('Negative Recall: {}'.format(neg_recall))

        f1 = 2 * precision * recall / (precision + recall)
        nf1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall)

        print('F1 score: {}'.format(f1))
        print('Negative F1 score: {}'.format(nf1))
        print()

    # print("displaying false positives")
    # num = 0
    # for i, pred in enumerate(predictions_0):
    #     if pred != 0:
    #         num += 1
    #         print('Showing false positive {}/{}'.format(num, fp))
    #         show(unnormalize_car(car_0s[i]).astype(np.uint8))
    #
    # print("displaying false negatives")
    # num = 0
    # for i, pred in enumerate(predictions_1):
    #     if pred != 1:
    #         num += 1
    #         print('Showing false negative {}/{}'.format(num, fn))
    #         show(unnormalize_car(car_1s[i]).astype(np.uint8))


if __name__ == '__main__':
    main()

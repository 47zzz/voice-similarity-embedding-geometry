from sklearn import metrics
import numpy


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE], fnr[idxE]) * 100

    return (tunedThreshold, eer, fpr, fnr)

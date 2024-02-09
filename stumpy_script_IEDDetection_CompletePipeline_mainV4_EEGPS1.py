from dataImport import applicationDidFinishLaunching, computeWaveforms, drawWaveformsMod, plotUpdateDetections, \
    plotUpdateDetectionsWithIsaAnnotations, importAnnotationOfIsa, computeRuptSamples

import matplotlib.pyplot as plt
import pickle
import numpy as np
import stumpy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from matplotlib.widgets import Cursor
import itertools
import scipy.stats as stats
import time
import copy
import scipy
from scipy import signal
from scipy.signal import find_peaks, peak_prominences, peak_widths, correlate
import multiprocessing
import multiprocessing as mp
from multiprocessing import freeze_support
from dtw import dtw, accelerated_dtw
from numba import guvectorize, float64, int64, njit, cuda, jit
import pandas as pd
import seaborn as sn
import xlsxwriter


def creationDuFichierExcelDeDetectionsPS1(fileName, nbEm, channelsNames):
    # Workbook() takes one, non-optional, argument
    # which is the filename that we want to create.
    fileName = fileName + 'Detections.xlsx'
    workbook = xlsxwriter.Workbook(fileName)

    # The workbook object is then used to add new
    # worksheet via the add_worksheet() method.

    worksheet = workbook.add_worksheet("detections")

    worksheet.write(0, 0, "Bipolar Channels")
    worksheet.write(0, 1, "Number of IEDs")

    # Start from the first cell. Rows and
    # columns are zero indexed.
    row = 1
    col = 0

    # Iterate over the data and write it out row by row.
    for i in range(len(nbEm)):
        worksheet.write(row, col, channelsNames[i])
        worksheet.write(row, col + 1, nbEm[i])
        row += 1

    workbook.close()
    return


def createExcelFileDetectorPerformances(channelsNames, results_FC5_C5_C3_FC1, results_AllChannels, resultsFileLocation):
    # Workbook() takes one, non-optional, argument
    # which is the filename that we want to create.
    fileName = resultsFileLocation + 'Report.xlsx'
    workbook = xlsxwriter.Workbook(fileName)

    # The workbook object is then used to add new
    # worksheet via the add_worksheet() method.

    worksheet_1 = workbook.add_worksheet("results_AllChannels")

    worksheet_1.write(0, 0, "EEG AVG Channel")
    worksheet_1.write(0, 1, "TPR = TP / (TP + FN)")
    worksheet_1.write(0, 2, "PPV = TP / (TP + FP)")
    worksheet_1.write(0, 3, "TS = TP / (TP + FN + FP)")

    # Iterate over the data and write it out row by row.
    for i in range(len(channelsNames)):
        col = 0
        worksheet_1.write(i + 1, col, channelsNames[i])
        col += 1
        worksheet_1.write(i + 1, col, results_AllChannels[i][0])
        col += 1
        worksheet_1.write(i + 1, col, results_AllChannels[i][1])
        col += 1
        worksheet_1.write(i + 1, col, results_AllChannels[i][2])

    worksheet_2 = workbook.add_worksheet("results_OnFewAnnotatedChannels")

    worksheet_2.write(0, 0, "EEG AVG Channel")
    worksheet_2.write(0, 1, "TPR = TP / (TP + FN)")
    worksheet_2.write(0, 2, "PPV = TP / (TP + FP)")
    worksheet_2.write(0, 3, "TS = TP / (TP + FN + FP)")

    channelsNames_Annotated = ["FC5AVGREF", "C5AVGREF", "C3AVGREF", "FC1AVGREF"]

    # Iterate over the data and write it out row by row.
    for i in range(len(channelsNames_Annotated)):
        col = 0
        worksheet_2.write(i + 1, col, channelsNames_Annotated[i])
        col += 1
        worksheet_2.write(i + 1, col, results_FC5_C5_C3_FC1[i][0])
        col += 1
        worksheet_2.write(i + 1, col, results_FC5_C5_C3_FC1[i][1])
        col += 1
        worksheet_2.write(i + 1, col, results_FC5_C5_C3_FC1[i][2])

    workbook.close()
    return


def verifyAnnnotationsRange(manualAnnotationsRange_samp, detections_samp, offSet_samp):
    TP = 0
    FP = 0
    for elementDet in detections_samp:
        found = 0
        for elementMan in manualAnnotationsRange_samp:
            if elementMan[0] - offSet_samp < elementDet < elementMan[1] + offSet_samp:
                TP += 1
                found = 1
                break
        if not found:
            FP += 1

    FN = 0
    for element in manualAnnotationsRange_samp:
        # a = np.where(element - offSet_samp < detections_samp < element + offSet_samp)
        a = detections_samp[
            (detections_samp > element[0] - offSet_samp) & (detections_samp < element[1] + offSet_samp)]
        if a.size == 0:
            FN += 1

    TPR = TP / (TP + FN) if (TP + FN) else 0.0  # sensitivity, recall, hit rate, or true positive rate (TPR) = 1 - FNR
    PPV = TP / (TP + FP) if (TP + FP) else 0.0  # precision or positive predictive value (PPV) = 1 - FDR
    TS = TP / (TP + FN + FP)  # threat score (TS) or critical success index (CSI)
    # FNR = FN / (FN + TP)  # miss rate or false negative rate (FNR) = 1 - TPR
    # FDR = FP / (FP + TP)  # false discovery rate (FDR) = 1 - PPV

    result_TPR_PPV_TS = [TPR, PPV, TS]

    return result_TPR_PPV_TS


def verifyAnnnotations(manualAnnotations_samp, detections_samp, offSet_samp):
    TP = 0
    FP = 0
    for element in detections_samp:
        # a = np.where(element - offSet_samp < manualAnnotations_samp < element + offSet_samp)
        # https://stackoverflow.com/questions/16343752/numpy-where-function-multiple-conditions
        a = manualAnnotations_samp[
            (manualAnnotations_samp > element - offSet_samp) & (manualAnnotations_samp < element + offSet_samp)]
        if a.size == 0:
            FP += 1
        else:
            TP += 1

    FN = 0
    for element in manualAnnotations_samp:
        # a = np.where(element - offSet_samp < detections_samp < element + offSet_samp)
        a = detections_samp[
            (detections_samp > element - offSet_samp) & (detections_samp < element + offSet_samp)]
        if a.size == 0:
            FN += 1

    TPR = TP / (TP + FN) if (TP + FN) else 0.0  # sensitivity, recall, hit rate, or true positive rate (TPR) = 1 - FNR
    PPV = TP / (TP + FP) if (TP + FP) else 0.0  # precision or positive predictive value (PPV) = 1 - FDR
    TS = TP / (TP + FN + FP)  # threat score (TS) or critical success index (CSI)
    # FNR = FN / (FN + TP)  # miss rate or false negative rate (FNR) = 1 - TPR
    # FDR = FP / (FP + TP)  # false discovery rate (FDR) = 1 - PPV

    result_TPR_PPV_TS = [TPR, PPV, TS]

    return result_TPR_PPV_TS


def compareDetections(inst_rupt, channelsNames, vector1_pointes_FC5ouC5_samp, vector2_pointes_C3_samp,
                      vector3_pointes_FC1_samp, vector_debut_fin_deschargesDePointes_samp, offSet_samp):
    results_FC5_C5_C3_FC1 = []
    annotatedSignals = ["FC5AVGREF", "C5AVGREF", "C3AVGREF", "FC1AVGREF"]

    isaAnnotations = (vector1_pointes_FC5ouC5_samp, vector1_pointes_FC5ouC5_samp, vector2_pointes_C3_samp,
                      vector3_pointes_FC1_samp)
    for i in range(len(annotatedSignals)):
        detections_samp = computeRuptSamples(inst_rupt[i, :])
        manualAnnotations_samp = isaAnnotations[i]
        result_TPR_PPV_TS = verifyAnnnotations(manualAnnotations_samp, detections_samp, offSet_samp)
        results_FC5_C5_C3_FC1.append(result_TPR_PPV_TS)

    results_AllChannels = []
    for i in range(len(channelsNames)):
        detections_samp = computeRuptSamples(inst_rupt[i, :])
        result_TPR_PPV_TS = verifyAnnnotationsRange(vector_debut_fin_deschargesDePointes_samp, detections_samp,
                                                    offSet_samp)
        results_AllChannels.append(result_TPR_PPV_TS)

    return results_FC5_C5_C3_FC1, results_AllChannels


as_strided = np.lib.stride_tricks.as_strided

"""  """


def plotIEDAndSelectMargins(t_wf, meanWaveFormPerChannnel, PROMINENCE_UP_VALUE, percCutBeginning, percCutEnd, display):
    if display:
        fig, ax = plt.subplots()
        ax.plot(t_wf, meanWaveFormPerChannnel)
        ax.set_ylabel('a.u.')
        ax.set_xlabel('seconds')
        ax.set_title("MeanWaveFormPerChannnel")
        ax.xaxis.visible = True
    else:
        fig = []
        ax = []

    peaks, _ = find_peaks(meanWaveFormPerChannnel, prominence=PROMINENCE_UP_VALUE)
    prominences = peak_prominences(meanWaveFormPerChannnel, peaks)[0]
    contour_heights = meanWaveFormPerChannnel[peaks] - prominences
    results_half = peak_widths(meanWaveFormPerChannnel, peaks, rel_height=0.5)
    results_full = peak_widths(meanWaveFormPerChannnel, peaks, rel_height=1)

    if display:
        # PEAKS
        plt.plot(t_wf[peaks], meanWaveFormPerChannnel[peaks], "o", color="r")
        # PROMINENCES
        plt.vlines(x=t_wf[peaks], ymin=contour_heights, ymax=meanWaveFormPerChannnel[peaks], color="k")
        # FWHM
        plt.hlines(results_half[1], t_wf[results_half[2].astype(int)], t_wf[results_half[3].astype(int)],
                   color="g")
        # BASELINE
        plt.hlines(results_full[1], t_wf[results_full[2].astype(int)], t_wf[results_full[3].astype(int)],
                   color="r")
        # PROMINENCES ANNOTATIONS
        style = dict(size=10, color='k')
        for j in range(len(prominences)):
            plt.annotate(np.round(prominences[j], 3), (t_wf[peaks[j]], meanWaveFormPerChannnel[peaks[j]]), **style)
        # FWHM ANNOTATIONS
        for j in range(len(results_half[1])):
            plt.annotate(np.round(results_half[0][j] / fEch, 3),
                         (t_wf[results_half[3][j].astype(int)], results_half[1][j]), color="g")
        # BASELINE ANNOTATIONS
        for j in range(len(results_full[1])):
            plt.annotate(np.round(results_full[0][j] / fEch, 3),
                         (t_wf[results_full[3][j].astype(int)], results_full[1][j]), color="r")

    """AUTOMATIC CUT meanWaveFormPerChannnel"""
    try:
        startFWHMSpike_samp = results_half[2][0].astype(int)
        lenFWHMSpike = results_half[3][0].astype(int) - results_half[2][0].astype(int)
        startCutMeanWaveform_samp = startFWHMSpike_samp - int(percCutBeginning * lenFWHMSpike)
    except:
        startCutMeanWaveform_samp = 0

    try:
        endFWHMWave = results_half[3][1].astype(int)
        lenFWHMWave = results_half[3][1].astype(int) - results_half[2][1].astype(int)
        stopCutMeanWaveform_samp = endFWHMWave + int(percCutEnd * lenFWHMWave)
    except:
        stopCutMeanWaveform_samp = len(meanWaveFormPerChannnel)

    if startCutMeanWaveform_samp < 0:
        startCutMeanWaveform_samp = 0
    if stopCutMeanWaveform_samp > len(meanWaveFormPerChannnel):
        stopCutMeanWaveform_samp = len(meanWaveFormPerChannnel)

    startCutMeanWaveform_sec = startCutMeanWaveform_samp / fEch
    stopCutMeanWaveform_sec = stopCutMeanWaveform_samp / fEch

    if display:
        ax.axvline(startCutMeanWaveform_sec, linestyle='--', color='0.75')
        ax.axvline(stopCutMeanWaveform_sec, linestyle='--', color='0.75')

    return fig, ax, startCutMeanWaveform_samp, stopCutMeanWaveform_samp


def computeDTW(s1, s2, metric, normInputSignal=True):
    if normInputSignal:
        # s1 = (s1 - np.mean(s1)) / (np.std(s1))
        # s2 = (s2 - np.mean(s2)) / (np.std(s2))
        s1 = stats.zscore(s1)
        s2 = stats.zscore(s2)
    distance = accelerated_dtw(s1, s2, dist=metric)[0]

    # manhattan_distance = lambda x, y: np.abs(x - y)
    # dtwMetric = [accelerated_dtw(meanWaveFormPerChannnel, element, dist=manhattan_distance)[0] for element in slicedSEEG]
    # dist = accelerated_dtw(meanWaveFormPerChannnel, element, dist='cityblock')[0]  # dist='euclidean')[0]

    return distance


def runDTWProcessPool(meanWaveFormPerChannnel, element, metric):  # , currentProcessInfo):
    # print(currentProcessInfo)
    distance = computeDTW(meanWaveFormPerChannnel, element, metric)
    return distance


def computePLV(s1, s2, normInputSignal=True):
    @guvectorize(["complex128[:], complex128[:], float64[:]"], '(n),(n)->()')
    def phase_locking_value(theta1, theta2, plv):
        complex_phase_diff = np.exp(np.complex(0, 1) * (theta1 - theta2))
        plv[0] = np.abs(np.sum(complex_phase_diff)) / len(theta1)

    if normInputSignal:
        # s1 = (s1 - np.mean(s1)) / (np.std(s1))
        # s2 = (s2 - np.mean(s2)) / (np.std(s2))
        s1 = stats.zscore(s1)
        s2 = stats.zscore(s2)

    s1_phase = np.unwrap(np.angle(signal.hilbert(s1)))
    s2_phase = np.unwrap(np.angle(signal.hilbert(s2)))

    plvp = np.zeros(2)
    phase_locking_value(s1_phase, s2_phase, plvp)

    return plvp[0]


def runPLVProcessPool(meanWaveFormPerChannnel, element):  # , currentProcessInfo):
    # print(currentProcessInfo)

    distance = computePLV(meanWaveFormPerChannnel, element)

    return distance


def computeXCORR(s1, s2, normInputSignal=True):
    if normInputSignal:
        # s1 = (s1 - np.mean(s1)) / (np.std(s1))
        # s2 = (s2 - np.mean(s2)) / (np.std(s2))
        s1 = stats.zscore(s1)
        s2 = stats.zscore(s2)

    xcorr = signal.correlate(s1, s2)
    distance = xcorr[(len(s1) - 1):]

    return distance


def runXCORRProcessPool(meanWaveFormPerChannnel, element):  # , currentProcessInfo):
    # print(currentProcessInfo)
    distance = computeXCORR(meanWaveFormPerChannnel, element)
    return distance


def writeMP(window_size_sec, m, t_seeg, seeg, channelName, IndexSignal2Process, resultsFileLocation):
    mp = stumpy.gpu_stump(seeg, m=m)

    if 0:
        filename_stump = resultsFileLocation + 'stump_' + str(IndexSignal2Process) + '_' + str(
            window_size_sec) + 's_' + channelName + '.pkl'
        with open(filename_stump, 'wb') as f:
            pickle.dump([mp, t_seeg, seeg, m], f, protocol=pickle.HIGHEST_PROTOCOL)

    all_chain_set, unanchored_chain = stumpy.allc(mp[:, 2], mp[:, 3])

    filename_mp = resultsFileLocation + 'mp_' + str(IndexSignal2Process) + '_' + str(
        window_size_sec) + 's_' + channelName + '.pkl'
    with open(filename_mp, 'wb') as f:
        pickle.dump([all_chain_set, unanchored_chain, mp, t_seeg, seeg, m], f, protocol=pickle.HIGHEST_PROTOCOL)

    return


""" STARTING MAIN """

if __name__ == '__main__':
    freeze_support()

    save = 0

    """ Matrix Profile computation for all Channnels """

    inputFilename = 'D:/DATA_MARSEILLE/DATA_MARSEILLE_PS1_P2/PS1_Patient2/EEGDetector/SigProcessing/exportDetEEGamad2Process.des'
    resultsFileLocation = "D:/DATA_MARSEILLE/DATA_MARSEILLE_PS1_P2/PS1_Patient2/EEGDetector/RunTime_Data/CompletePipeline_EEGPS1/" + \
                          inputFilename[:-4].rpartition('/')[2] + "_"

    inputFilename = 'D:/DATA_MARSEILLE/DATA_MARSEILLE_PS1_P2/PS1_Patient2/EEGDetector/ISA-DATA_SigProcessing/313Hz/02S1J1_64v_Galvani_0001_ICAcorr.des'
    inputFilename = 'D:/DATA_MARSEILLE/DATA_MARSEILLE_PS1_P2/PS1_Patient2/EEGDetector/ISA-DATA_SigProcessing/313Hz/02S1J1_64v_Galvani_0002_ICAcorr.des'
    inputFilename = 'D:/DATA_MARSEILLE/DATA_MARSEILLE_PS1_P2/PS1_Patient2/EEGDetector/ISA-DATA_SigProcessing/313Hz/02S4J1_64v_Galvani_0001_ICAcorr.des'
    inputFilename = 'D:/DATA_MARSEILLE/DATA_MARSEILLE_PS1_P2/PS1_Patient2/EEGDetector/ISA-DATA_SigProcessing/313Hz/02S4J1_64v_Galvani_0004_ICAcorr.des'

    resultsFileLocation = "D:/DATA_MARSEILLE/DATA_MARSEILLE_PS1_P2/PS1_Patient2/EEGDetector/ISA-DATA_RunTime_Data/CompletePipeline_313Hz/" + \
                          inputFilename[:-4].rpartition('/')[2] + "_"

    bias = 30
    patientTxtField, t_seeg, signal, nbChannels, channelsNames, nbPtsSig, fEch, debut, dateTxtField, heureTxtField \
        = applicationDidFinishLaunching(inputFilename, bias)

    """PLOT imported Signals"""
    if 0:
        signal2Plot = 0
        plotOneSignalForVerification(patientTxtField, t_seeg, signal, signal2Plot, channelsNames)
        plotImportedSignals(patientTxtField, t_seeg, signal, nbChannels, channelsNames)

    windowsSizes_array_sec = [0.4]  # [1.5, 1.25, 1, 0.75, 0.5, 0.25]
    numberOfSignals = np.size(signal, 0)

    if 0:

        # sig_size_sec = 60 * 30
        # sig_size = int(sig_size_sec * fEch)
        # t_seeg = t_seeg[:sig_size]
        sig_size = len(t_seeg)

        listInput = []
        for signal2Process_Index in range(numberOfSignals):
            seeg = signal[signal2Process_Index][:sig_size].astype(np.float64)

            for windowsSizes_array_sec_Index in range(len(windowsSizes_array_sec)):
                window_size_sec = windowsSizes_array_sec[windowsSizes_array_sec_Index]
                window_size = int(window_size_sec * fEch)

                listInput.append(tuple(
                    (window_size_sec, window_size, t_seeg, seeg, channelsNames[signal2Process_Index],
                     signal2Process_Index, resultsFileLocation)))

        print("Starting Pool\n")
        # pool = mp.Pool(mp.cpu_count())
        pool = mp.Pool(processes=40)
        resultsPool = pool.starmap(writeMP, listInput)
        pool.close()
        print("\nPool Closed.\n")

    """ Visualize all Extracted Motifs and Save the Figure """
    start = time.time()

    for windowsSizes_array_sec_Index in range(len(windowsSizes_array_sec)):

        filename_pattern = resultsFileLocation + 'pattern_' + str(
            windowsSizes_array_sec[windowsSizes_array_sec_Index]) + 'secWaveforms' + '.pkl'

        if 0:

            Cols = 3
            Rows = numberOfSignals // Cols
            if numberOfSignals - (Cols * Rows) > 0:
                Rows += 1

            fig = plt.figure(windowsSizes_array_sec_Index)
            k = 1

            nbWaveforms = []
            waveformsInSignal_touple = ()
            for signal2Process_Index in range(numberOfSignals):
                print("Creating main IED for signal: " + str(signal2Process_Index + 1) + "/" + str(numberOfSignals))

                ax = fig.add_subplot(Rows, Cols, k)

                filename_mp = resultsFileLocation + 'mp_' + str(signal2Process_Index) + '_' + str(
                    windowsSizes_array_sec[windowsSizes_array_sec_Index]) + 's_' + \
                              channelsNames[signal2Process_Index] + '.pkl'
                with open(filename_mp, 'rb') as f:
                    all_chain_set_orig, unanchored_chain, mp, t_seeg, seeg, m = pickle.load(f)
                all_chain_sets = copy.deepcopy(all_chain_set_orig)

                sx_limit = m
                dx_limit = int(1.5 * m)
                waveformsWindow_samp = sx_limit + dx_limit
                tempArray = []
                waveformsCounter = 0

                minSet = unanchored_chain.shape[0] - 1
                for z in range(len(all_chain_sets)):
                    if minSet <= len(all_chain_sets[z]) < unanchored_chain.shape[0]:
                        all_chain_set = all_chain_sets[z]
                        for w in range(len(all_chain_set)):

                            startSamp = all_chain_set[w] - sx_limit
                            endSamp = all_chain_set[w] + dx_limit
                            if startSamp < 0:
                                y = stats.zscore(seeg[0:endSamp])
                                y = list(itertools.chain(np.zeros(abs(startSamp)), y))
                            elif endSamp > len(seeg):
                                y = stats.zscore(seeg[startSamp - (endSamp - len(seeg)):endSamp])
                            else:
                                y = stats.zscore(seeg[startSamp:endSamp])

                            y = y - np.min(y)

                            tempArray.append(y)
                            waveformsCounter += 1
                            ax.plot(y, linewidth=0.1, color="blue")

                for z in range(unanchored_chain.shape[0]):

                    startSamp = unanchored_chain[z] - sx_limit
                    endSamp = unanchored_chain[z] + dx_limit
                    if startSamp < 0:
                        y = stats.zscore(seeg[0:endSamp])
                        y = list(itertools.chain(np.zeros(abs(startSamp)), y))
                    elif endSamp > len(seeg):
                        y = stats.zscore(seeg[startSamp - (endSamp - len(seeg)):endSamp])
                    else:
                        y = stats.zscore(seeg[startSamp:endSamp])

                    y = y - np.min(y)
                    tempArray.append(y)
                    waveformsCounter += 1
                    ax.plot(y, linewidth=0.3, color="red")

                ax.axis('on')
                ax.axvline(x=m, linestyle="dashed", color='yellow')
                rect = Rectangle((m, 0), m, 20, facecolor='lightgrey')
                ax.add_patch(rect)
                pltTitle = str(channelsNames[signal2Process_Index]) + ": " + str(waveformsCounter) + " waves"
                ax.set_title(pltTitle)
                k += 1

                waveformsInSignal_touple += (tempArray,)
                nbWaveforms.append(waveformsCounter)

            fig.suptitle("Extraction without Alignement - Window Size: %ssec" % windowsSizes_array_sec[
                windowsSizes_array_sec_Index])
            plt.tight_layout()

            """ STORE MeanWaveforms """
            meanWaveFormSPerChannnel = []
            representativeWaveFormPerChannel = []
            t_wf = np.arange(0, waveformsWindow_samp / fEch, 1 / fEch)
            iterations = 5
            for idx in range(len(nbWaveforms)):
                meanWaveForm, shifts_sec, normalizedWaveformsInSignal, representativeWaveForm = computeWaveforms(
                    waveformsInSignal_touple[idx], t_wf, iterations)
                meanWaveFormSPerChannnel.append(meanWaveForm)
                representativeWaveFormPerChannel.append(representativeWaveForm)

            with open(filename_pattern, 'wb') as f:
                pickle.dump([meanWaveFormSPerChannnel, representativeWaveFormPerChannel, t_wf, channelsNames,
                             fEch, waveformsWindow_samp, nbWaveforms, waveformsInSignal_touple,
                             windowsSizes_array_sec[windowsSizes_array_sec_Index]], f,
                            protocol=pickle.HIGHEST_PROTOCOL)

            """ 
                DOC:
                meanWaveFormPerChannnel: Spike and Wave to work on. NORMALIZED
                representativeWaveFormPerChannel: Just one example of extracted Spike and Wave (not important). NORMALIZED
                t_wf: temporal vector
                fEch: sample frequency
                seeg: entire sEEG bipolar channel, from which the Spike and Wave was extracted
            """

        """ VISUALIZE MeanWaveforms """
        if 0:
            with open(filename_pattern, 'rb') as f:
                meanWaveFormSPerChannnel, representativeWaveFormPerChannel, t_wf, channelsNames, \
                fEch, waveformsWindow_samp, nbWaveforms, waveformsInSignal_touple, \
                windowsSizes_array_sec[windowsSizes_array_sec_Index] = pickle.load(f)

            titlePlot = windowsSizes_array_sec[windowsSizes_array_sec_Index]
            titlePlot = resultsFileLocation + 'image_motifs'
            drawWaveformsMod(waveformsWindow_samp, fEch, channelsNames, nbWaveforms, waveformsInSignal_touple,
                             titlePlot)

    end = time.time()
    print("Finished creating IEDs")
    print('Elapsed time: ', round((end - start) / 60, 3), 'minutes')

    # Attention HERE - COMMENT this line exit(1)
    plt.show()
    # exit(1)

    """ Explore extracted IEDs by using Covariance """

    explorePatternsWithCovariance = 0
    if explorePatternsWithCovariance:
        for windowsSizes_array_sec_Index in range(len(windowsSizes_array_sec)):
            filename_pattern = resultsFileLocation + 'pattern_' + str(
                windowsSizes_array_sec[windowsSizes_array_sec_Index]) + 'secWaveforms' + '.pkl'

            with open(filename_pattern, 'rb') as f:
                meanWaveFormSPerChannnel, representativeWaveFormPerChannel, t_wf, channelsNames, \
                fEch, waveformsWindow_samp, nbWaveforms, waveformsInSignal_touple, \
                windowsSizes_array_sec[windowsSizes_array_sec_Index] = pickle.load(f)

            numberOfSignals = len(channelsNames)

            """ VISUALIZE MeanWaveforms """

            PROMINENCE_VALUE = 2
            Cols = 3
            Rows = numberOfSignals // Cols
            if numberOfSignals - (Cols * Rows) > 0:
                Rows += 1

            fig = plt.figure()
            k = 1
            for i in range(numberOfSignals):
                ax = fig.add_subplot(Rows, Cols, k)

                ax.plot(t_wf, meanWaveFormSPerChannnel[i], linewidth=1, color="r")
                # ax.plot(t_wf, representativeWaveFormPerChannel[i], linewidth=1, color="b")

                x = meanWaveFormSPerChannnel[i]
                if i == 3:
                    peaks, _ = find_peaks(x, prominence=1)
                else:
                    peaks, _ = find_peaks(x, prominence=PROMINENCE_VALUE)
                prominences = peak_prominences(x, peaks)[0]
                contour_heights = x[peaks] - prominences

                # PEAKS
                ax.plot(t_wf[peaks], x[peaks], "o", color="C3")

                # PROMINENCES
                plt.vlines(x=t_wf[peaks], ymin=contour_heights, ymax=x[peaks], color="black")

                # PROMINENCES ANNOTATIONS
                style = dict(size=10, color='black')
                for j in range(len(prominences)):
                    plt.annotate(np.round(prominences[j], 3), (t_wf[peaks[j]], x[peaks[j]]), **style)

                ax.axis('on')
                ax.set_title(channelsNames[i])
                k += 1

            shifts_sec = []
            shifts_samp = []

            usingPeaks = 1
            usingXCorr = 0

            if usingPeaks:
                alignedMeanWaveFormSPerChannnel = []
                centerOfWindow_samp = int(np.size(meanWaveFormSPerChannnel, 1) / 2)

                for i in range(np.size(meanWaveFormSPerChannnel, 0)):
                    dx = np.mean(np.diff(t_wf))

                    x = meanWaveFormSPerChannnel[i]
                    peaks, _ = find_peaks(x, prominence=PROMINENCE_VALUE)

                    if len(peaks) == 0:
                        shift_samp = 0
                    else:
                        shift_samp = centerOfWindow_samp - peaks[0]

                    shifts_samp.append(shift_samp)
                    shift_sec = shift_samp * dx
                    shifts_sec.append(shift_sec)

                    if shift_samp == 0:
                        newDataTemp = meanWaveFormSPerChannnel[i]
                    else:
                        nanArray = np.full(abs(shift_samp), np.nan)
                        if shift_samp > 0:  # Aggiungo all'inizio e taglio il segnale alla fine
                            newDataTemp = np.concatenate((nanArray, meanWaveFormSPerChannnel[i][:-abs(shift_samp)]),
                                                         axis=0)
                        elif shift_samp < 0:  # Taglio all'inizio e aggiungo il segnale alla fine
                            newDataTemp = np.concatenate((meanWaveFormSPerChannnel[i][abs(shift_samp):], nanArray),
                                                         axis=0)
                    alignedMeanWaveFormSPerChannnel.append(newDataTemp)

            if usingXCorr:
                alignedMeanWaveFormSPerChannnel = []
                normalizedMeanWaveFormSPerChannnel = stats.zscore(meanWaveFormSPerChannnel, axis=1)
                meanWaveForm = np.nanmean(normalizedMeanWaveFormSPerChannnel, axis=0)

                for i in range(len(meanWaveFormSPerChannnel)):
                    dx = np.mean(np.diff(t_wf))
                    xcorr = correlate(meanWaveForm, normalizedMeanWaveFormSPerChannnel[i])
                    xcorrMax = np.max(xcorr)
                    xcorrMaxSamp = np.argmax(xcorr)
                    shift_samp = (xcorrMaxSamp - (len(normalizedMeanWaveFormSPerChannnel[i]) - 1))
                    # shift = (np.argmax(signal.correlate(data0.y, target.y)) - (len(target.y)-1)) * dx

                    shifts_samp.append(shift_samp)
                    shift_sec = shift_samp * dx
                    shifts_sec.append(shift_sec)

                    if shift_samp == 0:
                        newDataTemp = normalizedMeanWaveFormSPerChannnel[i]
                    else:
                        nanArray = np.full(abs(shift_samp), np.nan)
                        if shift_samp > 0:  # Aggiungo all'inizio e taglio il segnale alla fine
                            newDataTemp = np.concatenate(
                                (nanArray, normalizedMeanWaveFormSPerChannnel[i][:-abs(shift_samp)]), axis=0)
                        elif shift_samp < 0:  # Taglio all'inizio e aggiungo il segnale alla fine
                            newDataTemp = np.concatenate(
                                (normalizedMeanWaveFormSPerChannnel[i][abs(shift_samp):], nanArray), axis=0)
                    alignedMeanWaveFormSPerChannnel.append(newDataTemp)

            fig = plt.figure()
            ax = fig.add_subplot()
            ts = np.arange(0, len(alignedMeanWaveFormSPerChannnel[0]) / fEch, 1 / fEch)
            for i in range(1, numberOfSignals - 1):
                ax.plot(ts, alignedMeanWaveFormSPerChannnel[i], linewidth=1, label=channelsNames[i])

            ax.axis('on')
            ax.set_title("Aligned Signals")
            ax.legend()

            data = np.array(alignedMeanWaveFormSPerChannnel)
            df = pd.DataFrame(data, index=channelsNames).transpose()

            covMatrix = df.corr(method='pearson')  # Verify if it's the same function
            #  covMatrix = np.corrcoef(data)  # To use only with data not containining NaN

            print(covMatrix)
            fig = plt.figure()
            ax = fig.add_subplot()
            matrix = np.triu(covMatrix)
            # ax.imshow(covMatrix, cmap='hot', interpolation='nearest')
            sn.heatmap(covMatrix, annot=False, fmt='g', xticklabels=channelsNames[1:-1],
                       yticklabels=channelsNames[1:-1])  # , mask=matrix)
            ax.set_title("Covariance Matrix")
            # plt.show()
            # exit(1)

            fig = plt.figure()
            ax = fig.add_subplot()
            # ax.imshow(covMatrix, cmap='hot', interpolation='nearest')
            covMatrixTresholded = covMatrix
            thresholdUP = 0.32
            thresholdDW = -0.34
            covMatrixTresholded[covMatrixTresholded > thresholdUP] = 1
            boolean_array = np.logical_and(thresholdDW > covMatrixTresholded, thresholdDW > thresholdUP)
            covMatrixTresholded[boolean_array] = 0.5
            covMatrixTresholded[covMatrixTresholded < thresholdDW] = 0
            sn.heatmap(covMatrixTresholded, annot=False, fmt='g', xticklabels=channelsNames[1:-1],
                       yticklabels=channelsNames[1:-1], mask=matrix)
            ax.set_title("thresholdUP: " + str(thresholdUP) + " - thresholdDW: " + str(thresholdDW))

            fig = plt.figure()
            ax = fig.add_subplot()
            # ax.imshow(covMatrix, cmap='hot', interpolation='nearest')
            covMatrixTresholded = covMatrix
            threshold = 0
            covMatrixTresholded[covMatrixTresholded >= threshold] = 1
            covMatrixTresholded[covMatrixTresholded < threshold] = 0
            sn.heatmap(covMatrixTresholded, annot=False, fmt='g', xticklabels=channelsNames[1:-1],
                       yticklabels=channelsNames[1:-1], mask=matrix)
            ax.set_title("threshold: " + str(threshold))

        plt.show()

        exit(1)

    """ Open Time Series Chains, Explore Motifs and perform Detection """

    start = time.time()

    for windowsSizes_array_sec_Index in range(len(windowsSizes_array_sec)):
        window_size_sec = windowsSizes_array_sec[windowsSizes_array_sec_Index]

        nbEm = []
        inst_rupt = []
        filename_pattern = resultsFileLocation + 'pattern_' + str(
            windowsSizes_array_sec[windowsSizes_array_sec_Index]) + 'secWaveforms' + '.pkl'
        filename_detections = resultsFileLocation + 'detections_' + str(
            windowsSizes_array_sec[windowsSizes_array_sec_Index]) + 'secWaveforms' + '.pkl'

        with open(filename_pattern, 'rb') as f:
            meanWaveFormSPerChannnel, representativeWaveFormPerChannel, t_wf, channelsNames, \
            fEch, waveformsWindow_samp, nbWaveforms, waveformsInSignal_touple, \
            windowsSizes_array_sec[windowsSizes_array_sec_Index] = pickle.load(f)

        # for signal2Process_Index in [0, 5, 10]:
        for signal2Process_Index in range(numberOfSignals):

            # MANUALLY DEFINE A CHANNEL TO PROCESS
            # signal2Process_Index = 2

            print("Processing bipolar channel " + str(signal2Process_Index + 1) + "/" + str(numberOfSignals) + " " +
                  channelsNames[signal2Process_Index] + " @ " + str(window_size_sec) + "sec")

            filename_mp = resultsFileLocation + 'mp_' + str(signal2Process_Index) + '_' + str(window_size_sec) + 's_' + \
                          channelsNames[signal2Process_Index] + '.pkl'
            with open(filename_mp, 'rb') as f:
                all_chain_set_orig, unanchored_chain, mp, t_seeg, seeg, m = pickle.load(f)

            motif_idx = np.argsort(mp[:, 0])[0]
            print(f"The motif is located at index {motif_idx}")
            nearest_neighbor_idx = mp[motif_idx, 1]
            print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")

            """ Main Motif Visualization per Bipolar Channel """

            if 0:
                motif_idx_s = motif_idx / fEch
                nearest_neighbor_idx_s = nearest_neighbor_idx / fEch
                m_s = m / fEch

                fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
                plt.suptitle('Motif (Pattern) Discovery', fontsize='30')
                axs[0].plot(t_seeg, seeg)
                axs[0].set_ylabel(channelsNames[signal2Process_Index], fontsize='20')
                rect = Rectangle((motif_idx_s, 0), m_s, 2500, facecolor='lightgrey')
                axs[0].add_patch(rect)
                rect = Rectangle((nearest_neighbor_idx_s, 0), m_s, 2500, facecolor='lightgrey')
                axs[0].add_patch(rect)
                axs[1].set_xlabel('Time', fontsize='20')
                axs[1].set_ylabel('Matrix Profile', fontsize='20')
                axs[1].axvline(x=motif_idx_s, linestyle="dashed")
                axs[1].axvline(x=nearest_neighbor_idx_s, linestyle="dashed")
                mp_plot = np.append(np.zeros(m - 1), mp[:, 0])
                axs[1].plot(t_seeg, mp_plot)
                plt.show()

            """ Main Motif Visualization using different windows sizes """
            # THIS TAKES SOME TIME !
            if 0:
                windowsSizes_array_sec = [1.5, 1.2, 1, 0.8, 0.6, 0.5, 0.4, 0.3]
                windowsSizes_array = [element * fEch for element in windowsSizes_array_sec]

                fig, axs = plt.subplots(8, sharex=True, gridspec_kw={'hspace': 0})
                fig.text(0.5, -0.1, 'Subsequence Start Date', ha='center', fontsize='20')
                fig.text(0.08, 0.5, 'Matrix Profile', va='center', rotation='vertical', fontsize='20')
                for i in range(len(windowsSizes_array)):
                    mp = stumpy.stump(seeg, m=windowsSizes_array[i])
                    axs[i].plot(mp[:, 0])
                    # axs[i].set_ylim(0, 9.5)
                    # axs[i].set_xlim(0, 3600)
                    title = f"m = {i}"
                    axs[i].set_title(title, fontsize=20, y=.5)
                plt.xticks(rotation=75)
                plt.suptitle('STUMP with Varying Window Sizes', fontsize='30')
                plt.show()

            """ VISUALIZE PRINCIPAL CHAIN """

            if 0:
                plt.figure()
                plt.plot(seeg, linewidth=0.5, color='black')
                for i in range(unanchored_chain.shape[0]):
                    y = seeg[unanchored_chain[i]:unanchored_chain[i] + m]
                    x = range(unanchored_chain[i], unanchored_chain[i] + m)
                    plt.plot(x, y, linewidth=1)
                plt.title("Principal Chain Fig 1")
                plt.figure()

                for i in range(unanchored_chain.shape[0]):
                    data = stats.zscore(seeg[unanchored_chain[i] - m:unanchored_chain[i] + int(1.5 * m)])
                    y = data
                    plt.plot(y - np.min(y), linewidth=0.3)
                plt.axis('off')
                plt.title("Principal Chain Fig 2")

                plt.figure()
                y = []
                x = []
                vl = []
                N = int(2.5 * m)
                baseTime = list(range(N))
                for i in range(unanchored_chain.shape[0]):
                    y = stats.zscore((seeg[unanchored_chain[i] - m:unanchored_chain[i] + int(1.5 * m)]))
                    x = ([x + (i * N) for x in baseTime])
                    vl = ([i * N])
                    plt.plot(x, y - np.min(y), linewidth=1)
                    plt.axvline(x=vl, alpha=0.3)
                plt.title("Principal Chain Fig 3")
                plt.show()

            """ VISUALIZE ALL CHAINS SETS """

            if 0:
                all_chain_sets = copy.deepcopy(all_chain_set_orig)

                numberOfPlots = 1
                # minSet = unanchored_chain.shape[0] - 1
                minSet = 6

                # Append element to a tuple
                # Highlight the elements contained in all_chain_set, on the original track
                fig = plt.figure()
                plt.plot(seeg, linewidth=0.5, color='black')
                for j in range(len(all_chain_sets)):
                    if len(all_chain_sets[j]) > minSet:
                        numberOfPlots += 1
                        all_chain_set = all_chain_sets[j]
                        col = tuple(np.random.choice(range(0, 2), size=3))
                        for i in range(len(all_chain_set)):
                            y = seeg[all_chain_set[i]:all_chain_set[i] + m]
                            x = range(all_chain_set[i], all_chain_set[i] + m)
                            plt.plot(x, y, linewidth=1, color=col)
                plt.title("Elements all_chain_set containing at least " + str(minSet) + " sets.")
                plt.tight_layout()
                # plt.show()

                Cols = 6
                Rows = numberOfPlots // Cols
                if numberOfPlots - (Cols * Rows) > 0:
                    Rows += 1

                fig = plt.figure()
                k = 1
                all_chain_sets = copy.deepcopy(all_chain_set_orig)
                for j in range(len(all_chain_sets)):
                    if len(all_chain_sets[j]) > minSet:
                        all_chain_set = all_chain_sets[j]
                        ax = fig.add_subplot(Rows, Cols, k)
                        for i in range(len(all_chain_set)):
                            y = stats.zscore(seeg[all_chain_set[i] - m:all_chain_set[i] + int(1.5 * m)])
                            ax.plot(y, linewidth=0.3, color="red")
                        k += 1

                plt.suptitle("All Chain Sets containing at least " + str(minSet) + " sets.")
                plt.tight_layout()
                plt.show()

            """ 
                TO DO:
                _ Aggiungere la selezione manuale dei margini della IED
                _ Add button to inverse the signal +++
                _ Testare la PDC in Python in SCAS
                _ Vedi se nella PLV ho un decalage di -1, (ritardo). Questo perche le detezioni si trovano dopo la Spike and Wave
                    Look at these lines: (should I adjust the detection time as i did for the Xcorr?)
                    if detectUsingXCorr:
                        correctedPeaksIndx = computedMetricPeaks + int(len(meanWaveFormPerChannnel) / 2)
                    and
                    if detectUsingXCORR:
                        correctedPeaksIndx = computedMetricPeaks
                _ Delete variables after each execution
        
                NOTES:
                _ The DTW metric results depend on the overlap I used on the signal
                _ Maybe I should not normalize signals because I do an extract of SW and I want to compare raw SWs: 
                    In this way I'll be more sensitive to true big SWs instead of small SWs.
                    I should not normalize the SW meanWaveform and the windowed seeg signal 
                _ Ho 2 versioni della Xcorr: una che fa la xcorr su tutto il segnale ed una 
                    che fa la xcorr per finestra, come per la DTW e PLV
                _ The entire signal seeg and the mean waveform have been normalized before computing the similarity metrics
                _ Before apply a similarity function, inside the similarity function, 
                    the s1 and s2 signals should be normalized (center and reduce). This would allow the algorythm to detect also 
                    small spikes and waves. If I don't want them, then I don't normalize inside the similarity function.
            """

            """ VISUALIZE MeanWaveforms and Choose if To Discard it """

            button_clicked = 0


            # defining function to add line plot
            def plotButtonPressedAndClose(val):
                global button_clicked
                button_clicked = 1
                plt.close()
                return


            if 0:
                """
                    VISUALIZE MeanWaveforms with Alignement and Choose to Discard it or Not
                """
                drawWaveformsMod(waveformsWindow_samp, fEch, [channelsNames[signal2Process_Index]],
                                 [nbWaveforms[signal2Process_Index]], (waveformsInSignal_touple[signal2Process_Index],),
                                 windowsSizes_array_sec[windowsSizes_array_sec_Index])

                # defining button and add its functionality
                axes = plt.axes([0.65, 0.1, 0.3, 0.075])
                bpress = Button(axes, 'Discard IED', color="yellow")
                bpress.on_clicked(plotButtonPressedAndClose)

                plt.show()

            discardIEDAnswer = button_clicked
            button_clicked = 0
            if discardIEDAnswer:
                print("Discarded IED")

            """ """" """ """ """
            """ DETECTION """
            """ """" """ """ """

            performVisualInspectionForDetection = 0  # To see motif per channel 1
            noSelectionOfPatternMargins = 1  # To select margin per motif 0

            detectUsingXCorr = 0
            detectUsingDTW = 0
            detectUsingPLV = 0
            detectUsingXCORR = 1
            detectUsingSimilarityMetric = 0
            if detectUsingDTW or detectUsingPLV or detectUsingXCORR:
                detectUsingSimilarityMetric = 1

            """ VISUALIZE MeanWaveforms """
            normInputSignal = 1
            if normInputSignal:
                # seeg = (seeg - np.mean(seeg)) / (np.std(seeg))
                seeg = stats.zscore(seeg)

            t_seeg = np.arange(0, len(seeg) / fEch, 1 / fEch)

            meanWaveFormPerChannnel = np.asarray(meanWaveFormSPerChannnel[signal2Process_Index]).squeeze()
            if normInputSignal:
                # meanWaveFormPerChannnel = (meanWaveFormPerChannnel - np.mean(meanWaveFormPerChannnel)) / (
                #     np.std(meanWaveFormPerChannnel))
                meanWaveFormPerChannnel = stats.zscore(meanWaveFormPerChannnel)

            t_wf = np.asarray(t_wf).squeeze()

            """PLOT meanWaveFormPerChannnel"""
            PROMINENCE_UP_VALUE = 1

            if detectUsingXCorr:
                percCutBeginning = 1
                percCutEnd = 0.3
            if detectUsingDTW:
                percCutBeginning = 2
                percCutEnd = 0.3
            if detectUsingPLV:
                percCutBeginning = 2
                percCutEnd = 0.3
            if detectUsingXCORR:
                percCutBeginning = 2
                percCutEnd = 0.3

            if noSelectionOfPatternMargins:
                startCutMeanWaveform_samp = 0
                stopCutMeanWaveform_samp = len(meanWaveFormPerChannnel)
            else:
                fig, ax, startCutMeanWaveform_samp, stopCutMeanWaveform_samp = plotIEDAndSelectMargins(t_wf,
                                                                                                       meanWaveFormPerChannnel,
                                                                                                       PROMINENCE_UP_VALUE,
                                                                                                       percCutBeginning,
                                                                                                       percCutEnd,
                                                                                                       performVisualInspectionForDetection)

            if performVisualInspectionForDetection and not noSelectionOfPatternMargins:
                # defining button and add its functionality
                buttonAxes = fig.add_axes([0.65, 0.1, 0.3, 0.075])
                bpress = Button(buttonAxes, 'Manually Define IED Range', color="yellow")
                bpress.on_clicked(plotButtonPressedAndClose)

                plt.tight_layout()
                plt.show()
                """
                    This shows 1 plot: 
                    1) You can manually define the IEDs limit or keep it automatic
                """

            manualIEDRangeDefinition = button_clicked
            button_clicked = 0

            # startCutMeanWaveform_sec = 0.32
            # stopCutMeanWaveform_sec = 0.49
            # startCutMeanWaveform_samp = int(startCutMeanWaveform_sec * fEch)
            # stopCutMeanWaveform_samp = int(stopCutMeanWaveform_sec * fEch)
            if not manualIEDRangeDefinition:
                meanWaveFormPerChannnel = meanWaveFormPerChannnel[startCutMeanWaveform_samp:stopCutMeanWaveform_samp]
                t_wf_mod = t_wf[startCutMeanWaveform_samp:stopCutMeanWaveform_samp]
            else:
                # rifai il plot di tutto e recupera gli input
                fig, ax, startCutMeanWaveform_samp, stopCutMeanWaveform_samp = plotIEDAndSelectMargins(t_wf,
                                                                                                       meanWaveFormPerChannnel,
                                                                                                       PROMINENCE_UP_VALUE,
                                                                                                       percCutBeginning,
                                                                                                       percCutEnd,
                                                                                                       performVisualInspectionForDetection)
                # Creating an annotating box
                annot = ax.annotate("", xy=(0, 0), xytext=(-40, 40), textcoords="offset points",
                                    bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                                    arrowprops=dict(arrowstyle='-|>'))
                annot.set_visible(False)

                # https://www.geeksforgeeks.org/matplotlib-cursor-widget
                cursor = Cursor(ax, color='r', lw=1, horizOn=False, vertOn=True)

                coord = []
                collectedPoints = 0


                def onclick(event):
                    global coord
                    global collectedPoints
                    global ax

                    if collectedPoints > 1:
                        # plt.close()
                        return
                    coord.append((event.xdata, event.ydata))
                    collectedPoints += 1

                    x = event.xdata
                    y = event.ydata
                    text = "({:.2g}, {:.2g})".format(x, y)
                    print(text)

                    annot.xy = (x, y)
                    annot.set_text(text)
                    annot.set_visible(True)
                    fig.canvas.draw()  # redraw the figure

                    ax.axvline(x=x, color='blue', linestyle='--', axes=ax)


                fig.canvas.mpl_connect('button_press_event', onclick)
                plt.tight_layout()
                plt.show()

                try:
                    sxmargin = coord[0][0]
                    dxmargin = coord[1][0]
                    print("({:.2g}, {:.2g})".format(sxmargin, dxmargin))
                    startCutMeanWaveform_samp = int(sxmargin * fEch)
                    stopCutMeanWaveform_samp = int(dxmargin * fEch)
                    meanWaveFormPerChannnel = meanWaveFormPerChannnel[
                                              startCutMeanWaveform_samp:stopCutMeanWaveform_samp]
                    t_wf_mod = t_wf[startCutMeanWaveform_samp:stopCutMeanWaveform_samp]
                except:
                    pass

            if manualIEDRangeDefinition or not noSelectionOfPatternMargins:
                fig, ax = plt.subplots()
                ax.plot(t_wf_mod, meanWaveFormPerChannnel)
                ax.set_ylabel('a.u.')
                ax.set_xlabel('seconds')
                ax.set_title("Verify selected range of MeanWaveFormPerChannnel")
                ax.xaxis.visible = True
                plt.tight_layout()
                plt.show()
                # Show selected pattern

            """ """" """ """ """
            """COMPUTE XCORR"""
            """ """" """ """ """

            if detectUsingXCorr:
                xcorr = scipy.signal.correlate(seeg, meanWaveFormPerChannnel)
                xcorr = xcorr[(len(meanWaveFormPerChannnel) - 1):]

                similarityMetric = xcorr

            """ """" """ """ """
            """COMPUTE SIMILARITY METRIC"""
            """ """" """ """ """

            if detectUsingSimilarityMetric:
                # seconds2CutSignal = 15 * 60
                # seeg = seeg[:(fEch * seconds2CutSignal)]
                # t_seeg = t_seeg[:(fEch * seconds2CutSignal)]
                customOverlap = 1

                """COMPUTE SIMILARITY METRIC with CUSTOM Overlap"""
                if customOverlap:
                    similarityOffSet_perc = 10
                    windowFrame_Samples = len(meanWaveFormPerChannnel)
                    overlap_Samples = int((similarityOffSet_perc / 100) * windowFrame_Samples)
                    sigLen = len(seeg)
                    totalNumberOfTimeFrames = int(
                        (sigLen - windowFrame_Samples) / (windowFrame_Samples - overlap_Samples)) + 1

                    processingFrame = 0
                    slicedSEEG = []
                    while processingFrame < totalNumberOfTimeFrames:
                        startingSample = processingFrame * (windowFrame_Samples - overlap_Samples)
                        endingSample = processingFrame * (windowFrame_Samples - overlap_Samples) + windowFrame_Samples
                        slicedSEEG.append([seeg[startingSample:endingSample]])
                        processingFrame += 1
                    slicedSEEG = np.asarray(slicedSEEG).squeeze()

                """COMPUTE SIMILARITY METRIC with one sample Overlap"""
                if not customOverlap:
                    windowFrame_Samples = len(meanWaveFormPerChannnel)
                    totalNumberOfTimeFrames = len(seeg) - (windowFrame_Samples - 1)
                    seeg2treat = seeg.copy()
                    slicedSEEG = as_strided(seeg2treat, (totalNumberOfTimeFrames, windowFrame_Samples),
                                            seeg2treat.strides * 2)

                """COMPUTE SIMILARITY METRIC with MULTIPROCESSING"""
                poolProcess = 1

                # DISTANCES:
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
                # cityblock or Manhattan distance (same distance different names)

                """COMPUTE SIMILARITY METRIC PARALLEL"""

                if poolProcess:
                    listPoolInput = []
                    iteration = 1
                    for element in slicedSEEG:
                        # currentProcessInfo = "Iteration: " + str(iteration) + "/" + str(slicedSEEG.shape[0])
                        if detectUsingDTW:
                            listPoolInput.append(
                                tuple((meanWaveFormPerChannnel, element, 'cityblock')))  # 'euclidean')))
                        if detectUsingPLV:
                            listPoolInput.append(tuple((meanWaveFormPerChannnel, element)))
                        if detectUsingXCORR:
                            listPoolInput.append(tuple((meanWaveFormPerChannnel, element)))
                        iteration += 1

                    start = time.time()
                    if detectUsingDTW:
                        print("Starting DTW Pool")
                    if detectUsingPLV:
                        print("Starting PLV Pool")
                    if detectUsingXCORR:
                        print("Starting XCORR Pool")
                    pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count()))
                    if detectUsingDTW:
                        resultsPool = pool.starmap(runDTWProcessPool, listPoolInput)
                    if detectUsingPLV:
                        resultsPool = pool.starmap(runPLVProcessPool, listPoolInput)
                    if detectUsingXCORR:
                        resultsPool = pool.starmap(runXCORRProcessPool, listPoolInput)
                    pool.close()

                    del pool

                    print("Pool Closed.")
                    end = time.time()
                    print('Elapsed time: ', round((end - start) / 60, 3), 'minutes')

                    similarityMetric = np.array([a_tuple for a_tuple in resultsPool])

                """COMPUTE SIMILARITY METRIC SERIAL"""

                if not poolProcess:
                    similarityMetric = []
                    iteration = 1
                    for element in slicedSEEG:
                        print("Iteration: ", iteration, "/", slicedSEEG.shape[0])
                        if detectUsingDTW:
                            dist = computeDTW(meanWaveFormPerChannnel, element, 'cityblock')
                        if detectUsingPLV:
                            dist = computePLV(meanWaveFormPerChannnel, element)
                        if detectUsingXCORR:
                            dist = computeXCORR(meanWaveFormPerChannnel, element)
                        similarityMetric.append(dist)
                        iteration += 1

                """SAVE SIMILARITY METRIC"""
                if detectUsingDTW:
                    filename_simMetric = resultsFileLocation + 'similarityMetric' + str(
                        signal2Process_Index) + '_' + str(
                        window_size_sec) + 's_' + channelsNames[signal2Process_Index] + 'stump_DTW' + '.pkl'
                if detectUsingPLV:
                    filename_simMetric = resultsFileLocation + 'similarityMetric' + str(
                        signal2Process_Index) + '_' + str(
                        window_size_sec) + 's_' + channelsNames[signal2Process_Index] + 'stump_PLV' + '.pkl'
                if detectUsingXCORR:
                    filename_simMetric = resultsFileLocation + 'similarityMetric' + str(
                        signal2Process_Index) + '_' + str(
                        window_size_sec) + 's_' + channelsNames[signal2Process_Index] + 'stump_XCORR' + '.pkl'
                if 1:
                    with open(filename_simMetric, 'wb') as f:
                        pickle.dump([meanWaveFormPerChannnel, similarityMetric], f,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(filename_simMetric, 'rb') as f:
                        meanWaveFormPerChannnel, similarityMetric = pickle.load(f)

                """ADJUST SIMILARITY METRIC Metric if COMPUTE SIMILARITY METRIC with CUSTOM Overlap"""
                if customOverlap:
                    tempSimilarity = np.zeros(len(seeg)).squeeze()
                    processingFrame = 0
                    while processingFrame < len(similarityMetric):
                        startingSample = processingFrame * (windowFrame_Samples - overlap_Samples)
                        endingSample = processingFrame * (windowFrame_Samples - overlap_Samples) + windowFrame_Samples
                        tempSimilarity[startingSample:endingSample] = similarityMetric[processingFrame]
                        processingFrame += 1
                    similarityMetric = tempSimilarity

            """ """" """ """ """
            """PLOT COMPUTED METRIC"""
            """ """" """ """ """
            performVisualInspectionForDetection = 0  # To see detections per channel 1
            if performVisualInspectionForDetection:

                fig = plt.figure()

                """PLOT SEEG"""
                dimRow = 3
                dimCol = 1
                nbPlot = 1
                subax1 = fig.add_subplot(dimRow, dimCol, nbPlot)
                subax1.plot(t_seeg, seeg, color='b')
                subax1.set_ylabel('a.u.')
                subax1.set_xlabel('seconds')
                subax1.set_title("Signal")

                """PLOT IED"""
                nbPlot = 3
                subax3 = fig.add_subplot(dimRow, dimCol, nbPlot)
                subax3.plot(t_wf_mod, meanWaveFormPerChannnel, color='b')
                subax3.set_ylabel('a.u.')
                subax3.set_xlabel('seconds')
                subax3.set_title("IED")
                subax3.set_xlim([0, 1])

                """PLOT Computed Metric"""
                nbPlot = 2
                subax2 = fig.add_subplot(dimRow, dimCol, nbPlot, sharex=subax1)
                subax2.plot(t_seeg, similarityMetric, color='k')
                subax2.set_ylabel('a.u.')
                if detectUsingDTW:
                    subax2.set_title("DTW Metric")
                if detectUsingPLV:
                    subax2.set_title("PLV Metric")
                if detectUsingXCORR:
                    subax2.set_title("XCORR Metric")

            """PLOT Computed Metric PEAKS"""
            percThreshold = 0.85
            threshold = min(similarityMetric) + percThreshold * (max(similarityMetric) - min(similarityMetric))

            computedMetricPeaks, _ = find_peaks(similarityMetric, height=threshold)
            if detectUsingXCorr:
                correctedPeaksIndx = computedMetricPeaks + int(len(meanWaveFormPerChannnel) / 2)
            if detectUsingDTW:
                correctedPeaksIndx = computedMetricPeaks
            if detectUsingPLV:
                correctedPeaksIndx = computedMetricPeaks
            if detectUsingXCORR:
                correctedPeaksIndx = computedMetricPeaks

            if performVisualInspectionForDetection:
                subax2.axhline(threshold, color='b')
                subax1.plot(t_seeg[correctedPeaksIndx], seeg[correctedPeaksIndx], 'ro')
                subax2.plot(t_seeg[computedMetricPeaks], similarityMetric[computedMetricPeaks], 'go')

                """PLOT DETECTED IEDS"""
                for element in correctedPeaksIndx:
                    offSet_samp = int(len(meanWaveFormPerChannnel) / 2)
                    start_samp = element - offSet_samp
                    end_samp = element + offSet_samp
                    subax1.plot(t_seeg[start_samp:end_samp], seeg[start_samp:end_samp], color='r')

                plt.tight_layout()
                plt.show()

            # Create boolean array from correctedPeaksIndx ma attento perche qui ho considerato solo i 30 minuti di segnale
            # per fare il pattern recognition
            # Create e number of detection per channnel
            nbEm.append(len(correctedPeaksIndx))
            booleanDetection = np.zeros(len(seeg), dtype=bool)
            booleanDetection[correctedPeaksIndx] = True
            inst_rupt.append(booleanDetection)

        # TRY TO TRANSFORM DETECTIONS HERE TO a Compatible format with MySCAS

        with open(filename_detections, 'wb') as f:
            pickle.dump([nbEm, inst_rupt], f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(filename_detections, 'rb') as f:
            nbEm, inst_rupt = pickle.load(f)

        creationDuFichierExcelDeDetectionsPS1(resultsFileLocation, nbEm, channelsNames)

        end = time.time()
        print("Finished creating IEDs")
        print('Elapsed time: ', round((end - start) / 60, 3), 'minutes')

        """PLOT Detections on Signals"""
        bias = 30
        patientTxtField, t_seeg, signal, nbChannels, channelsNames, nbPtsSig, fEch, debut, dateTxtField, heureTxtField \
            = applicationDidFinishLaunching(inputFilename, bias)

        inst_rupt = np.asarray(inst_rupt)
        coefAmplitudeOfSignals = 0.3

        # plotUpdateDetections(patientTxtField, t_seeg, signal, nbChannels, channelsNames, inst_rupt,
        #                      coefAmplitudeOfSignals)

        vector1_pointes_FC5ouC5_samp, vector2_pointes_C3_samp, vector3_pointes_FC1_samp, \
        vector4_debut_deschargesDePointes_samp, vector5_fin_deschargesDePointes_samp, vector_debut_fin_deschargesDePointes_samp \
            = importAnnotationOfIsa(inputFilename, fEch, t_seeg)

        offSet_sec = 1
        offSet_samp = offSet_sec * fEch
        results_FC5_C5_C3_FC1, results_AllChannels = compareDetections(inst_rupt, channelsNames,
                                                                       vector1_pointes_FC5ouC5_samp,
                                                                       vector2_pointes_C3_samp,
                                                                       vector3_pointes_FC1_samp,
                                                                       vector_debut_fin_deschargesDePointes_samp,
                                                                       offSet_samp)

        createExcelFileDetectorPerformances(channelsNames, results_FC5_C5_C3_FC1, results_AllChannels,
                                            resultsFileLocation)

        plotUpdateDetectionsWithIsaAnnotations(patientTxtField, t_seeg, signal, nbChannels, channelsNames, inst_rupt,
                                               coefAmplitudeOfSignals, vector1_pointes_FC5ouC5_samp,
                                               vector2_pointes_C3_samp, vector3_pointes_FC1_samp,
                                               vector4_debut_deschargesDePointes_samp,
                                               vector5_fin_deschargesDePointes_samp,
                                               vector_debut_fin_deschargesDePointes_samp)

        exit(1)

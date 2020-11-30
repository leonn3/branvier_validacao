# -*- coding: utf-8 -*-
"""
@author: Woldson_Leonne
"""
import numpy
from statistics import *
import numpy as np
import numpy as np
import scipy as sp
import scipy.stats
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
     

SAPO_IND_1_AI_FRONTAL = [13.7, 14.5, 179, 165, 178.1, 167.9]
SAPO_IND_1_AI_POSTERIOR = [18.3, 16.8, 163.4, 167.8, 177.5, 181.7]
SAPO_IND_1_AI_LAT_DIR = [10, 161.4, 174.9]
SAPO_IND_1_AI_LAT_ESQ = [9.3, 155, 173.9]

SAPO_IND_1_BI_FRONTAL = [14.2, 16.1, 174.3, 167.1, 177.9, 169.4]
SAPO_IND_1_BI_POSTERIOR = [16.1, 17.1, 156.8, 173.6, 179.8, 180.3]
SAPO_IND_1_BI_LAT_DIR = [10.8, 165.1, 176.2]
SAPO_IND_1_BI_LAT_ESQ = [8.1, 155.9, 170.8]

SAPO_IND_2_AI_FRONTAL = [11.8, 13.8, 179.1, 173.9, 178.1, 176.3]
SAPO_IND_2_AI_POSTERIOR = [16.2, 14.3, 181, 180.1, 172, 178.6]
SAPO_IND_2_AI_LAT_DIR = [7.5, 159.1, 177.5]
SAPO_IND_2_AI_LAT_ESQ = [6.7, 159.9, 174.2]

SAPO_IND_3_AI_FRONTAL = [16.3, 17.9, 163.2, 168.6, 182, 179.8]
SAPO_IND_3_AI_POSTERIOR = [17.6, 19.4, 162, 158.9, 181.7, 179.7]
SAPO_IND_3_AI_LAT_DIR = [8, 166.3, 173.8]
SAPO_IND_3_AI_LAT_ESQ = [6.7, 160.4, 172.1]

SAPO_IND_3_BI_FRONTAL = [17.6, 15, 173, 172.4, 182.1, 180.9]
SAPO_IND_3_BI_POSTERIOR = [17.2, 19.1, 161.2, 160.4, 182.4, 183]
SAPO_IND_3_BI_LAT_DIR = [10.6, 158.4, 169.1]
SAPO_IND_3_BI_LAT_ESQ = [9.1, 154.6, 173.8]

##############################################################################

ANTERIOR_POSE_AI_I1_A1 = [19.67, 19.65, 19.45, 19.34, 18.89, 18.96, 18.87, 19.22, 19.34, 19.10]
ANTERIOR_POSE_AI_I1_A2 = [18.55, 18.32, 18.12, 18.78, 18.80, 18.31, 18.15, 20.2, 19.5, 18.78]
ANTERIOR_POSE_AI_I1_A3 = [170.23, 168.7, 164.2, 159.7, 163.6, 161.9, 172.5, 168.1, 167.4, 171.3]
ANTERIOR_POSE_AI_I1_A4 = [160.9, 162.8, 164.7, 155.2, 162, 169.6, 167.4, 165, 159.5, 160.2]
ANTERIOR_POSE_AI_I1_A5 = [175.1, 172, 177, 172.8, 173.1, 179, 173.8, 177, 176.4, 172.9]
ANTERIOR_POSE_AI_I1_A6 = [172.9, 168.8, 166.7, 174, 176, 169.7, 168.4, 173.8, 177.2, 174.4]

POSTERIOR_POSE_AI_I1_A1 = [18.76, 18.42, 16.64, 16.89, 18.46, 16.00, 17.45, 15.24,15.85,17.58]
POSTERIOR_POSE_AI_I1_A2 = [16.02, 17.33,16.00,16.35,13.92,14.38,14.24,17.36,13.69,13.30]
POSTERIOR_POSE_AI_I1_A3 = [150.16,153.56,157.71,169.47,150.55,159.16,152.55,161.48,167.82,152.37]
POSTERIOR_POSE_AI_I1_A4 = [151.03,165.20,164.91,164.67,167.31,161.17,166.34,162.90,160.19,168.93]
POSTERIOR_POSE_AI_I1_A5 = [172.58,171.85,169.73,174.66,168.04,169.80,170.73,170.91,171.16,169.05]
POSTERIOR_POSE_AI_I1_A6 = [176.05,173.25,174.83,178.50,174.08,171.04,170.96,156.57,171.13,176.46]

LD_POSE_AI_I1_A1 = [6.60,2.94,6.81,13.57,5.55,9.37,8.71,4.69,5.52,9.67]
LD_POSE_AI_I1_A2 = [161.31,156.97,157.63,168.30,155.09,164.06,163.47,157.59,157.13,161.47]
LD_POSE_AI_I1_A3 = [176.31,178.49,174.38,171.86,170.48,170.75,170.37,174.59,173.56,166.91]

LE_POSE_AI_I1_A1 = [0.51,1.19,0.64,3.75,3.43,0.40,0.63,1.71,2.23,4.15]
LE_POSE_AI_I1_A2 = [136.66,135.45,147.24,151.44,142.22,133.78,148.94,141.68,146.89,146.59]
LE_POSE_AI_I1_A3 = [172.26,174.30,170.86,169.65,176.96,177.52,172.04,171.68,172.29,169.34]

#############################################################################

ANTERIOR_POSE_BI_I1_A1 = [19.88,19.74,18.89,18.88,19.27,19.92,19.26,19.38,20.75,21.07]
ANTERIOR_POSE_BI_I1_A2 = [20.39,19.19,18.55,18.30,18.29,20.16,17.21,18.10,19.70,20.30]
ANTERIOR_POSE_BI_I1_A3 = [171.51,169.53,171.41,172.41,172.34,170.21,170.35,168.56,171.28,170.80]
ANTERIOR_POSE_BI_I1_A4 = [162.76,163.27,166.17,164.89,164.61,160.80,165.00,167.95,165.05,162.53]
ANTERIOR_POSE_BI_I1_A5 = [175.74,176.61,177.36,177.32,177.64,175.94,174.92,174.15,175.46,175.84]
ANTERIOR_POSE_BI_I1_A6 = [174.18,175.71,174.12,172.09,171.07,172.01,171.13,170.80,171.00,169.84]

POSTERIOR_POSE_BI_I1_A1 = [17.65,20.51,19.57,18.62,19.01,15.98,15.35,16.93,17.63,17.04]
POSTERIOR_POSE_BI_I1_A2 = [14.01,13.84,14.22,13.39,14.30,16.50,16.02,15.99,16.56,14.28]
POSTERIOR_POSE_BI_I1_A3 = [150.21,148.94,151.44,153.13,152.81,155.58,158.63,151.30,148.46,152.96]
POSTERIOR_POSE_BI_I1_A4 = [168.99,166.11,165.12,166.84,163.66,156.24,163.65,157.95,165.67,162.23]
POSTERIOR_POSE_BI_I1_A5 = [171.80,174.11,173.69,170.09,172.88,173.14,171.56,172.73,170.90,173.90]
POSTERIOR_POSE_BI_I1_A6 = [174.68,171.70,173.44,172.05,172.11,173.50,175.23,175.69,176.91,171.96]

LD_POSE_BI_I1_A1 = [7.57,7.04,12.95,7.96,6.43,9.35,7.37,3.47,5.96,7.24]
LD_POSE_BI_I1_A2 = [158.85,156.39,172.18,152.46,154.97,167.44,163.63,150.08,156.11,157.22]
LD_POSE_BI_I1_A3 = [177.74,173.81,172.26,175.71,173.65,178.23,177.60,175.26,176.05,176.74]

LE_POSE_BI_I1_A1 = [8.19,6.38,6.38,5.90,7.47,5.05,3.27,5.19,4.44,6.02]
LE_POSE_BI_I1_A2 = [158.46,153.25,151.94,152.52,153.67,149.78,145.86,150.07,148.97,156.87]
LE_POSE_BI_I1_A3 = [168.42,167.25,172.78,171.02,167.28,168.01,167.76,166.19,171.64,170.37]

############################################################################# ALTA ILUMINAÇÃO

MEDIA_ANTERIOR_POSE_AI_I1_A1 = numpy.mean(ANTERIOR_POSE_AI_I1_A1)
MEDIA_ANTERIOR_POSE_AI_I1_A2 = numpy.mean(ANTERIOR_POSE_AI_I1_A2)
MEDIA_ANTERIOR_POSE_AI_I1_A3 = numpy.mean(ANTERIOR_POSE_AI_I1_A3)
MEDIA_ANTERIOR_POSE_AI_I1_A4 = numpy.mean(ANTERIOR_POSE_AI_I1_A4)
MEDIA_ANTERIOR_POSE_AI_I1_A5 = numpy.mean(ANTERIOR_POSE_AI_I1_A5)
MEDIA_ANTERIOR_POSE_AI_I1_A6 = numpy.mean(ANTERIOR_POSE_AI_I1_A6)
MEDIA_ANTERIOR_POSE_AI_I1 = [MEDIA_ANTERIOR_POSE_AI_I1_A1, MEDIA_ANTERIOR_POSE_AI_I1_A2, ...and
                             MEDIA_ANTERIOR_POSE_AI_I1_A3, MEDIA_ANTERIOR_POSE_AI_I1_A4, ...and
                             MEDIA_ANTERIOR_POSE_AI_I1_A5, MEDIA_ANTERIOR_POSE_AI_I1_A6]
print("Média dos pontos anterior com alta iluminação:")
print("%.2f" %MEDIA_ANTERIOR_POSE_AI_I1_A1, "%.2f" %MEDIA_ANTERIOR_POSE_AI_I1_A2, "%.2f" %MEDIA_ANTERIOR_POSE_AI_I1_A3, ...and
      "%.2f" %MEDIA_ANTERIOR_POSE_AI_I1_A4, "%.2f" %MEDIA_ANTERIOR_POSE_AI_I1_A5, "%.2f" %MEDIA_ANTERIOR_POSE_AI_I1_A6)

#############################################################################

MEDIA_POSTERIOR_POSE_AI_I1_A1 = numpy.mean(POSTERIOR_POSE_AI_I1_A1)
MEDIA_POSTERIOR_POSE_AI_I1_A2 = numpy.mean(POSTERIOR_POSE_AI_I1_A2)
MEDIA_POSTERIOR_POSE_AI_I1_A3 = numpy.mean(POSTERIOR_POSE_AI_I1_A3)
MEDIA_POSTERIOR_POSE_AI_I1_A4 = numpy.mean(POSTERIOR_POSE_AI_I1_A4)
MEDIA_POSTERIOR_POSE_AI_I1_A5 = numpy.mean(POSTERIOR_POSE_AI_I1_A5)
MEDIA_POSTERIOR_POSE_AI_I1_A6 = numpy.mean(POSTERIOR_POSE_AI_I1_A6)
MEDIA_POSTERIOR_POSE_AI_I1 = [MEDIA_POSTERIOR_POSE_AI_I1_A1, MEDIA_POSTERIOR_POSE_AI_I1_A2, ...and
                             MEDIA_POSTERIOR_POSE_AI_I1_A3, MEDIA_POSTERIOR_POSE_AI_I1_A4, ...and
                             MEDIA_POSTERIOR_POSE_AI_I1_A5, MEDIA_POSTERIOR_POSE_AI_I1_A6]
print("Média dos pontos posterior com alta iluminação:")
print("%.2f" %MEDIA_POSTERIOR_POSE_AI_I1_A1, "%.2f" %MEDIA_POSTERIOR_POSE_AI_I1_A2, "%.2f" %MEDIA_POSTERIOR_POSE_AI_I1_A3, ...and
      "%.2f" %MEDIA_POSTERIOR_POSE_AI_I1_A4, "%.2f" %MEDIA_POSTERIOR_POSE_AI_I1_A5, "%.2f" %MEDIA_POSTERIOR_POSE_AI_I1_A6)

#############################################################################

MEDIA_LD_POSE_AI_I1_A1 = numpy.mean(LD_POSE_AI_I1_A1)
MEDIA_LD_POSE_AI_I1_A2 = numpy.mean(LD_POSE_AI_I1_A2)
MEDIA_LD_POSE_AI_I1_A3 = numpy.mean(LD_POSE_AI_I1_A3)

MEDIA_LD_POSE_AI_I1 = [MEDIA_LD_POSE_AI_I1_A1, MEDIA_LD_POSE_AI_I1_A2, ...and
                             MEDIA_LD_POSE_AI_I1_A3]
print("Média dos pontos lateral direita com alta iluminação:")
print("%.2f" %MEDIA_LD_POSE_AI_I1_A1, "%.2f" %MEDIA_LD_POSE_AI_I1_A2, "%.2f" %MEDIA_LD_POSE_AI_I1_A3)

#############################################################################

MEDIA_LE_POSE_AI_I1_A1 = numpy.mean(LE_POSE_AI_I1_A1)
MEDIA_LE_POSE_AI_I1_A2 = numpy.mean(LE_POSE_AI_I1_A2)
MEDIA_LE_POSE_AI_I1_A3 = numpy.mean(LE_POSE_AI_I1_A3)

MEDIA_LE_POSE_AI_I1 = [MEDIA_LE_POSE_AI_I1_A1, MEDIA_LE_POSE_AI_I1_A2, ...and
                             MEDIA_LE_POSE_AI_I1_A3]
print("Média dos pontos lateral esquerda com alta iluminação:")
print("%.2f" %MEDIA_LE_POSE_AI_I1_A1, "%.2f" %MEDIA_LE_POSE_AI_I1_A2, "%.2f" %MEDIA_LE_POSE_AI_I1_A3)     

############################################################################# BAIXA ILUMINAÇÃO
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
MEDIA_ANTERIOR_POSE_BI_I1_A1 = numpy.mean(ANTERIOR_POSE_BI_I1_A1)
MEDIA_ANTERIOR_POSE_BI_I1_A2 = numpy.mean(ANTERIOR_POSE_BI_I1_A2)
MEDIA_ANTERIOR_POSE_BI_I1_A3 = numpy.mean(ANTERIOR_POSE_BI_I1_A3)
MEDIA_ANTERIOR_POSE_BI_I1_A4 = numpy.mean(ANTERIOR_POSE_BI_I1_A4)
MEDIA_ANTERIOR_POSE_BI_I1_A5 = numpy.mean(ANTERIOR_POSE_BI_I1_A5)
MEDIA_ANTERIOR_POSE_BI_I1_A6 = numpy.mean(ANTERIOR_POSE_BI_I1_A6)
MEDIA_ANTERIOR_POSE_BI_I1 = [MEDIA_ANTERIOR_POSE_BI_I1_A1, MEDIA_ANTERIOR_POSE_BI_I1_A2, ...and
                             MEDIA_ANTERIOR_POSE_BI_I1_A3, MEDIA_ANTERIOR_POSE_BI_I1_A4, ...and
                             MEDIA_ANTERIOR_POSE_BI_I1_A5, MEDIA_ANTERIOR_POSE_BI_I1_A6]
print("Média dos pontos anterior com baixa iluminação:")
print("%.2f" %MEDIA_ANTERIOR_POSE_BI_I1_A1, "%.2f" %MEDIA_ANTERIOR_POSE_BI_I1_A2, "%.2f" %MEDIA_ANTERIOR_POSE_BI_I1_A3, ...and
      "%.2f" %MEDIA_ANTERIOR_POSE_BI_I1_A4, "%.2f" %MEDIA_ANTERIOR_POSE_BI_I1_A5, "%.2f" %MEDIA_ANTERIOR_POSE_BI_I1_A6)

#############################################################################

MEDIA_POSTERIOR_POSE_BI_I1_A1 = numpy.mean(POSTERIOR_POSE_BI_I1_A1)
MEDIA_POSTERIOR_POSE_BI_I1_A2 = numpy.mean(POSTERIOR_POSE_BI_I1_A2)
MEDIA_POSTERIOR_POSE_BI_I1_A3 = numpy.mean(POSTERIOR_POSE_BI_I1_A3)
MEDIA_POSTERIOR_POSE_BI_I1_A4 = numpy.mean(POSTERIOR_POSE_BI_I1_A4)
MEDIA_POSTERIOR_POSE_BI_I1_A5 = numpy.mean(POSTERIOR_POSE_BI_I1_A5)
MEDIA_POSTERIOR_POSE_BI_I1_A6 = numpy.mean(POSTERIOR_POSE_BI_I1_A6)
MEDIA_POSTERIOR_POSE_BI_I1 = [MEDIA_POSTERIOR_POSE_BI_I1_A1, MEDIA_POSTERIOR_POSE_BI_I1_A2, ...and
                             MEDIA_POSTERIOR_POSE_BI_I1_A3, MEDIA_POSTERIOR_POSE_BI_I1_A4, ...and
                             MEDIA_POSTERIOR_POSE_BI_I1_A5, MEDIA_POSTERIOR_POSE_BI_I1_A6]
print("Média dos pontos posterior com baixa iluminação:")
print("%.2f" %MEDIA_POSTERIOR_POSE_BI_I1_A1, "%.2f" %MEDIA_POSTERIOR_POSE_BI_I1_A2, "%.2f" %MEDIA_POSTERIOR_POSE_BI_I1_A3, ...and
      "%.2f" %MEDIA_POSTERIOR_POSE_BI_I1_A4, "%.2f" %MEDIA_POSTERIOR_POSE_BI_I1_A5, "%.2f" %MEDIA_POSTERIOR_POSE_BI_I1_A6)

#############################################################################

MEDIA_LD_POSE_BI_I1_A1 = numpy.mean(LD_POSE_BI_I1_A1)
MEDIA_LD_POSE_BI_I1_A2 = numpy.mean(LD_POSE_BI_I1_A2)
MEDIA_LD_POSE_BI_I1_A3 = numpy.mean(LD_POSE_BI_I1_A3)

MEDIA_LD_POSE_BI_I1 = [MEDIA_LD_POSE_BI_I1_A1, MEDIA_LD_POSE_BI_I1_A2, ...and
                             MEDIA_LD_POSE_BI_I1_A3]
print("Média dos pontos lateral direita com baixa iluminação:")
print("%.2f" %MEDIA_LD_POSE_BI_I1_A1, "%.2f" %MEDIA_LD_POSE_BI_I1_A2, "%.2f" %MEDIA_LD_POSE_BI_I1_A3)

#############################################################################

MEDIA_LE_POSE_BI_I1_A1 = numpy.mean(LE_POSE_BI_I1_A1)
MEDIA_LE_POSE_BI_I1_A2 = numpy.mean(LE_POSE_BI_I1_A2)
MEDIA_LE_POSE_BI_I1_A3 = numpy.mean(LE_POSE_BI_I1_A3)

MEDIA_LE_POSE_BI_I1 = [MEDIA_LE_POSE_BI_I1_A1, MEDIA_LE_POSE_BI_I1_A2, ...and
                             MEDIA_LE_POSE_BI_I1_A3]
print("Média dos pontos lateral esquerda com baixa iluminação:")
print("%.2f" %MEDIA_LE_POSE_BI_I1_A1, "%.2f" %MEDIA_LE_POSE_BI_I1_A2, "%.2f" %MEDIA_LE_POSE_BI_I1_A3)     

############################################################################ DESVIO PADRÃO
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("DP dos pontos anterior com alta iluminação:")
print("%.2f" %stdev(numpy.array(ANTERIOR_POSE_AI_I1_A1)), "%.2f" %stdev(numpy.array(ANTERIOR_POSE_AI_I1_A2)), "%.2f" %stdev(numpy.array(ANTERIOR_POSE_AI_I1_A3)), ...and
      "%.2f" %stdev(numpy.array(ANTERIOR_POSE_AI_I1_A4)), "%.2f" %stdev(numpy.array(ANTERIOR_POSE_AI_I1_A5)), "%.2f" %stdev(numpy.array(ANTERIOR_POSE_AI_I1_A6)))
print("DP dos pontos posterior com alta iluminação:")
print("%.2f" %stdev(POSTERIOR_POSE_AI_I1_A1), "%.2f" %stdev(POSTERIOR_POSE_AI_I1_A2), "%.2f" %stdev(POSTERIOR_POSE_AI_I1_A3), ...and
      "%.2f" %stdev(POSTERIOR_POSE_AI_I1_A4), "%.2f" %stdev(POSTERIOR_POSE_AI_I1_A5), "%.2f" %stdev(POSTERIOR_POSE_AI_I1_A6))
print("DP dos pontos lateral direito com alta iluminação:")
print("%.2f" %stdev(LD_POSE_AI_I1_A1), "%.2f" %stdev(LD_POSE_AI_I1_A2), "%.2f" %stdev(LD_POSE_AI_I1_A3))
print("DP dos pontos lateral esquerdo com alta iluminação:")
print("%.2f" %stdev(LE_POSE_AI_I1_A1), "%.2f" %stdev(LE_POSE_AI_I1_A2), "%.2f" %stdev(LE_POSE_AI_I1_A3))     

print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("DP dos pontos anterior com baixa iluminação:")
print("%.2f" %stdev(ANTERIOR_POSE_BI_I1_A1), "%.2f" %stdev(ANTERIOR_POSE_BI_I1_A2), "%.2f" %stdev(ANTERIOR_POSE_BI_I1_A3), ...and
      "%.2f" %stdev(ANTERIOR_POSE_BI_I1_A4), "%.2f" %stdev(ANTERIOR_POSE_BI_I1_A5), "%.2f" %stdev(ANTERIOR_POSE_BI_I1_A6))
print("DP dos pontos posterior com baixa iluminação:")
print("%.2f" %stdev(POSTERIOR_POSE_BI_I1_A1), "%.2f" %stdev(POSTERIOR_POSE_BI_I1_A2), "%.2f" %stdev(POSTERIOR_POSE_BI_I1_A3), ...and
      "%.2f" %stdev(POSTERIOR_POSE_BI_I1_A4), "%.2f" %stdev(POSTERIOR_POSE_BI_I1_A5), "%.2f" %stdev(POSTERIOR_POSE_BI_I1_A6))
print("DP dos pontos lateral direito com baixa iluminação:")
print("%.2f" %stdev(LD_POSE_BI_I1_A1), "%.2f" %stdev(LD_POSE_BI_I1_A2), "%.2f" %stdev(LD_POSE_BI_I1_A3))
print("DP dos pontos lateral esquerdo com baixa iluminação:")
print("%.2f" %stdev(LE_POSE_BI_I1_A1), "%.2f" %stdev(LE_POSE_BI_I1_A2), "%.2f" %stdev(LE_POSE_BI_I1_A3))    

############################################################################ CALCULO DE ERRO (RMSE)
RMS_FRONTAL_AI = sqrt(mean_squared_error(np.array(SAPO_IND_1_AI_FRONTAL), np.transpose(MEDIA_ANTERIOR_POSE_AI_I1)))
RMS_POSTERIOR_AI = sqrt(mean_squared_error(SAPO_IND_1_AI_POSTERIOR, MEDIA_POSTERIOR_POSE_AI_I1))
RMS_LD_AI = sqrt(mean_squared_error(SAPO_IND_1_AI_LAT_DIR, MEDIA_LD_POSE_AI_I1))
RMS_LE_AI = sqrt(mean_squared_error(SAPO_IND_1_AI_LAT_ESQ, MEDIA_LE_POSE_AI_I1))

RMS_FRONTAL_BI = sqrt(mean_squared_error(SAPO_IND_1_BI_FRONTAL, MEDIA_ANTERIOR_POSE_BI_I1))
RMS_POSTERIOR_BI = sqrt(mean_squared_error(SAPO_IND_1_BI_POSTERIOR, MEDIA_POSTERIOR_POSE_BI_I1))
RMS_LD_BI = sqrt(mean_squared_error(SAPO_IND_1_BI_LAT_DIR, MEDIA_LD_POSE_BI_I1))
RMS_LE_BI = sqrt(mean_squared_error(SAPO_IND_1_BI_LAT_ESQ, MEDIA_LE_POSE_BI_I1))

print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
############################################################################ CALCULO DE ERRO (Erro Relativo-> [(real-aprox)/aprox]*100). ((np.array()-np.array())/np.array())*100
ER_FRONTAL_AI = ((np.array(SAPO_IND_1_AI_FRONTAL)-np.array(MEDIA_ANTERIOR_POSE_AI_I1))/np.array(SAPO_IND_1_AI_FRONTAL))*100
ER_POSTERIOR_AI = ((np.array(SAPO_IND_1_AI_POSTERIOR)-np.array(MEDIA_POSTERIOR_POSE_AI_I1))/np.array(SAPO_IND_1_AI_POSTERIOR))*100
ER_LD_AI = ((np.array(SAPO_IND_1_AI_LAT_DIR)-np.array(MEDIA_LD_POSE_AI_I1))/np.array(SAPO_IND_1_AI_LAT_DIR))*100
ER_LE_AI = ((np.array(SAPO_IND_1_AI_LAT_ESQ)-np.array(MEDIA_LE_POSE_AI_I1))/np.array(SAPO_IND_1_AI_LAT_ESQ))*100

ER_FRONTAL_BI = ((np.array(SAPO_IND_1_BI_FRONTAL)-np.array(MEDIA_ANTERIOR_POSE_BI_I1))/np.array(SAPO_IND_1_BI_FRONTAL))*100
ER_POSTERIOR_BI = ((np.array(SAPO_IND_1_BI_POSTERIOR)-np.array(MEDIA_POSTERIOR_POSE_BI_I1))/np.array(SAPO_IND_1_BI_POSTERIOR))*100
ER_LD_BI = ((np.array(SAPO_IND_1_BI_LAT_DIR)-np.array(MEDIA_LD_POSE_BI_I1))/np.array(SAPO_IND_1_BI_LAT_DIR))*100
ER_LE_BI = ((np.array(SAPO_IND_1_BI_LAT_ESQ)-np.array(MEDIA_LE_POSE_BI_I1))/np.array(SAPO_IND_1_BI_LAT_ESQ))*100

print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("Erro dos pontos anterior com alta iluminação:")
print(abs(np.round(ER_FRONTAL_AI,2)))
print("Erro dos pontos posterior com alta iluminação:")
print(abs(np.round(ER_POSTERIOR_AI,2)))
print("Erro dos pontos lateral direito com alta iluminação:")
print(abs(np.round(ER_LD_AI,2)))
print("Erro dos pontos lateral esquerdo com alta iluminação:")
print(abs(np.round(ER_LE_AI,2)))   

print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("Erro dos pontos anterior com baixa iluminação:")
print(abs(np.round(ER_FRONTAL_BI,2)))
print("Erro dos pontos posterior com baixa iluminação:")
print(abs(np.round(ER_POSTERIOR_BI,2)))
print("Erro dos pontos lateral direito com baixa iluminação:")
print(abs(np.round(ER_LD_BI,2)))
print("Erro dos pontos lateral esquerdo com baixa iluminação:")
print(abs(np.round(ER_LE_BI,2)))

print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
############################################################################ INTERVALO DE CONFIANÇA
CONF_INT_AI_FRONTAL_A1 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_AI_I1_A1, scale=stdev(numpy.array(ANTERIOR_POSE_AI_I1_A1))) 
CONF_INT_AI_FRONTAL_A2 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_AI_I1_A2, scale=stdev(numpy.array(ANTERIOR_POSE_AI_I1_A2))) 
CONF_INT_AI_FRONTAL_A3 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_AI_I1_A3, scale=stdev(numpy.array(ANTERIOR_POSE_AI_I1_A3))) 
CONF_INT_AI_FRONTAL_A4 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_AI_I1_A4, scale=stdev(numpy.array(ANTERIOR_POSE_AI_I1_A4))) 
CONF_INT_AI_FRONTAL_A5 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_AI_I1_A5, scale=stdev(numpy.array(ANTERIOR_POSE_AI_I1_A5))) 
CONF_INT_AI_FRONTAL_A6 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_AI_I1_A6, scale=stdev(numpy.array(ANTERIOR_POSE_AI_I1_A6))) 
print("FRONTAL AI - A1 = ", np.round(CONF_INT_AI_FRONTAL_A1,2))
print("FRONTAL AI - A2 = ", np.round(CONF_INT_AI_FRONTAL_A2,2))
print("FRONTAL AI - A3 = ", np.round(CONF_INT_AI_FRONTAL_A3,2))
print("FRONTAL AI - A4 = ", np.round(CONF_INT_AI_FRONTAL_A4,2))
print("FRONTAL AI - A5 = ", np.round(CONF_INT_AI_FRONTAL_A5,2))
print("FRONTAL AI - A6 = ", np.round(CONF_INT_AI_FRONTAL_A6,2))
CONF_INT_AI_POSTERIOR_A1 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_AI_I1_A1, scale=stdev(numpy.array(POSTERIOR_POSE_AI_I1_A1))) 
CONF_INT_AI_POSTERIOR_A2 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_AI_I1_A2, scale=stdev(numpy.array(POSTERIOR_POSE_AI_I1_A2))) 
CONF_INT_AI_POSTERIOR_A3 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_AI_I1_A3, scale=stdev(numpy.array(POSTERIOR_POSE_AI_I1_A3))) 
CONF_INT_AI_POSTERIOR_A4 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_AI_I1_A4, scale=stdev(numpy.array(POSTERIOR_POSE_AI_I1_A4))) 
CONF_INT_AI_POSTERIOR_A5 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_AI_I1_A5, scale=stdev(numpy.array(POSTERIOR_POSE_AI_I1_A5))) 
CONF_INT_AI_POSTERIOR_A6 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_AI_I1_A6, scale=stdev(numpy.array(POSTERIOR_POSE_AI_I1_A6))) 
print("POSTERIOR AI - A1 = ", np.round(CONF_INT_AI_POSTERIOR_A1,2))
print("POSTERIOR AI - A2 = ", np.round(CONF_INT_AI_POSTERIOR_A2,2))
print("POSTERIOR AI - A3 = ", np.round(CONF_INT_AI_POSTERIOR_A3,2))
print("POSTERIOR AI - A4 = ", np.round(CONF_INT_AI_POSTERIOR_A4,2))
print("POSTERIOR AI - A5 = ", np.round(CONF_INT_AI_POSTERIOR_A5,2))
print("POSTERIOR AI - A6 = ", np.round(CONF_INT_AI_POSTERIOR_A6,2))
CONF_INT_AI_LD_A1 = scipy.stats.norm.interval(0.95, loc=MEDIA_LD_POSE_AI_I1_A1, scale=stdev(numpy.array(LD_POSE_AI_I1_A1))) 
CONF_INT_AI_LD_A2 = scipy.stats.norm.interval(0.95, loc=MEDIA_LD_POSE_AI_I1_A2, scale=stdev(numpy.array(LD_POSE_AI_I1_A2))) 
CONF_INT_AI_LD_A3 = scipy.stats.norm.interval(0.95, loc=MEDIA_LD_POSE_AI_I1_A3, scale=stdev(numpy.array(LD_POSE_AI_I1_A3))) 
print("LATERAL DIREITA AI - A1 = ", np.round(CONF_INT_AI_LD_A1,2))
print("LATERAL DIREITA AI - A2 = ", np.round(CONF_INT_AI_LD_A2,2))
print("LATERAL DIREITA AI - A3 = ", np.round(CONF_INT_AI_LD_A3,2))
CONF_INT_AI_LE_A1 = scipy.stats.norm.interval(0.95, loc=MEDIA_LE_POSE_AI_I1_A1, scale=stdev(numpy.array(LE_POSE_AI_I1_A1))) 
CONF_INT_AI_LE_A2 = scipy.stats.norm.interval(0.95, loc=MEDIA_LE_POSE_AI_I1_A2, scale=stdev(numpy.array(LE_POSE_AI_I1_A2))) 
CONF_INT_AI_LE_A3 = scipy.stats.norm.interval(0.95, loc=MEDIA_LE_POSE_AI_I1_A3, scale=stdev(numpy.array(LE_POSE_AI_I1_A3))) 
print("LATERAL ESQUERDA AI - A1 = ", np.round(CONF_INT_AI_LE_A1,2))
print("LATERAL ESQUERDA AI - A2 = ", np.round(CONF_INT_AI_LE_A2,2))
print("LATERAL ESQUERDA AI - A3 = ", np.round(CONF_INT_AI_LE_A3,2))
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
CONF_INT_BI_FRONTAL_A1 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_BI_I1_A1, scale=stdev(numpy.array(ANTERIOR_POSE_BI_I1_A1))) 
CONF_INT_BI_FRONTAL_A2 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_BI_I1_A2, scale=stdev(numpy.array(ANTERIOR_POSE_BI_I1_A2))) 
CONF_INT_BI_FRONTAL_A3 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_BI_I1_A3, scale=stdev(numpy.array(ANTERIOR_POSE_BI_I1_A3))) 
CONF_INT_BI_FRONTAL_A4 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_BI_I1_A4, scale=stdev(numpy.array(ANTERIOR_POSE_BI_I1_A4))) 
CONF_INT_BI_FRONTAL_A5 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_BI_I1_A5, scale=stdev(numpy.array(ANTERIOR_POSE_BI_I1_A5))) 
CONF_INT_BI_FRONTAL_A6 = scipy.stats.norm.interval(0.95, loc=MEDIA_ANTERIOR_POSE_BI_I1_A6, scale=stdev(numpy.array(ANTERIOR_POSE_BI_I1_A6))) 
print("FRONTAL BI - A1 = ", np.round(CONF_INT_BI_FRONTAL_A1,2))
print("FRONTAL BI - A2 = ", np.round(CONF_INT_BI_FRONTAL_A2,2))
print("FRONTAL BI - A3 = ", np.round(CONF_INT_BI_FRONTAL_A3,2))
print("FRONTAL BI - A4 = ", np.round(CONF_INT_BI_FRONTAL_A4,2))
print("FRONTAL BI - A5 = ", np.round(CONF_INT_BI_FRONTAL_A5,2))
print("FRONTAL BI - A6 = ", np.round(CONF_INT_BI_FRONTAL_A6,2))
CONF_INT_BI_POSTERIOR_A1 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_BI_I1_A1, scale=stdev(numpy.array(POSTERIOR_POSE_BI_I1_A1))) 
CONF_INT_BI_POSTERIOR_A2 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_BI_I1_A2, scale=stdev(numpy.array(POSTERIOR_POSE_BI_I1_A2))) 
CONF_INT_BI_POSTERIOR_A3 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_BI_I1_A3, scale=stdev(numpy.array(POSTERIOR_POSE_BI_I1_A3))) 
CONF_INT_BI_POSTERIOR_A4 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_BI_I1_A4, scale=stdev(numpy.array(POSTERIOR_POSE_BI_I1_A4))) 
CONF_INT_BI_POSTERIOR_A5 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_BI_I1_A5, scale=stdev(numpy.array(POSTERIOR_POSE_BI_I1_A5))) 
CONF_INT_BI_POSTERIOR_A6 = scipy.stats.norm.interval(0.95, loc=MEDIA_POSTERIOR_POSE_BI_I1_A6, scale=stdev(numpy.array(POSTERIOR_POSE_BI_I1_A6))) 
print("POSTERIOR BI - A1 = ", np.round(CONF_INT_BI_POSTERIOR_A1,2))
print("POSTERIOR BI - A2 = ", np.round(CONF_INT_BI_POSTERIOR_A2,2))
print("POSTERIOR BI - A3 = ", np.round(CONF_INT_BI_POSTERIOR_A3,2))
print("POSTERIOR BI - A4 = ", np.round(CONF_INT_BI_POSTERIOR_A4,2))
print("POSTERIOR BI - A5 = ", np.round(CONF_INT_BI_POSTERIOR_A5,2))
print("POSTERIOR BI - A6 = ", np.round(CONF_INT_BI_POSTERIOR_A6,2))
CONF_INT_BI_LD_A1 = scipy.stats.norm.interval(0.95, loc=MEDIA_LD_POSE_BI_I1_A1, scale=stdev(numpy.array(LD_POSE_BI_I1_A1))) 
CONF_INT_BI_LD_A2 = scipy.stats.norm.interval(0.95, loc=MEDIA_LD_POSE_BI_I1_A2, scale=stdev(numpy.array(LD_POSE_BI_I1_A2))) 
CONF_INT_BI_LD_A3 = scipy.stats.norm.interval(0.95, loc=MEDIA_LD_POSE_BI_I1_A3, scale=stdev(numpy.array(LD_POSE_BI_I1_A3))) 
print("LATERAL DIREITA BI - A1 = ", np.round(CONF_INT_BI_LD_A1,2))
print("LATERAL DIREITA BI - A2 = ", np.round(CONF_INT_BI_LD_A2,2))
print("LATERAL DIREITA BI - A3 = ", np.round(CONF_INT_BI_LD_A3,2))
CONF_INT_BI_LE_A1 = scipy.stats.norm.interval(0.95, loc=MEDIA_LE_POSE_BI_I1_A1, scale=stdev(numpy.array(LE_POSE_BI_I1_A1))) 
CONF_INT_BI_LE_A2 = scipy.stats.norm.interval(0.95, loc=MEDIA_LE_POSE_BI_I1_A2, scale=stdev(numpy.array(LE_POSE_BI_I1_A2))) 
CONF_INT_BI_LE_A3 = scipy.stats.norm.interval(0.95, loc=MEDIA_LE_POSE_BI_I1_A3, scale=stdev(numpy.array(LE_POSE_BI_I1_A3))) 
print("LATERAL ESQUERDA BI - A1 = ", np.round(CONF_INT_BI_LE_A1,2))
print("LATERAL ESQUERDA BI - A2 = ", np.round(CONF_INT_BI_LE_A2,2))
print("LATERAL ESQUERDA BI - A3 = ", np.round(CONF_INT_BI_LE_A3,2))



########################################################################################## PLOT

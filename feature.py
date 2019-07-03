import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from scipy.io.wavfile import read


def extract_wav_feature(file_path, select=None):
    """
    提取wav文件的特征
    :param file_path:带提取特征的wav文件路径
    :param select:选择的部分特征。字符串列表，必须在支持的特征名称内选择。
        可提取的全部特征按顺序为：
            ['zcr', 'energy', 'energy_entropy', 'spectral_centroid',
            'spectral_spread', 'spectral_entropy', 'spectral_flux',
            'spectral_rolloff', 'mfcc_1', 'mfcc_2', 'mfcc_3',
            'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8',
            'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12',
            'mfcc_13', 'chroma_1', 'chroma_2', 'chroma_3',
            'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7',
            'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11',
            'chroma_12', 'chroma_std']
        select示例：
            select=['zcr', 'spectral_centroid', 'energy', 'mfcc_1', 'spectral_rolloff', 'chroma_std']
    :return:特征字典。
        返回值示意：
            {[特征名称1]:[特征值1],[特征名称2]:[特征值2]...}
    """
    names, features = _get_feature(file_path)
    feature = {n: f for (n, f) in zip(names, features)}
    if select:
        return {s_n: feature[s_n] for s_n in select}

    return feature


def _get_feature(file_path):
    """使用pyAudioAnalysis获取特征名称和对应的特征值"""
    # [fs, x] = audioBasicIO.readAudioFile(file_path)
    sr, x = read(file_path)
    x = audioBasicIO.stereo2mono(x)
    f, f_names = audioFeatureExtraction.stFeatureExtraction(x, sr, 0.050 * sr, 0.025 * sr)

    return f_names, [np.mean(fm) for fm in f]




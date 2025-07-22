import numpy as np
from scipy.signal import savgol_filter
from scipy.signal.windows import general_gaussian
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FastICA, KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from source.analysis.base import BaseSklearnPipelineModel
from xgboost import XGBClassifier

# Preprocessing methods
###############################################################################


class PLSRegressionWrapper(PLSRegression):
    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X)


class SavgolWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, win_length=7, polyorder=2, deriv=2):
        self.win_length = win_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        signatures_sav = []
        sp = [self.win_length, self.polyorder, self.deriv]
        for signal in X:
            if self.win_length != 0:
                signal = savgol_filter(signal, sp[0], sp[1], sp[2])
            signatures_sav.append(signal)
        return np.array(signatures_sav)


class FFTWrapper(BaseEstimator, TransformerMixin):
    """
    https://nirpyresearch.com/fourier-spectral-smoothing-method/
    for derivatives Fourier derivative theorem used
    """

    def __init__(self, shape_param=1, sigma=10, deriv=2):
        self.shape_param = shape_param
        self.sigma = sigma
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        signatures_fft = []
        for signal in X:
            signal = self.FFT_filter(signal)
            signatures_fft.append(signal)
        return np.array(signatures_fft)

    def FFT_filter(self, signal):
        XX = np.hstack((signal, np.flip(signal)))
        win = np.roll(
            general_gaussian(XX.shape[0], self.shape_param, self.sigma),
            XX.shape[0] // 2,
        )
        fXX = np.fft.fft(XX)

        if self.deriv != 0:
            qq = (
                2
                * np.pi
                * np.arange(-XX.shape[0] // 2, XX.shape[0] // 2, 1)
                / XX.shape[0]
            )
            if self.deriv == 1:
                fXX = np.roll(
                    np.roll(fXX, -XX.shape[0] // 2) * (np.complex(0, 1) * qq),
                    XX.shape[0] // 2,
                )
            elif self.deriv == 2:
                fXX = np.roll(
                    np.roll(fXX, -XX.shape[0] // 2) * (-(qq**2)), XX.shape[0] // 2
                )

        return np.real(np.fft.ifft(fXX * win))[: signal.shape[0]]


class SNVTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply SNV to each sample
        return np.array([(x - np.mean(x)) / np.std(x) for x in X])


# Example models
###############################################################################


class SavgolXGB(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("savgol", SavgolWrapper()),
            ("xgb", XGBClassifier(random_state=0)),
        ]
    )


class SavgolSVC(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("savgol", SavgolWrapper()),
            ("svc", SVC(random_state=0)),
        ]
    )


class SavgolPLSXGB(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("savgol", SavgolWrapper()),
            ("pls", PLSRegressionWrapper()),
            ("xgb", XGBClassifier(random_state=0)),
        ]
    )


class SavgolPLSSVC(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("savgol", SavgolWrapper()),
            ("pls", PLSRegressionWrapper()),
            ("svc", SVC(random_state=0)),
        ]
    )


class SavgolICASVC(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("savgol", SavgolWrapper()),
            ("ica", FastICA(random_state=0)),
            ("svc", SVC(random_state=0)),
        ]
    )


class SavgolICAXGB(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("savgol", SavgolWrapper()),
            ("ica", FastICA(random_state=0)),
            ("xgb", XGBClassifier(random_state=0)),
        ]
    )


class SavgolKPCASVC(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("savgol", SavgolWrapper()),
            ("kpca", KernelPCA(kernel="rbf", random_state=0)),
            ("svc", SVC(random_state=0)),
        ]
    )


class SavgolKPCAXGB(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("savgol", SavgolWrapper()),
            ("kpca", KernelPCA(kernel="rbf", random_state=0)),
            ("xgb", XGBClassifier(random_state=0)),
        ]
    )


class FFTXGB(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fft", FFTWrapper()),
            ("xgb", XGBClassifier(random_state=0)),
        ]
    )


class FFTSVC(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fft", FFTWrapper()),
            ("svc", SVC(random_state=0)),
        ]
    )


class FFTPLSXGB(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fft", FFTWrapper()),
            ("pls", PLSRegressionWrapper()),
            ("xgb", XGBClassifier(random_state=0)),
        ]
    )


class FFTPLSSVC(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fft", FFTWrapper()),
            ("pls", PLSRegressionWrapper()),
            ("svc", SVC(random_state=0)),
        ]
    )


class FFTICASVC(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fft", FFTWrapper()),
            ("ica", FastICA(random_state=0)),
            ("svc", SVC(random_state=0)),
        ]
    )


class FFTICAXGB(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fft", FFTWrapper()),
            ("ica", FastICA(random_state=0)),
            ("xgb", XGBClassifier(random_state=0)),
        ]
    )


class FFTKPCASVC(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fft", FFTWrapper()),
            ("kpca", KernelPCA(kernel="rbf", random_state=0)),
            ("svc", SVC(random_state=0)),
        ]
    )


class FFTKPCAXGB(BaseSklearnPipelineModel):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("fft", FFTWrapper()),
            ("kpca", KernelPCA(kernel="rbf", random_state=0)),
            ("xgb", XGBClassifier(random_state=0)),
        ]
    )

from flask import Flask, request, jsonify
from flask_cors import CORS
import scipy.io
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis, entropy
import pywt  # For wavelet transform
import joblib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://stress-detector-five.vercel.app"}})


# Load pre-trained models and scalers
model = joblib.load('./model/stress_detection_model.pkl')
label_encoder = joblib.load('./model/label_encoder.pkl')
scaler = joblib.load('./model/scaler.pkl')

fs = 128  # Sampling frequency


# Utility: Bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# Utility: Calculate entropy
def calc_entropy(signal):
    power = np.abs(np.fft.fft(signal)) ** 2
    power /= np.sum(power)
    return entropy(power)


# Utility: Calculate band power
def band_power(data, fs):
    freqs, psd = welch(data, fs)
    idx_band = np.logical_and(freqs >= 13, freqs <= 30)  # Band range: 13–30 Hz
    return np.trapz(psd[idx_band], freqs[idx_band])


# Utility: Hjorth parameters
def hjorth_params(signal):
    d1 = np.diff(signal)
    d2 = np.diff(d1)
    var0 = np.var(signal)
    var1 = np.var(d1)
    var2 = np.var(d2)
    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    return activity, mobility, complexity


# Utility: Wavelet transform for heatmap
def wavelet_transform(data, fs):
    wavelet = 'cmor'  # Complex Morlet wavelet
    scales = np.linspace(1, 128, 256)  # Increase resolution with more scales
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period=1 / fs)
    coefficients = np.abs(coefficients)
    
    # Normalize coefficients for better visualization
    coefficients /= np.max(coefficients)

    # Apply logarithmic scaling (optional)
    coefficients = np.log1p(coefficients)

    return {
        "z": coefficients.tolist(),  # Magnitude of wavelet coefficients
        "x": list(range(len(data))),  # Time axis
        "y": frequencies.tolist()  # Frequency axis
    }


@app.route('/process', methods=['POST'])
def process():
    # Load the .mat file
    mat = scipy.io.loadmat(request.files['file'].stream)
    raw = mat['Clean_data']  # Replace 'Clean_data' with the key in your .mat file
    num_channels = raw.shape[0]
    time = list(range(len(raw[0])))
    all_features = []

    # Feature extraction for all channels
    for ch in raw:
        ch_filtered = bandpass_filter(ch, 13, 30, fs)
        stats = [
            float(np.mean(ch_filtered)), float(np.var(ch_filtered)), float(np.max(ch_filtered)), float(np.min(ch_filtered)),
            float(np.std(ch_filtered)), float(skew(ch_filtered)), float(kurtosis(ch_filtered)), float(calc_entropy(ch_filtered)),
            float(band_power(ch_filtered, fs)), *map(float, hjorth_params(ch_filtered))
        ]
        all_features.extend(stats)

    # Prepare the feature matrix
    X = np.array(all_features).reshape(1, -1)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        return jsonify({'error': 'Invalid input features: NaN or Inf detected'}), 400

    # Normalize the features
    X_scaled = scaler.transform(X)
    pred_encoded = model.predict(X_scaled)
    pred_label = label_encoder.inverse_transform(pred_encoded)[0]

    # Graph data for the first channel
    original = raw[0]
    filtered = bandpass_filter(original, 13, 30, fs)
    fft_vals = np.abs(np.fft.fft(original))[:len(original) // 2]
    freq = np.fft.fftfreq(len(original), 1 / fs)[:len(original) // 2]

    filtered_fft_vals = np.abs(np.fft.fft(filtered))[:len(filtered) // 2]
    filtered_freq = np.fft.fftfreq(len(filtered), 1 / fs)[:len(filtered) // 2]

    # Compute wavelet heatmap for the first channel
    wavelet_heatmap = wavelet_transform(original, fs)

    # Return all data as JSON
    return jsonify({
        'graphs': {
            'time': {'x': time, 'y': original.tolist(), 'type': 'scatter', 'name': 'Time'},
            'freq': {'x': freq.tolist(), 'y': fft_vals.tolist(), 'type': 'scatter', 'name': 'FFT'},
            'filtered': {'x': time, 'y': filtered.tolist(), 'type': 'scatter', 'name': 'Filtered'},
            'filteredFFT': {'x': filtered_freq.tolist(), 'y': filtered_fft_vals.tolist(), 'type': 'scatter', 'name': 'Filtered FFT'},
            'wavelet': wavelet_heatmap  # Improved wavelet heatmap
        },
        'features': {f'ch{ch+1}_{name}': val for ch in range(num_channels) for name, val in zip(
            ['mean', 'var', 'max', 'min', 'std', 'skew', 'kurtosis', 'entropy',
             'bandpower', 'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'],
            all_features[ch*12:(ch+1)*12])
        },
        'prediction': pred_label
    })


if __name__ == '__main__':
    app.run(debug=True)

import numpy as np
import matplotlib.pyplot as plt
import tda_ecg as tda

def generate_synthetic_ecg(duration=10, fs=100, bpm=60):
    """
    Generate a simple synthetic ECG-like signal using sine and random noise
    """
    t = np.linspace(0, duration, duration * fs)
    # Basic heartbeat rhythm (QRS approximation with narrowed sine)
    ecg = np.sin(2 * np.pi * (bpm / 60) * t) ** 10
    # Add some noise
    ecg += 0.05 * np.random.normal(size=len(t))
    return ecg

def main():
    print("Generating synthetic ECG signal...")
    ecg_signal = generate_synthetic_ecg(duration=20, fs=50, bpm=60)

    print("Searching for optimal delay (delta t)...")
    # We test delays from 1 to 25
    delays, distances = tda.find_optimal_delay(ecg_signal, dimension=3, max_delay=25, homology_dim=1)

    print(f"Delays: {delays}")
    print(f"Distances: {distances}")

    plt.figure(figsize=(10, 6))
    # We skip the first distance as it's 0.0 placeholder
    plt.plot(delays[1:], distances[1:], 'o-', label='Bottleneck Distance (H1)')
    plt.xlabel('Delay (tau)')
    plt.ylabel('Bottleneck Distance between tau and tau-1')
    plt.title('Topological Stability for Takens Embedding')
    plt.grid(True)
    plt.legend()

    output_path = 'bottleneck_stability.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()

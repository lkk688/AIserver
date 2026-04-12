import wave
import struct
import math

sample_rate = 16000
duration = 3.0  # seconds

wave_file = wave.open('test_audio.wav', 'w')
wave_file.setnchannels(1)
wave_file.setsampwidth(2)
wave_file.setframerate(sample_rate)

# Generate a simple 440Hz sine wave beep
for i in range(int(sample_rate * duration)):
    value = int(32767.0 * math.cos(2.0 * math.pi * 440.0 * i / sample_rate))
    wave_file.writeframes(struct.pack('<h', value))

wave_file.close()
print("Generated test_audio.wav")

import Foundation
import Testing
@testable import SlurperKit

struct DemucsTests {
    @Test func inverseSpectrogramReconstructsTheSegment() {
        // _spec drops the Nyquist bin and two frames at each end, so the signal stays well below Nyquist
        // and only the interior comes back.
        let segment = (0..<Demucs.segment).map { k in
            let t = Double(k) / 44_100
            return Float(0.4 * sin(2 * Double.pi * 55 * t) + 0.3 * sin(2 * Double.pi * 1_234.5 * t) + 0.2 * sin(2 * Double.pi * 9_876 * t))
        }
        let (real, imaginary) = Demucs.spectrogram(segment)
        let back = Demucs.inverseSpectrogram(real: real[...], imaginary: imaginary[...])
        #expect(back.count == segment.count)
        let interior = 8_192..<(segment.count - 8_192)
        let worst = interior.map { abs(back[$0] - segment[$0]) }.max()!
        #expect(worst < 1e-4, "\(worst)")
    }

    @Test func segmentsCenterTheLastChunkOnRealAudio() {
        let audio = [(0..<400_000).map(Float.init), (0..<400_000).map { -Float($0) }]
        let first = Demucs.segment(of: audio, at: 0)
        #expect(first[0].count == Demucs.segment && first[0][0] == 0 && first[0][Demucs.segment - 1] == Float(Demucs.segment - 1))

        // From 257,985 only 142,015 samples remain; demucs centers them, pulling earlier audio in front.
        let last = Demucs.segment(of: audio, at: 257_985)
        let start = 257_985 - (Demucs.segment - 142_015) / 2
        #expect(last[0][0] == Float(start))
        #expect(last[1][399_999 - start] == -399_999 && last[1][400_000 - start] == 0)
    }

    @Test func spectrogramPutsASineInItsBinAtTorchScale() {
        // Bin 100 of a 4096-sample FFT; torch's normalized STFT of a full-scale sine peaks at sqrt(4096) / 4.
        let sine = (0..<Demucs.segment).map { Float(sin(2 * Double.pi * 100 * Double($0) / 4096)) }
        let (real, imaginary) = Demucs.spectrogram(sine)
        let frame = 168
        let magnitudes = (0..<2048).map { bin in hypot(real[bin * 336 + frame], imaginary[bin * 336 + frame]) }
        #expect(magnitudes.firstIndex(of: magnitudes.max()!) == 100)
        #expect(abs(magnitudes[100] - 16) < 0.01, "\(magnitudes[100])")
    }
}

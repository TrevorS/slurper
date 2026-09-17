import Foundation
@testable import SlurperKit

/// SplitMix64, so test audio is the same on every run and a failure reproduces.
struct SeededRandom: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

func noise(frames: Int, channels: Int, amplitude: Float = 0.5, seed: UInt64 = 1) -> [[Float]] {
    var random = SeededRandom(seed: seed)
    return (0..<channels).map { _ in (0..<frames).map { _ in Float.random(in: -amplitude...amplitude, using: &random) } }
}

func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appending(path: "SlurperTests-\(UUID().uuidString)", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

let testRate = 44_100.0

/// White noise 80 dB down.
func noiseFloor(seconds: Double) -> [Float] {
    noise(frames: Int(seconds * testRate), channels: 1, amplitude: 1e-4, seed: 2)[0]
}

/// Adds `length` seconds of `sample(t, random)`, from `seconds` into `signal`, cut off at its end. The random
/// numbers are seeded by the start time.
private func add(_ signal: inout [Float], at seconds: Double, length: Double, _ sample: (Double, inout SeededRandom) -> Float) {
    let start = Int(seconds * testRate)
    var random = SeededRandom(seed: UInt64(start) &+ 3)
    for i in 0..<max(0, min(Int(length * testRate), signal.count - start)) {
        signal[start + i] += sample(Double(i) / testRate, &random)
    }
}

/// Exponentially decaying white noise.
func burst(_ signal: inout [Float], at seconds: Double, amplitude: Float, decay: Double, length: Double) {
    add(&signal, at: seconds, length: length) { t, random in
        amplitude * Float(exp(-t / decay)) * Float.random(in: -1...1, using: &random)
    }
}

/// A sine falling from 120 to 45 Hz, with a 2 ms click.
func kick(_ signal: inout [Float], at seconds: Double, amplitude: Float) {
    var phase = 0.0
    add(&signal, at: seconds, length: 0.25) { t, random in
        phase += 2 * Double.pi * (45 + 75 * exp(-t / 0.03)) / testRate
        let click = t < 0.002 ? Float.random(in: -0.3...0.3, using: &random) : 0
        return amplitude * (Float(sin(phase) * exp(-t / 0.08)) + click)
    }
}

/// Decaying noise over a 190 Hz tone.
func snare(_ signal: inout [Float], at seconds: Double, amplitude: Float) {
    add(&signal, at: seconds, length: 0.2) { t, random in
        let rattle = 0.6 * Float.random(in: -1...1, using: &random) * Float(exp(-t / 0.05))
        return amplitude * (rattle + 0.4 * Float(sin(2 * Double.pi * 190 * t) * exp(-t / 0.03)))
    }
}

/// Short high-passed noise.
func hat(_ signal: inout [Float], at seconds: Double, amplitude: Float) {
    var previous: Float = 0
    add(&signal, at: seconds, length: 0.06) { t, random in
        let next = Float.random(in: -1...1, using: &random)
        defer { previous = next }
        return amplitude * (next - previous) / 2 * Float(exp(-t / 0.012))
    }
}

let vocalModelInstalled = FileManager.default.fileExists(atPath: StemSplitter.vocalModel.path)
let hornsModelInstalled = FileManager.default.fileExists(atPath: StemSplitter.hornsModel.path)
let demucsModelInstalled = FileManager.default.fileExists(atPath: StemSplitter.demucsModel.path)
let modelsInstalled = vocalModelInstalled && hornsModelInstalled && demucsModelInstalled

private final class BundleMarker {}

/// The `slurper` executable built next to this test bundle.
let slurperBinary = Bundle(for: BundleMarker.self).bundleURL.deletingLastPathComponent().appending(path: "slurper")

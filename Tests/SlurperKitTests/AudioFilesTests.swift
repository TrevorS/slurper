import AVFoundation
import Testing
@testable import SlurperKit

struct AudioFilesTests {
    @Test func float32RoundTripIsExact() throws {
        let audio = noise(frames: 4_410, channels: 2)
        let url = try temporaryDirectory().appending(path: "float.wav")
        try AudioFiles.writeWAV(audio, sampleRate: 44_100, to: url)
        #expect(try AudioFiles.readStereo(url, sampleRate: 44_100) == audio)
    }

    @Test func monoFileReadsAsTwoEqualChannels() throws {
        let mono = noise(frames: 4_410, channels: 1)
        let url = try temporaryDirectory().appending(path: "mono.wav")
        try AudioFiles.writeWAV(mono, sampleRate: 44_100, to: url)
        let stereo = try AudioFiles.readStereo(url, sampleRate: 44_100)
        #expect(stereo.count == 2)
        #expect(stereo[0] == stereo[1])
        #expect(zip(stereo[0], mono[0]).allSatisfy { abs($0 - $1) < 1e-6 })
    }

    @Test func resampleKeepsDurationAndLevel() throws {
        let tone = (0..<44_100).map { Float(0.5 * sin(2 * Double.pi * 1_000 * Double($0) / 44_100)) }
        let resampled = try AudioFiles.resample([tone, tone], from: 44_100, to: 48_000)
        #expect(resampled.count == 2)
        #expect(abs(resampled[0].count - 48_000) <= 2)
        let peak = resampled[0][12_000..<36_000].map(abs).max() ?? 0
        #expect(abs(peak - 0.5) < 0.01)
    }

    @Test func int16DitherStaysWithinOneAndAHalfSteps() throws {
        let audio = noise(frames: 48_000, channels: 2, amplitude: 0.9)
        let url = try temporaryDirectory().appending(path: "int16.wav")
        try AudioFiles.writeWAV(audio, sampleRate: 48_000, encoding: .int16Dithered, to: url)

        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatInt16, interleaved: false)
        let format = file.fileFormat.streamDescription.pointee
        #expect(format.mBitsPerChannel == 16)
        #expect(format.mFormatFlags & kAudioFormatFlagIsFloat == 0)

        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let samples = UnsafeBufferPointer(start: buffer.int16ChannelData![0], count: Int(buffer.frameLength))
        let worst = zip(samples, audio[0]).map { abs(Float($0) - $1 * 32767) }.max() ?? .infinity
        #expect(worst <= 1.5)
    }

    @Test func int16DitherKeepsLevelsBelowOneStep() throws {
        // A constant quarter step would round to silence without dither.
        let url = try temporaryDirectory().appending(path: "quarter.wav")
        try AudioFiles.writeWAV([[Float](repeating: 0.25 / 32767, count: 48_000)], sampleRate: 48_000, encoding: .int16Dithered, to: url)

        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatInt16, interleaved: false)
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let samples = UnsafeBufferPointer(start: buffer.int16ChannelData![0], count: Int(buffer.frameLength))
        let mean = Double(samples.reduce(0) { $0 + Int($1) }) / Double(samples.count)
        #expect(abs(mean - 0.25) < 0.05, "\(mean)")
    }

    @Test func readsOtherSampleRatesAtTheRequestedRate() throws {
        let tone = (0..<48_000).map { Float(0.5 * sin(2 * Double.pi * 1_000 * Double($0) / 48_000)) }
        let url = try temporaryDirectory().appending(path: "48k.wav")
        try AudioFiles.writeWAV([tone, tone], sampleRate: 48_000, to: url)
        let audio = try AudioFiles.readStereo(url, sampleRate: 44_100)
        #expect(abs(audio[0].count - 44_100) <= 2)
        let peak = audio[0][11_000..<33_000].map(abs).max() ?? 0
        #expect(abs(peak - 0.5) < 0.01)
    }
}

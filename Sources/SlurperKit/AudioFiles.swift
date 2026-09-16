import AVFoundation

enum AudioFiles {
    enum Encoding {
        case float32
        /// 16-bit PCM with TPDF dither.
        case int16Dithered
    }

    /// Decodes an audio file to stereo float samples at `sampleRate`.
    static func readStereo(_ url: URL, sampleRate: Double) throws -> [[Float]] {
        let file = try AVAudioFile(forReading: url)
        let source = file.processingFormat
        guard let target = floatFormat(sampleRate: sampleRate, channels: 2) else {
            throw SplitError.audio("cannot convert \(url.lastPathComponent) to \(Int(sampleRate)) Hz stereo")
        }
        guard let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw SplitError.audio("cannot allocate \(file.length) frames to read \(url.lastPathComponent)")
        }
        try file.read(into: input)
        let output = source.sampleRate == sampleRate && source.channelCount == 2 ? input : try convert(input, to: target)
        return channels(of: output)
    }

    /// Resamples `[channel][sample]` float audio with the mastering-quality converter.
    static func resample(_ audio: [[Float]], from sourceRate: Double, to targetRate: Double) throws -> [[Float]] {
        guard sourceRate != targetRate, let frames = audio.first?.count, frames > 0 else { return audio }
        guard let source = floatFormat(sampleRate: sourceRate, channels: audio.count),
              let target = floatFormat(sampleRate: targetRate, channels: audio.count),
              let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(frames))
        else { throw SplitError.audio("cannot allocate \(frames) frames for resampling") }
        input.frameLength = AVAudioFrameCount(frames)
        for (c, samples) in audio.enumerated() {
            samples.withUnsafeBufferPointer { input.floatChannelData![c].update(from: $0.baseAddress!, count: frames) }
        }
        return try channels(of: convert(input, to: target))
    }

    static func writeWAV(_ audio: [[Float]], sampleRate: Double, encoding: Encoding = .float32, to url: URL) throws {
        let frames = audio.first?.count ?? 0
        let common: AVAudioCommonFormat = encoding == .float32 ? .pcmFormatFloat32 : .pcmFormatInt16
        guard let format = AVAudioFormat(
                commonFormat: common, sampleRate: sampleRate, channels: AVAudioChannelCount(audio.count), interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(max(frames, 1)))
        else { throw SplitError.audio("cannot allocate output buffer for \(url.lastPathComponent)") }
        buffer.frameLength = AVAudioFrameCount(frames)

        switch encoding {
        case .float32:
            for (c, samples) in audio.enumerated() {
                samples.withUnsafeBufferPointer { buffer.floatChannelData![c].update(from: $0.baseAddress!, count: frames) }
            }
        case .int16Dithered:
            // Seed per file so stems summed back together don't share a dither pattern.
            var noise = Xorshift(seed: UInt64(truncatingIfNeeded: url.path.hashValue) | 1)
            for (c, samples) in audio.enumerated() {
                let output = buffer.int16ChannelData![c]
                for k in 0..<frames {
                    let dither = noise.unit() + noise.unit() - 1
                    let value = (samples[k] * 32767 + dither).rounded()
                    output[k] = Int16(max(-32768, min(32767, value)))
                }
            }
        }
        // Not format.settings: it carries AVLinearPCMIsNonInterleaved, which WAV files reject with a log line.
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: audio.count,
            AVLinearPCMBitDepthKey: encoding == .float32 ? 32 : 16,
            AVLinearPCMIsFloatKey: encoding == .float32,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: common, interleaved: false)
        try file.write(from: buffer)
    }

    private static func floatFormat(sampleRate: Double, channels: Int) -> AVAudioFormat? {
        AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: AVAudioChannelCount(channels), interleaved: false)
    }

    private static func convert(_ input: AVAudioPCMBuffer, to target: AVAudioFormat) throws -> AVAudioPCMBuffer {
        let source = input.format
        guard let converter = AVAudioConverter(from: source, to: target) else {
            throw SplitError.audio("cannot convert \(source) to \(target)")
        }
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        if source.channelCount == 1, target.channelCount == 2 { converter.channelMap = [0, 0] }
        let capacity = AVAudioFrameCount((Double(input.frameLength) * target.sampleRate / source.sampleRate).rounded(.up)) + 8192
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
            throw SplitError.audio("cannot allocate \(capacity) frames")
        }
        nonisolated(unsafe) var supplied = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, inputStatus in
            if supplied {
                inputStatus.pointee = .endOfStream
                return nil
            }
            supplied = true
            inputStatus.pointee = .haveData
            return input
        }
        if status == .error { throw error ?? SplitError.audio("sample rate conversion failed") }
        return output
    }

    private static func channels(of buffer: AVAudioPCMBuffer) -> [[Float]] {
        let frames = Int(buffer.frameLength)
        return (0..<Int(buffer.format.channelCount)).map {
            Array(UnsafeBufferPointer(start: buffer.floatChannelData![$0], count: frames))
        }
    }
}

private struct Xorshift {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    /// Uniform in [0, 1).
    mutating func unit() -> Float {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return Float(state >> 40) / Float(1 << 24)
    }
}

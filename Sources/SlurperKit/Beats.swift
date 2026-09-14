import Accelerate

/// Tempo and bar lines from a drum stem: an autocorrelation tempo, Ellis's dynamic-programming beat tracker
/// ("Beat Tracking by Dynamic Programming", 2007), and the beat of the bar with the most kick as the downbeat.
enum Beats {
    struct Grid: Sendable {
        var bpm: Double
        /// Sample positions where bars start.
        var downbeats: [Int]
    }

    static let beatsPerBar = 4
    /// RMS band-level difference, in dB, under which a loop repeats an earlier one. On a 170 BPM drum and bass
    /// track, 4-bar bass loops fell into repeats at 1.7-3.6 dB from their nearest neighbor and changes at 6.2 dB
    /// and up; the drum loops had no clear gap.
    static let repeatThreshold: Float = 4.5

    /// `nil` when the drums have no steady beat. `bpm` skips the tempo estimate, for when it lands on half or
    /// double the tempo you count in.
    static func grid(of drums: [[Float]], sampleRate: Double, bpm: Double? = nil) -> Grid? {
        let mono = Transients.mono(drums)
        let kickBins = 1..<max(2, Int(150 * Double(Transients.frameSize) / sampleRate) + 1)
        let flux = Transients.spectralFlux(mono, bands: [0..<Transients.frameSize / 2, kickBins])
        let envelope = smoothed(flux[0], deviation: 2)
        let fps = sampleRate / Double(Transients.hop)
        guard let period = bpm.map({ 60 * fps / $0 }) ?? period(of: envelope, framesPerSecond: fps) else { return nil }
        let beats = track(envelope, period: period)
        guard beats.count >= 2 * beatsPerBar else { return nil }

        let kick = flux[1]
        let phase = (0..<beatsPerBar).max { p, q in
            let strength = { (phase: Int) -> Float in
                let frames = stride(from: phase, to: beats.count, by: beatsPerBar).map { beats[$0] }
                return frames.map { kick[max(0, $0 - 2)...min(kick.count - 1, $0 + 2)].max()! }.reduce(0, +) / Float(frames.count)
            }
            return strength(p) < strength(q)
        }!

        // Frames are coarse and flux peaks as a hit enters the frame, so bar lines move to the nearby attack.
        let attacks = Transients.onsets(of: mono, sampleRate: sampleRate)
        let reach = Int(0.03 * sampleRate)
        let downbeats = stride(from: phase, to: beats.count, by: beatsPerBar).map { i in
            let position = beats[i] * Transients.hop
            let nearest = attacks.min { abs($0 - position) < abs($1 - position) }
            return nearest.map { abs($0 - position) <= reach ? $0 : position } ?? position
        }
        let span = Double((beats.last! - beats.first!) * Transients.hop)
        return Grid(bpm: 60 * sampleRate * Double(beats.count - 1) / span, downbeats: downbeats)
    }

    /// Frames per beat: the autocorrelation peak of the onset envelope between 70 and 180 BPM, weighted toward
    /// 120 BPM by a log-normal prior an octave wide. Double that tempo wins if it correlates at least 80% as well,
    /// since drum and bass music repeats every two beats nearly as strongly as every beat.
    static func period(of envelope: [Float], framesPerSecond fps: Double) -> Double? {
        let lags = Int(60 * fps / 180)...Int(60 * fps / 70)
        guard envelope.count > 4 * lags.upperBound else { return nil }
        let centered = vDSP.add(-vDSP.mean(envelope), envelope)
        let n = centered.count
        let correlations = lags.map { lag in vDSP.dot(centered[0..<n - lag], centered[lag..<n]) / Float(n - lag) }
        let weighted = correlations.indices.map { i in
            correlations[i] * Float(exp(-0.5 * pow(log2(60 * fps / Double(lags.lowerBound + i) / 120), 2)))
        }
        guard var best = weighted.indices.max(by: { weighted[$0] < weighted[$1] }), weighted[best] > 0 else { return nil }
        let half = (lags.lowerBound + best) / 2 - lags.lowerBound
        if half >= 1, half + 1 < correlations.count,
           let faster = (half - 1...half + 1).max(by: { correlations[$0] < correlations[$1] }),
           correlations[faster] >= 0.8 * correlations[best] {
            best = faster
        }
        guard best > 0, best < correlations.count - 1 else { return Double(lags.lowerBound + best) }
        let (left, middle, right) = (correlations[best - 1], correlations[best], correlations[best + 1])
        let curve = left - 2 * middle + right
        return Double(lags.lowerBound + best) + (curve < 0 ? Double(0.5 * (left - right) / curve) : 0)
    }

    /// Beat frames that trade onset strength against spacing away from `period`, trimmed of weak beats at
    /// either end. Tightness 400 (librosa uses 100) makes a 10% short beat cost about 4 average onsets, which
    /// programmed music never needs and a live drummer's drift doesn't reach.
    static func track(_ envelope: [Float], period: Double, tightness: Float = 400) -> [Int] {
        let n = envelope.count
        let spread = vDSP.rootMeanSquare(vDSP.add(-vDSP.mean(envelope), envelope))
        guard n > 2, spread > 0 else { return [] }
        let local = smoothed(vDSP.divide(envelope, spread), deviation: period / 32)
        let offsets = Array(-Int((2 * period).rounded())...(-Int((period / 2).rounded())))
        let penalties = offsets.map { -tightness * pow(log(Float(-$0) / Float(period)), 2) }

        var cumulative = [Float](repeating: 0, count: n)
        var backlink = [Int](repeating: -1, count: n)
        let quiet = 0.01 * local.max()!
        var started = false
        for i in 0..<n {
            var (best, link) = (-Float.infinity, -1)
            for (k, offset) in offsets.enumerated() {
                let score = penalties[k] + (i + offset >= 0 ? cumulative[i + offset] : 0)
                if score > best { (best, link) = (score, i + offset) }
            }
            cumulative[i] = local[i] + best
            // Nothing before the music starts can be a beat.
            if started || local[i] >= quiet { (backlink[i], started) = (link, true) }
        }

        // The last beat is the last local maximum of the cumulative score that reaches half their median.
        let maxima = (1..<n - 1).filter { cumulative[$0] > cumulative[$0 - 1] && cumulative[$0] >= cumulative[$0 + 1] }
        guard !maxima.isEmpty else { return [] }
        let median = maxima.map { cumulative[$0] }.sorted()[maxima.count / 2]
        guard var beat = maxima.last(where: { cumulative[$0] >= 0.5 * median }) else { return [] }
        var beats = [beat]
        while backlink[beat] >= 0 {
            beat = backlink[beat]
            beats.append(beat)
        }
        beats.reverse()

        let strengths = beats.map { local[$0] }
        let weak = 0.5 * vDSP.rootMeanSquare(strengths)
        guard let first = strengths.firstIndex(where: { $0 >= weak }), let last = strengths.lastIndex(where: { $0 >= weak }) else { return [] }
        return Array(beats[first...last])
    }

    /// Gaussian smoothing, the same length as `signal`.
    static func smoothed(_ signal: [Float], deviation: Double) -> [Float] {
        let radius = max(1, Int((3 * deviation).rounded(.up)))
        let kernel = (-radius...radius).map { Float(exp(-0.5 * pow(Double($0) / deviation, 2))) }
        // vDSP.convolve returns the input length less the kernel length, so one extra zero keeps the signal's length.
        let padding = [Float](repeating: 0, count: radius)
        return vDSP.convolve(padding + signal + padding + [0], withKernel: vDSP.divide(kernel, vDSP.sum(kernel)))
    }
}

extension Beats.Grid {
    /// `bars`-bar loops between bar lines, named by their first bar, leaving out silent loops and loops that
    /// repeat an earlier one.
    func loops(of audio: [[Float]], bars: Int, sampleRate: Double) -> Chops {
        let mono = Transients.mono(audio)
        let floor = vDSP.maximumMagnitude(mono) * Transients.silence
        let ranges = stride(from: 0, to: downbeats.count - bars, by: bars)
            .map { (bar: $0, range: downbeats[$0]..<min(mono.count, downbeats[$0 + bars])) }
            .filter { !$0.range.isEmpty && vDSP.maximumMagnitude(mono[$0.range]) > floor }

        // Band levels at the start of each 16th note.
        let steps = bars * Beats.beatsPerBar * 4
        let bands = BandLevels(sampleRate: sampleRate)
        let features = ranges.map { loop in
            Kit.normalized((0..<steps).flatMap { step in
                let from = loop.range.lowerBound + loop.range.count * step / steps
                return bands.levels(Array(mono[from..<min(loop.range.upperBound, from + 2048)]))
            })
        }
        let kept = Kit.group(features, order: Array(ranges.indices), threshold: Beats.repeatThreshold).map { $0.min()! }.sorted()
        return Chops(
            folder: "loops_\(Int(bpm.rounded()))bpm",
            pieces: kept.map {
                Chops.Piece(name: "bar" + Chops.label(ranges[$0].bar + 1, of: downbeats.count, digits: 3), range: ranges[$0].range)
            },
            fadeIn: 0.001, fadeOut: 0.002)
    }
}

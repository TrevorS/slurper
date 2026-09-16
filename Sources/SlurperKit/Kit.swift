import Accelerate

/// Groups a stem's hits by sound and keeps the cleanest example of each.
enum Kit {
    /// RMS band-level difference, in dB, under which two hits are the same sound.
    static let threshold: Float = 6
    /// Sounds heard fewer times than this are taken for fills or separation artifacts and dropped, unless no sound
    /// reaches this count.
    static let fewest = 3

    /// One piece per sound, most frequent first, named `01`, `02`, ...
    static func chops(of audio: [[Float]], sampleRate: Double) -> Chops {
        let mono = Transients.mono(audio)
        let attacks = Transients.onsets(of: mono, sampleRate: sampleRate)
        let hits = Transients.hits(at: attacks, in: mono, sampleRate: sampleRate)
        let bands = BandLevels(sampleRate: sampleRate)
        let gap = Int(Transients.minimumGap * sampleRate)
        let features = attacks.indices.map { i in
            fingerprint(mono, attack: attacks[i], end: i + 1 < attacks.count ? attacks[i + 1] : mono.count, bands: bands)
        }
        let peaks = attacks.map { vDSP.maximumMagnitude(mono[$0..<min(mono.count, $0 + gap)]) }

        // Loudest first, so each group starts from a clear hit.
        var groups = group(features, order: attacks.indices.sorted { peaks[$0] > peaks[$1] }, threshold: threshold)
        if groups.contains(where: { $0.count >= fewest }) { groups.removeAll { $0.count < fewest } }
        groups.sort { ($0.count, -$0.min()!) > ($1.count, -$1.min()!) }

        let preroll = Int(Transients.preroll * sampleRate)
        /// Tail of the previous hit in the preroll, in dB against this hit's peak, less 6 dB per doubling of the
        /// room before the next hit (from 50 to 400 ms). Lower is cleaner.
        func mess(_ i: Int) -> Double {
            let tail = mono[max(0, attacks[i] - preroll)..<attacks[i]]
            let tailLevel = tail.isEmpty ? 0 : vDSP.rootMeanSquare(tail)
            let next = i + 1 < attacks.count ? attacks[i + 1] : mono.count
            let room = min(max(Double(next - attacks[i]) / sampleRate, 0.05), 0.4)
            return 20 * log10(Double(max(tailLevel, 1e-9) / max(peaks[i], 1e-9))) - 6 * log2(room / 0.05)
        }
        let pieces = groups.enumerated().map { n, members in
            let mean = vDSP.divide(members.dropFirst().reduce(features[members[0]]) { vDSP.add($0, features[$1]) }, Float(members.count))
            let distances = Dictionary(uniqueKeysWithValues: members.map { ($0, distance(features[$0], mean)) })
            let typical = members.sorted { distances[$0]! < distances[$1]! }.prefix((members.count + 1) / 2)
            let cleanest = typical.min { mess($0) < mess($1) }!
            return Chops.Piece(name: Chops.label(n + 1, of: groups.count, digits: 2), range: hits[cleanest])
        }
        return Chops(folder: "kit", pieces: pieces, fadeIn: Transients.preroll, fadeOut: 0.005)
    }

    /// Band levels over the 0-1024, 1024-2048 and 2048-4096 samples after the attack (23, 46 and 93 ms at
    /// 44.1 kHz), with the next hit onward silenced.
    static func fingerprint(_ mono: [Float], attack: Int, end: Int, bands: BandLevels) -> [Float] {
        normalized([0..<1024, 1024..<2048, 2048..<4096].flatMap { window in
            var samples = [Float](repeating: 0, count: window.count)
            let (from, to) = (attack + window.lowerBound, min(end, mono.count, attack + window.upperBound))
            if from < to { samples.replaceSubrange(0..<to - from, with: mono[from..<to]) }
            return bands.levels(samples)
        })
    }

    /// Levels floored 60 dB below their loudest, less their mean, so loudness alone doesn't tell sounds apart.
    static func normalized(_ levels: [Float]) -> [Float] {
        guard let top = levels.max() else { return levels }
        let floored = vDSP.clip(levels, to: (top - 60)...top)
        return vDSP.add(-vDSP.mean(floored), floored)
    }

    /// RMS difference in dB.
    static func distance(_ a: [Float], _ b: [Float]) -> Float {
        (vDSP.distanceSquared(a, b) / Float(a.count)).squareRoot()
    }

    /// Taken in `order`, each vector joins the group with the nearest mean if that is within `threshold`, or starts
    /// a group. Then groups with means within `threshold` merge, closest first. Returns indices into `vectors`.
    static func group(_ vectors: [[Float]], order: [Int], threshold: Float) -> [[Int]] {
        var members: [[Int]] = []
        var means: [[Float]] = []
        for i in order {
            let distances = means.map { distance($0, vectors[i]) }
            if let nearest = distances.indices.min(by: { distances[$0] < distances[$1] }), distances[nearest] < threshold {
                members[nearest].append(i)
                let step = vDSP.multiply(1 / Float(members[nearest].count), vDSP.subtract(vectors[i], means[nearest]))
                means[nearest] = vDSP.add(means[nearest], step)
            } else {
                members.append([i])
                means.append(vectors[i])
            }
        }

        while true {
            var closest: (a: Int, b: Int, distance: Float)?
            for a in means.indices {
                for b in a + 1..<means.count {
                    let d = distance(means[a], means[b])
                    if d < threshold, d < closest?.distance ?? .infinity { closest = (a, b, d) }
                }
            }
            guard let pair = closest else { break }
            let (weightA, weightB) = (Float(members[pair.a].count), Float(members[pair.b].count))
            means[pair.a] = vDSP.divide(vDSP.add(vDSP.multiply(weightA, means[pair.a]), vDSP.multiply(weightB, means[pair.b])), weightA + weightB)
            members[pair.a] += members.remove(at: pair.b)
            means.remove(at: pair.b)
        }
        return members
    }
}

/// Levels in dB of 24 mel-spaced bands up to 16 kHz, from a 2048-sample FFT.
final class BandLevels {
    static let count = 24
    private static let size = 2048
    private let transform = try! vDSP.DiscreteFourierTransform(
        count: size, direction: .forward, transformType: .complexComplex, ofType: Float.self)
    /// Bin edges of the bands, skipping DC, at least one bin wide.
    private let edges: [Int]

    init(sampleRate: Double) {
        let mel = { (hz: Double) in 2595 * log10(1 + hz / 700) }
        let top = mel(min(16_000, sampleRate / 2))
        var edges = [1]
        for k in 1...Self.count {
            let hz = 700 * (pow(10, top * Double(k) / Double(Self.count) / 2595) - 1)
            edges.append(max(edges[k - 1] + 1, Int((hz * Double(Self.size) / sampleRate).rounded())))
        }
        self.edges = edges
    }

    /// Up to 2048 samples, Hann-windowed at their own length and zero-padded.
    func levels(_ samples: [Float]) -> [Float] {
        let n = min(samples.count, Self.size)
        var input = [Float](repeating: 0, count: Self.size)
        if n > 1 {
            let window = vDSP.window(ofType: Float.self, usingSequence: .hanningDenormalized, count: n, isHalfWindow: false)
            input.replaceSubrange(0..<n, with: vDSP.multiply(samples[0..<n], window))
        }
        var real = [Float](repeating: 0, count: Self.size), imaginary = real
        transform.transform(
            inputReal: input, inputImaginary: [Float](repeating: 0, count: Self.size), outputReal: &real, outputImaginary: &imaginary)
        let power = vDSP.add(vDSP.square(real[0..<Self.size / 2]), vDSP.square(imaginary[0..<Self.size / 2]))
        return (0..<Self.count).map { 10 * log10(vDSP.sum(power[edges[$0]..<min(power.count, edges[$0 + 1])]) + 1e-12) }
    }
}

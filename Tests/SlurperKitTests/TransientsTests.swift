import Foundation
import Testing
@testable import SlurperKit

struct TransientsTests {
    /// Refinement lands within one 64-sample block of the attack; two milliseconds leaves room.
    private static let tolerance = 88

    @Test func findsHitsWithinTwoMilliseconds() throws {
        var audio = noiseFloor(seconds: 2)
        // A ghost note, a long tail with a hit on top of it, and plain hits.
        let hits: [(seconds: Double, amplitude: Float, decay: Double)] = [(0.2, 0.8, 0.04), (0.55, 0.1, 0.04), (0.8, 0.6, 0.3), (1.1, 0.6, 0.04), (1.6, 0.3, 0.04)]
        for hit in hits { burst(&audio, at: hit.seconds, amplitude: hit.amplitude, decay: hit.decay, length: hit.decay * 4) }

        let onsets = Transients.onsets(of: audio, sampleRate: testRate)
        try #require(onsets.count == hits.count, "\(onsets)")
        for (onset, hit) in zip(onsets, hits) {
            #expect(abs(onset - Int(hit.seconds * testRate)) <= Self.tolerance, "\(hit.seconds) s")
        }
    }

    @Test func findsEveryHitInDenseSixteenths() throws {
        var audio = noiseFloor(seconds: 10)
        let step = 60.0 / 170 / 4
        let hits = (0..<Int(9.9 / step)).map { Double($0) * step }
        for (k, seconds) in hits.enumerated() {
            burst(&audio, at: seconds, amplitude: k % 4 == 0 ? 0.9 : 0.25, decay: 0.02, length: 0.08)
        }

        let onsets = Transients.onsets(of: audio, sampleRate: testRate)
        try #require(onsets.count == hits.count)
        #expect(zip(onsets, hits).allSatisfy { abs($0 - Int($1 * testRate)) <= Self.tolerance })
        let ranges = Transients.hits(at: onsets, in: audio, sampleRate: testRate)
        #expect(zip(ranges, ranges.dropFirst()).allSatisfy { $0.upperBound == $1.lowerBound })
    }

    @Test func ignoresSilenceSteadyTonesAndStops() {
        #expect(Transients.onsets(of: [Float](repeating: 0, count: 44_100), sampleRate: testRate).isEmpty)
        // A tone from 0.2 s that stops dead at 1 s: an onset where it starts, none where it stops.
        var audio = noiseFloor(seconds: 2)
        for k in Int(0.2 * testRate)..<Int(testRate) {
            audio[k] += Float(0.5 * sin(2 * Double.pi * 440 * Double(k) / testRate))
        }
        let onsets = Transients.onsets(of: audio, sampleRate: testRate)
        #expect(onsets.count == 1 && abs(onsets[0] - Int(0.2 * testRate)) <= Self.tolerance, "\(onsets)")
    }

    @Test func gatesQuietHitsAndMergesFlams() {
        // A hit 46 dB below the loudest is dropped; one 20 dB below is kept.
        for (amplitude, count) in [(Float(0.004), 1), (0.08, 2)] {
            var audio = noiseFloor(seconds: 1)
            burst(&audio, at: 0.1, amplitude: 0.8, decay: 0.04, length: 0.15)
            burst(&audio, at: 0.5, amplitude: amplitude, decay: 0.04, length: 0.15)
            #expect(Transients.onsets(of: audio, sampleRate: testRate).count == count, "\(amplitude)")
        }

        var flam = noiseFloor(seconds: 1)
        burst(&flam, at: 0.1, amplitude: 0.5, decay: 0.005, length: 0.02)
        burst(&flam, at: 0.125, amplitude: 0.5, decay: 0.005, length: 0.02)
        #expect(Transients.onsets(of: flam, sampleRate: testRate).count == 1)
    }

    @Test func hitsStartBeforeTheAttackAndDropTrailingSilence() throws {
        var audio = noiseFloor(seconds: 1)
        burst(&audio, at: 0.2, amplitude: 0.8, decay: 0.04, length: 0.15)
        burst(&audio, at: 0.6, amplitude: 0.8, decay: 0.04, length: 0.15)

        let onsets = Transients.onsets(of: audio, sampleRate: testRate)
        try #require(onsets.count == 2)
        let ranges = Transients.hits(at: onsets, in: audio, sampleRate: testRate)
        let preroll = Int(Transients.preroll * testRate)
        #expect(ranges.map(\.lowerBound) == onsets.map { $0 - preroll })
        // The bursts end at 0.35 and 0.75 s.
        #expect((Int(0.35 * testRate)...Int(0.37 * testRate)).contains(ranges[0].upperBound))
        #expect((Int(0.75 * testRate)...Int(0.77 * testRate)).contains(ranges[1].upperBound))
    }

    @Test func hitsRunAtMostTwoSeconds() {
        let tone = (0..<220_500).map { Float(0.5 * sin(2 * Double.pi * 440 * Double($0) / testRate)) }
        #expect(Transients.hits(at: [44_100], in: tone, sampleRate: testRate) == [(44_100 - 132)..<(44_100 + 88_200)])
    }
}

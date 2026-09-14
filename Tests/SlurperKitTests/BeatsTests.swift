import Foundation
import Testing
@testable import SlurperKit

struct BeatsTests {
    @Test func findsTempoAndBarLines() throws {
        let (audio, bars) = groove(bars: 16, from: 0.4, seconds: 32) { _ in 128 }
        let grid = try #require(Beats.grid(of: [audio, audio], sampleRate: testRate))
        #expect(abs(grid.bpm - 128) < 0.5, "\(grid.bpm)")
        #expect(grid.downbeats.count >= 14)
        for downbeat in grid.downbeats {
            #expect(bars.contains { abs($0 - downbeat) <= 441 }, "\(Double(downbeat) / testRate) s")
        }
    }

    @Test func followsADriftingTempo() throws {
        let (audio, bars) = groove(bars: 14, from: 0.4, seconds: 30) { 120 + 6 * $0 / 28 }
        let grid = try #require(Beats.grid(of: [audio, audio], sampleRate: testRate))
        #expect(grid.downbeats.count >= 12)
        for downbeat in grid.downbeats {
            #expect(bars.contains { abs($0 - downbeat) <= 441 }, "\(Double(downbeat) / testRate) s")
        }
    }

    @Test func givenTempoOverridesTheEstimate() throws {
        let (audio, bars) = groove(bars: 16, from: 0.4, seconds: 32) { _ in 128 }
        let grid = try #require(Beats.grid(of: [audio, audio], sampleRate: testRate, bpm: 64))
        #expect(abs(grid.bpm - 64) < 0.5, "\(grid.bpm)")
        // A bar at half tempo spans two real bars. Which beat it starts on is a guess: every other real beat is a snare.
        #expect(grid.downbeats.count >= 6)
        let twoBars = Double(bars[2] - bars[0])
        #expect(zip(grid.downbeats, grid.downbeats.dropFirst()).allSatisfy { abs(Double($1 - $0) - twoBars) < 0.02 * twoBars })
    }

    @Test func prefersTheFasterOfTwoTemposThatCorrelateAlike() throws {
        // Beats at 170 BPM, every other one louder: 85 BPM correlates best, 170 nearly as well.
        let fps = testRate / 256
        let period = 60 * fps / 170
        var envelope = [Float](repeating: 0, count: Int(fps * 30))
        for beat in 0..<Int(Double(envelope.count - 1) / period) {
            envelope[Int((Double(beat) * period).rounded())] = beat % 2 == 0 ? 1 : 0.8
        }
        let found = try #require(Beats.period(of: Beats.smoothed(envelope, deviation: 2), framesPerSecond: fps))
        #expect(abs(found - period) < 0.5, "\(60 * fps / found) BPM")
    }

    @Test func silenceHasNoGrid() {
        #expect(Beats.grid(of: [[Float](repeating: 0, count: 441_000)], sampleRate: testRate) == nil)
    }

    @Test func loopsLeaveOutSilenceAndRepeats() {
        // Two-second bars: 1-6 repeat a groove, 7-8 are hats alone, 9-10 are silent.
        var audio = noiseFloor(seconds: 20)
        for bar in 0..<8 {
            for beat in 0..<4 {
                let seconds = Double(bar * 4 + beat) * 0.5
                if bar < 6 {
                    if beat % 2 == 0 { kick(&audio, at: seconds, amplitude: 0.9) } else { snare(&audio, at: seconds, amplitude: 0.7) }
                }
                hat(&audio, at: seconds + 0.25, amplitude: 0.3)
            }
        }
        let grid = Beats.Grid(bpm: 120, downbeats: Array(stride(from: 0, through: 882_000, by: 88_200)))
        let loops = grid.loops(of: [audio, audio], bars: 2, sampleRate: testRate)
        #expect(loops.folder == "loops_120bpm")
        #expect(loops.pieces == [.init(name: "bar001", range: 0..<176_400), .init(name: "bar007", range: 529_200..<705_600)])
    }

    /// Kick on 1, snare on 2 and 4, a softer kick on 3, and hats between beats, for `bars` bars from `from`
    /// seconds at `bpm(time)`. Returns the audio and where each bar starts.
    private func groove(bars: Int, from start: Double, seconds: Double, bpm: (Double) -> Double) -> ([Float], [Int]) {
        var audio = noiseFloor(seconds: seconds)
        var starts: [Int] = []
        var time = start
        for beat in 0..<(bars * 4) {
            let length = 60 / bpm(time)
            switch beat % 4 {
            case 0:
                kick(&audio, at: time, amplitude: 0.9)
                starts.append(Int(time * testRate))
            case 2:
                kick(&audio, at: time, amplitude: 0.45)
            default:
                snare(&audio, at: time, amplitude: 0.7)
            }
            hat(&audio, at: time + length / 2, amplitude: 0.3)
            time += length
        }
        return (audio, starts)
    }
}

import Foundation
import Testing
@testable import SlurperKit

struct KitTests {
    @Test func keepsOneExampleOfEachSound() {
        // 8 bars at 120 BPM: kicks on 1 and 3, snares on 2 and 4, hats on the offbeat 8ths, at varying velocity.
        var audio = noiseFloor(seconds: 17)
        var sounds: [(sample: Int, name: String)] = []
        var random = SeededRandom(seed: 5)
        for step in 0..<(8 * 16) {
            let seconds = 0.25 + Double(step) * 60 / 120 / 4
            let velocity = Float.random(in: 0.5...1, using: &random)
            switch step % 16 {
            case 0, 8:
                kick(&audio, at: seconds, amplitude: 0.9 * velocity)
                sounds.append((Int(seconds * testRate), "kick"))
            case 4, 12:
                snare(&audio, at: seconds, amplitude: 0.8 * velocity)
                sounds.append((Int(seconds * testRate), "snare"))
            case 2, 6, 10, 14:
                hat(&audio, at: seconds, amplitude: 0.4 * velocity)
                sounds.append((Int(seconds * testRate), "hat"))
            default:
                continue
            }
        }

        let kit = Kit.chops(of: [audio, audio], sampleRate: testRate)
        let names = kit.pieces.map { piece in
            sounds.min { abs($0.sample - piece.range.lowerBound) < abs($1.sample - piece.range.lowerBound) }!.name
        }
        #expect(names.first == "hat", "\(names)")
        #expect(names.sorted() == ["hat", "kick", "snare"])
        #expect(kit.pieces.map(\.name) == ["01", "02", "03"])
        #expect(kit.folder == "kit")
    }

    @Test func keepsRareSoundsWhenNoSoundIsCommon() {
        // Two kicks and two hats: neither reaches `Kit.fewest`, so both stay.
        var audio = noiseFloor(seconds: 3)
        for (index, seconds) in [0.2, 0.9, 1.6, 2.3].enumerated() {
            if index % 2 == 0 {
                kick(&audio, at: seconds, amplitude: 0.9)
            } else {
                hat(&audio, at: seconds, amplitude: 0.5)
            }
        }
        #expect(Kit.chops(of: [audio], sampleRate: testRate).pieces.count == 2)
    }

    @Test func silenceHasNoKit() {
        #expect(Kit.chops(of: [[Float](repeating: 0, count: 44_100)], sampleRate: testRate).pieces.isEmpty)
    }

    @Test func groupsNearVectorsAndMergesGroupsThatDriftTogether() {
        let vectors: [[Float]] = [[0, 0], [1, 0], [10, 10], [0, 1], [10, 11]]
        let groups = Kit.group(vectors, order: Array(vectors.indices), threshold: 2)
        #expect(groups.map { $0.sorted() }.sorted { $0[0] < $1[0] } == [[0, 1, 3], [2, 4]])

        // [3, 0] starts its own group, but once [1.5, 0] pulls the first group's mean over, the two merge.
        #expect(Kit.group([[0, 0], [3, 0], [1.5, 0]], order: [0, 1, 2], threshold: 2).map { $0.sorted() } == [[0, 1, 2]])
    }
}

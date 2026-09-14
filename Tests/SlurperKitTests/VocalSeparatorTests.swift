import Accelerate
import Foundation
import Testing
@testable import SlurperKit

struct VocalSeparatorTests {
    @Test func reflectPadMatchesNumpy() {
        #expect(reflectPad([1, 2, 3], 2) == [3, 2, 1, 2, 3, 2, 1])
        #expect(reflectPad([1, 2, 3], 4) == [1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3])
    }

    /// Compares one 8 s chunk against the golden vocals published in the Core AI model's repository.
    @Test(.enabled(if: modelsInstalled))
    func matchesPublishedGolden() async throws {
        let raw = try await golden("golden_raw.f32")
        let expected = try await golden("golden_vocals.f32")
        let n = VocalSeparator.chunkSamples

        let separator = try await VocalSeparator(model: StemSplitter.vocalModel, chunksAtOnce: 1)
        let vocals = try await separator.vocals(ofChunk: [Array(raw[0..<n]), Array(raw[n..<2 * n])])

        let got = vocals[0] + vocals[1]
        let cosine = vDSP.dot(got, expected) / (vDSP.sumOfSquares(got) * vDSP.sumOfSquares(expected)).squareRoot()
        #expect(cosine > 0.9999)
    }

    @Test(.enabled(if: modelsInstalled))
    func parallelChunksGiveIdenticalVocals() async throws {
        let mix = noise(frames: 44_100 * 20, channels: 2, amplitude: 0.1)
        let serial = try await VocalSeparator(model: StemSplitter.vocalModel, chunksAtOnce: 1).vocals(of: mix) { _ in }
        let parallel = try await VocalSeparator(model: StemSplitter.vocalModel, chunksAtOnce: 3).vocals(of: mix) { _ in }
        #expect(serial[0].count == mix[0].count)
        #expect(parallel == serial)
    }

    private func golden(_ name: String) async throws -> [Float] {
        let cached = URL.cachesDirectory.appending(path: "SlurperTests/\(StemSplitter.roformerRevision)/\(name)")
        if !FileManager.default.fileExists(atPath: cached.path) {
            let remote = URL(string: "https://huggingface.co/\(StemSplitter.roformerRepo)/resolve/\(StemSplitter.roformerRevision)/\(name)")!
            let (download, _) = try await URLSession.shared.download(from: remote)
            try FileManager.default.createDirectory(at: cached.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.moveItem(at: download, to: cached)
        }
        return try Data(contentsOf: cached).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}

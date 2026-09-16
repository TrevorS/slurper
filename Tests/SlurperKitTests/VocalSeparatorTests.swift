import Accelerate
import Foundation
import Testing
@testable import SlurperKit

struct VocalSeparatorTests {
    @Test func reflectPadMatchesNumpy() {
        #expect(reflectPad([1, 2, 3], 2) == [3, 2, 1, 2, 3, 2, 1])
        #expect(reflectPad([1, 2, 3], 4) == [1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3])
    }

    /// Compares one 8 s chunk against PyTorch's vocals, written next to the model by
    /// `scripts/convert_melband_roformer.py` and downloaded with it.
    @Test(.enabled(if: modelsInstalled))
    func matchesPublishedGolden() async throws {
        let raw = try golden("golden_raw.f32")
        let expected = try golden("golden_vocals.f32")
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

    private func golden(_ name: String) throws -> [Float] {
        let file = StemSplitter.vocalModel.deletingLastPathComponent().appending(path: name)
        return try Data(contentsOf: file).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}

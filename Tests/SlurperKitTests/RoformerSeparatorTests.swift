import Accelerate
import Foundation
import Testing
@testable import SlurperKit

struct RoformerSeparatorTests {
    @Test func reflectPadMatchesNumpy() {
        #expect(reflectPad([1, 2, 3], 2) == [3, 2, 1, 2, 3, 2, 1])
        #expect(reflectPad([1, 2, 3], 4) == [1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3])
    }

    /// Compares one 8 s chunk against PyTorch's vocals, written next to the model by
    /// `scripts/convert_melband_roformer.py` and downloaded with it.
    @Test(.enabled(if: vocalModelInstalled))
    func vocalsMatchPublishedGolden() async throws {
        try await expectGolden(StemSplitter.vocals, stem: "golden_vocals.f32")
    }

    /// Compares one 8 s chunk against PyTorch's horns, written next to the model by
    /// `scripts/convert_bs_roformer.py` and downloaded with it.
    @Test(.enabled(if: hornsModelInstalled))
    func hornsMatchPublishedGolden() async throws {
        try await expectGolden(StemSplitter.horns, stem: "golden_horns.f32")
    }

    @Test(.enabled(if: hornsModelInstalled))
    func rejectsTheWrongHop() async throws {
        await #expect(throws: SplitError.self) {
            try await RoformerSeparator(model: StemSplitter.horns.package, hop: StemSplitter.vocals.hop, chunksAtOnce: 1)
        }
    }

    @Test(.enabled(if: vocalModelInstalled))
    func parallelChunksGiveIdenticalStems() async throws {
        let mix = noise(frames: 44_100 * 20, channels: 2, amplitude: 0.1)
        let serial = try await RoformerSeparator(StemSplitter.vocals, chunksAtOnce: 1).stem(of: mix) { _ in }
        let parallel = try await RoformerSeparator(StemSplitter.vocals, chunksAtOnce: 3).stem(of: mix) { _ in }
        #expect(serial[0].count == mix[0].count)
        #expect(parallel == serial)
    }

    /// Runs the model on `golden_raw.f32` next to it and compares against `stem`, PyTorch's output for that chunk.
    private func expectGolden(_ model: StemSplitter.RoformerModel, stem: String) async throws {
        let raw = try golden("golden_raw.f32", by: model.package)
        let expected = try golden(stem, by: model.package)
        let n = RoformerSeparator.chunkSamples

        let separator = try await RoformerSeparator(model, chunksAtOnce: 1)
        let got = try await separator.stem(ofChunk: [Array(raw[0..<n]), Array(raw[n..<2 * n])])
        let flat = got[0] + got[1]
        let cosine = vDSP.dot(flat, expected) / (vDSP.sumOfSquares(flat) * vDSP.sumOfSquares(expected)).squareRoot()
        #expect(cosine > 0.9999)
    }

    private func golden(_ name: String, by model: URL) throws -> [Float] {
        let file = model.deletingLastPathComponent().appending(path: name)
        return try Data(contentsOf: file).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}

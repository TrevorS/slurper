import Foundation

/// Pieces cut from one stem, written together to `<stem>_<folder>/<stem>_<name>.wav`.
struct Chops: Sendable {
    struct Piece: Sendable, Equatable {
        var name: String
        var range: Range<Int>
    }

    var folder: String
    var pieces: [Piece]
    /// Fade lengths in seconds. A piece that starts the audio has nothing to fade in from.
    var fadeIn: Double
    var fadeOut: Double

    /// The same pieces for a copy of the audio at `ratio` times the sample rate, `frames` long.
    func scaled(by ratio: Double, frames: Int) -> Chops {
        var copy = self
        copy.pieces = pieces.map { piece in
            let bounds = [piece.range.lowerBound, piece.range.upperBound].map { min(frames, Int((Double($0) * ratio).rounded())) }
            return Piece(name: piece.name, range: bounds[0]..<bounds[1])
        }
        return copy
    }

    func cut(_ audio: [[Float]], _ range: Range<Int>, sampleRate: Double) -> [[Float]] {
        let fadeIn = range.lowerBound == 0 ? 0 : min(range.count, Int(self.fadeIn * sampleRate))
        let fadeOut = min(range.count - fadeIn, Int(self.fadeOut * sampleRate))
        return audio.map { channel in
            var samples = Array(channel[range])
            for i in 0..<fadeIn { samples[i] *= Float(i) / Float(fadeIn) }
            for i in 0..<fadeOut { samples[samples.count - 1 - i] *= Float(i) / Float(fadeOut) }
            return samples
        }
    }

    /// Writes each piece of `audio` to `<prefix>_<folder>/<prefix>_<name>.wav` in `directory`.
    func write(_ audio: [[Float]], sampleRate: Double, encoding: AudioFiles.Encoding = .float32, prefix: String, in directory: URL) throws {
        guard !pieces.isEmpty else { return }
        let folder = directory.appending(path: "\(prefix)_\(self.folder)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for piece in pieces {
            try AudioFiles.writeWAV(
                cut(audio, piece.range, sampleRate: sampleRate), sampleRate: sampleRate, encoding: encoding,
                to: folder.appending(path: "\(prefix)_\(piece.name).wav"))
        }
    }

    /// `number` zero-padded to fit every number up to `count`, and at least `digits` wide.
    static func label(_ number: Int, of count: Int, digits: Int) -> String {
        String(format: "%0*d", max(digits, String(count).count), number)
    }
}

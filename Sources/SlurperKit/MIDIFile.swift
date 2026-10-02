import Foundation

/// Standard MIDI files laid out as pretty_midi writes SheetSage2's: format 1 at 960 ticks a quarter note and
/// 120 bpm, so seconds map to ticks at 1920 a second, with a tempo track and then one track per part on its own
/// channel.
enum MIDIFile {
    struct Note: Equatable {
        var pitch: Int
        var start: Double
        var end: Double
        var velocity: Int
    }

    struct Track {
        var name: String
        var notes: [Note]
    }

    static let ticksPerQuarter = 960
    static let ticksPerSecond = 1920.0

    static func data(_ tracks: [Track]) -> Data {
        var file = Data("MThd".utf8)
        file += bigEndian(6, bytes: 4) + bigEndian(1, bytes: 2) + bigEndian(tracks.count + 1, bytes: 2)
            + bigEndian(ticksPerQuarter, bytes: 2)
        file += chunk([0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20])  // 500,000 µs a quarter note
        for (index, track) in tracks.enumerated() {
            // Channel 10 is drums in General MIDI.
            let channel = UInt8(index < 9 ? index : index + 1)
            let name = Array(track.name.utf8)
            var events: [UInt8] = [0x00, 0xFF, 0x03] + variableLength(name.count) + name + [0x00, 0xC0 | channel, 0x00]
            // Each note's off precedes any on at the same tick, so a repeated pitch restarts rather than stops.
            let messages = track.notes.flatMap { note -> [(tick: Int, on: Bool, pitch: Int, velocity: Int)] in
                let start = Int((note.start * ticksPerSecond).rounded())
                let end = Int((note.end * ticksPerSecond).rounded())
                return [(start, true, note.pitch, note.velocity), (end, false, note.pitch, 0)]
            }.sorted { ($0.tick, $0.on ? 1 : 0, $0.pitch) < ($1.tick, $1.on ? 1 : 0, $1.pitch) }
            var tick = 0
            for message in messages {
                events += variableLength(message.tick - tick)
                events += [(message.on ? 0x90 : 0x80) | channel, UInt8(message.pitch), UInt8(message.velocity)]
                tick = message.tick
            }
            file += chunk(events)
        }
        return file
    }

    /// A track chunk holding `events`, ended.
    private static func chunk(_ events: [UInt8]) -> Data {
        let body = events + [0x00, 0xFF, 0x2F, 0x00]
        return Data("MTrk".utf8) + bigEndian(body.count, bytes: 4) + Data(body)
    }

    private static func bigEndian(_ value: Int, bytes: Int) -> Data {
        Data((0..<bytes).reversed().map { UInt8(truncatingIfNeeded: value >> (8 * $0)) })
    }

    /// MIDI's variable-length quantity: seven bits a byte, most significant first, the high bit marking more.
    static func variableLength(_ value: Int) -> [UInt8] {
        var bytes = [UInt8(value & 0x7F)]
        var rest = value >> 7
        while rest > 0 {
            bytes.insert(UInt8(rest & 0x7F) | 0x80, at: 0)
            rest >>= 7
        }
        return bytes
    }
}

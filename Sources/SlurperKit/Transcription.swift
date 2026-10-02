import Foundation

/// What SheetSage2 heard in a song, as events stitched across its windows with times in the song, and the
/// annotations made from them as SheetSage2's `exports_sheetsage2.py` makes them.
public struct Transcription: Sendable {
    let events: [SheetSageTokens.Event]
    let duration: Double

    struct Beat: Equatable {
        var time: Double
        /// 1 on the bar line.
        var number: Int
        var numerator: Int
        var denominator: Int
    }

    /// Each event with an eighth-note position under a known meter. The position counts eighth notes, so a
    /// beat of the meter's denominator is `position * denominator / 8`; events off that grid are left out.
    var beats: [Beat] {
        var meter: (numerator: Int, denominator: Int)?
        return events.compactMap { event in
            meter = event.meter ?? meter
            guard let eighth = event.eighthPosition, let meter, eighth * meter.denominator % 8 == 0 else { return nil }
            let position = eighth * meter.denominator / 8
            guard position < meter.numerator else { return nil }
            return Beat(time: event.time, number: position + 1, numerator: meter.numerator, denominator: meter.denominator)
        }
    }

    var downbeats: [Double] { beats.filter { $0.number == 1 }.map(\.time) }

    /// Spans from each labeled event to the next one (or the song's end), leaving out empty spans.
    func spans(_ label: (SheetSageTokens.Event) -> String?) -> [(start: Double, end: Double, label: String)] {
        let labeled = events.compactMap { event in label(event).map { (event.time, $0) } }
        return labeled.indices.compactMap { i in
            let end = i + 1 < labeled.count ? labeled[i + 1].0 : duration
            return end > labeled[i].0 ? (labeled[i].0, end, labeled[i].1) : nil
        }
    }

    var keys: [(start: Double, end: Double, label: String)] { spans(\.key) }
    var sections: [(start: Double, end: Double, label: String)] { spans(\.section) }

    /// Chords spelled for the key at each chord's midpoint; a midpoint on a key change takes the earlier key.
    var chords: [(start: Double, end: Double, label: String)] {
        let keys = keys
        return spans(\.chord).map { chord in
            guard !keys.isEmpty else { return chord }
            let middle = (chord.start + chord.end) / 2
            let key = keys[min(keys.dropLast().count { $0.end < middle }, keys.count - 1)]
            return (chord.start, chord.end, Spelling.chord(chord.label, inKey: key.label))
        }
    }

    /// Melody notes, ending at the song's end at the latest; track 0 is the vocal line, 1 instrumental.
    var notes: [(start: Double, end: Double, pitch: Int, track: Int)] {
        events.flatMap { event in
            zip(event.notes, event.noteEnds).compactMap { note, end in
                let end = min(duration, end)
                return end > event.time ? (event.time, end, note.pitch, note.track) : nil
            }
        }.sorted { ($0.start, $0.end, $0.pitch, $0.track) < ($1.start, $1.end, $1.pitch, $1.track) }
    }

    /// The tempo in quarter notes a minute: the median over consecutive beats, each counted in its meter's unit.
    var bpm: Double? {
        let beats = beats
        let tempos = zip(beats, beats.dropFirst()).compactMap { a, b -> Double? in
            b.time > a.time ? 60 / (b.time - a.time) * 4 / Double(a.denominator) : nil
        }.sorted()
        guard !tempos.isEmpty else { return nil }
        return tempos.count % 2 == 1 ? tempos[tempos.count / 2] : (tempos[tempos.count / 2 - 1] + tempos[tempos.count / 2]) / 2
    }

    /// Bar lines for loops at `sampleRate`, each moved to a drum attack within 30 ms; nil with fewer than two.
    func grid(snappedTo drums: [[Float]], sampleRate: Double) -> Beats.Grid? {
        let downbeats = downbeats
        guard downbeats.count >= 2, let bpm else { return nil }
        let positions = downbeats.map { Int(($0 * sampleRate).rounded()) }
        return Beats.Grid(bpm: bpm, downbeats: Beats.snapped(positions, toAttacksIn: Transients.mono(drums), sampleRate: sampleRate))
    }

    /// "Ab major, 2/4, 88 bpm, 191 bars", from what the song mostly is.
    var summary: String {
        func longest(_ spans: [(start: Double, end: Double, label: String)]) -> String? {
            Dictionary(grouping: spans, by: \.label).mapValues { $0.reduce(0) { $0 + $1.end - $1.start } }
                .max { ($0.value, $1.key) < ($1.value, $0.key) }?.key
        }
        let meters = beats.map { "\($0.numerator)/\($0.denominator)" }
        let meter = Dictionary(grouping: meters, by: { $0 }).max { ($0.value.count, $1.key) < ($1.value.count, $0.key) }?.key
        return [
            longest(keys).map { $0.replacingOccurrences(of: ":", with: " ") }, meter,
            bpm.map { "\(Int($0.rounded())) bpm" }, "\(downbeats.count) bars",
        ].compactMap { $0 }.joined(separator: ", ")
    }

    /// `transcription/` in `folder`: beat, downbeat, key, chord and section annotations as tab-separated LAB
    /// files, the melody's two parts and the chords as MIDI, and all three in transcription.mid.
    func write(to folder: URL) throws {
        let directory = folder.appending(path: "transcription", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        func lab(_ name: String, _ rows: [[Any]]) throws {
            let text = rows.map { $0.map { "\($0)" }.joined(separator: "\t") + "\n" }.joined()
            try text.write(to: directory.appending(path: name), atomically: true, encoding: .utf8)
        }
        try lab("beat.lab", beats.map { [$0.time, $0.number, $0.numerator, $0.denominator] })
        try lab("downbeat.lab", downbeats.map { [$0] })
        try lab("key.lab", keys.map { [$0.start, $0.end, $0.label] })
        try lab("chord.lab", chords.map { [$0.start, $0.end, $0.label] })
        try lab("structure.lab", sections.map { [$0.start, $0.end, $0.label] })

        let parts = (0..<2).map { track in
            notes.filter { $0.track == track }.map { MIDIFile.Note(pitch: $0.pitch, start: $0.start, end: $0.end, velocity: 100) }
        }
        let vocal = MIDIFile.Track(name: "Vocal", notes: parts[0])
        let instrumental = MIDIFile.Track(name: "Ins", notes: parts[1])
        let chordTrack = MIDIFile.Track(name: "Chords", notes: chordNotes)
        for (name, tracks) in [
            ("melody_vocal.mid", [vocal]), ("melody_instrumental.mid", [instrumental]), ("chords.mid", [chordTrack]),
            ("transcription.mid", [vocal, instrumental, chordTrack]),
        ] {
            try MIDIFile.data(tracks).write(to: directory.appending(path: name))
        }
    }

    /// Each chord's pitches held for its span, struck again at every bar line inside it.
    var chordNotes: [MIDIFile.Note] {
        let downbeats = downbeats
        return chords.flatMap { chord -> [MIDIFile.Note] in
            let (start, end) = (max(0, chord.start), min(duration, chord.end))
            let cuts = [start] + downbeats.filter { $0 > start && $0 < end } + [end]
            return zip(cuts, cuts.dropFirst()).filter { $1 > $0 }.flatMap { a, b in
                Spelling.pitches(ofChord: chord.label).map { MIDIFile.Note(pitch: $0, start: a, end: b, velocity: 48) }
            }
        }
    }
}

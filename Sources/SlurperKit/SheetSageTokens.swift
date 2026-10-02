/// SheetSage2's token vocabulary (schema v1: 300 s windows, 100 time steps a second) and the event grammar its
/// decoding is constrained to, ported from `tokenization_sheetsage2.py` and `generation_sheetsage2.py`.
///
/// A sequence is `<|sos|>`, the task prompts, `<|out|>`, then events, each a run of subbeat shifts followed by
/// fields in order: a timestamp, the rhythm (an optional meter, then the eighth-note position in the bar), a
/// section label, a key, a chord, and melody notes (a pitch, then an optional duration).
enum SheetSageTokens {
    static let pad = 0, sos = 1, eos = 2, out = 3
    /// `<|sos|>`, the prompts slurper asks for (timestamp, downbeat_meter, structure, key, chord_full,
    /// melody_full), `<|out|>`.
    static let prompt = [sos, 4, 5, 6, 7, 9, 11, out]

    /// Shifts of 0...256 subbeats.
    static let shift = 260..<517
    /// 0.00 to 299.99 s into the window.
    static let time = 517..<30_517
    /// Numerators 1...32, each with denominators 1, 2, 4, 8, 16, 32.
    static let meter = 30_517..<30_709
    /// Eighth notes since the bar line.
    static let eighth = 30_709..<30_965
    static let structure = 30_965..<30_988
    /// Major keys on C...B, then minor keys.
    static let key = 30_988..<31_012
    /// Maj/min chords, which slurper's prompts never produce.
    static let majminChord = 31_012..<31_037
    static let chord = 31_037..<31_398
    /// MIDI pitches 0...127 on the vocal track, then on the instrumental track.
    static let pitch = 31_398..<31_654
    static let duration = 31_654..<31_678
    static let count = 31_678
    static let timeHz = 100.0

    static let sharps = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    static let sections = [
        "silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus", "post-chorus", "interlude", "fade-out",
        "loop", "rap", "preshot", "irregular", "instrumental", "intro and verse", "pre-chorus and chorus",
        "verse and pre-chorus", "solo", "theme", "development", "variation", "pre-outro",
    ]
    /// Note lengths in subbeats, sixteenth notes.
    static let durations = [
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
    ]

    /// "N", then every quality on every root, each root's inversions before its root position.
    static let chords: [String] = {
        let qualities: [(String, [String])] = [
            ("maj", ["/2", "/3", "/5"]), ("min", ["/2", "/b3", "/5"]), ("dim", []), ("aug", []),
            ("maj7", ["/3", "/5", "/7"]), ("min7", ["/b3", "/5", "/b7"]), ("7", ["/3", "/5", "/b7"]), ("hdim7", []),
            ("dim7", []), ("minmaj7", []), ("sus2", []), ("sus4", []), ("sus4(b7)", []), ("maj6", []), ("min6", []),
        ]
        return ["N"] + qualities.flatMap { quality, inversions in
            sharps.flatMap { root in (inversions + [""]).map { "\(root):\(quality)\($0)" } }
        }
    }()

    enum Field: Int, CaseIterable, Comparable {
        case timestamp, rhythm, structure, key, chord, melody

        static func < (a: Field, b: Field) -> Bool { a.rawValue < b.rawValue }
    }

    static func field(of token: Int) -> Field? {
        switch token {
        case time: .timestamp
        case meter, eighth: .rhythm
        case structure: .structure
        case key: .key
        case majminChord, chord: .chord
        case pitch, duration: .melody
        default: nil
        }
    }

    /// Shift tokens for `subbeats`, in steps of at most 256.
    static func shifts(_ subbeats: Int) -> [Int] {
        var left = subbeats
        var tokens: [Int] = []
        while left > shift.count - 1 {
            tokens.append(shift.upperBound - 1)
            left -= shift.count - 1
        }
        return tokens + [shift.lowerBound + left]
    }

    static func timeToken(seconds: Double) -> Int {
        time.lowerBound + min(max(Int((seconds * timeHz).rounded()), 0), time.count - 1)
    }

    /// The tokens that may come next, as `PromptGrammarState` allows them.
    struct Grammar {
        private var inShift = true
        private var shiftRun = 0
        private var payload = 0
        private var lastField: Field?
        private var waiting: Waiting?

        /// A meter waits for its eighth-note position; a pitch may take a duration or another pitch.
        private enum Waiting { case eighth, duration }

        func allowed() -> [Range<Int>] {
            var ranges: [Range<Int>] = []
            if payload > 0 { ranges.append(eos..<eos + 1) }
            if (payload > 0 || inShift) && shiftRun < 4 { ranges.append(shift) }
            switch waiting {
            case .eighth: return ranges + [eighth]
            case .duration: return ranges + [duration, pitch]
            case nil: break
            }
            let last = lastField?.rawValue ?? -1
            if last < Field.timestamp.rawValue { ranges.append(time) }
            if last < Field.rhythm.rawValue { ranges += [meter, eighth] }
            if last < Field.structure.rawValue { ranges.append(structure) }
            if last < Field.key.rawValue { ranges.append(key) }
            if last < Field.chord.rawValue { ranges.append(chord) }
            ranges.append(pitch)
            return ranges
        }

        /// Takes `token` and returns whether the sequence has ended.
        mutating func update(_ token: Int) -> Bool {
            if token == eos { return true }
            if shift.contains(token) {
                if !inShift && payload > 0 { (payload, lastField, waiting) = (0, nil, nil) }
                inShift = true
                shiftRun += 1
                return false
            }
            (inShift, shiftRun) = (false, 0)
            payload += 1
            lastField = SheetSageTokens.field(of: token)
            waiting = meter.contains(token) ? .eighth : pitch.contains(token) ? .duration : nil
            return false
        }
    }

    /// One event of a window's sequence: its position in subbeats since the window's first event and its tokens
    /// by field. Stitched events also carry their time in the song.
    struct Event: Equatable {
        var subbeat: Int
        var tokens: [Field: [Int]]
        var time = 0.0
        var globalSubbeat = 0
        /// For melody notes, the end of each note in the song, in seconds.
        var noteEnds: [Double] = []

        var timestamp: Double? { tokens[.timestamp].map { Double($0[0] - SheetSageTokens.time.lowerBound) / timeHz } }
        var meter: (numerator: Int, denominator: Int)? {
            guard let token = tokens[.rhythm]?.first(where: SheetSageTokens.meter.contains) else { return nil }
            let index = token - SheetSageTokens.meter.lowerBound
            return (index / 6 + 1, 1 << (index % 6))
        }
        var eighthPosition: Int? {
            tokens[.rhythm]?.first(where: eighth.contains).map { $0 - eighth.lowerBound }
        }
        var section: String? { tokens[.structure].map { sections[$0[0] - structure.lowerBound] } }
        /// Spelled the way SheetSage2 names keys, as "Eb:major" rather than "D#:major".
        var key: String? {
            tokens[Field.key].map { token in
                let index = token[0] - SheetSageTokens.key.lowerBound
                return Spelling.keyName(tonic: index % 12, minor: index >= 12)
            }
        }
        /// As the model predicts it, with sharps: "A#:7/3".
        var chord: String? {
            tokens[.chord].map { token in
                SheetSageTokens.chord.contains(token[0])
                    ? chords[token[0] - SheetSageTokens.chord.lowerBound]
                    : token[0] == majminChord.lowerBound ? "N" : majminChords[token[0] - majminChord.lowerBound - 1]
            }
        }
        /// The notes starting here: MIDI pitch, track (0 vocal, 1 instrumental) and length in subbeats.
        var notes: [(pitch: Int, track: Int, subbeats: Int)] {
            guard let melody = tokens[.melody] else { return [] }
            var notes: [(Int, Int, Int)] = []
            var index = 0
            while index < melody.count {
                let id = melody[index] - pitch.lowerBound
                var bin = 0
                if index + 1 < melody.count, duration.contains(melody[index + 1]) {
                    bin = melody[index + 1] - duration.lowerBound
                    index += 2
                } else {
                    index += 1
                }
                notes.append((id % 128, id >= 128 ? 1 : 0, durations[bin]))
            }
            return notes
        }

        static func == (a: Event, b: Event) -> Bool {
            a.subbeat == b.subbeat && a.tokens == b.tokens && a.time == b.time && a.globalSubbeat == b.globalSubbeat
                && a.noteEnds == b.noteEnds
        }
    }

    private static let majminChords = sharps.map { "\($0):maj" } + sharps.map { "\($0):min" }

    /// The events after `<|out|>`, up to `<|eos|>`, as `decode_sequence(strict=False)` reads them. Tokens outside
    /// any field are skipped, and a payload with no shift before it starts an event where the last one was.
    static func events(_ tokens: [Int]) -> [Event] {
        guard let start = tokens.firstIndex(of: out) else { return [] }
        var events: [Event] = []
        var subbeat = 0
        var current: Event?
        for token in tokens[(start + 1)...] {
            if token == eos { break }
            if shift.contains(token) {
                if let event = current { events.append(event) }
                current = nil
                subbeat += token - shift.lowerBound
                continue
            }
            guard let field = field(of: token) else { continue }
            if current == nil { current = Event(subbeat: subbeat, tokens: [:]) }
            current?.tokens[field, default: []].append(token)
        }
        if let event = current { events.append(event) }
        return events
    }

    /// The prompt, then `events` re-encoded (`encode_decoded_sequence` without `<|eos|>`).
    static func encode(_ events: [Event]) -> [Int] {
        var tokens = prompt
        var previous = 0
        for event in events {
            tokens += shifts(event.subbeat - previous)
            previous = event.subbeat
            for field in Field.allCases { tokens += event.tokens[field] ?? [] }
        }
        return tokens
    }
}

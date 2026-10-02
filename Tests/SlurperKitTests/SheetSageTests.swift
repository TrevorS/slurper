import Foundation
import Testing
@testable import SlurperKit

struct SheetSageTests {
    /// The start of the golden window's tokens: the prompt, then a 4/4 bar line at 0.00 s in a verse in Ab major
    /// under G#:maj7, a beat at 0.35 s, and a beat at 1.06 s with a vocal Ab4 two subbeats long.
    static let opening = [
        1, 4, 5, 6, 7, 9, 11, 3, 260, 517, 30537, 30709, 30968, 30996, 31193, 264, 552, 30709, 31193, 264, 588, 30711,
        264, 623, 30713, 31466, 31655, 262, 31466, 31658,
    ]

    @Test func vocabularyMatchesTheCheckpoint() {
        #expect(SheetSageTokens.count == 31_678)
        #expect(SheetSageTokens.chords.count == SheetSageTokens.chord.count)
        #expect(SheetSageTokens.chords[156] == "G#:maj7")
        #expect(SheetSageTokens.duration.upperBound == SheetSageTokens.count)
        #expect(SheetSageTokens.shifts(0) == [260])
        #expect(SheetSageTokens.shifts(600) == [516, 516, 348])
    }

    @Test func decodesEvents() throws {
        let events = SheetSageTokens.events(Self.opening + [SheetSageTokens.eos, 260, 517])
        #expect(events.map(\.subbeat) == [0, 4, 8, 12, 14])
        let first = events[0]
        #expect(first.timestamp == 0)
        #expect(first.meter! == (4, 4))
        #expect(first.eighthPosition == 0)
        #expect(first.section == "verse")
        #expect(first.key == "Ab:major")
        #expect(first.chord == "G#:maj7")
        #expect(events[3].timestamp == 1.06)
        #expect(events[3].eighthPosition == 4)
        #expect(events[3].notes.map(\.pitch) == [68] && events[3].notes.map(\.subbeats) == [2])
        #expect(events[4].notes.map(\.subbeats) == [6])
        #expect(SheetSageTokens.encode(events) == Self.opening)
    }

    @Test func grammarAllowsOnlyWellFormedEvents() {
        var grammar = SheetSageTokens.Grammar()
        func allows(_ token: Int) -> Bool { grammar.allowed().contains { $0.contains(token) } }
        #expect(allows(260) && allows(517) && !allows(SheetSageTokens.eos))
        _ = grammar.update(260)
        _ = grammar.update(517)
        #expect(!allows(517) && allows(30537) && allows(31466) && allows(SheetSageTokens.eos))
        _ = grammar.update(30537)
        #expect(allows(30709) && !allows(30968) && !allows(31466))
        _ = grammar.update(30709)
        _ = grammar.update(31466)
        #expect(allows(31655) && allows(31466) && !allows(30996))
        for _ in 0..<4 { _ = grammar.update(260) }
        #expect(!allows(260))
        let ended = grammar.update(SheetSageTokens.eos)
        #expect(ended)
    }

    @Test func spellsKeysAndChordsLikeSheetSage2() {
        let major = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        let minor = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
        #expect((0..<12).map { Spelling.keyName(tonic: $0, minor: false) } == major.map { "\($0):major" })
        #expect((0..<12).map { Spelling.keyName(tonic: $0, minor: true) } == minor.map { "\($0):minor" })

        // chord_spelling_sheetsage2's tests.
        for (raw, expected) in [("D#:maj", "Eb:maj"), ("G#:maj", "Ab:maj"), ("A#:7/3", "Bb:7/3"),
                                ("D#:min7/b7", "Eb:min7/b7"), ("C:min", "C:min"), ("N", "N"), ("X", "X")] {
            #expect(Spelling.chord(raw, inKey: "C:minor") == expected)
        }
        for (raw, key, expected) in [("C:maj", "E:major", "C:maj"), ("A#:maj", "G:major", "Bb:maj"),
                                     ("D#:maj", "G:major", "Eb:maj"), ("G#:7", "C:major", "Ab:7"),
                                     ("A#:min", "C:major", "Bb:min"), ("C#:hdim7", "G:major", "C#:hdim7"),
                                     ("F:hdim7", "G#:minor", "E#:hdim7")] {
            #expect(Spelling.chord(raw, inKey: key) == expected, "\(raw) in \(key)")
        }
        // Spelling never changes what sounds.
        for key in ["C:minor", "E:major", "A#:minor", "C#:major", "F#:major"] {
            for chord in SheetSageTokens.chords {
                #expect(Spelling.pitches(ofChord: Spelling.chord(chord, inKey: key)) == Spelling.pitches(ofChord: chord))
            }
        }
    }

    @Test func chordPitchesPutTheBassBelow() {
        #expect(Spelling.pitches(ofChord: "C:maj") == [36, 48, 52, 55])
        #expect(Spelling.pitches(ofChord: "C:maj/3") == [40, 48, 52, 55])
        #expect(Spelling.pitches(ofChord: "Bb:7/b7") == [44, 58, 62, 65, 68])
        #expect(Spelling.pitches(ofChord: "N").isEmpty)
    }

    /// sliding_window_plan(duration, 300, 200, 100).
    @Test func plansWindowsLikeSheetSage2() {
        #expect(SheetSage.windows(duration: 255).map(\.start) == [0])
        #expect(SheetSage.windows(duration: 255)[0].generationStop == nil)
        let plan = SheetSage.windows(duration: 360)
        #expect(plan == [
            .init(start: 0, end: 300, acceptStart: 0, acceptEnd: 200, prefixEnd: 0, generationStop: 200),
            .init(start: 60, end: 360, acceptStart: 200, acceptEnd: 360, prefixEnd: 200, generationStop: nil),
        ])
        #expect(SheetSage.windows(duration: 650).map(\.start) == [0, 100, 200, 300, 350])
    }

    @Test func timesSubbeatsThroughTheTimestamps() {
        let events = SheetSageTokens.events(Self.opening)
        let time = SheetSage.timeMap(events)
        #expect(abs(time(4) - 0.35) < 1e-9)
        #expect(abs(time(6) - 0.53) < 1e-9)
        // Past the last timestamp the median rate, 0.0875 s a subbeat, carries on.
        #expect(abs(time(16) - (1.06 + 4 * 0.0875)) < 1e-9)
        #expect(SheetSage.timeMap([])(16) == 2)
    }

    @Test func carriesContextIntoTheNextWindow() throws {
        var events = SheetSage.stitch(SheetSageTokens.events(Self.opening), window: SheetSage.windows(duration: 400)[0],
                                      duration: 400, base: 0)
        #expect(events.map(\.time) == [0, 0.35, 0.71, 1.06, 1.06 + 2 * 0.0875])
        // A later window starting at 0.5 s begins with the beat at 0.71 s, re-timed to 0.21 s, under the key,
        // section, chord and meter in force before it.
        events = events.map { var event = $0; event.globalSubbeat = event.subbeat; return event }
        let (tokens, base) = try #require(SheetSage.overlapPrefix(events, start: 0.5, end: 2))
        #expect(base == 8)
        #expect(Array(tokens.prefix(16)) == SheetSageTokens.prompt + [260, 538, 30537, 30711, 30968, 30996, 31193, 264])
    }

    @Test func spellsChordsInTheirLocalKey() {
        let gSharp = SheetSageTokens.chord.lowerBound + SheetSageTokens.chords.firstIndex(of: "G#:maj")!
        func event(_ time: Double, key: Int? = nil, chord: Bool = true) -> SheetSageTokens.Event {
            var tokens: [SheetSageTokens.Field: [Int]] = chord ? [.chord: [gSharp]] : [:]
            if let key { tokens[.key] = [SheetSageTokens.key.lowerBound + key] }
            return SheetSageTokens.Event(subbeat: 0, tokens: tokens, time: time)
        }
        // chord_spelling_sheetsage2's test: C minor until 4 s, then E major, and the chord from 2 to 6 s, whose
        // midpoint is on the change, takes C minor.
        let song = Transcription(events: [event(0, key: 12), event(2), event(4, key: 4, chord: false), event(6)], duration: 8)
        #expect(song.chords.map(\.label) == ["Ab:maj", "Ab:maj", "G#:maj"])
        #expect(song.keys.map(\.label) == ["C:minor", "E:major"])
    }

    @Test func writesMIDILikePrettyMIDI() {
        #expect(MIDIFile.variableLength(0) == [0])
        #expect(MIDIFile.variableLength(0x80) == [0x81, 0x00])
        #expect(MIDIFile.variableLength(0x0FFF_FFFF) == [0xFF, 0xFF, 0xFF, 0x7F])
        let data = MIDIFile.data([.init(name: "Vocal", notes: [.init(pitch: 60, start: 0.5, end: 1, velocity: 100)])])
        #expect(Array(data.prefix(14)) == Array("MThd".utf8) + [0, 0, 0, 6, 0, 1, 0, 2, 0x03, 0xC0])
        // Note on at 960 ticks, off 960 later.
        #expect(data.range(of: Data([0x87, 0x40, 0x90, 60, 100, 0x87, 0x40, 0x80, 60, 0])) != nil)
    }

    /// Decodes the golden window written next to the models by `scripts/convert_sheetsage2.py` and requires
    /// PyTorch's tokens.
    @Test(.enabled(if: sheetSageInstalled))
    func goldenWindowMatchesPyTorch() async throws {
        let folder = StemSplitter.sheetSageEncoder.package.deletingLastPathComponent()
        let audio = try Data(contentsOf: folder.appending(path: "golden_audio.f32")).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        let expected = try JSONDecoder().decode([Int].self, from: Data(contentsOf: folder.appending(path: "golden_tokens.json")))
        let model = try await SheetSage(encoder: StemSplitter.sheetSageEncoder.package, decoder: StemSplitter.sheetSageDecoder.package)
        let samples = audio + [Float](repeating: 0, count: SheetSage.windowSamples - audio.count)
        let tokens = try await model.generate(samples, prefix: SheetSageTokens.prompt, stop: Double(audio.count) / SheetSage.sampleRate) { _ in }
        #expect(tokens == expected)
    }
}

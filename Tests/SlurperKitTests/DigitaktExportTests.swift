import AVFoundation
import Testing
@testable import SlurperKit

struct DigitaktExportTests {
    @Test func writes48kHz16BitStereo() throws {
        let folder = try temporaryDirectory()
        try StemSplitter.exportDigitakt("drums", noise(frames: 44_100, channels: 2), title: "Test Song", sampleRate: 44_100, to: folder)

        let file = try AVAudioFile(forReading: folder.appending(path: "Digitakt II/Test_Song_drums.wav"))
        let format = file.fileFormat.streamDescription.pointee
        #expect(format.mSampleRate == 48_000)
        #expect(format.mBitsPerChannel == 16)
        #expect(format.mChannelsPerFrame == 2)
        #expect(format.mFormatFlags & kAudioFormatFlagIsFloat == 0)
        #expect(abs(file.length - 48_000) <= 2)
    }

    @Test func foldsIdenticalChannelsToMono() throws {
        let folder = try temporaryDirectory()
        let channel = noise(frames: 44_100, channels: 1)[0]
        try StemSplitter.exportDigitakt("bass", [channel, channel], title: "Mono", sampleRate: 44_100, to: folder)

        let file = try AVAudioFile(forReading: folder.appending(path: "Digitakt II/Mono_bass.wav"))
        #expect(file.fileFormat.channelCount == 1)
    }

    @Test func keepsStemsLongerThanFiveMinutesInOneFile() throws {
        let folder = try temporaryDirectory()
        let frames = 44_100 * 301
        let stem = [[Float](repeating: 0.25, count: frames), [Float](repeating: -0.25, count: frames)]
        try StemSplitter.exportDigitakt("other", stem, title: "Long", sampleRate: 44_100, to: folder)

        let directory = folder.appending(path: "Digitakt II")
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["Long_other.wav"])
        let file = try AVAudioFile(forReading: directory.appending(path: "Long_other.wav"))
        #expect(abs(file.length - 48_000 * 301) <= 2)
    }

    @Test func writesChopsAtTheSameTimesAsTheStem() throws {
        let folder = try temporaryDirectory()
        let stem = noise(frames: 44_100, channels: 2)
        let pieces = [Chops.Piece(name: "01", range: 4_410..<22_050), Chops.Piece(name: "02", range: 22_050..<44_100)]
        let kit = Chops(folder: "kit", pieces: pieces, fadeIn: 0.003, fadeOut: 0.005)
        try kit.write(stem, sampleRate: 44_100, prefix: "drums", in: folder)
        try StemSplitter.exportDigitakt("drums", stem, title: "Test Song", sampleRate: 44_100, chops: [kit], to: folder)

        let native = folder.appending(path: "drums_kit")
        #expect(try FileManager.default.contentsOfDirectory(atPath: native.path).sorted() == ["drums_01.wav", "drums_02.wav"])
        let piece = try AudioFiles.readStereo(native.appending(path: "drums_01.wav"), sampleRate: 44_100)
        #expect(piece[0].count == 17_640)
        // Past the 3 ms fade-in, the piece is the stem from sample 4,410 on.
        #expect(piece[0][200] == stem[0][4_610] && piece[1][9_000] == stem[1][13_410])

        let digitakt = folder.appending(path: "Digitakt II/Test_Song_drums_kit")
        #expect(try FileManager.default.contentsOfDirectory(atPath: digitakt.path).sorted() == ["Test_Song_drums_01.wav", "Test_Song_drums_02.wav"])
        let first = try AVAudioFile(forReading: digitakt.appending(path: "Test_Song_drums_01.wav"))
        #expect(first.length == 19_200)
        #expect(first.fileFormat.streamDescription.pointee.mBitsPerChannel == 16)
    }

    @Test(arguments: [
        ("Big Buck Bunny 60fps 4K - Official Blender Foundation Short Film", "Big_Buck_Bunny_60fps"),
        ("Café: Live!", "Caf_Live"),
        ("???", "song"),
    ])
    func filenamePrefix(title: String, prefix: String) {
        #expect(StemSplitter.digitaktPrefix(title) == prefix)
    }
}

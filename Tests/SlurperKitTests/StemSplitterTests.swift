import Foundation
import Testing
@testable import SlurperKit

struct StemSplitterTests {
    @Test func parsesYTDLPLines() {
        #expect(StemSplitter.parse(ytdlpLine: "slurper-title\tSong") == .title("Song"))
        #expect(StemSplitter.parse(ytdlpLine: "slurper-file\t/tmp/source.wav") == .file("/tmp/source.wav"))
        #expect(StemSplitter.parse(ytdlpLine: "slurper-progress\t50\t200") == .progress(0.25))
        #expect(StemSplitter.parse(ytdlpLine: "slurper-progress\t50\tNA") == nil)
        #expect(StemSplitter.parse(ytdlpLine: "[youtube] aqz-KE-bpKQ: Downloading webpage") == nil)
    }

    @Test func uniqueFolderCleansTitleAndNumbersRepeats() throws {
        let root = try temporaryDirectory()
        let first = try StemSplitter.uniqueFolder(in: root, named: "AC/DC: Live.")
        let second = try StemSplitter.uniqueFolder(in: root, named: "AC/DC: Live.")
        #expect(first.lastPathComponent == "AC-DC- Live")
        #expect(second.lastPathComponent == "AC-DC- Live 2")
        #expect(try StemSplitter.uniqueFolder(in: root, named: " ... ").lastPathComponent == "untitled")
    }
}

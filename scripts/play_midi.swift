#!/usr/bin/env swift
// Plays a MIDI file through macOS's built-in General MIDI sounds, such as slurper's transcription.mid.
//
// Usage: swift scripts/play_midi.swift FILE.mid

import AVFoundation

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: play_midi.swift FILE.mid\n".utf8))
    exit(2)
}
let bank = URL(filePath: "/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls")
let player = try AVMIDIPlayer(contentsOf: URL(filePath: CommandLine.arguments[1]), soundBankURL: bank)
player.prepareToPlay()
print("playing \(Int(player.duration)) s, ctrl-c to stop")
let finished = DispatchSemaphore(value: 0)
player.play { finished.signal() }
finished.wait()

import Foundation
import ScreenCaptureKit
import CoreMedia

class AudioOutput: NSObject, SCStreamOutput {
    let fileHandle = FileHandle.standardOutput
    var errorLogged = false

    var firstFrame = true

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        
        if firstFrame {
            if let format = CMSampleBufferGetFormatDescription(sampleBuffer) {
                if let desc = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee {
                    let isFloat = (desc.mFormatFlags & kAudioFormatFlagIsFloat) != 0
                    let header = "RATE:\(Int(desc.mSampleRate))\nCHANNELS:\(desc.mChannelsPerFrame)\nBITS:\(desc.mBitsPerChannel)\nFLOAT:\(isFloat ? 1 : 0)\n"
                    if let headerData = header.data(using: .utf8) {
                        fileHandle.write(headerData)
                    }
                }
            }
            firstFrame = false
        }

        guard let dataBuffer = sampleBuffer.dataBuffer else {
            if !errorLogged { fputs("Error: sampleBuffer has no dataBuffer\n", stderr); errorLogged = true }
            return
        }
        
        var lengthAtOffset: Int = 0
        var totalLength: Int = 0
        var dataPointer: UnsafeMutablePointer<Int8>?
        
        let status = CMBlockBufferGetDataPointer(dataBuffer, atOffset: 0, lengthAtOffsetOut: &lengthAtOffset, totalLengthOut: &totalLength, dataPointerOut: &dataPointer)
        
        if status == kCMBlockBufferNoErr, let dataPointer = dataPointer, totalLength > 0 {
            fileHandle.write(Data(bytes: dataPointer, count: totalLength))
        } else {
            if !errorLogged { fputs("Error: CMBlockBufferGetDataPointer failed with status \(status)\n", stderr); errorLogged = true }
        }
    }
}

// MUST retain SCStream and output globally so ARC doesn't destroy them instantly
var globalStream: SCStream?
var globalOutput: AudioOutput?

func startCapture() {
    SCShareableContent.getExcludingDesktopWindows(false, onScreenWindowsOnly: false) { content, error in
        guard let content = content, error == nil else {
            fputs("Error getting SCShareableContent: \(String(describing: error))\n", stderr)
            exit(1)
        }
        guard let display = content.displays.first else {
            fputs("No display found\n", stderr)
            exit(1)
        }

        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let config = SCStreamConfiguration()
        config.width = display.width
        config.height = display.height
        config.showsCursor = false
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 16000
        config.channelCount = 1

        globalOutput = AudioOutput()
        globalStream = SCStream(filter: filter, configuration: config, delegate: nil)
        
        do {
            try globalStream!.addStreamOutput(globalOutput!, type: .audio, sampleHandlerQueue: DispatchQueue(label: "sck_audio_queue"))
            globalStream!.startCapture { error in
                if let error = error {
                    fputs("Capture failed to start: \(error)\n", stderr)
                    exit(1)
                }
                fputs("SCK Capture started: 16kHz Mono PCM streaming to stdout\n", stderr)
            }
        } catch {
            fputs("Failed to add stream output: \(error)\n", stderr)
            exit(1)
        }
    }
}

startCapture()
RunLoop.main.run()

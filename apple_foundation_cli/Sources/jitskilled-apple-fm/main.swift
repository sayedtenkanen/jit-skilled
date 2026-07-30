// jitskilled-apple-fm: a minimal CLI bridge between jitskilled (Python)
// and Apple's on-device FoundationModels framework.
//
// Contract: reads ONE JSON object from stdin --
//   {"system": "...", "user": "...", "max_tokens": 800}
// and writes ONE JSON object to stdout on success --
//   {"text": "..."}
// or writes a human-readable error to stderr and exits non-zero on failure.
//
// Built against the FoundationModels API as publicly documented following
// WWDC 2025 (macOS 26 "Tahoe" / Xcode 26). Apple's API surface here is
// young and may have moved since -- if this fails to compile against your
// SDK, check Apple's current FoundationModels documentation for the
// current shape of LanguageModelSession / GenerationOptions and adjust
// accordingly; the stdin/stdout JSON contract above is the part the
// Python side depends on and should be kept stable.

import Foundation
import FoundationModels

struct CLIRequest: Codable {
    let system: String
    let user: String
    let max_tokens: Int?
}

struct CLIResponse: Codable {
    let text: String
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

@main
struct JitskilledAppleFM {
    static func main() async {
        let stdinData = FileHandle.standardInput.readDataToEndOfFile()
        guard !stdinData.isEmpty else {
            fail("No input on stdin. Expected JSON: "
                 + "{\"system\": ..., \"user\": ..., \"max_tokens\": ...}")
        }

        let request: CLIRequest
        do {
            request = try JSONDecoder().decode(CLIRequest.self, from: stdinData)
        } catch {
            fail("Could not parse stdin as JSON: \(error)")
        }

        let availability = SystemLanguageModel.default.availability
        guard case .available = availability else {
            fail("Apple on-device model is not available (\(availability)). "
                 + "Check Settings > Apple Intelligence & Siri, and that the "
                 + "on-device model has finished downloading.")
        }

        do {
            let session = LanguageModelSession(instructions: request.system)
            let options = GenerationOptions(maxTokens: request.max_tokens ?? 800)
            let response = try await session.respond(to: request.user, options: options)

            let output = CLIResponse(text: response.content)
            let outData = try JSONEncoder().encode(output)
            FileHandle.standardOutput.write(outData)
            FileHandle.standardOutput.write("\n".data(using: .utf8)!)
        } catch {
            fail("Foundation Models request failed: \(error)")
        }
    }
}
